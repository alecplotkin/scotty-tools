import logging
import anndata as ad
import numpy as np
import numpy.typing as npt
import pandas as pd
import igraph as ig
import leidenalg
import warnings
from typing import Dict, Tuple, TypeVar
from time import perf_counter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsTransformer, KNeighborsClassifier
from src.sctrat.models.trajectory import (
    OTModel, coarsen_ot_model
)
from src.sctrat.tools.trajectories import compute_trajectories
from sketchKH import sketch
from scipy.sparse import csr_matrix, coo_matrix
from numba.core.errors import NumbaDeprecationWarning

logger = logging.getLogger('tracter_kh')
logger.handlers.clear()
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s")
)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


IntArray = TypeVar('IntArray', bound=npt.NDArray[np.int32 | np.int64])
FloatArray = TypeVar('FloatArray', bound=npt.NDArray[np.float32 | np.float64])


class TRACTER_knn:
    """TRAjectory Clustering using Timepoint Embedding Representations.

    This version uses k-nearest neighbors and leiden clusters to embed time
    points.

    This method creates single cell representations at each timepoint,
    and then samples cells from the embedding space and approximates their
    trajectories. This results in a trajectory matrix, which is used to fit
    clusters of related trajectories.

    The default embedding and clustering methods are both Gaussian Mixture
    Models.

    Args:
        rep_dim (int): dimension of the embedding space at each time point.
        n_clusters (int): number of clusters in the trajectory clustering model.
    """

    def __init__(
        self,
        rep_resolution: float = 1,
        n_neighbors: int = 20,
        n_clusters: int = 10,
        n_subsamples: int = 200,
        random_state: int = None,
        scale_trajectory_reps: bool = True,
    ):
        self.rep_resolution = rep_resolution
        self.n_neighbors = n_neighbors
        self.n_clusters = n_clusters
        self.n_subsamples = n_subsamples
        self.random_state = random_state
        self.scale_trajectory_reps = scale_trajectory_reps

    def fit(self, ot_model: OTModel, data: ad.AnnData):
        """Fit TRACTER using trajectory model and single-cell data."""

        self.time_var = ot_model.time_var
        logger.info('Fitting time point embeddings...')
        _, data_train = sketch(
            data,
            sample_set_key=self.time_var,
            num_subsamples=self.n_subsamples,
            frequency_seed=self.random_state,
        )
        self.ix_train = data_train.obs_names
        embedding_models = dict()
        time_embeddings = dict()
        for time in ot_model.timepoints:
            embedding_models[time] = self.fit_embeddings(
                data_train[data_train.obs[self.time_var] == time]
            )
        self.embedding_models = embedding_models
        time_embeddings = self.embed_data(data)
        logger.info('Done')

        logger.info('Computing trajectory representations...')
        trajectory_matrix = self.compute_trajectory_matrix(
                ot_model, time_embeddings
        )
        self.trajectory_matrix = trajectory_matrix
        logger.info('Done')

        logger.info('Fitting trajectory clusters...')
        cluster_model = self.fit_clusters(
            trajectory_matrix[self.ix_train],
            scale_trajectory_reps=self.scale_trajectory_reps,
        )
        self.cluster_model = cluster_model
        logger.info('Done')

    def predict(self, data: ad.AnnData) -> pd.Series:
        """Predict trajectory cluster assignments for data."""
        X = self.trajectory_matrix[data.obs_names, :].to_df()
        preds = self.cluster_model.predict(X)
        preds = pd.Series(preds, index=X.index)
        return preds

    def fit_embeddings(
            self, data: ad.AnnData, cluster_suffix: str = None
    ) -> Dict:
        """Fit time point embeddings using a GMM.

        Args:
            data (ad.AnnData): data to fit embeddings on.

        Returns:
            dict: dictionary of fitted time point GMMs.
        """

        X = data.obsm['X_pca']
        # This assumes that the number of data points is reasonably small.
        knn = KNeighborsTransformer(n_neighbors=self.n_neighbors)
        distances = knn.fit_transform(X)
        knn_ix, knn_dist = _get_ix_distances_from_sparse_matrix(
            distances, self.n_neighbors
        )
        connectivities = umap(
            knn_ix,
            knn_dist,
            n_obs=data.shape[0],
            n_neighbors=self.n_neighbors
        )
        g = get_igraph_from_adjacency(connectivities, directed=True)
        weights = np.array(g.es['weight']).astype(np.float64)
        part = leidenalg.find_partition(
            g,
            partition_type=leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=self.rep_resolution,
            n_iterations=2,
            weights=weights,
            seed=self.random_state,
        )
        clusters = pd.Series(part.membership, dtype=str)
        if cluster_suffix is not None:
            clusters += cluster_suffix
        knn_class = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights='distance',
        )
        knn_class.fit(X, clusters)
        return knn_class
        # TODO: format leiden clusters, fit KNeighborsClassifier to embed new data.

    def embed_data(self, data: ad.AnnData) -> Dict:
        """Embed each timepoint in data using fitted embedding models.

        Args:
            data (ad.AnnData): data to embed.

        Returns:
            dict: dictionary of embedding dataframes for each timepoint.
        """

        embedding_dict = dict()
        for tp, df in data.obs.groupby(self.time_var):
            model = self.embedding_models[tp]
            rep = model.predict_proba(data[df.index].obsm['X_pca'])
            rep = pd.DataFrame(rep, index=df.index)
            embedding_dict[tp] = rep
        return embedding_dict

    def compute_trajectory_matrix(self, ot_model: OTModel, timepoint_embeddings: Dict):
        """Compute matrix of trajectories with all timepoints."""

        # Need to get subset names from coarsened ot_model in order for
        # trajectory var_names to make sense.
        meta = ot_model.meta
        traj_by_ref_tp = dict()
        for tp in ot_model.timepoints:
            ix_day = meta.index[meta[self.time_var] == tp]
            emb = timepoint_embeddings[tp].loc[ix_day, :]
            traj = compute_trajectories(
                ot_model, emb, ref_time=tp,
                normalize=True,  norm_strategy='expected_value',
            )
            traj_by_ref_tp[tp] = traj
        traj_matrix = ad.concat(
            traj_by_ref_tp.values(),
            axis=1,
            keys=traj_by_ref_tp.keys(),
            label=self.time_var
        )
        return traj_matrix

    def fit_clusters(
            self,
            trajectory_matrix: ad.AnnData,
            scale_trajectory_reps: bool = True,
    ) -> Pipeline:
        """Fit clusters on trajectory representations."""
        X = trajectory_matrix.to_df()
        pipeline_steps = []
        if scale_trajectory_reps:
            pipeline_steps.append(('scaler', StandardScaler()))
        pipeline_steps.append(('cluster', GaussianMixture(
                n_components=self.n_clusters, random_state=self.random_state
            )
        ))
        model = Pipeline(pipeline_steps)
        model.fit(X)
        return model


def _get_ix_distances_from_sparse_matrix(
        D: csr_matrix, n_neighbors: int
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Stolen from scanpy.neighbors, does what the function says."""

    if (shortcut := _ind_dist_shortcut(D)) is not None:
        indices, distances = shortcut
    else:
        indices, distances = _ind_dist_slow(D, n_neighbors)

    # handle RAPIDS style indices_distances lacking the self-column
    if not _has_self_column(indices, distances):
        indices = np.hstack([np.arange(indices.shape[0])[:, None], indices])
        distances = np.hstack([np.zeros(distances.shape[0])[:, None], distances])

    # If using the shortcut or adding the self column resulted in too many neighbors,
    # restrict the output matrices to the correct size
    if indices.shape[1] > n_neighbors:
        indices, distances = indices[:, :n_neighbors], distances[:, :n_neighbors]

    return indices, distances


def _ind_dist_slow(
    D: csr_matrix, n_neighbors: int
) -> Tuple[IntArray, FloatArray]:
    indices = np.zeros((D.shape[0], n_neighbors), dtype=int)
    distances = np.zeros((D.shape[0], n_neighbors), dtype=D.dtype)
    n_neighbors_m1 = n_neighbors - 1
    for i in range(indices.shape[0]):
        neighbors = D[i].nonzero()  # 'true' and 'spurious' zeros
        indices[i, 0] = i
        distances[i, 0] = 0
        # account for the fact that there might be more than n_neighbors
        # due to an approximate search
        # [the point itself was not detected as its own neighbor during the search]
        if len(neighbors[1]) > n_neighbors_m1:
            sorted_indices = np.argsort(D[i][neighbors].A1)[:n_neighbors_m1]
            indices[i, 1:] = neighbors[1][sorted_indices]
            distances[i, 1:] = D[i][
                neighbors[0][sorted_indices], neighbors[1][sorted_indices]
            ]
        else:
            indices[i, 1:] = neighbors[1]
            distances[i, 1:] = D[i][neighbors]
    return indices, distances


def _ind_dist_shortcut(
    D: csr_matrix,
) -> Tuple[IntArray, FloatArray] | None:
    """Shortcut for scipy or RAPIDS style distance matrices."""
    # Check if each row has the correct number of entries
    nnzs = D.getnnz(axis=1)
    if not np.allclose(nnzs, nnzs[0]):
        msg = (
            "Sparse matrix has no constant number of neighbors per row. "
            "Cannot efficiently get indices and distances."
        )
        logger.warn(msg, category=RuntimeWarning)
        return None
    n_obs, n_neighbors = D.shape[0], int(nnzs[0])
    return (
        D.indices.reshape(n_obs, n_neighbors),
        D.data.reshape(n_obs, n_neighbors),
    )


# some algorithms have some messed up reordering.
def _has_self_column(indices: IntArray, distances: FloatArray) -> bool:
    return (indices[:, 0] == np.arange(indices.shape[0])).any()


def umap(
    knn_indices: IntArray,
    knn_dists: FloatArray,
    *,
    n_obs: int,
    n_neighbors: int,
    set_op_mix_ratio: float = 1.0,
    local_connectivity: float = 1.0,
) -> csr_matrix:
    """\
    This is from umap.fuzzy_simplicial_set :cite:p:`McInnes2018`.

    Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.
    """
    with warnings.catch_warnings():
        # umap 0.5.0
        warnings.filterwarnings("ignore", message=r"Tensorflow not installed")
        from umap.umap_ import fuzzy_simplicial_set

    X = coo_matrix(([], ([], [])), shape=(n_obs, 1))
    with warnings.catch_warnings():
        # umap 0.5.0
        warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
        connectivities, _sigmas, _rhos = fuzzy_simplicial_set(
            X,
            n_neighbors,
            None,
            None,
            knn_indices=knn_indices,
            knn_dists=knn_dists,
            set_op_mix_ratio=set_op_mix_ratio,
            local_connectivity=local_connectivity,
        )

    return connectivities.tocsr()


def get_igraph_from_adjacency(adjacency, directed=None):
    """Get igraph graph from adjacency matrix."""

    sources, targets = adjacency.nonzero()
    weights = adjacency[sources, targets]
    if isinstance(weights, np.matrix):
        weights = weights.A1
    g = ig.Graph(directed=directed)
    g.add_vertices(adjacency.shape[0])  # this adds adjacency.shape[0] vertices
    g.add_edges(list(zip(sources, targets)))
    try:
        g.es["weight"] = weights
    except KeyError:
        pass
    if g.vcount() != adjacency.shape[0]:
        logger.warning(
            f"The constructed graph has only {g.vcount()} nodes. "
            "Your adjacency matrix contained redundant nodes."
        )
    return g
