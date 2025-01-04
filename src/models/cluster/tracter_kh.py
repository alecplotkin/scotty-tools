import anndata as ad
import numpy as np
import numpy.typing as npt
import pandas as pd
import logging
from typing import Dict
from time import perf_counter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from src.sctrat.models.trajectory import (
    OTModel, coarsen_ot_model
)
from src.sctrat.tools.trajectories import compute_trajectories
from sketchKH import kernel_herding
from scipy.special import softmax


handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s")
)
logger = logging.getLogger('tracter_kh')
logger.setLevel(logging.INFO)
logger.addHandler(handler)


# TODO: write method docstrings
class TRACTER_kh:
    """TRAjectory Clustering using Timepoint Embedding Representations.

    This one will use kernel herding/kernel embedding.

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
        n_clusters: int = 10,
        n_subsamples: int = 200,
        random_state: int = None,
        gamma: float = 1,
    ):
        self.n_clusters = n_clusters
        self.n_subsamples = n_subsamples
        self.random_state = random_state
        self.gamma = gamma

    def fit(self, ot_model: OTModel, data: ad.AnnData) -> 'TRACTER_kh':
        scale = 1 / self.gamma
        generator = np.random.default_rng(self.random_state)
        self.W = generator.normal(scale=scale, size=(data.shape[1], 1000))
        self.time_var = ot_model.time_var
        # ix_ = {tp: df.index for tp, df in ot_model.meta.groupby(self.time_var)}
        landmark_feats_ = dict()
        responsibilities = dict()
        landmark_ix_ = dict()
        logger.info('Embedding timepoints using landmarks')
        for time, df in ot_model.meta.groupby(self.time_var):
            ix_time = df.index
            X = _parse_input(data[ix_time, :])
            phi = self._make_random_feats(X)
            ix_kh = kernel_herding(phi, num_subsamples=self.n_subsamples)
            phi_lm = phi[ix_kh, :]
            landmark_feats_[time] = phi_lm
            kernel_distance = np.matmul(phi, phi_lm.T)
            responsibilities[time] = pd.DataFrame(
                softmax(-kernel_distance, axis=1),
                index=ix_time, columns=ix_time[ix_kh],
            )
            landmark_ix_[time] = ix_time[ix_kh]
        # TODO: concatenate these into an array?
        self.landmark_feats_ = landmark_feats_
        self.landmark_ix_ = landmark_ix_
        logger.info('Computing trajectory representation')
        self.ot_model = coarsen_ot_model(ot_model, responsibilities)
        # TODO: Only need to cache either ot_model or trajectory_matrix...
        # Decide which one make most sense.
        trajectory_matrix = self.compute_trajectory_matrix(self.ot_model)
        logger.info('Fitting clusters')
        self.cluster_model = self.fit_clusters(trajectory_matrix.X)
        self.trajectory_matrix = trajectory_matrix

    def predict(self, data: ad.AnnData) -> pd.Series:
        clusters_ = []
        for time, df in data.obs.groupby(self.time_var):
            ix_time = df.index
            X = _parse_input(data[ix_time, :])
            phi = self._make_random_feats(X)
            phi_lm = self.landmark_feats_[time]
            kernel_distance = np.matmul(phi, phi_lm.T)
            responsibilities = softmax(-kernel_distance, axis=1)
            ix_lm = self.landmark_ix_[time]
            traj_lm = self.trajectory_matrix[ix_lm, :].X
            traj_X = np.matmul(responsibilities, traj_lm)
            clusters = self.cluster_model.predict(traj_X)
            clusters_.append(pd.Series(clusters, index=ix_time))
        clusters_ = pd.concat(clusters_, axis=0)
        return clusters_

    def _make_random_feats(self, X: npt.NDArray) -> npt.NDArray:
        XW = np.matmul(X, self.W)
        phi = np.concatenate((np.cos(XW), np.sin(XW)), axis=1)
        return phi

    def compute_trajectory_matrix(self, ot_model: OTModel):
        """Compute matrix of trajectories with all timepoints."""

        # Need to get subset names from coarsened ot_model in order for
        # trajectory var_names to make sense.
        meta = ot_model.meta
        traj_by_ref_tp = dict()
        for tp in ot_model.timepoints:
            ix_day = meta.index[meta[self.time_var] == tp]
            subset_weights = pd.DataFrame(
                np.eye(len(ix_day)), index=ix_day, columns=ix_day
            )
            traj = compute_trajectories(
                ot_model, subset_weights, ref_time=tp,
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

    def fit_clusters(self, X: npt.NDArray) -> Pipeline:
        """Fit clusters on trajectory representations."""
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=30)),
            ('cluster', GaussianMixture(
                n_components=self.n_clusters, random_state=self.random_state
            ))
        ])
        model.fit(X)
        return model


def _parse_input(adata: ad.AnnData) -> npt.NDArray:
    """accesses and parses data from adata object

    Parameters
    adata: anndata.AnnData
        annotated data object where adata.X is the attribute for preprocessed data

    ----------

    Returns
    X: np.ndarray
        array of data (dimensions = cells x features)
    ----------
    """
    try:
        if isinstance(adata, ad.AnnData):
            X = adata.X.copy()
        if isinstance(X, scipy.sparse.csr_matrix):
            X = np.asarray(X.todense())
        if is_numeric_dtype(adata.obs_names):
            logger.warning('Converting cell IDs to strings.')
            adata.obs_names = adata.obs_names.astype('str')
    except NameError:
        pass

    return X
