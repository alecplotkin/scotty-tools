import logging
import anndata as ad
import numpy as np
import pandas as pd
from typing import Dict, Literal, Union, Iterable
from tqdm import tqdm
from sketchKH import sketch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.mixture import GaussianMixture
from src.sctrat.models.trajectory import OTModel
from src.sctrat.tools.trajectories import compute_trajectories
from copy import copy


logger = logging.getLogger('tracter')
logger.handlers.clear()
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s")
)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


class TRACTER:
    """TRAjectory Clustering using Time-Encoded Representations.

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
        rep_dims: int = 10,
        n_clusters: int = 10,
        n_subsamples: int = 200,
        random_state: int = None,
        log_trajectory_reps: bool = True,
        scale_trajectory_reps: bool = True,
        norm_strategy: Literal['joint', 'expected_value'] = 'expected_value',
    ):
        self.rep_dims = rep_dims
        self.n_clusters = n_clusters
        self.n_subsamples = n_subsamples
        self.random_state = random_state
        self.log_trajectory_reps = log_trajectory_reps
        self.scale_trajectory_reps = scale_trajectory_reps
        self.norm_strategy = norm_strategy

    def fit(
        self,
        model: OTModel,
        data: ad.AnnData,
        encodings: Union[pd.Series, pd.DataFrame] = None,
    ):
        """Fit TRACTER using trajectory model and single-cell data."""

        self.time_var = model.time_var
        logger.info('Subsampling data...')
        _, data_train = sketch(
            data,
            sample_set_key=self.time_var,
            num_subsamples=self.n_subsamples,
            frequency_seed=self.random_state,
        )
        self.ix_train = data_train.obs_names

        if encodings is None:
            logger.info('Fitting time point embeddings...')
            self.embedding_models = self.fit_embeddings(data[self.ix_train])
            timepoint_embeddings = self.embed_data(data)
        else:
            if isinstance(encodings, pd.Series):
                encodings = pd.get_dummies(encodings)
            timepoint_embeddings = dict()
            # Need to split encodings by time point.
            for tp, df in data.obs.groupby(self.time_var):
                rep = encodings.loc[df.index, :]
                rep.columns.names = ['component']
                rep.columns = rep.columns.astype(str) + f'_{tp}'
                timepoint_embeddings[tp] = rep

        logger.info('Computing trajectory representations...')
        trajectory_matrix = self.compute_trajectory_matrix(
                model, timepoint_embeddings, norm_strategy=self.norm_strategy
        )
        self.trajectory_matrix = trajectory_matrix

        logger.info('Fitting trajectory clusters...')
        cluster_model = self.fit_clusters(
            trajectory_matrix[self.ix_train],
            log_trajectory_reps=self.log_trajectory_reps,
            scale_trajectory_reps=self.scale_trajectory_reps,
        )
        self.cluster_model = cluster_model

    def predict(self, data: ad.AnnData) -> pd.Series:
        """Predict trajectory cluster assignments for data."""
        X = self.trajectory_matrix[data.obs_names, :].to_df()
        preds = self.cluster_model.predict(X)
        preds = pd.Series(preds, index=X.index)
        return preds

    def get_training_ix(
            self,
            model: OTModel,
            n_subsamples: int,
            random_state: int = None,
    ) -> Dict:
        """Get subsampled training ix for each time point."""
        training_ix = dict()
        for day, df in model.meta.groupby(model.time_var):
            generator = np.random.default_rng(random_state)
            ix_ = generator.choice(
                df.index, size=n_subsamples, replace=False
            )
            training_ix[day] = ix_
        return training_ix

    def fit_embeddings(self, data: ad.AnnData) -> Dict:
        """Fit time point embeddings using a GMM.

        Args:
            data (ad.AnnData): data to fit embeddings on.

        Returns:
            dict: dictionary of fitted time point GMMs.
        """

        model_dict = dict()
        for time, df in tqdm(data.obs.groupby(self.time_var)):
            X = data[df.index].obsm['X_pca']
            model = GaussianMixture(
                n_components=self.rep_dims, random_state=self.random_state
            )
            model.fit(X)
            model_dict[time] = model
        return model_dict

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
            rep.columns.names = ['component']
            rep.columns = rep.columns.astype(str) + f'_{tp}'
            embedding_dict[tp] = rep
        return embedding_dict

    def compute_trajectory_matrix(
            self,
            ot_model: OTModel,
            timepoint_embeddings: Dict,
            norm_strategy: Literal['joint', 'expected_value'] = 'expected_value',
    ) -> ad.AnnData:
        """Compute matrix of trajectories with all timepoints."""

        meta = ot_model.meta
        traj_by_ref_tp = dict()
        for tp in tqdm(ot_model.timepoints):
            ix_day = meta.index[meta[self.time_var] == tp]
            emb = timepoint_embeddings[tp].loc[ix_day, :]
            traj = compute_trajectories(
                ot_model, emb, ref_time=tp,
                normalize=True,  norm_strategy=norm_strategy,
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
            log_trajectory_reps: bool = True,
            scale_trajectory_reps: bool = True,
    ) -> Pipeline:
        """Fit clusters on trajectory representations."""
        X = trajectory_matrix.to_df()
        pipeline_steps = []
        if log_trajectory_reps:
            pipeline_steps.append(
                ('log_transform', FunctionTransformer(np.log1p, validate=True))
            )
        if scale_trajectory_reps:
            pipeline_steps.append(('scaler', StandardScaler()))
        pipeline_steps.append(('cluster', GaussianMixture(
                n_components=self.n_clusters, random_state=self.random_state
            )))
        model = Pipeline(pipeline_steps)
        model.fit(X)
        return model


def prune_tracter_model(
        tracter: TRACTER,
        clusters: pd.Series,
        keep_clusters: Union[int | Iterable],
) -> TRACTER:
    """Prune the cluster model to use only the top clusters."""

    tracter_pruned = copy(tracter)
    # Select most frequent clusters if keep_clusters is an int, otherwise
    # use the cluster indices within keep_clusters if it is an iterable.
    if isinstance(keep_clusters, int):
        tracter_pruned.n_clusters = keep_clusters
        cluster_counts = pd.value_counts(clusters)
        keep_clusters = cluster_counts.index[0:keep_clusters].to_numpy()

    # Get params for clusters from old model so we can use a warm start.
    gmm = tracter.cluster_model['cluster']
    weights_init = gmm.weights_[keep_clusters]
    weights_init = weights_init / weights_init.sum()
    means_init = gmm.means_[keep_clusters, :]
    precisions_init = gmm.precisions_[keep_clusters, :, :]

    pipeline_steps = []
    if tracter_pruned.scale_trajectory_reps:
        pipeline_steps.append(('scaler', StandardScaler()))
    pipeline_steps.append(('cluster', GaussianMixture(
            n_components=len(keep_clusters),
            random_state=tracter.random_state,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init,
        )
    ))
    gmm_pruned = Pipeline(pipeline_steps)

    # Fit new clusters starting from top old clusters.
    ix_train = tracter_pruned.ix_train
    X = tracter_pruned.trajectory_matrix[ix_train, :].to_df()
    gmm_pruned.fit(X)
    tracter_pruned.cluster_model = gmm_pruned
    return tracter_pruned
