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
from sklearn.cluster import KMeans
from sklearn.kernel_approximation import Nystroem, RBFSampler
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


class TRACTER_kernel:
    def __init__(
        self,
        embedding_size: int = 20,
        n_clusters: int = 10,
        n_subsamples: int = None,
        random_state: int = None,
        gamma: float = 1,
        log1p_transform: bool = True,
        cluster_method: Literal['kmeans', 'gmm'] = 'gmm',
    ):

        self.embedding_size = embedding_size
        self.n_clusters = n_clusters
        self.n_subsamples = n_subsamples
        self.random_state = random_state
        self.gamma = gamma
        self.log1p_transform = log1p_transform
        self.cluster_method = cluster_method

    def fit(
        self,
        model: OTModel,
        data: ad.AnnData,
    ):
        self.time_var = model.time_var
        logger.info('Fitting time point embeddings...')
        self.embedding_models = self.fit_embeddings(data)
        time_embeddings = self.embed_data(data)
        # TODO: implement kernel herding on embeddings for subsampling.
        self.ix_train = self.get_training_ix(
            model,
            n_subsamples=self.n_subsamples,
            random_state=self.random_state,
        )

        logger.info('Computing trajectory representations...')
        trajectory_matrix = self.compute_trajectory_matrix(
            model, time_embeddings, log1p_transform=self.log1p_transform
        )
        self.trajectory_matrix = trajectory_matrix

        logger.info('Fitting trajectory clusters...')
        cluster_model = self.fit_clusters(
            trajectory_matrix[self.ix_train],
            log1p_transform=self.log1p_transform,
            cluster_method=self.cluster_method,
        )
        self.cluster_model = cluster_model

    def predict(self, data: ad.AnnData) -> pd.Series:
        """Predict cluster assignments for data."""
        X = self.trajectory_matrix[data.obs_names, :].to_df()
        preds = self.cluster_model.predict(X)
        preds = pd.Series(preds, index=X.index)
        return preds

    def get_training_ix(
        self,
        model: OTModel,
        n_subsamples: int = None,
        random_state: int = None,
    ) -> Dict:
        ix_train = []
        if n_subsamples is None:
            n_subsamples = pd.value_counts(model.meta[model.time_var]).min()
            self.n_subsamples = n_subsamples
        for day, df in model.meta.groupby(model.time_var):
            generator = np.random.default_rng(random_state)
            ix_ = generator.choice(
                df.index, size=n_subsamples, replace=False
            )
            ix_train += list(ix_)
        return ix_train

    def fit_embeddings(self, data: ad.AnnData) -> Dict:
        """Fit time point embeddings using kernel approximation."""

        model_dict = dict()
        for time, df in tqdm(data.obs.groupby(self.time_var)):
            X = data[df.index].obsm['X_pca']
            model = Nystroem(
                n_components=self.embedding_size,
                gamma=self.gamma,
                random_state=self.random_state,
            )
            model.fit(X)
            model_dict[time] = model
        return model_dict

    def embed_data(self, data: ad.AnnData) -> Dict:
        """Embed data at each time point."""

        embedding_dict = dict()
        for time, df in data.obs.groupby(self.time_var):
            model = self.embedding_models[time]
            emb = model.transform(data[df.index].obsm['X_pca'])
            emb = pd.DataFrame(emb, index=df.index)
            emb.columns.names = ['component']
            emb.columns = emb.columns.astype(str) + f'_{time}'
            embedding_dict[time] = emb
        return embedding_dict

    def compute_trajectory_matrix(
            self,
            ot_model: OTModel,
            time_embeddings: Dict,
            log1p_transform: bool = False,
    ) -> ad.AnnData:
        """Compute matrix of trajectories for all timepoint pairs."""

        meta = ot_model.meta
        traj_embeddings = dict()
        for time in tqdm(ot_model.timepoints):
            ix_day = meta.index[meta[self.time_var] == time]
            emb = time_embeddings[time].loc[ix_day, :]
            traj = compute_trajectories(
                ot_model, emb, ref_time=time,
                normalize=True, norm_strategy='expected_value',
            )
            if log1p_transform:
                g = np.sum(traj.X ** 2, axis=1)
                log_g = np.log2(1 + g)
                alpha = np.sqrt(log_g / g).reshape(-1, 1)
                traj.X = traj.X * alpha
            traj_embeddings[time] = traj
        traj_matrix = ad.concat(
            traj_embeddings.values(),
            axis=1,
            keys=traj_embeddings.keys(),
            label=self.time_var,
        )
        return traj_matrix

    def fit_clusters(
            self,
            trajectory_matrix: ad.AnnData,
            log1p_transform: bool = False,  # Not sure how to do this with kernel embeddings.
            cluster_method: Literal['kmeans', 'gmm'] = 'gmm',
    ) -> Pipeline:
        """Fit clusters on trajectory embeddings."""
        X = trajectory_matrix.to_df()
        pipeline_steps = []
        if cluster_method == 'gmm':
            pipeline_steps.append(('cluster', GaussianMixture(
                    n_components=self.n_clusters,
                    random_state=self.random_state,
                )))
        elif cluster_method == 'kmeans':
            pipeline_steps.append(('cluster', KMeans(
                    n_clusters=self.n_clusters,
                    random_state=self.random_state,
                )))
        else:
            raise ValueError('cluster_method not recognized')
        model = Pipeline(pipeline_steps)
        model.fit(X)
        return model
