import logging
import anndata as ad
import pandas as pd
from typing import Dict
from tqdm import tqdm
from sklearn.kernel_approximation import Nystroem
from scotty.models.trajectory import OTModel
from scotty.tools.trajectories import compute_trajectory_expectation


logger = logging.getLogger('kernel_trajectory_featurization')
logger.handlers.clear()
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s")
)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


# TODO: allow user to specify whether to fit kernel embeddings separately for each time point.
class KernelTrajectoryFeaturizer:
    def __init__(
        self,
        embedding_size: int = 20,
        gamma: float = 0.01,
        random_state: int = None,
    ):
        self.embedding_size = embedding_size
        self.gamma = gamma
        self.random_state = random_state

    def fit_transform(
        self,
        model: OTModel,
        data: ad.AnnData,
        embedding_rep: str = 'X_pca',
    ):
        self.time_var = model.time_var
        logger.info('Fitting time point embeddings...')
        self.embedding_models = self.fit_embeddings(data, embedding_rep=embedding_rep)
        time_embeddings = self.embed_data(data, embedding_rep=embedding_rep)
        logger.info('Computing kernel trajectory featurization...')
        trajectory_feat = self.compute_trajectory_featurization(
            model, time_embeddings,
        )
        return trajectory_feat

    # TODO: make it so that embeddings is a DataFrame instead of a dict
    def fit_embeddings(
        self,
        data: ad.AnnData,
        embedding_rep: str = 'X_pca',
    ) -> Dict:
        """Fit data embeddings by time point using kernel approximation."""

        model_dict = dict()
        for time, df in tqdm(data.obs.groupby(self.time_var, observed=True)):
            X = data[df.index].obsm[embedding_rep]
            model = Nystroem(
                n_components=self.embedding_size,
                gamma=self.gamma,
                random_state=self.random_state,
            )
            model.fit(X)
            model_dict[time] = model
        return model_dict

    def embed_data(
        self,
        data: ad.AnnData,
        embedding_rep: str = 'X_pca',
    ) -> Dict:
        """Embed data using feature map at each time point."""

        embedding_dict = dict()
        for time, df in data.obs.groupby(self.time_var, observed=True):
            model = self.embedding_models[time]
            emb = model.transform(data[df.index].obsm[embedding_rep])
            emb = pd.DataFrame(emb, index=df.index)
            emb.columns.names = ['component']
            emb.columns = emb.columns.astype(str) + f'_{time}'
            embedding_dict[time] = emb
        return embedding_dict

    # TODO: make it so that embeddings is a DataFrame instead of a dict
    def compute_trajectory_featurization(
        self,
        ot_model: OTModel,
        embeddings: Dict,
    ):
        """Compute matrix of kernel trajectories for each cell."""

        meta = ot_model.meta
        traj_embeddings = dict()
        for time in tqdm(ot_model.timepoints):
            ix_day = meta.index[meta[self.time_var] == time]
            emb = embeddings[time].loc[ix_day]
            traj = compute_trajectory_expectation(ot_model, emb, ref_time=time)
            traj_embeddings[time] = traj
        traj_matrix = ad.concat(
            traj_embeddings.values(),
            axis=1,
            keys=traj_embeddings.keys(),
            label=self.time_var,
        )
        return traj_matrix
