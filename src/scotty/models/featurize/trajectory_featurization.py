import logging
import numpy as np
import anndata as ad
import pandas as pd

from typing import Dict, Literal, Union
from tqdm import tqdm
from scipy.spatial import distance_matrix
from sklearn.kernel_approximation import Nystroem, RBFSampler
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
class TrajectoryKMEFeaturizer:
    def __init__(
        self,
        embedding_size: int = 20,
        gamma: float = None,
        random_state: int = None,
        embedding_model: Literal['Nystroem', 'RBFSampler'] = 'Nystroem',
    ):
        self.embedding_size = embedding_size
        self.gamma = gamma
        self.random_state = random_state
        self.embedding_model = embedding_model

    def fit_transform(
        self,
        model: OTModel,
        data: ad.AnnData,
        use_rep: str = 'X_pca',
    ):
        self.time_var = model.time_var
        logger.info('Fitting time point embeddings...')
        self.embedding_model_ = self.fit_embeddings(data, use_rep=use_rep)
        embeddings = self.embed_data(data, use_rep=use_rep)
        self.embeddings_ = embeddings
        logger.info('Computing kernel trajectory featurization...')
        trajectory_feat = featurize_trajectories(embeddings, model)
        return trajectory_feat

    # TODO: make it so that embeddings is a DataFrame instead of a dict
    def fit_embeddings(
        self,
        data: ad.AnnData,
        use_rep: str = 'X_pca',
    ) -> Dict:
        """Fit data embeddings by time point using kernel approximation."""

        if self.embedding_model == 'Nystroem':
            model_class = Nystroem
        elif self.embedding_model == 'RBFSampler':
            model_class = RBFSampler

        model_dict = dict()
        for time, df in tqdm(data.obs.groupby(self.time_var, observed=True)):
            X = data[df.index].obsm[use_rep]

            gamma = self.gamma
            if gamma is None:  # Use heuristic to determine optimal gamma.
                D = distance_matrix(X, X)
                sig = np.median(D[np.triu_indices_from(D, k=1)])
                gamma = 1 / sig**2

            model = model_class(
                n_components=self.embedding_size,
                gamma=gamma,
                random_state=self.random_state,
            )
            model.fit(X)
            model_dict[time] = model
        return model_dict

    def embed_data(
        self,
        data: ad.AnnData,
        use_rep: str = 'X_pca',
    ) -> Dict:
        """Embed data using feature map at each time point."""

        embedding_dict = dict()
        for time, df in data.obs.groupby(self.time_var, observed=True):
            model = self.embedding_model_[time]
            emb = model.transform(data[df.index].obsm[use_rep])
            emb = pd.DataFrame(emb, index=df.index)
            emb.columns.names = ['component']
            emb.columns = emb.columns.astype(str) + f'_{time}'
            embedding_dict[time] = emb
        return embedding_dict


def featurize_trajectories(
    feats: Union[pd.DataFrame, Dict[int, pd.DataFrame]],
    ot_model: OTModel,
):
    """Compute trajectory featurization vector for each cell."""

    traj_feats = dict()
    logger.info("Estimating trajectory mean embeddings...")
    for time in tqdm(ot_model.timepoints):
        X = feats[time] if isinstance(feats, dict) else feats
        X.columns = [f'{col}_{time}' for col in X.columns]
        traj_feats[time] = compute_trajectory_expectation(ot_model, X, time)
    traj_feats = ad.concat(
        traj_feats.values(),
        axis=1,
        keys=traj_feats.keys(),
        label=ot_model.time_var,
    )
    traj_feats.obs = ot_model.meta.loc[traj_feats.obs_names, :]
    return traj_feats
