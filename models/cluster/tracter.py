import anndata as ad
import numpy as np
import pandas as pd
from typing import Dict
# TODO: create a class that encapsulates all potential types of TRAjectoryModels.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from src.sctrat.models.trajectory import (
    WOTModel, GenericOTModel, embed_ot_model
)


class TRACTER:
    """TRAjectory Clustering using Timepoint Embedding Representations.

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
    ):
        self.rep_dims = rep_dims
        self.n_clusters = n_clusters
        self.n_subsamples = n_subsamples
        self.random_state = random_state

    def fit(self, model: WOTModel, data: ad.AnnData):
        """Fit TRACTER using trajectory model and single-cell data."""

        self.time_var = model.time_var
        print('Fitting time point embeddings...')
        self.embedding_models = self.fit_embeddings(data)
        timepoint_embeddings = self.embed_data(data)
        print('Done.')
        print('Coarsening OT model...')
        self.ot_model = embed_ot_model(model, timepoint_embeddings)
        print('Done.')
        print('Computing trajectory representations...')
        self.trajectory_matrix = self.compute_trajectory_matrix(self.ot_model)
        trajectory_reprs = self.compute_trajectory_reprs(timepoint_embeddings)
        print('Done.')
        print('Fitting trajectory clusters...')
        self.cluster_model = self.fit_clusters(trajectory_reprs)
        print('Done.')

    def predict(self, data: ad.AnnData) -> pd.Series:
        """Predict trajectory cluster assignments for data."""

        timepoint_embeddings = self.embed_data(data)
        trajectory_reprs = self.compute_trajectory_reprs(timepoint_embeddings)
        trajectory_reprs = pd.concat(trajectory_reprs.values(), axis=0)
        preds = self.cluster_model.predict(trajectory_reprs)
        preds = pd.Series(preds, index=trajectory_reprs.index)
        return preds

    def fit_embeddings(self, data: ad.AnnData) -> Dict:
        """Fit time point embeddings using a GMM.

        Args:
            data (ad.AnnData): data to fit embeddings on.

        Returns:
            dict: dictionary of fitted time point GMMs.
        """

        model_dict = dict()
        for tp, df in data.obs.groupby(self.time_var):
            model = GaussianMixture(
                n_components=self.rep_dims, random_state=self.random_state
            )
            model.fit(data[df.index].obsm['X_pca'])
            model_dict[tp] = model
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

    def compute_trajectory_matrix(self, ot_model: GenericOTModel):
        """Compute matrix of trajectories with all timepoints."""

        # Need to get subset names from coarsened ot_model in order for
        # trajectory var_names to make sense.
        meta = ot_model.meta
        traj_by_ref_tp = dict()
        weight_vecs = []
        for tp in ot_model.timepoints:
            ix_day = meta.index[meta[self.time_var] == tp]
            weights = self.embedding_models[tp].weights_
            weight_vecs.append(pd.Series(1/weights, index=ix_day))
            subset_weights = pd.DataFrame(
                np.diag(weights), index=ix_day, columns=ix_day
            )
            traj = ot_model.compute_trajectories(
                subset_weights, ref_time=tp,
                normalize=True, norm_to_days=True
            )
            traj_by_ref_tp[tp] = traj
        weight_vec = pd.concat(weight_vecs, axis=0)
        traj_matrix = ad.concat(
            traj_by_ref_tp.values(),
            axis=1,
            keys=traj_by_ref_tp.keys(),
            label=self.time_var
        )
        # This should fix the weighting of each component.
        # TODO: this should be fixable from within the trajectory function.
        traj_matrix = traj_matrix[weight_vec.index, weight_vec.index]
        w = weight_vec.to_numpy()
        traj_matrix.X = (w.reshape(-1, 1) * traj_matrix.X)  # * w.reshape(1, -1)
        return traj_matrix

    def compute_trajectory_reprs(self, timepoint_embeddings):
        """Compute trajectory representations given timepoint embeddings."""

        rep_dict = dict()
        for tp, df in timepoint_embeddings.items():
            R = self.trajectory_matrix[df.columns, :].to_df()
            rep_dict[tp] = df @ R
        return rep_dict

    def fit_clusters(self, trajectory_reprs):
        """Fit clusters on trajectory representations."""

        train_reprs = []
        for tp, rep in trajectory_reprs.items():
            generator = np.random.default_rng(self.random_state)
            ix_train = generator.choice(
                rep.index, size=self.n_subsamples, replace=False
            )
            train_reprs.append(rep.loc[ix_train, :])
        X = pd.concat(train_reprs, axis=0)
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('cluster', GaussianMixture(
                n_components=self.n_clusters, random_state=self.random_state
            ))
        ])
        model.fit(X)
        return model
