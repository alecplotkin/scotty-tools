import anndata as ad
from typing import Dict
# TODO: create a class that encapsulates all potential types of TRAjectoryModels.
from sklearn.mixture import GaussianMixture
from src.sctrat.models.trajectory import WOTModel


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
    ):
        self.rep_dims = rep_dims
        self.n_clusters = n_clusters

    def fit(self, model: WOTModel, data: ad.AnnData):
        """Fit TRACTER using trajectory model and single-cell data."""

        self.time_var = model.time_var
        self.embedding_models = self.fit_embeddings(data)
        data_embeddings = self.embed_data(data)
        # # TODO: make option to coarsen tmap model, sample new data points from
        # # embedding space.
        # self._coarsen_tmap(model)
        # y = self.sample_embedding_space()
        traj_matrix = self.compute_trajectory_matrix(data_embeddings, subsample=True)

        # traj_matrix = self.compute_trajectory_matrix(y)
        # self.cluster_trajectories(traj_matrix)

    def fit_embeddings(self, data: ad.AnnData) -> Dict:
        """Fit time point embeddings using a GMM.

        Args:
            data (ad.AnnData): data to fit embeddings on.

        Returns:
            dict: dictionary of fitted time point GMMs.
        """

        model_dict = dict()
        for tp, df in data.obs.groupby(self.time_var):
            model = GaussianMixture(n_components=self.rep_dims)
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
            embedding_dict[tp] = rep
        return embedding_dict
