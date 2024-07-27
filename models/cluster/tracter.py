import anndata as ad
import numpy as np
import pandas as pd
from typing import Dict
from time import perf_counter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from src.sctrat.models.trajectory import (
    OTModel, coarsen_ot_model
)
from src.sctrat.tools.trajectories import compute_trajectories


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

    def fit(self, model: OTModel, data: ad.AnnData):
        """Fit TRACTER using trajectory model and single-cell data."""

        self.time_var = model.time_var
        print('Fitting time point embeddings...')
        start = perf_counter()
        self.ix_train = self.get_training_ix(
            model,
            n_subsamples=self.n_subsamples,
            random_state=self.random_state,
        )
        self.embedding_models = self.fit_embeddings(data, self.ix_train)
        timepoint_embeddings = self.embed_data(data)
        stop = perf_counter()
        print(f'Done in {stop - start:.1f}s')
        print('Computing trajectory representations...')
        start = perf_counter()
        self.trajectory_matrix = self.compute_trajectory_matrix(
                model, timepoint_embeddings
        )
        stop = perf_counter()
        print(f'Done in {stop - start:.1f}s')
        print('Fitting trajectory clusters...')
        start = perf_counter()
        self.cluster_model = self.fit_clusters(
            self.trajectory_matrix,
            ix_train=np.concatenate(list(self.ix_train.values()), axis=0),
        )
        stop = perf_counter()
        print(f'Done in {stop - start:.1f}s')

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

    def fit_embeddings(self, data: ad.AnnData, ix_tp: Dict) -> Dict:
        """Fit time point embeddings using a GMM.

        Args:
            data (ad.AnnData): data to fit embeddings on.
            ix_tp (Dict[pd.Index]): dict of indices to train on for each tp.

        Returns:
            dict: dictionary of fitted time point GMMs.
        """

        model_dict = dict()
        for tp, ix in ix_tp.items():
            model = GaussianMixture(
                n_components=self.rep_dims, random_state=self.random_state
            )
            model.fit(data[ix].obsm['X_pca'])
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

    def fit_clusters(self, trajectory_matrix, ix_train):
        """Fit clusters on trajectory representations."""
        X = trajectory_matrix[ix_train, :].to_df()
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('cluster', GaussianMixture(
                n_components=self.n_clusters, random_state=self.random_state
            ))
        ])
        model.fit(X)
        return model


class TRACTER_coarse:
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

    def fit(self, model: OTModel, data: ad.AnnData):
        """Fit TRACTER using trajectory model and single-cell data."""

        self.time_var = model.time_var
        print('Fitting time point embeddings...')
        start = perf_counter()
        self.embedding_models = self.fit_embeddings(data)
        timepoint_embeddings = self.embed_data(data)
        stop = perf_counter()
        print(f'Done in {stop - start:.1f}s')
        print('Computing trajectory representations...')
        start = perf_counter()
        self.ot_model = coarsen_ot_model(model, timepoint_embeddings)
        self.trajectory_matrix = self.compute_trajectory_matrix(self.ot_model)
        trajectory_reprs = self.compute_trajectory_reprs(timepoint_embeddings)
        stop = perf_counter()
        print(f'Done in {stop - start:.1f}s')
        print('Fitting trajectory clusters...')
        start = perf_counter()
        self.cluster_model = self.fit_clusters(trajectory_reprs)
        stop = perf_counter()
        print(f'Done in {stop - start:.1f}s')

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


def prune_tracter_model(
        tracter: TRACTER,
        clusters: pd.Series,
        n_clusters_pruned: int,
) -> TRACTER:
    """Prune the cluster model to use only the top clusters."""

    # Initialize new TRACTER object with same metadata as old.
    tracter_pruned = TRACTER(
        rep_dims=tracter.rep_dims,
        n_clusters=n_clusters_pruned,
        n_subsamples=tracter.n_subsamples,
        random_state=tracter.random_state,
    )
    tracter_pruned.time_var = tracter.time_var
    tracter_pruned.ix_train = tracter.ix_train
    tracter_pruned.embedding_models = tracter.embedding_models
    tracter_pruned.trajectory_matrix = tracter.trajectory_matrix

    # Get params for top clusters from old model so we can use a warm start.
    cluster_counts = pd.value_counts(clusters)
    ix_keep = cluster_counts.index[0:n_clusters_pruned].to_numpy()
    gmm = tracter.cluster_model['cluster']
    weights_init = gmm.weights_[ix_keep]
    weights_init = weights_init / weights_init.sum()
    means_init = gmm.means_[ix_keep, :]
    precisions_init = gmm.precisions_[ix_keep, :, :]
    gmm_pruned = Pipeline([
        ('scaler', StandardScaler()),
        ('cluster', GaussianMixture(
            n_components=n_clusters_pruned,
            random_state=tracter.random_state,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init,
        ))
    ])

    # Fit new clusters starting from top old clusters.
    ix_train = np.concatenate(list(tracter_pruned.ix_train.values()), axis=0)
    X = tracter_pruned.trajectory_matrix[ix_train, :].to_df()
    gmm_pruned.fit(X)
    tracter_pruned.cluster_model = gmm_pruned
    return tracter_pruned
