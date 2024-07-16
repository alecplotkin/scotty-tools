import anndata as ad
import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Tuple


class TrajectoryMixIn(ad.AnnData):
    def __init__(
        self,
        trajectory: ad.AnnData,
        ref_time: float,
        time_var: str = 'day',
    ):
        super().__init__(trajectory)
        self.ref_time = ref_time
        self.time_var = time_var

    def __repr__(self):
        rep = f"{self.__class__.__name__} with data:\n{super().__repr__()}"
        rep += f"\n    ref_time: {self.ref_time}"
        rep += f"\n    time_var: '{self.time_var}'"
        return rep


class SubsetTrajectory(TrajectoryMixIn):
    def __init__(
        self,
        trajectory: ad.AnnData,
        ref_time: float,
        time_var: str = 'day',
    ):
        super().__init__(trajectory, ref_time, time_var)


class GeneTrajectory(TrajectoryMixIn):
    def __init__(
        self,
        trajectory: ad.AnnData,
        ref_time: float,
        time_var: str = 'day',
        subset_var: str = 'subset',
    ):
        super().__init__(trajectory, ref_time, time_var)
        self.subset_var = subset_var

    @staticmethod
    def from_subset_trajectory(
        trajectory: SubsetTrajectory,
        features: ad.AnnData,
        subset_var: str = 'subset',
        layer: str = None,
    ) -> "GeneTrajectory":
        X = features.to_df()
        if layer is not None:
            X.loc[:, :] = features.layers[layer]
        means, vars, obs = GeneTrajectory.compute_tractory_stats(
            trajectory, X, subset_var=subset_var
        )
        gene_trajectory = ad.AnnData(
            X=means,
            obs=obs,
            var=features.var,
            layers={'vars': vars}
        )
        gene_trajectory = GeneTrajectory(
            gene_trajectory, ref_time=trajectory.ref_time,
            time_var=trajectory.time_var, subset_var=subset_var
        )
        return gene_trajectory

    @staticmethod
    def compute_tractory_stats(
        trajectory: SubsetTrajectory,
        features: pd.DataFrame,
        subset_var: str = 'subset',
    ):
        """Compute trajectory-weighted expression stats at each timepoint."""

        means = dict()
        vars = dict()
        nobs = dict()
        for tp, df in trajectory.obs.groupby(trajectory.time_var):
            # TODO: make sig2 conditional on normalization of trajectory?
            # i.e. to test effect of different effective population sizes.
            mu, sig2 = GeneTrajectory._compute_weighted_stats(
                trajectory[df.index].X.T,
                features.loc[df.index, :].to_numpy()
            )
            means[tp] = pd.DataFrame(
                mu, index=trajectory.var_names, columns=features.columns
            )
            vars[tp] = pd.DataFrame(
                sig2, index=trajectory.var_names, columns=features.columns
            )
            # TODO: make nobs conditional on normalization of trajectory?
            # i.e. to test effect of different effective population sizes.
            nobs[tp] = pd.DataFrame(index=trajectory.var_names)
            nobs[tp]['nobs'] = len(df.index)

        means = np.concatenate(list(means.values()), axis=0)
        vars = np.concatenate(list(vars.values()), axis=0)
        nobs = pd.concat(
            nobs.values(), axis=0,
            keys=nobs.keys(),
        ).reset_index(names=[trajectory.time_var, subset_var])
        return means, vars, nobs

    @staticmethod
    def _compute_weighted_stats(
            W: npt.NDArray,
            X: npt.NDArray
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Compute weighted mean and variance of gene expression.

        Uses the sample sizes at the starting time point to compute the
        variance, with Bessel bias correction.

        Args:
            W (np.ndarray): subsets by cells array of weights.
            X (np.ndarray): cells by features array of genes.

        Returns:
            (np.ndarray, np.ndarray): subsets by features arrays of means and
                variances.

        TODOs:
            Double check assumptions about sample size... it may be possible to
            estimate population sample size from the unnormalized transition
            matrix, however this would require leaving the subset trajectories
            unnormalized.
        """

        W = W / W.sum(1, keepdims=True)
        mu = W @ X
        Xm = X[np.newaxis, :, :] - mu[:, np.newaxis, :]
        sig2 = W[:, np.newaxis, :] @ (Xm**2) / (W.shape[1] - 1)
        sig2 = np.squeeze(sig2, axis=1)
        return mu, sig2

    def __repr__(self):
        rep = super().__repr__()
        rep += f"\n    subset_var: '{self.subset_var}'"
        return rep
