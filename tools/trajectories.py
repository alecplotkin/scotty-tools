import anndata as ad
import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Tuple, Union, Literal
from src.sctrat.models.trajectory import OTModel
from src.sctrat.utils import window


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
        norm_strategy: Literal['joint', 'expected_value'],
        time_var: str = 'day',
    ):
        super().__init__(trajectory, ref_time, time_var)
        self.norm_strategy = norm_strategy


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


# TODO: work on normalization behavior.
def compute_trajectories(
        ot_model: OTModel,
        subsets: Union[pd.Series, pd.DataFrame, npt.NDArray],
        ref_time: int,
        use_ancestor_growth: bool = True,
        normalize: bool = True,
        norm_strategy: Literal['joint', 'expected_value'] = 'joint',
) -> ad.AnnData:

    ix_day = ot_model.meta.index[ot_model.meta[ot_model.time_var] == ref_time]
    if isinstance(subsets, pd.Series):
        subsets = pd.get_dummies(subsets[ix_day], dtype=float)
    elif isinstance(subsets, pd.DataFrame):
        subsets = subsets.loc[ix_day, :].astype(float)
    elif subsets.shape[0] != len(ix_day):
        raise ValueError(
            'subsets and model have different number of cells at ref_time'
        )
    traj = ad.AnnData(subsets)
    norm_to_days = True if norm_strategy == 'expected_value' else False
    if normalize and not norm_to_days:
        traj.X = traj.X / traj.shape[0]
    traj_by_tp = {ref_time: traj.copy()}

    # # Get ancestor trajectories starting with ref_time.
    ancestor_days = [tp for tp in ot_model.timepoints if tp <= ref_time]
    ancestor_day_pairs = window(ancestor_days, k=2)
    for t0, t1 in ancestor_day_pairs[::-1]:
        traj = ot_model.pull_back(traj, t0, t1, normalize=False)
        # TODO: abstract this into a static method...
        if normalize:
            norm_factor = traj.shape[0] if norm_to_days else 1
            traj.X = traj.X / traj.X.sum(keepdims=True) * norm_factor
        traj_by_tp[t0] = traj.copy()

    # Get descendant trajectories starting with ref_time.
    traj = traj_by_tp[ref_time]  # reset traj back to ref_time
    descendant_days = [tp for tp in ot_model.timepoints if tp >= ref_time]
    descendant_day_pairs = window(descendant_days, k=2)
    for t0, t1 in descendant_day_pairs:
        traj = ot_model.push_forward(traj, t0, t1, normalize=False)
        if normalize:
            norm_factor = traj.shape[0] if norm_to_days else 1
            traj.X = traj.X / traj.X.sum(keepdims=True) * norm_factor
        traj_by_tp[t1] = traj.copy()

    # Combine all trajectories into single SubsetTrajectory object
    traj = SubsetTrajectory(
        ad.concat(
            traj_by_tp.values(),
            axis=0,
            keys=traj_by_tp.keys(),
            label=ot_model.time_var,
        ),
        ref_time=ref_time,
        norm_strategy=norm_strategy,
        time_var=ot_model.time_var
    )
    return traj
