import anndata as ad
import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Tuple, Union, Literal, Iterable
from scotty.models.trajectory import OTModel
from scotty.utils import window, adjust_pvalues
from itertools import combinations
from scipy.stats import ttest_ind_from_stats, pearsonr, spearmanr
# from statsmodels.stats.multitest import multipletests


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


class TrajectoryExpectation(TrajectoryMixIn):
    def __init__(
        self,
        trajectory: ad.AnnData,
        ref_time: float,
        time_var: str = 'day',
    ):
        super().__init__(trajectory, ref_time, time_var)

    def __getitem__(self, index):
        subset = super().__getitem__(index)
        return TrajectoryExpectation(
            subset.to_memory(),
            ref_time=self.ref_time,
            time_var=self.time_var,
        )

    def __copy__(self):
        return TrajectoryExpectation(
            super().copy(),
            ref_time=self.ref_time,
            time_var=self.time_var,
        )

    def copy(self):
        return self.__copy__()


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

    # TODO: fix subsetting behavior.
    # This behavior works differently than anndata.__getitem__
    # It will always return a copy of the subset rather than a view, which
    # means that code like `traj[ix_, :].X = ...` will not modify the original
    # traj object.
    def __getitem__(self, index):
        subset = super().__getitem__(index)
        return SubsetTrajectory(
            subset.to_memory(),
            ref_time=self.ref_time,
            norm_strategy=self.norm_strategy,
            time_var=self.time_var,
        )

    def __copy__(self):
        return SubsetTrajectory(
            super().copy(),
            ref_time=self.ref_time,
            norm_strategy=self.norm_strategy,
            time_var=self.time_var,
        )

    def copy(self):
        return self.__copy__()

    # TODO: think of a better name for this function.
    def compute_alternative(self):
        traj_alt = self.copy()
        traj_alt.X = traj_alt.X.sum(1, keepdims=True) - traj_alt.X
        traj_alt.var_names = '~ ' + traj_alt.var_names
        return traj_alt


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

    # TODO: fix subsetting behavior.
    # This behavior works differently than anndata.__getitem__
    # It will always return a copy of the subset rather than a view, which
    # means that code like `traj[ix_, :].X = ...` will not modify the original
    # traj object.
    def __getitem__(self, index):
        subset = super().__getitem__(index)
        return GeneTrajectory(
            subset.to_memory(),
            ref_time=self.ref_time,
            time_var=self.time_var,
            subset_var=self.subset_var,
        )

    def __copy__(self):
        return GeneTrajectory(
            super().copy(),
            ref_time=self.ref_time,
            time_var=self.time_var,
            subset_var=self.subset_var
        )

    def copy(self):
        return self.__copy__()

    def compare_means(self, comparisons: Iterable = None, log_base: float = None) -> pd.DataFrame:
        unique_subsets = self.obs[self.subset_var].unique()
        if comparisons is None:
            # Comparisons must be turned into a list, otherwise iterator will
            # be exhausted after first inner for loop.
            comparisons = list(combinations(unique_subsets, 2))
        results = []
        for day, df in self.obs.groupby(self.time_var):
            for group1, group2 in comparisons:
                ix1 = df.loc[df[self.subset_var] == group1, :].index
                ix2 = df.loc[df[self.subset_var] == group2, :].index
                mean1 = self[ix1].X.squeeze()
                mean2 = self[ix2].X.squeeze()
                std1 = self[ix1].layers['std'].squeeze()
                std2 = self[ix2].layers['std'].squeeze()
                nobs1 = self[ix1].obs['nobs'].to_numpy()
                nobs2 = self[ix2].obs['nobs'].to_numpy()

                # TODO: add check for if std error is 0
                stats, pvals = ttest_ind_from_stats(
                    mean1=mean1, std1=std1, nobs1=nobs1,
                    mean2=mean2, std2=std2, nobs2=nobs2,
                    equal_var=False, alternative='two-sided',
                )
                # Not sure if this is the cleanest way to accomplish this...
                # Assume data is not log-transformed if log_base is None
                if log_base is None:
                    log2_fc = np.log2(mean1) - np.log2(mean2)
                # Switch to log2 if data is log-transformed
                else:
                    log2_fc = (mean1 - mean2) / np.log2(log_base)

                df_comp = pd.DataFrame({
                    'gene': self.var_names,
                    'day': day,
                    'group1': group1,
                    'group2': group2,
                    't-statistic': stats,
                    'pval': pvals,
                    'log2_fc': log2_fc,
                })
                results.append(df_comp)
        results = pd.concat(results, axis=0, ignore_index=True)
        results['padj'] = adjust_pvalues(results['pval'], method='fdr_bh')
        return results


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
        means, stds, obs = GeneTrajectory.compute_tractory_stats(
            trajectory, X, subset_var=subset_var
        )
        gene_trajectory = ad.AnnData(
            X=means,
            obs=obs,
            var=features.var,
            layers={'std': stds}
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
        stdvs = dict()
        nobs = dict()
        for tp, df in trajectory.obs.groupby(trajectory.time_var):
            # TODO: make sig conditional on normalization of trajectory?
            # i.e. to test effect of different effective population sizes.
            mu, sig = GeneTrajectory._compute_weighted_stats(
                trajectory[df.index].X.T,
                features.loc[df.index, :].to_numpy()
            )
            means[tp] = pd.DataFrame(
                mu, index=trajectory.var_names, columns=features.columns
            )
            stdvs[tp] = pd.DataFrame(
                sig, index=trajectory.var_names, columns=features.columns
            )
            if trajectory.norm_strategy == 'expected_value':
                n = trajectory[df.index].X.sum(0)
            elif trajectory.norm_strategy == 'joint':
                n = trajectory[df.index].X.sum(0) * len(df.index)
            else:
                print('norm_strategy not recognized, falling back to using population size for nobs')
                n = len(df.index)
            nobs[tp] = pd.DataFrame({'nobs': n}, index=trajectory.var_names)

        means = np.concatenate(list(means.values()), axis=0)
        stdvs = np.concatenate(list(stdvs.values()), axis=0)
        nobs = pd.concat(
            nobs.values(), axis=0,
            keys=nobs.keys(),
        ).reset_index(names=[trajectory.time_var, subset_var])
        return means, stdvs, nobs

    @staticmethod
    def _compute_weighted_stats(
            W: npt.NDArray,
            X: npt.NDArray
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Compute weighted mean and standard deviation of gene expression.

        Uses the sample sizes at the starting time point to compute the
        variance, with Bessel bias correction.

        Args:
            W (np.ndarray): subsets by cells array of weights.
            X (np.ndarray): cells by features array of genes.

        Returns:
            (np.ndarray, np.ndarray): subsets by features arrays of means and
                std deviations.

        TODOs:
            Double check assumptions about sample size... it may be possible to
            estimate population sample size from the unnormalized transition
            matrix, however this would require leaving the subset trajectories
            unnormalized.
        """

        W = W / W.sum(1, keepdims=True)
        mu = W @ X
        Xm = X[np.newaxis, :, :] - mu[:, np.newaxis, :]
        sig2 = W[:, np.newaxis, :] @ (Xm**2) * W.shape[1] / (W.shape[1] - 1)
        sig = np.sqrt(np.squeeze(sig2, axis=1))
        return mu, sig

    def __repr__(self):
        rep = super().__repr__()
        rep += f"\n    subset_var: '{self.subset_var}'"
        return rep


def _propagate_trajectory(traj, prop_func, t0, t1, normalize, normalize_to_population_size):
    traj = prop_func(traj, t0, t1, normalize=normalize)
    if normalize:
        if normalize_to_population_size:
            norm_factor = traj.shape[0]
        else:
            norm_factor = 1 / traj.X.sum()
        traj.X = traj.X * norm_factor
    return traj


# TODO: work on normalization behavior.
# TODO: add docstring
def compute_trajectories(
        ot_model: OTModel,
        subsets: Union[pd.Series, pd.DataFrame, npt.NDArray],
        ref_time: int,
        condition_on: Literal['reference', 'ancestors', 'descendants'] = 'reference',
        normalize: bool = True,
        normalize_to_population_size: bool = True,
) -> SubsetTrajectory:
    """
    # TODO
    """

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
    if normalize and not normalize_to_population_size:
        traj.X = traj.X / traj.X.sum()
    traj_by_tp = {ref_time: traj.copy()}

    # # Get ancestor trajectories starting with ref_time.
    ancestor_days = [tp for tp in ot_model.timepoints if tp <= ref_time]
    ancestor_day_pairs = window(ancestor_days, k=2)
    for t0, t1 in ancestor_day_pairs[::-1]:
        traj = _propagate_trajectory(
            traj, ot_model.pull_back, t0, t1, normalize=normalize,
            normalize_to_population_size=normalize_to_population_size,
        )
        traj_by_tp[t0] = traj.copy()

    # Get descendant trajectories starting with ref_time.
    traj = traj_by_tp[ref_time]  # reset traj back to ref_time
    descendant_days = [tp for tp in ot_model.timepoints if tp >= ref_time]
    descendant_day_pairs = window(descendant_days, k=2)
    for t0, t1 in descendant_day_pairs:
        traj = _propagate_trajectory(
            traj, ot_model.push_forward, t0, t1, normalize=normalize,
            normalize_to_population_size=normalize_to_population_size,
        )
        traj_by_tp[t1] = traj.copy()

    if normalize:
        if normalize_to_population_size:
            norm_strategy = 'expected_value'
        else:
            norm_strategy = 'joint'
    else:
        norm_strategy = None

    # Sort time points before concatenating, otherwise will be in order added.
    traj_by_tp = {key: traj_by_tp[key] for key in sorted(traj_by_tp.keys())}
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


# TODO: add docstring
def compute_trajectory_expectation(
    ot_model: OTModel,
    features: Union[pd.Series, pd.DataFrame, npt.NDArray],
    ref_time: float,
) -> TrajectoryExpectation:
    """
    # TODO
    """

    ix_day = ot_model.meta.index[ot_model.meta[ot_model.time_var] == ref_time]
    if isinstance(features, pd.Series):
        features = pd.get_dummies(features[ix_day], dtype=float)
    elif isinstance(features, pd.DataFrame):
        features = features.loc[ix_day, :].astype(float)
    elif features.shape[0] != len(ix_day):
        raise ValueError(
            'features and model have different number of cells at ref_time'
        )
    traj = ad.AnnData(features)
    traj_by_tp = {ref_time: traj.copy()}

    # Get ancestor trajectories starting with ref_time.
    ancestor_days = [tp for tp in ot_model.timepoints if tp <= ref_time]
    ancestor_day_pairs = window(ancestor_days, k=2)
    for t0, t1 in ancestor_day_pairs[::-1]:
        traj = ot_model.pull_back(traj, t0, t1, normalize=True, norm_axis=1)
        traj_by_tp[t0] = traj.copy()

    # Get descendant trajectories starting with ref_time.
    traj = traj_by_tp[ref_time]  # reset traj back to ref_time
    descendant_days = [tp for tp in ot_model.timepoints if tp >= ref_time]
    descendant_day_pairs = window(descendant_days, k=2)
    for t0, t1 in descendant_day_pairs:
        traj = ot_model.push_forward(traj, t0, t1, normalize=True, norm_axis=0)
        traj_by_tp[t1] = traj.copy()

    # Sort time points before concatenating, otherwise will be in order added.
    traj_by_tp = {key: traj_by_tp[key] for key in sorted(traj_by_tp.keys())}
    # Combine all trajectories into single SubsetTrajectory object
    traj = TrajectoryExpectation(
        ad.concat(
            traj_by_tp.values(),
            axis=0,
            keys=traj_by_tp.keys(),
            label=ot_model.time_var,
        ),
        ref_time=ref_time,
        time_var=ot_model.time_var
    )
    return traj


def compute_subset_frequency_table(
    ot_model: OTModel, subsets: pd.Series,
) -> pd.DataFrame:
    freqs = dict()
    for i, ref_time in enumerate(ot_model.timepoints):
        # Usually we're only interested in forecasting the frequencies
        # into the future, since the frequencies into the past are constant
        # (for now... until I can implement the correction for expected
        # population size). However, we are calculating trajectories for all
        # pairs of time points. This is very inefficient.
        # TODO: implement method to compute only the necessary trajectories to
        # forecast into the future.
        traj = compute_trajectories(
            ot_model,
            subsets=subsets,
            ref_time=ref_time,
            normalize=True,
            normalize_to_population_size=False,
        )
        # Not sure why, but the time_var is ending up as a pandas categorical.
        # Probably something that's happening upstream in compute_trajectories.
        time_dtype = type(ref_time)
        traj.obs[traj.time_var] = traj.obs[traj.time_var].astype(time_dtype)
        traj = traj[traj.obs[traj.time_var] >= ref_time]
        df_traj = traj.to_df().join(traj.obs)
        df_freq = df_traj.groupby(traj.time_var).sum().reset_index()
        freqs[ref_time] = df_freq
    freqs = pd.concat(
        freqs.values(), axis=0, keys=freqs.keys(), names=['ref_time']
    ).reset_index(level='ref_time')
    freqs[ot_model.time_var] = freqs[ot_model.time_var].astype(time_dtype)
    return freqs


def _format_results_df(stats: npt.NDArray, value_name: str, traj1: GeneTrajectory, traj2: GeneTrajectory) -> pd.DataFrame:
    meta = pd.DataFrame({
        traj1.time_var: traj1.obs[traj1.time_var],
        'group1': traj1.obs[traj1.subset_var],
        'group2': traj2.obs[traj2.subset_var],
    })
    res = pd.DataFrame(stats, index=traj1.obs_names, columns=traj1.var_names)
    res = res.join(meta).melt(id_vars=meta.columns, var_name='gene', value_name=value_name)
    return res


def compare_trajectory_means(traj1: GeneTrajectory, traj2: GeneTrajectory, log_base: float = None) -> pd.DataFrame:
    # TODO: ensure that traj1 and traj2 are compatible, i.e. that genes and
    # days match.
    stats, pvals = ttest_ind_from_stats(
        mean1=traj1.X,
        std1=traj1.layers['std'],
        nobs1=traj1.obs['nobs'].to_numpy().reshape(-1, 1),
        mean2=traj2.X,
        std2=traj2.layers['std'],
        nobs2=traj2.obs['nobs'].to_numpy().reshape(-1, 1),
        equal_var=False,
        alternative='two-sided',
    )
    # Not sure if this is the cleanest way to accomplish this...
    # Assume data is not log-transformed if log_base is None
    if log_base is None:
        logfc = np.log2(traj1.X) - np.log2(traj2.X)
    # Switch to log2 if data is log-transformed
    else:
        logfc = (traj1.X - traj2.X) / np.log2(log_base)

    stats = _format_results_df(stats, 't-statistic', traj1, traj2)
    pvals = _format_results_df(pvals, 'pval', traj1, traj2)
    logfc = _format_results_df(logfc, 'log2_fc', traj1, traj2)
    stats['pval'] = pvals['pval']
    stats['padj'] = adjust_pvalues(stats['pval'], method='fdr_bh')
    stats['log2_fc'] = logfc['log2_fc']
    return stats


def calculate_feature_correlation(
    ot_model: OTModel,
    adata: ad.AnnData,
    source_day: float,
    target_day: float,
    method: Literal['pearson', 'spearman'] = 'pearson',
    axis: Literal[0, 1] = 1,
):
    if method == 'pearson':
        corr_fun = pearsonr
    elif method == 'spearman':
        corr_fun = spearmanr
    else:
        raise NotImplementedError(f"Correlation method {method} not implemented.")

    tmap = ot_model.get_coupling(source_day, target_day)
    tmap.X = tmap.X / tmap.X.sum(axis=axis, keepdims=True)

    # Pullback measure
    if axis == 1:
        X = adata[tmap.obs_names].X
        Y = tmap.X.dot(adata[tmap.var_names].X)

    # Pushforward measure
    elif axis == 0:
        X = tmap.X.dot(adata[tmap.obs_names].X)
        Y = adata[tmap.var_names].X

    else:
        raise ValueError("axis must either be 0 (pushforward) or 1 (pullback).")

    corr = np.array([corr_fun(X[:, i], Y[:, i]) for i in range(X.shape[1])])
    corr = pd.DataFrame(corr, index=adata.var_names, columns=['corr', 'pval'])
    return corr


def compute_trajectory_entropy(ot_model: OTModel, subsets: Iterable) -> pd.DataFrame:
    subsets = pd.get_dummies(subsets)
    fates = dict()
    for day_pair in ot_model.day_pairs:
        tmap = ot_model.get_coupling(*day_pair)
        tmap.X = tmap.X / tmap.X.sum(1, keepdims=True)
        fates[day_pair] = tmap.to_df().dot(subsets.loc[tmap.var_names, :])
    fates = pd.concat(fates.values(), keys=fates.keys(), names=['source_day', 'target_day'])
    entropy = fates.apply(lambda x: -np.nansum(x * np.log(x)), axis=1)
    entropy.name = 'entropy'
    return entropy.reset_index(level=['source_day', 'target_day'])
