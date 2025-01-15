import anndata as ad
import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Union, Literal, Dict
from scrtt.models.trajectory import OTModel
from scrtt.tools.trajectories import SubsetTrajectory


def compute_cluster_entropy(df: pd.DataFrame) -> float:
    p = df.sum(axis=0) / df.shape[0]
    p = p[p != 0]
    return -np.sum(p * np.log(p))


# TODO: Allow day_pair to be specified.
def compute_trajectory_entropy(
    ot_model: OTModel,
    # day_pair: Tuple[float, float],
    direction: Literal['forward', 'backward'] = 'forward',
    subsets: Union[pd.Series, pd.DataFrame] = None,
    compute_ratio: bool = True,
) -> pd.Series:
    """Compute trajectory entropy with respect to subsets."""

    if subsets is not None:
        if isinstance(subsets, pd.Series):
            subsets = pd.get_dummies(subsets).astype(float)
        elif isinstance(subsets, pd.DataFrame):
            subsets = subsets.astype(float)
        # AnnData will send a warning if var_names are not str.
        subsets.columns = subsets.columns.astype(str)

    if direction == 'forward':
        traj_func = getattr(ot_model, 'pull_back')
        ix_ref_t = 1
    elif direction == 'backward':
        traj_func = getattr(ot_model, 'push_forward')
        ix_ref_t = 0
    else:
        raise ValueError('direction must be either forward or backward')

    results = []
    for day_pair in ot_model.day_pairs:
        t_ref = day_pair[ix_ref_t]
        ix_ = ot_model.meta.index[ot_model.meta[ot_model.time_var] == t_ref]
        if subsets is not None:
            s = ad.AnnData(subsets.loc[ix_, :])
            s = s[:, s.X.sum(0) > 0]
            traj = traj_func(s, *day_pair, normalize=False)
        else:
            traj = ot_model.get_coupling(*day_pair)
            if direction == 'backward':
                traj = traj.T

        fates = traj.X / traj.X.sum(1, keepdims=True)
        entropy = -np.sum(fates * np.log(fates), axis=1)
        if compute_ratio:
            growth = traj.X.sum(0)
            weights = growth / growth.sum()
            expected_entropy = -np.sum(weights * np.log(weights))
            entropy = entropy / expected_entropy
        results.append(pd.Series(entropy, index=traj.obs_names))
    results = pd.concat(results, axis=0)
    return results


def calculate_trajectory_divergence(
        traj: SubsetTrajectory,
        sub1: str,
        sub2: str,
        metric: Literal['jensen_shannon', 'total_variation', 'mmd'] = 'jensen_shannon',
        feature_key: str = None,
) -> npt.NDArray:
    """Calculate divergence between 2 subsets' trajectories."""
    divs = []
    for day, df in traj.obs.groupby(traj.time_var):
        X = traj[df.index].X
        X = X / X.sum(0, keepdims=True)
        p1 = X[:, traj.var_names == sub1]
        p2 = X[:, traj.var_names == sub2]
        if metric == 'jensen_shannon':
            kl1 = np.sum(p1 * np.log(p1) - p1 * np.log(p2))
            kl2 = np.sum(p2 * np.log(p2) - p2 * np.log(p1))
            div = kl1/2 + kl2/2
        elif metric == 'total_variation':
            div = 0.5 * np.sum(np.abs(p1 - p2))
        elif metric == 'mmd':
            feat = traj[df.index].obsm[feature_key]
            delta = (p1.T @ feat) - (p2.T @ feat)
            div = np.dot(delta, delta.T).squeeze()
        divs.append(div)
    return np.array(divs)
