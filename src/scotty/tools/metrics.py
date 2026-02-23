import anndata as ad
import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Union, Literal, Dict
from scotty.models.trajectory import OTModel
from scotty.tools.trajectories import SubsetTrajectory, compute_trajectory_entropy


def _entropy(x):
    return -np.nansum(x * np.log(x))


def fate_consistency(ot_model: OTModel, subsets: pd.Series) -> pd.Series:
    fate_entropy = compute_trajectory_entropy(ot_model, subsets)

    time_var = ot_model.time_var
    freqs = ot_model.meta.join(subsets).groupby(time_var).value_counts(normalize=True)
    null_entropy = freqs.reset_index().pivot(index=time_var, columns=subsets.name).apply(_entropy, axis=1)
    null_entropy.name = 'null_entropy'

    # return fate_entropy, null_entropy
    fate_entropy = pd.merge(
        fate_entropy,
        null_entropy.reset_index().astype(float),
        left_on=f'target_{time_var}',
        right_on=time_var,
    ).set_index(fate_entropy.index)
    return 1 - fate_entropy['entropy'] / fate_entropy['null_entropy']


def compute_cluster_entropy(df: pd.DataFrame) -> float:
    p = df.sum(axis=0) / df.shape[0]
    p = p[p != 0]
    return -np.sum(p * np.log(p))


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
