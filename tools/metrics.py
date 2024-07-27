import anndata as ad
import numpy as np
import pandas as pd
from typing import Union, Literal
from src.sctrat.models.trajectory import OTModel


# # TODO: Allow day_pair to be specified.
# def compute_trajectory_entropy(
#     ot_model: OTModel,
#     # day_pair: Tuple[float, float],
#     subsets: Union[pd.Series, pd.DataFrame] = None,
# ) -> pd.Series:
#     """Compute trajectory entropy with respect to subsets."""
# 
#     if subsets is not None:
#         if isinstance(subsets, pd.Series):
#             subsets = pd.get_dummies(subsets).astype(float)
#         elif isinstance(subsets, pd.DataFrame):
#             subsets = subsets.astype(float)
#         # AnnData will send a warning if var_names are not str.
#         subsets.columns = subsets.columns.astype(str)
# 
#     results = []
#     for t0, t1 in ot_model.day_pairs:
#         ix_t1 = ot_model.meta.index[ot_model.meta[ot_model.time_var] == t1]
#         if subsets is not None:
#             s1 = ad.AnnData(subsets.loc[ix_t1, :])
#             s1 = s1[:, s1.X.sum(0) > 0]
#             traj = ot_model.pull_back(s1, t0, t1, normalize=True, norm_axis=0)
#         else:
#             traj = ot_model.get_coupling(t0, t1)
#             traj.X = traj.X / traj.X.sum(0, keepdims=True)
# 
#         fates = traj.X / traj.X.sum(1, keepdims=True)
# 
#         # # this corrects for maximum possible entropy
#         # norm_factor = np.log(traj.shape[1])
#         # entropy = -np.sum(fates * np.log(fates), axis=1) / norm_factor
# 
#         # # This corrects for relative sizes of clusters.
#         # cluster_weights = s1.X.sum(0, keepdims=True) / s1.X.sum()
#         # entropy = -np.sum(cluster_weights * fates * np.log(fates), axis=1)
# 
#         # # This corrects for relative growth rates of cells.
#         # weights = traj.X.sum(1)
#         # entropy = weights * -np.sum(fates * np.log(fates), axis=1)
# 
#         # This is the un-adjusted entropy.
#         entropy = -np.sum(fates * np.log(fates), axis=1)
# 
#         results.append(pd.Series(entropy, index=traj.obs_names))
#     results = pd.concat(results, axis=0)
#     return results


# TODO: Allow day_pair to be specified.
def compute_trajectory_entropy(
    ot_model: OTModel,
    # day_pair: Tuple[float, float],
    direction: Literal['forward', 'backward'] = 'forward',
    subsets: Union[pd.Series, pd.DataFrame] = None,
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
        results.append(pd.Series(entropy, index=traj.obs_names))
    results = pd.concat(results, axis=0)
    return results
