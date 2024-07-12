import anndata as ad
import pandas as pd
import numpy.typing as npt
from typing import (
        Union,
        Literal,
)
import wot
from src.wot.utils import window
from src.sctrat.tools.trajectories import CellBySubsetTrajectory


# class TrajectoryMixIn:
#     """Container for various types of trajectory models."""
#
#     def __init__(self, model):
#         self.model = model
#         return
#
#     def load(self, path): ...
#
#     def fit(self, data: ad.AnnData): ...
#
#     def compute_trajectories(
#             self,
#             subsets: Union[pd.Series, pd.DataFrame, npt.NDArray],
#             ref_time: int,
#     ): ...


class WOTModel:
    """WOT trajectory model"""

    def __init__(
        self,
        model: wot.tmap.TransportMapModel = None,
        time_var: str = 'day',
    ):
        self.model = model
        self.time_var = time_var

    @staticmethod
    def load(path) -> "WOTModel":
        return WOTModel(wot.tmap.TransportMapModel.from_directory(path))

    # TODO
    def fit(self, data: ad.AnnData) -> "WOTModel": ...

    def compute_trajectories(
        self,
        subsets: Union[pd.Series, pd.DataFrame, npt.NDArray],
        ref_time: int,
        use_ancestor_growth: bool = True,
        normalize: bool = True,
        norm_to_days: bool = True,
    ) -> ad.AnnData:

        model = self.model
        ix_day = model.meta.index[model.meta[self.time_var] == ref_time]
        if isinstance(subsets, pd.Series):
            subsets = pd.get_dummies(subsets[ix_day], dtype=float)
        elif isinstance(subsets, pd.DataFrame):
            subsets = subsets.loc[ix_day, :].astype(float)
        elif subsets.shape[0] != len(ix_day):
            raise ValueError(
                'subsets and model have different number of cells at ref_time'
            )
        trajectory = ad.AnnData(subsets)
        if normalize and not norm_to_days:
            trajectory.X = trajectory.X / trajectory.shape[0]
        traj_by_tp = {ref_time: trajectory.copy()}
        ancestor_norm_axis = 0 if use_ancestor_growth else 1
        ancestor_days = [tp for tp in model.timepoints if tp <= ref_time]
        ancestor_day_pairs = window(ancestor_days, k=2)
        for day_pair in ancestor_day_pairs[::-1]:
            trajectory = self.pull_back(
                trajectory, *day_pair, norm_axis=ancestor_norm_axis
            )
            if normalize:
                norm_factor = trajectory.shape[0] if norm_to_days else 1
                trajectory.X = trajectory.X / trajectory.X.sum(keepdims=True) * norm_factor
            traj_by_tp[day_pair[0]] = trajectory.copy()

        trajectory = traj_by_tp[ref_time]  # reset trajectory back to ref_time
        descendant_days = [tp for tp in model.timepoints if tp >= ref_time]
        descendant_day_pairs = window(descendant_days, k=2)
        for day_pair in descendant_day_pairs:
            trajectory = self.push_forward(
                trajectory, *day_pair, norm_axis=0
            )
            if normalize:
                norm_factor = trajectory.shape[0] if norm_to_days else 1
                trajectory.X = trajectory.X / trajectory.X.sum(keepdims=True) * norm_factor
            traj_by_tp[day_pair[1]] = trajectory.copy()

        trajectories = ad.concat(
            traj_by_tp.values(),
            axis=0,
            keys=traj_by_tp.keys(),
            label=self.time_var,
        )
        return CellBySubsetTrajectory(trajectories, time_var=self.time_var)

    def push_forward(
        self,
        p: ad.AnnData,
        t0: int,
        t1: int,
        norm_axis: Literal[0, 1] = 0,
    ) -> ad.AnnData:
        tmap = self.model.get_coupling(t0, t1)
        tmap.X = tmap.X / tmap.X.sum(norm_axis, keepdims=True)
        p1 = ad.AnnData(pd.DataFrame(
            tmap.X.T @ p.X, columns=p.var_names, index=tmap.var_names
        ))
        return p1

    def pull_back(
        self,
        p: ad.AnnData,
        t0: int,
        t1: int,
        norm_axis: Literal[0, 1] = 0,
    ) -> ad.AnnData:
        tmap = self.model.get_coupling(t0, t1)
        tmap.X = tmap.X / tmap.X.sum(norm_axis, keepdims=True)
        p1 = ad.AnnData(pd.DataFrame(
            tmap.X @ p.X, columns=p.var_names, index=tmap.obs_names
        ))
        return p1
