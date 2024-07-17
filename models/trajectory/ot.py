import anndata as ad
import pandas as pd
import numpy.typing as npt
from typing import (
        Union,
        Literal,
        List,
        Dict,
        Tuple,
        TypeVar
)
import wot
from src.wot.utils import window
from src.sctrat.tools.trajectories import SubsetTrajectory


class BaseOTModel:
    """Container for various types of trajectory models."""

    def __init__(
        self,
        meta: pd.DataFrame,
        timepoints: List,
        day_pairs: List,
        time_var: str,
    ):
        self.meta = meta
        self.timepoints = timepoints
        self.day_pairs = day_pairs
        self.time_var = time_var

    # TODO: create TransportMap class to be returned by get_coupling?
    def get_coupling(self, t0: float, t1: float) -> ad.AnnData: ...

    def push_forward(
        self,
        p: ad.AnnData,
        t0: int,
        t1: int,
        normalize: bool = True,
        norm_axis: int = None,
    ) -> ad.AnnData:
        tmap = self.get_coupling(t0, t1)
        if normalize:
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
        normalize: bool = True,
        norm_axis: int = None,
    ) -> ad.AnnData:
        tmap = self.get_coupling(t0, t1)
        if normalize:
            tmap.X = tmap.X / tmap.X.sum(norm_axis, keepdims=True)
        p1 = ad.AnnData(pd.DataFrame(
            tmap.X @ p.X, columns=p.var_names, index=tmap.obs_names
        ))
        return p1

    # TODO: work on normalization behavior.
    def compute_trajectories(
        self,
        subsets: Union[pd.Series, pd.DataFrame, npt.NDArray],
        ref_time: int,
        use_ancestor_growth: bool = True,
        normalize: bool = True,
        # norm_strategy: Literal['joint', 'ancestors', 'descendants'] = 'joint',
        norm_to_days: bool = False,
    ) -> ad.AnnData:

        ix_day = self.meta.index[self.meta[self.time_var] == ref_time]
        if isinstance(subsets, pd.Series):
            subsets = pd.get_dummies(subsets[ix_day], dtype=float)
        elif isinstance(subsets, pd.DataFrame):
            subsets = subsets.loc[ix_day, :].astype(float)
        elif subsets.shape[0] != len(ix_day):
            raise ValueError(
                'subsets and model have different number of cells at ref_time'
            )
        traj = ad.AnnData(subsets)
        if normalize and not norm_to_days:
            traj.X = traj.X / traj.shape[0]
        traj_by_tp = {ref_time: traj.copy()}

        # # Get ancestor trajectories starting with ref_time.
        # norm_axis = 0 if use_ancestor_growth else 1
        ancestor_days = [tp for tp in self.timepoints if tp <= ref_time]
        ancestor_day_pairs = window(ancestor_days, k=2)
        for t0, t1 in ancestor_day_pairs[::-1]:
            traj = self.pull_back(
                # traj, t0, t1, normalize=normalize, norm_axis=norm_axis,
                traj, t0, t1, normalize=False,
            )
            # TODO: abstract this into a static method...
            if normalize:
                norm_factor = traj.shape[0] if norm_to_days else 1
                traj.X = traj.X / traj.X.sum(keepdims=True) * norm_factor
            traj_by_tp[t0] = traj.copy()

        # Get descendant trajectories starting with ref_time.
        traj = traj_by_tp[ref_time]  # reset traj back to ref_time
        descendant_days = [tp for tp in self.timepoints if tp >= ref_time]
        descendant_day_pairs = window(descendant_days, k=2)
        for t0, t1 in descendant_day_pairs:
            traj = self.push_forward(
                # If normalizing, always use norm_axis=0 for descendants, since
                # propagation of growth rates is important.
                # traj, t0, t1, normalize=normalize, norm_axis=0,
                traj, t0, t1, normalize=False,
            )
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
                label=self.time_var,
            ),
            ref_time=ref_time,
            time_var=self.time_var
        )
        return traj


class WOTModel(BaseOTModel):
    """WOT trajectory model"""

    def __init__(
        self,
        model: wot.tmap.TransportMapModel = None,
        time_var: str = 'day',
    ):
        super().__init__(
            meta=model.meta,
            timepoints=list(sorted(model.timepoints)),
            day_pairs=model.day_pairs,
            time_var=time_var,
        )
        self.wot_model = model

    @staticmethod
    def load(path) -> "WOTModel":
        return WOTModel(wot.tmap.TransportMapModel.from_directory(path))

    # TODO
    def fit(self, data: ad.AnnData) -> "WOTModel": ...

    # TODO: create TransportMap class to be returned by get_coupling?
    def get_coupling(self, t0: float, t1: float) -> ad.AnnData:
        return self.wot_model.get_coupling(t0, t1)


class GenericOTModel(BaseOTModel):
    """Generic OTModel, with tmaps explicitly stored in object."""

    def __init__(
        self,
        tmaps: Dict[Tuple[float, float], ad.AnnData],
        meta: pd.DataFrame,
        timepoints: List,
        day_pairs: List,
        time_var: str,
    ):
        super().__init__(
            meta=meta,
            timepoints=timepoints,
            day_pairs=day_pairs,
            time_var=time_var,
        )
        self.tmaps = tmaps

    def get_coupling(self, t0: float, t1: float) -> ad.AnnData:
        return self.tmaps[(t0, t1)]


OTModel = TypeVar('OTModel', bound=BaseOTModel)


def embed_ot_model(
    model: OTModel,
    timepoint_embeddings: Dict[float, pd.DataFrame],
) -> GenericOTModel:
    """Embed an OT model into the dimensions given by time_embeddings.

    Args:
        model (OTModel): model to embed.
        timepoint_embeddings (dict): dictionary of single cell embeddings with
            timepoints as keys.

    Returns:
        GenericOTModel: OTModel with embedded tmaps.
    """

    # First loop makes embedding columns unique in case they aren't already.
    # Also generates new meta df.
    meta = []
    for tp in model.timepoints:
        df = timepoint_embeddings[tp].T
        df[model.time_var] = tp
        meta.append(df[[model.time_var]])
    meta = pd.concat(meta, axis=0)
    tmaps = dict()
    # Second loop embeds the transport maps.
    for t0, t1 in model.day_pairs:
        tmap = model.get_coupling(t0, t1)
        emb0 = timepoint_embeddings[t0].loc[tmap.obs_names, :]
        emb1 = timepoint_embeddings[t1].loc[tmap.var_names, :]
        tmap_emb = ad.AnnData((emb0.values.T @ tmap.X) @ emb1.values)
        tmap_emb.obs_names = emb0.columns
        tmap_emb.var_names = emb1.columns
        tmaps[(t0, t1)] = tmap_emb
    model_emb = GenericOTModel(
        tmaps=tmaps,
        meta=meta,
        timepoints=model.timepoints,
        day_pairs=model.day_pairs,
        time_var=model.time_var,
    )
    return model_emb
