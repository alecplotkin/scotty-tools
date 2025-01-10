import anndata as ad
import numpy as np
import pandas as pd
from typing import (
        List,
        Dict,
        Tuple,
        TypeVar,
        TYPE_CHECKING,
)


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
        self.timepoints = list(sorted(timepoints))
        self.day_pairs = list(sorted(day_pairs))
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


class WOTModel(BaseOTModel):
    """WOT trajectory model"""

    def __init__(
        self,
        model: 'wot.tmap.TransportMapModel',
        time_var: str = 'day',
    ):
        import wot
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


def coarsen_ot_model(
    model: OTModel,
    timepoint_mixtures: Dict[float, pd.DataFrame],
) -> GenericOTModel:
    """Embed an OT model into the dimensions given by time_mixtures.

    Args:
        model (OTModel): model to embed.
        timepoint_mixtures (dict): dictionary of single cell mixtures with
            timepoints as keys.

    Returns:
        GenericOTModel: OTModel with embedded tmaps.
    """

    # First loop makes embedding columns unique in case they aren't already.
    # Also generates new meta df.
    meta = []
    for tp in model.timepoints:
        df = timepoint_mixtures[tp].T
        df[model.time_var] = tp
        meta.append(df[[model.time_var]])
    meta = pd.concat(meta, axis=0)
    tmaps = dict()
    # Second loop embeds the transport maps.
    for t0, t1 in model.day_pairs:
        tmap = model.get_coupling(t0, t1)
        # Correct for population size in mix0 but not mix1
        mix0 = timepoint_mixtures[t0].loc[tmap.obs_names, :]
        mix0 = mix0 / mix0.values.sum(axis=0, keepdims=True)
        mix1 = timepoint_mixtures[t1].loc[tmap.var_names, :]
        tmap_mix = ad.AnnData(
            np.linalg.multi_dot((mix0.values.T, tmap.X, mix1.values)),
        )
        tmap_mix.obs_names = mix0.columns
        tmap_mix.var_names = mix1.columns
        tmaps[(t0, t1)] = tmap_mix
    model_mix = GenericOTModel(
        tmaps=tmaps,
        meta=meta,
        timepoints=model.timepoints,
        day_pairs=model.day_pairs,
        time_var=model.time_var,
    )
    return model_mix
