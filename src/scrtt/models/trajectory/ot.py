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

if TYPE_CHECKING:
    import wot
    import moscot


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
            tmap[p.obs_names, :].X.T @ p.X,
            columns=p.var_names,
            index=tmap.var_names,
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
            tmap[:, p.obs_names].X @ p.X,
            columns=p.var_names,
            index=tmap.obs_names,
        ))
        return p1


class MoscotModel(BaseOTModel):
    """Moscot trajectory model"""

    def __init__(self, model: 'moscot.problems.TemporalProblem'):
        meta = model.adata.obs[[model.temporal_key]]
        timepoints = list(sorted(meta[model.temporal_key].unique()))
        day_pairs = list(model.problems.keys())
        super().__init__(
            meta=meta,
            timepoints=timepoints,
            day_pairs=day_pairs,
            time_var=model.temporal_key,
        )
        self.moscot_model = model

    @staticmethod
    def load(path) -> "MoscotModel":
        from moscot.problems import TemporalProblem
        return MoscotModel(TemporalProblem.load(path))

    # TODO: Override push_forward / pull_back behavior to use native push / pull methods.

    # TODO: create TransportMap class to be returned by get_coupling?
    def get_coupling(self, t0: float, t1: float) -> ad.AnnData:
        problem = self.moscot_model[(t0, t1)]
        tmap = ad.AnnData(np.asarray(problem.solution.transport_matrix))
        tmap.obs_names = problem.adata_src.obs_names
        tmap.var_names = problem.adata_tgt.obs_names
        return tmap

    def estimate_population_sizes(self, init_size: float = None, init_day: float = None, compartment_key: str = None, freqs: Dict = None):
        adata = self.moscot_model.adata

        if compartment_key is not None:
            M = pd.get_dummies(adata.obs[compartment_key], dtype=float).to_numpy()
        else:
            M = np.ones((adata.shape[0], 1))

        if init_day is None:
            init_day = adata.obs[self.time_var].cat.categories.min()
        if init_size is None:
            init_size = adata[adata.obs[self.time_var] == init_day].shape[0]
        if freqs is not None:
            init_size = init_size * freqs[init_day]

        pop_sizes = {init_day: init_size}
        tp = self.moscot_model
        masks = tp._policy.create_masks()
        for day_pair in sorted(tp):
            src_day, tgt_day = day_pair
            problem = tp[day_pair]
            src_masks, tgt_masks = masks[day_pair]

            cell_weights = M[src_masks, :] / M[src_masks, :].sum(0, keepdims=True)
            cell_growth = problem.prior_growth_rates ** problem.delta
            pop_growth = np.dot(cell_growth, cell_weights)

            tgt_size = np.nansum(pop_growth * pop_sizes[src_day])
            if freqs is not None:
                tgt_size = tgt_size * freqs[tgt_day]
            pop_sizes[tgt_day] = tgt_size

        return pop_sizes


class WOTModel(BaseOTModel):
    """WOT trajectory model"""

    def __init__(
        self,
        model: 'wot.tmap.TransportMapModel',
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
        import wot
        return WOTModel(wot.tmap.TransportMapModel.from_directory(path))

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
