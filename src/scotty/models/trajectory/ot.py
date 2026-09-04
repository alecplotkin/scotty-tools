import anndata as ad
import json
import numpy as np
import numpy.typing as npt
import pandas as pd
from pathlib import Path
from scipy.sparse import issparse
from typing import (
        List,
        Dict,
        Tuple,
        Literal,
        TypeVar,
        TYPE_CHECKING,
)
from scotty.utils import window


if TYPE_CHECKING:
    import wot
    import moscot


def _patch_anndata_file_backing_setstate() -> None:
    """Make ``AnnDataFileManager.__setstate__`` tolerant of legacy pickles.

    Models pickled with older anndata versions store the dereferenced AnnData
    under the key ``_adata``, whereas anndata >= 0.11 expects ``_adata_ref``.
    Unpickling such a file raises ``KeyError: '_adata_ref'``. This shim renames
    the legacy key so old tmaps load under the current anndata. Idempotent.
    """
    import weakref
    from anndata._core.file_backing import AnnDataFileManager

    if getattr(AnnDataFileManager.__setstate__, "_scotty_compat", False):
        return

    def __setstate__(self, state):
        state = dict(state)
        adata = state.pop("_adata_ref", None)
        if adata is None:
            adata = state.pop("_adata", None)
        self.__dict__ = state
        self.__dict__["_adata_ref"] = weakref.ref(adata) if adata is not None else None
        self.__dict__.setdefault("_file", None)

    __setstate__._scotty_compat = True
    AnnDataFileManager.__setstate__ = __setstate__


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

    @classmethod
    def load(cls, path: str) -> "BaseOTModel":
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Save the model to a directory in a portable, pickle-independent format.

        Saves each transport coupling as an h5ad file, the meta DataFrame as CSV,
        and timepoint metadata as JSON. Can be reloaded with GenericOTModel.load(path).
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        for t0, t1 in self.day_pairs:
            coupling = self.get_coupling(t0, t1)
            coupling.write_h5ad(path / f"coupling_{t0}_{t1}.h5ad")

        self.meta.to_csv(path / "meta.csv")

        metadata = {
            "time_var": self.time_var,
            "timepoints": self.timepoints,
            "day_pairs": [list(p) for p in self.day_pairs],
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f)

    # TODO: create TransportMap class to be returned by get_coupling?
    def get_coupling(self, t0: float, t1: float) -> ad.AnnData: ...

    @staticmethod
    def _normalized_coupling(
        tmap: ad.AnnData,
        normalize: bool,
        norm_axis: int,
    ) -> npt.NDArray:
        """Return the (optionally normalized) coupling matrix as a dense array.

        Never writes back to ``tmap``: some models (e.g. GenericOTModel) return
        the stored coupling itself from ``get_coupling``, so normalizing in
        place would corrupt the model and make repeated pushes/pulls depend on
        call order.
        """
        X = tmap.X.toarray() if issparse(tmap.X) else np.asarray(tmap.X)
        if normalize:
            tmap_sum = X.sum(norm_axis, keepdims=True)
            X = X / np.where(tmap_sum == 0, 1e-9, tmap_sum)
        return X

    @staticmethod
    def _get_indexer(names: pd.Index, wanted: pd.Index) -> npt.NDArray:
        """Positional index of ``wanted`` within ``names``, erroring if absent."""
        ix = names.get_indexer(wanted)
        if (ix < 0).any():
            missing = list(pd.Index(wanted)[ix < 0][:5])
            raise KeyError(f'cells not found in transport map: {missing}')
        return ix

    def push_forward(
        self,
        p: ad.AnnData,
        t0: int,
        t1: int,
        normalize: bool = True,
        norm_axis: int = None,
    ) -> ad.AnnData:
        tmap = self.get_coupling(t0, t1)
        X = self._normalized_coupling(tmap, normalize, norm_axis)
        ix = self._get_indexer(tmap.obs_names, p.obs_names)
        p1 = ad.AnnData(pd.DataFrame(
            X[ix, :].T @ p.X,
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
        X = self._normalized_coupling(tmap, normalize, norm_axis)
        ix = self._get_indexer(tmap.var_names, p.obs_names)
        p1 = ad.AnnData(pd.DataFrame(
            X[:, ix] @ p.X,
            columns=p.var_names,
            index=tmap.obs_names,
        ))
        return p1


class MoscotModel(BaseOTModel):
    """Moscot trajectory model"""

    def __init__(self, model: 'moscot.problems.TemporalProblem', compartment_key: str | None = None):
        meta_cols = [model.temporal_key]
        if compartment_key is not None:
            meta_cols.append(compartment_key)
        meta = model.adata.obs[meta_cols]
        timepoints = list(sorted(meta[model.temporal_key].unique()))
        day_pairs = list(model.problems.keys())
        super().__init__(
            meta=meta,
            timepoints=timepoints,
            day_pairs=day_pairs,
            time_var=model.temporal_key,
        )
        self.compartment_key = compartment_key
        self.moscot_model = model
        self._base_initialized = True

    @classmethod
    def from_adata(
        cls,
        adata: ad.AnnData,
        problem_type: Literal["temporal", "lineage"] = "temporal",
        compartment_key: str | None = None,
    ) -> "MoscotModel":
        """Create an unfitted MoscotModel from raw AnnData.

        Use the fluent API to prepare and solve before passing to other scotty functions:

            ot_model = (
                MoscotModel.from_adata(adata)
                .score_genes_for_marginals(...)
                .prepare(time_key='day', ...)
                .clip_growth_rates(upper_quantile=0.95)
                .solve(epsilon=0.01, ...)
            )
        """
        from moscot.problems.time import TemporalProblem, LineageProblem
        instance = object.__new__(cls)
        if problem_type == "temporal":
            instance.moscot_model = TemporalProblem(adata)
        elif problem_type == "lineage":
            instance.moscot_model = LineageProblem(adata)
        else:
            raise ValueError("Unsupported problem type.")
        instance.compartment_key = compartment_key
        instance._base_initialized = False
        return instance

    def _initialize_base(self):
        model = self.moscot_model
        compartment_key = self.compartment_key
        meta_cols = [model.temporal_key]
        if compartment_key is not None:
            meta_cols.append(compartment_key)
        meta = model.adata.obs[meta_cols]

        self.meta = meta
        self.timepoints = list(sorted(meta[model.temporal_key].unique()))
        self.day_pairs = list(model.problems.keys())
        self.time_var = model.temporal_key
        self._base_initialized = True

    def score_genes_for_marginals(self, **kwargs) -> "MoscotModel":
        self.moscot_model.score_genes_for_marginals(**kwargs)
        return self

    def prepare(self, time_key: str, **kwargs) -> "MoscotModel":
        self.moscot_model.prepare(time_key=time_key, **kwargs)
        self._initialize_base()
        return self

    def clip_growth_rates(
        self,
        lower_quantile: float = 0.,
        upper_quantile: float = 0.95,
        group_key: str | None = None,
    ) -> "MoscotModel":
        """Clip per-problem prior growth rates to the given quantile range.

        Parameters
        ----------
        lower_quantile, upper_quantile:
            Quantiles of the prior growth rates used as clipping bounds.
        group_key:
            Column of ``problem.adata_src.obs`` defining groups of cells (e.g. a
            tissue compartment). When given, the bounds are computed and applied
            within each group, so a group whose growth rates are systematically
            lower is not clipped against another group's outliers. When ``None``
            (default) the bounds are computed over all source cells at once.
        """
        for day_pair in self.moscot_model:
            problem = self.moscot_model[day_pair]
            prior_growth = problem._prior_growth
            clipped = prior_growth.copy()
            for ix in self._group_indices(problem.adata_src, group_key):
                lo = np.quantile(prior_growth[ix], lower_quantile)
                hi = np.quantile(prior_growth[ix], upper_quantile)
                clipped[ix] = np.clip(prior_growth[ix], lo, hi)
            problem._prior_growth = clipped
            problem._a = clipped / clipped.sum()
        return self

    def temper_growth_rates(
        self,
        max_log_spread: float = 8.0,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
        group_key: str | None = None,
    ) -> "MoscotModel":
        """Shrink the spread of the per-interval log growth field toward its mean.

        ``problem._prior_growth`` is the growth factor for the whole interval,
        ``exp((beta - delta) * dt)``, so its spread across cells grows with the
        interval length. Over a long interval a per-cell growth field estimated
        from a noisy expression score is amplified to many orders of magnitude,
        and a handful of cells come to carry nearly all of the source mass. Every
        quantity computed off the coupling then reflects those few cells: the
        population estimate, the compartment split, and the migration rates.

        This rescales the log growth field about its mean so that the
        ``lower_quantile``-to-``upper_quantile`` spread is at most
        ``max_log_spread``:

        .. code-block:: text

            L  = log(prior_growth)
            s  = min(1, max_log_spread / (quantile(L, uq) - quantile(L, lq)))
            L' = mean(L) + s * (L - mean(L))

        Properties, which are the justification for using it:

        - When the spread is already within ``max_log_spread``, ``s == 1`` and
          the method is an **exact no-op**. Short intervals are therefore left
          untouched and only the long ones are tempered.
        - The **rank ordering** of cells and the **mean of log growth** (i.e. the
          geometric mean of the growth factor) are preserved.
        - ``mean(prior_growth)`` is deliberately *not* preserved -- that is the
          quantity a heavy upper tail blows up, and the one being brought back
          under control.
        - Because ``_prior_growth`` is the per-interval factor, tempering it is
          equivalent to tempering the per-day rate by the same ``s``; there is no
          ``dt`` bookkeeping to get wrong.

        Parameters
        ----------
        max_log_spread:
            Maximum admissible spread of ``log(prior_growth)`` between the two
            quantiles, i.e. ``log`` of the largest fold difference in growth
            allowed between cells within one interval.
        lower_quantile, upper_quantile:
            Quantiles delimiting the spread that is measured. Defaults ignore the
            extreme 1% at each end so a single outlier cannot set ``s``.
        group_key:
            Column of ``problem.adata_src.obs`` defining groups of cells (e.g. a
            tissue compartment). When given, the spread is measured and the
            shrinkage applied within each group, so one group's spread does not
            set another's. When ``None`` (default) all source cells are tempered
            together.

        Returns
        -------
        Self, with ``problem._prior_growth`` tempered and ``problem._a``
        renormalized.
        """
        if max_log_spread <= 0:
            raise ValueError(
                f'max_log_spread must be positive, got {max_log_spread!r}.'
            )
        for day_pair in self.moscot_model:
            problem = self.moscot_model[day_pair]
            prior_growth = np.asarray(problem._prior_growth, dtype=float)
            tempered = prior_growth.copy()
            for ix in self._group_indices(problem.adata_src, group_key):
                log_growth = np.log(prior_growth[ix])
                lo, hi = np.quantile(log_growth, (lower_quantile, upper_quantile))
                spread = hi - lo
                # A degenerate spread needs no tempering and would divide by zero.
                scale = 1.0 if spread <= 0 else min(1.0, max_log_spread / spread)
                if scale == 1.0:
                    continue
                mean = log_growth.mean()
                tempered[ix] = np.exp(mean + scale * (log_growth - mean))
            problem._prior_growth = tempered
            problem._a = tempered / tempered.sum()
        return self

    def growth_rate_log_spread(
        self,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
        group_key: str | None = None,
    ) -> pd.DataFrame:
        """Per-interval spread of the log growth field, for auditing tempering.

        Returns one row per (day pair, group) with the quantile spread of
        ``log(prior_growth)`` and the shrinkage ``temper_growth_rates`` would
        apply at a given ``max_log_spread`` (reported as ``spread``; the caller
        computes ``min(1, cap / spread)``).
        """
        rows = []
        for day_pair in sorted(self.moscot_model):
            problem = self.moscot_model[day_pair]
            prior_growth = np.asarray(problem._prior_growth, dtype=float)
            obs = problem.adata_src.obs
            for ix in self._group_indices(problem.adata_src, group_key):
                log_growth = np.log(prior_growth[ix])
                lo, hi = np.quantile(log_growth, (lower_quantile, upper_quantile))
                rows.append({
                    't0': day_pair[0],
                    't1': day_pair[1],
                    'group': (str(obs[group_key].iloc[ix[0]])
                              if group_key is not None else 'all'),
                    'n': int(len(ix)),
                    'spread': float(hi - lo),
                    'fold_spread': float(np.exp(hi - lo)),
                })
        return pd.DataFrame(rows)
    def set_death_rates(self, rates: Dict[float, float]) -> "MoscotModel":
        """Apply a per-timepoint death rate to the prior growth rates.

        Multiplies each day pair's prior growth by ``exp(-rates[t0] * dt)``, i.e.
        subtracts a constant ``rates[t0]`` from the log growth rate of every source
        cell at ``t0``. Use it when the death rate is known at the population level
        (e.g. from measured cell counts) but cannot be resolved per cell.

        .. note::
            A rate that is constant within a timepoint **does not change the
            couplings**. ``exp(-rate * dt)`` is a common factor across the source
            cells, and the source marginal is renormalized (``_a``), so it cancels
            exactly. What it does change is :meth:`estimate_population_sizes` and
            :attr:`prior_growth_rates`, which read the unnormalized
            ``_prior_growth``. To make death affect the transport plan it has to
            vary between cells.

        Parameters
        ----------
        rates:
            ``{source_timepoint: rate}`` in units of log growth per unit time.
            Positive values shrink the population, negative values grow it. Every
            source timepoint of the model must have an entry.

        Returns
        -------
        Self, with ``_prior_growth`` scaled and ``_a`` renormalized.
        """
        missing = {t0 for t0, _ in self.moscot_model} - set(rates)
        if missing:
            raise KeyError(
                f'No death rate given for source timepoint(s) {sorted(missing)}.'
            )
        for day_pair in self.moscot_model:
            t0, t1 = day_pair
            problem = self.moscot_model[day_pair]
            scaled = problem._prior_growth * np.exp(-rates[t0] * (t1 - t0))
            problem._prior_growth = scaled
            problem._a = scaled / scaled.sum()
        return self

    def rescale_marginals(
        self,
        target_freqs: Dict[float, Dict[str, float]],
        group_key: str,
    ) -> "MoscotModel":
        """Rescale marginals so cell groups carry reference amounts of mass.

        The number of cells sampled from a group (e.g. a tissue compartment) at
        a given timepoint reflects the experiment, not the size of that group in
        the animal. This divides each cell's marginal weight by its group's
        over-representation (observed cell frequency / reference frequency), so
        that after renormalization every group carries the mass given by
        ``target_freqs``.

        Parameters
        ----------
        target_freqs:
            ``{timepoint: {group: frequency}}``. Frequencies need not sum to one;
            only their ratios within a timepoint matter. Every timepoint of the
            model, and every group present at that timepoint, must have an entry.
        group_key:
            Column of ``adata.obs`` defining the groups.

        Returns
        -------
        Self, with ``problem._a`` / ``problem._b`` rescaled and renormalized.
        """
        for day_pair in self.moscot_model:
            problem = self.moscot_model[day_pair]
            problem._a = self._rescale_marginal(
                problem.a, problem.adata_src, target_freqs, day_pair[0], group_key
            )
            problem._b = self._rescale_marginal(
                problem.b, problem.adata_tgt, target_freqs, day_pair[1], group_key
            )
        return self

    @staticmethod
    def _group_indices(adata: ad.AnnData, group_key: str | None) -> List[npt.NDArray]:
        """Positional indices of each group of cells, or of all cells if no key."""
        if group_key is None:
            return [np.arange(adata.n_obs)]
        return [
            adata.obs_names.get_indexer(df.index)
            for _, df in adata.obs.groupby(group_key, observed=True)
        ]

    @staticmethod
    def _rescale_marginal(
        marginal: npt.NDArray,
        adata: ad.AnnData,
        target_freqs: Dict[float, Dict[str, float]],
        timepoint: float,
        group_key: str,
    ) -> npt.NDArray:
        """Divide out group over-representation from one marginal; renormalize."""
        if timepoint not in target_freqs:
            raise KeyError(
                f'target_freqs has no entry for timepoint {timepoint!r}; '
                'provide reference frequencies for every timepoint of the model.'
            )
        targets = target_freqs[timepoint]
        # Groups with no cells at this timepoint are simply not represented in
        # the marginal (e.g. gut cells before they appear), so they are skipped.
        obs_freqs = adata.obs[group_key].value_counts(normalize=True)
        obs_freqs = obs_freqs[obs_freqs > 0]

        missing = set(obs_freqs.index) - set(targets)
        if missing:
            raise KeyError(
                f'target_freqs[{timepoint!r}] is missing groups {sorted(missing)}.'
            )
        nonpositive = [g for g in obs_freqs.index if targets[g] <= 0]
        if nonpositive:
            raise ValueError(
                f'target_freqs[{timepoint!r}] must be positive for groups present '
                f'in the data, got non-positive values for {sorted(nonpositive)}.'
            )

        scale = {g: freq / targets[g] for g, freq in obs_freqs.items()}
        scale = adata.obs[group_key].map(scale).to_numpy(dtype=float)
        rescaled = np.asarray(marginal, dtype=float) / scale
        return rescaled / rescaled.sum()

    def solve(self, **kwargs) -> "MoscotModel":
        self.moscot_model.solve(**kwargs)
        return self

    @staticmethod
    def load(path) -> "MoscotModel":
        from moscot.problems import TemporalProblem
        _patch_anndata_file_backing_setstate()
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

    def estimate_population_sizes_by_group(
        self,
        group_key: str,
        init_sizes: Dict[str, float],
        init_day: float = None,
    ) -> Dict[float, Dict[str, float]]:
        """Propagate per-group population sizes through growth **and** migration.

        A group's size changes both because its cells divide and die, and because cells move
        between groups. This pushes each source cell's share of its group's population through the
        row-stochastic coupling, weighted by that cell's prior growth, and re-aggregates by the
        group of the *target* cell::

            n_c(t1) = sum_{j in c} sum_i  w_i(t0) * g_i^dt * P(j|i)

        where ``w_i(t0)`` is cell ``i``'s share of its group's population at ``t0`` and
        ``P(j|i)`` is the coupling normalized over targets.

        .. note::
            This is not what :meth:`estimate_population_sizes` does with ``compartment_key``. That
            method sums over groups (``np.nansum``) and, if ``freqs`` is given, redistributes the
            total by those frequencies -- so the split between groups is an *input* rather than a
            prediction. (Its ``freqs`` path also expects a per-day scalar, so the
            ``{day: {group: frac}}`` produced by ``src.population_kinetics.fractions_at`` raises a
            ``TypeError``.) Use this method when the split should be predicted and compared against
            an independent measurement.

        Parameters
        ----------
        group_key:
            Column of ``adata.obs`` defining the groups, e.g. a tissue compartment.
        init_sizes:
            ``{group: size}`` at ``init_day``. Groups absent from the data at a timepoint simply
            carry no mass.
        init_day:
            Timepoint the sizes refer to; defaults to the earliest.

        Returns
        -------
        ``{timepoint: {group: size}}``.
        """
        adata = self.moscot_model.adata
        if group_key not in adata.obs:
            raise KeyError(f'No group column {group_key!r} in `adata.obs`.')

        groups = list(adata.obs[group_key].astype(str).unique())
        missing = set(init_sizes) - set(groups)
        if missing:
            raise KeyError(f'init_sizes names unknown group(s) {sorted(missing)}.')

        if init_day is None:
            init_day = min(t for pair in self.moscot_model for t in pair)

        sizes = {init_day: {g: float(init_sizes.get(g, 0.0)) for g in groups}}
        for day_pair in sorted(self.moscot_model):
            src_day, tgt_day = day_pair
            problem = self.moscot_model[day_pair]
            src_obs = problem.adata_src.obs
            src_groups = src_obs[group_key].astype(str)

            # Each source cell carries its group's population, split evenly across the cells
            # observed for that group, then scaled by its own growth over the interval.
            counts = src_groups.value_counts()
            per_cell = src_groups.map(
                {g: sizes[src_day].get(g, 0.0) / counts[g] for g in counts.index}
            ).to_numpy(dtype=float)
            weights = per_cell * np.asarray(problem._prior_growth, dtype=float)

            pushed = self.push_forward(
                ad.AnnData(pd.DataFrame(weights[:, None], index=src_obs.index,
                                        columns=['size'])),
                src_day, tgt_day, normalize=True, norm_axis=1,
            )
            tgt_groups = problem.adata_tgt.obs[group_key].astype(str)
            arrived = pd.Series(np.asarray(pushed.X).ravel(), index=pushed.obs_names)
            by_group = arrived.groupby(tgt_groups.reindex(arrived.index)).sum()
            sizes[tgt_day] = {g: float(by_group.get(g, 0.0)) for g in groups}

        return sizes


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
        time_var: str,
    ):
        timepoints = list(sorted(meta[time_var].unique()))
        day_pairs = window(timepoints, 2)
        super().__init__(
            meta=meta,
            timepoints=timepoints,
            day_pairs=day_pairs,
            time_var=time_var,
        )
        self.tmaps = tmaps

    @classmethod
    def load(cls, path: str) -> "GenericOTModel":
        """Load a GenericOTModel from a directory written by BaseOTModel.save()."""
        path = Path(path)

        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        tmaps = {}
        for t0, t1 in metadata["day_pairs"]:
            coupling = ad.read_h5ad(path / f"coupling_{t0}_{t1}.h5ad")
            tmaps[tuple([t0, t1])] = coupling

        meta = pd.read_csv(path / "meta.csv", index_col=0)

        return cls(tmaps=tmaps, meta=meta, time_var=metadata["time_var"])

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
