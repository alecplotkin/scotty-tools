import numpy as np
import anndata as ad
import pytest

from scotty.models.trajectory.ot import coarsen_ot_model, GenericOTModel


def test_timepoints_sorted(ot_model):
    assert ot_model.timepoints == [1.0, 2.0, 3.0]


def test_day_pairs(ot_model):
    assert list(ot_model.day_pairs) == [(1.0, 2.0), (2.0, 3.0)]


def test_get_coupling_shape(ot_model):
    tmap = ot_model.get_coupling(1.0, 2.0)
    assert tmap.shape == (15, 15)


def test_get_coupling_obs_names(ot_model, cell_ids):
    tmap = ot_model.get_coupling(1.0, 2.0)
    assert list(tmap.obs_names) == cell_ids[1.0]


def test_get_coupling_var_names(ot_model, cell_ids):
    tmap = ot_model.get_coupling(1.0, 2.0)
    assert list(tmap.var_names) == cell_ids[2.0]


def _make_p(obs_names, var_name='x'):
    p = ad.AnnData(np.ones((len(obs_names), 1)))
    p.obs_names = obs_names
    p.var_names = [var_name]
    return p


def test_push_forward_output_shape(ot_model, cell_ids):
    p = _make_p(cell_ids[1.0])
    result = ot_model.push_forward(p, 1.0, 2.0)
    assert result.n_obs == 15


def test_push_forward_preserves_var_names(ot_model, cell_ids):
    p = _make_p(cell_ids[1.0])
    result = ot_model.push_forward(p, 1.0, 2.0)
    assert list(result.var_names) == ['x']


def test_pull_back_output_shape(ot_model, cell_ids):
    p = _make_p(cell_ids[2.0])
    result = ot_model.pull_back(p, 1.0, 2.0)
    assert result.n_obs == 15


def test_pull_back_preserves_var_names(ot_model, cell_ids):
    p = _make_p(cell_ids[2.0])
    result = ot_model.pull_back(p, 1.0, 2.0)
    assert list(result.var_names) == ['x']


def test_push_forward_non_negative(ot_model, cell_ids):
    p = _make_p(cell_ids[1.0])
    result = ot_model.push_forward(p, 1.0, 2.0)
    assert (result.X >= 0).all()


def test_pull_back_non_negative(ot_model, cell_ids):
    p = _make_p(cell_ids[2.0])
    result = ot_model.pull_back(p, 1.0, 2.0)
    assert (result.X >= 0).all()


def test_push_forward_does_not_mutate_coupling(unbalanced_ot_model, cell_ids):
    before = unbalanced_ot_model.get_coupling(1.0, 2.0).X.copy()
    unbalanced_ot_model.push_forward(_make_p(cell_ids[1.0]), 1.0, 2.0, norm_axis=1)
    after = unbalanced_ot_model.get_coupling(1.0, 2.0).X
    assert np.allclose(before, after)


def test_pull_back_does_not_mutate_coupling(unbalanced_ot_model, cell_ids):
    before = unbalanced_ot_model.get_coupling(1.0, 2.0).X.copy()
    unbalanced_ot_model.pull_back(_make_p(cell_ids[2.0]), 1.0, 2.0, norm_axis=1)
    after = unbalanced_ot_model.get_coupling(1.0, 2.0).X
    assert np.allclose(before, after)


def test_push_forward_repeatable(unbalanced_ot_model, cell_ids):
    p = _make_p(cell_ids[1.0])
    first = unbalanced_ot_model.push_forward(p, 1.0, 2.0, norm_axis=1).X.copy()
    second = unbalanced_ot_model.push_forward(p, 1.0, 2.0, norm_axis=1).X
    assert np.allclose(first, second)


def test_push_forward_norm_axes_are_independent(unbalanced_ot_model, cell_ids):
    """Pushing with one norm_axis must not change what the other norm_axis gives."""
    p = _make_p(cell_ids[1.0])
    expected = unbalanced_ot_model.push_forward(p, 1.0, 2.0, norm_axis=0).X.copy()
    unbalanced_ot_model.push_forward(p, 1.0, 2.0, norm_axis=1)
    after = unbalanced_ot_model.push_forward(p, 1.0, 2.0, norm_axis=0).X
    assert np.allclose(expected, after)


def test_push_forward_matches_row_stochastic_coupling(unbalanced_ot_model, cell_ids):
    K = unbalanced_ot_model.get_coupling(1.0, 2.0).X.copy()
    p = _make_p(cell_ids[1.0])
    result = unbalanced_ot_model.push_forward(p, 1.0, 2.0, norm_axis=1)
    expected = (K / K.sum(1, keepdims=True)).T @ p.X
    assert np.allclose(result.X, expected)


def test_pull_back_matches_column_stochastic_coupling(unbalanced_ot_model, cell_ids):
    K = unbalanced_ot_model.get_coupling(1.0, 2.0).X.copy()
    p = _make_p(cell_ids[2.0])
    result = unbalanced_ot_model.pull_back(p, 1.0, 2.0, norm_axis=1)
    expected = (K / K.sum(1, keepdims=True)) @ p.X
    assert np.allclose(result.X, expected)


def test_push_forward_honors_obs_name_order(unbalanced_ot_model, cell_ids):
    """p may be ordered differently from the coupling's obs_names."""
    p = _make_p(cell_ids[1.0])
    shuffled = p[list(reversed(cell_ids[1.0])), :].copy()
    shuffled.X = np.arange(shuffled.n_obs, dtype=float).reshape(-1, 1)
    aligned = shuffled[cell_ids[1.0], :].copy()
    a = unbalanced_ot_model.push_forward(shuffled, 1.0, 2.0, norm_axis=1)
    b = unbalanced_ot_model.push_forward(aligned, 1.0, 2.0, norm_axis=1)
    assert np.allclose(a.X, b.X)


def test_push_forward_unknown_cell_raises(unbalanced_ot_model):
    p = _make_p(['not_a_cell'])
    with pytest.raises(KeyError):
        unbalanced_ot_model.push_forward(p, 1.0, 2.0)


@pytest.mark.xfail(
    strict=True,
    reason="known bug: coarsen_ot_model passes unexpected kwargs to GenericOTModel",
)
def test_coarsen_ot_model(ot_model, cell_ids):
    import pandas as pd

    timepoint_mixtures = {}
    for tp, ids in cell_ids.items():
        n = len(ids)
        timepoint_mixtures[tp] = pd.DataFrame(
            np.eye(n),
            index=ids,
            columns=[f'cluster_{i}' for i in range(n)],
        )
    coarsen_ot_model(ot_model, timepoint_mixtures)


def test_save_creates_expected_files(ot_model, tmp_path):
    ot_model.save(str(tmp_path))
    assert (tmp_path / "meta.csv").exists()
    assert (tmp_path / "metadata.json").exists()
    assert (tmp_path / "coupling_1.0_2.0.h5ad").exists()
    assert (tmp_path / "coupling_2.0_3.0.h5ad").exists()


def test_save_metadata_json_contents(ot_model, tmp_path):
    import json
    ot_model.save(str(tmp_path))
    with open(tmp_path / "metadata.json") as f:
        metadata = json.load(f)
    assert metadata["time_var"] == "day"
    assert metadata["timepoints"] == [1.0, 2.0, 3.0]
    assert metadata["day_pairs"] == [[1.0, 2.0], [2.0, 3.0]]


def test_generic_load_roundtrip_timepoints(ot_model, tmp_path):
    ot_model.save(str(tmp_path))
    loaded = GenericOTModel.load(str(tmp_path))
    assert loaded.timepoints == ot_model.timepoints


def test_generic_load_roundtrip_day_pairs(ot_model, tmp_path):
    ot_model.save(str(tmp_path))
    loaded = GenericOTModel.load(str(tmp_path))
    assert loaded.day_pairs == ot_model.day_pairs


def test_generic_load_roundtrip_time_var(ot_model, tmp_path):
    ot_model.save(str(tmp_path))
    loaded = GenericOTModel.load(str(tmp_path))
    assert loaded.time_var == ot_model.time_var


def test_generic_load_roundtrip_coupling_values(ot_model, tmp_path):
    ot_model.save(str(tmp_path))
    loaded = GenericOTModel.load(str(tmp_path))
    for t0, t1 in ot_model.day_pairs:
        orig = ot_model.get_coupling(t0, t1)
        reloaded = loaded.get_coupling(t0, t1)
        assert np.allclose(orig.X, reloaded.X)


def test_generic_load_roundtrip_coupling_obs_names(ot_model, tmp_path, cell_ids):
    ot_model.save(str(tmp_path))
    loaded = GenericOTModel.load(str(tmp_path))
    tmap = loaded.get_coupling(1.0, 2.0)
    assert list(tmap.obs_names) == cell_ids[1.0]


def test_generic_load_roundtrip_coupling_var_names(ot_model, tmp_path, cell_ids):
    ot_model.save(str(tmp_path))
    loaded = GenericOTModel.load(str(tmp_path))
    tmap = loaded.get_coupling(1.0, 2.0)
    assert list(tmap.var_names) == cell_ids[2.0]


def test_generic_load_roundtrip_meta(ot_model, tmp_path):
    ot_model.save(str(tmp_path))
    loaded = GenericOTModel.load(str(tmp_path))
    assert list(loaded.meta.index) == list(ot_model.meta.index)
    assert list(loaded.meta["day"]) == list(ot_model.meta["day"])


def test_base_load_raises(ot_model, tmp_path):
    from scotty.models.trajectory.ot import BaseOTModel
    ot_model.save(str(tmp_path))
    with pytest.raises(NotImplementedError):
        BaseOTModel.load(str(tmp_path))


def test_save_creates_directory_if_missing(ot_model, tmp_path):
    dest = tmp_path / "nested" / "subdir"
    ot_model.save(str(dest))
    assert dest.is_dir()
    assert (dest / "metadata.json").exists()


# --- MoscotModel marginal adjustments -------------------------------------
# Exercised through duck-typed stand-ins for moscot's problems, so the tests
# neither solve an OT problem nor depend on moscot internals.

import pandas as pd

from scotty.models.trajectory.ot import MoscotModel


class _FakeProblem:
    """Minimal stand-in for moscot's BirthDeathProblem."""

    def __init__(self, adata_src, adata_tgt, prior_growth):
        self.adata_src = adata_src
        self.adata_tgt = adata_tgt
        self._prior_growth = np.asarray(prior_growth, dtype=float)
        self._a = self._prior_growth / self._prior_growth.sum()
        self._b = np.full(adata_tgt.n_obs, 1 / adata_tgt.n_obs)

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b


def _group_adata(groups):
    """AnnData whose obs carries a 'compartment' column (values in `groups`)."""
    groups = list(groups)
    adata = ad.AnnData(np.zeros((len(groups), 1)))
    adata.obs_names = [f'cell_{i}' for i in range(len(groups))]
    adata.obs['compartment'] = pd.Categorical(groups)
    return adata


def _fake_moscot_model(problems):
    model = MoscotModel.__new__(MoscotModel)
    model.moscot_model = problems
    model.compartment_key = 'compartment'
    return model


@pytest.fixture
def growth_problem():
    """One day pair; group A growth rates 1-5, group B growth rates 10-50."""
    groups = ['A'] * 5 + ['B'] * 5
    prior_growth = np.array([1., 2., 3., 4., 5., 10., 20., 30., 40., 50.])
    adata = _group_adata(groups)
    problem = _FakeProblem(adata, adata.copy(), prior_growth)
    return _fake_moscot_model({(1.0, 2.0): problem}), problem


def test_clip_growth_rates_ungrouped_uses_global_bounds(growth_problem):
    model, problem = growth_problem
    model.clip_growth_rates(upper_quantile=0.5)
    # Global median of the 10 values is 7.5; every value above it is clipped.
    assert problem._prior_growth.max() == pytest.approx(7.5)
    assert problem._prior_growth[:5].tolist() == [1., 2., 3., 4., 5.]


def test_clip_growth_rates_grouped_bounds_each_group(growth_problem):
    model, problem = growth_problem
    model.clip_growth_rates(upper_quantile=0.5, group_key='compartment')
    clipped = problem._prior_growth
    # Group A median is 3, group B median is 30: neither group is clipped
    # against the other group's scale.
    assert clipped[:5].max() == pytest.approx(3.)
    assert clipped[5:].max() == pytest.approx(30.)
    assert clipped[:5].min() == pytest.approx(1.)


def test_clip_growth_rates_renormalizes_a(growth_problem):
    model, problem = growth_problem
    model.clip_growth_rates(upper_quantile=0.5, group_key='compartment')
    assert problem._a.sum() == pytest.approx(1.)
    assert np.allclose(problem._a, problem._prior_growth / problem._prior_growth.sum())


@pytest.fixture
def rescale_problem():
    """Day pair with 5 'A' and 5 'B' cells on both sides (observed 50/50)."""
    adata = _group_adata(['A'] * 5 + ['B'] * 5)
    problem = _FakeProblem(adata, adata.copy(), np.ones(10))
    return _fake_moscot_model({(1.0, 2.0): problem}), problem


def test_rescale_marginals_hits_target_ratio(rescale_problem):
    model, problem = rescale_problem
    target = {1.0: {'A': 0.9, 'B': 0.1}, 2.0: {'A': 0.25, 'B': 0.75}}
    model.rescale_marginals(target, group_key='compartment')
    assert problem.a[:5].sum() == pytest.approx(0.9)
    assert problem.a[5:].sum() == pytest.approx(0.1)
    assert problem.b[:5].sum() == pytest.approx(0.25)
    assert problem.b[5:].sum() == pytest.approx(0.75)


def test_rescale_marginals_normalizes(rescale_problem):
    model, problem = rescale_problem
    model.rescale_marginals(
        {1.0: {'A': 3., 'B': 1.}, 2.0: {'A': 1., 'B': 1.}}, group_key='compartment'
    )
    assert problem.a.sum() == pytest.approx(1.)
    assert problem.b.sum() == pytest.approx(1.)
    # Unnormalized frequencies are fine: only their ratio matters.
    assert problem.a[:5].sum() == pytest.approx(0.75)


def test_rescale_marginals_preserves_within_group_ratios():
    adata = _group_adata(['A'] * 2 + ['B'] * 2)
    problem = _FakeProblem(adata, adata.copy(), np.array([1., 3., 1., 1.]))
    model = _fake_moscot_model({(1.0, 2.0): problem})
    model.rescale_marginals(
        {1.0: {'A': 0.5, 'B': 0.5}, 2.0: {'A': 0.5, 'B': 0.5}}, group_key='compartment'
    )
    assert problem.a[1] / problem.a[0] == pytest.approx(3.)


def test_rescale_marginals_group_absent_at_timepoint():
    """A compartment missing on one side (e.g. gut before day 4) is skipped."""
    adata_src = _group_adata(['A'] * 4)
    adata_tgt = _group_adata(['A'] * 2 + ['B'] * 2)
    problem = _FakeProblem(adata_src, adata_tgt, np.ones(4))
    model = _fake_moscot_model({(1.0, 2.0): problem})
    model.rescale_marginals(
        {1.0: {'A': 0.6, 'B': 0.4}, 2.0: {'A': 0.6, 'B': 0.4}}, group_key='compartment'
    )
    assert problem.a.sum() == pytest.approx(1.)
    assert np.allclose(problem.a, 0.25)  # only group A present -> uniform
    assert problem.b[:2].sum() == pytest.approx(0.6)


def test_rescale_marginals_missing_timepoint_raises(rescale_problem):
    model, _ = rescale_problem
    with pytest.raises(KeyError, match='timepoint'):
        model.rescale_marginals({1.0: {'A': 0.5, 'B': 0.5}}, group_key='compartment')


def test_rescale_marginals_missing_group_raises(rescale_problem):
    model, _ = rescale_problem
    with pytest.raises(KeyError, match='missing groups'):
        model.rescale_marginals(
            {1.0: {'A': 1.0}, 2.0: {'A': 0.5, 'B': 0.5}}, group_key='compartment'
        )


def test_rescale_marginals_nonpositive_target_raises(rescale_problem):
    model, _ = rescale_problem
    with pytest.raises(ValueError, match='positive'):
        model.rescale_marginals(
            {1.0: {'A': 1.0, 'B': 0.0}, 2.0: {'A': 0.5, 'B': 0.5}},
            group_key='compartment',
        )


@pytest.fixture
def two_pair_growth_model():
    """Two day pairs with different dt, so per-timepoint scaling is distinguishable."""
    adata = _group_adata(['A'] * 4)
    p1 = _FakeProblem(adata, adata.copy(), np.array([1., 2., 3., 4.]))
    p2 = _FakeProblem(adata.copy(), adata.copy(), np.array([2., 4., 6., 8.]))
    # dt = 1 and dt = 3 respectively.
    return _fake_moscot_model({(0.0, 1.0): p1, (1.0, 4.0): p2}), p1, p2


def test_set_death_rates_scales_prior_growth(two_pair_growth_model):
    model, p1, p2 = two_pair_growth_model
    before1, before2 = p1._prior_growth.copy(), p2._prior_growth.copy()
    model.set_death_rates({0.0: 0.5, 1.0: 0.25})
    # dt = 1 for the first pair, dt = 3 for the second.
    np.testing.assert_allclose(p1._prior_growth, before1 * np.exp(-0.5 * 1))
    np.testing.assert_allclose(p2._prior_growth, before2 * np.exp(-0.25 * 3))


def test_set_death_rates_leaves_marginal_unchanged(two_pair_growth_model):
    """The cancellation property: a constant rate cannot change the couplings."""
    model, p1, p2 = two_pair_growth_model
    a1_before, a2_before = p1.a.copy(), p2.a.copy()
    model.set_death_rates({0.0: 0.5, 1.0: -1.5})
    np.testing.assert_allclose(p1.a, a1_before)
    np.testing.assert_allclose(p2.a, a2_before)


def test_set_death_rates_marginal_stays_normalized(two_pair_growth_model):
    model, p1, p2 = two_pair_growth_model
    model.set_death_rates({0.0: 2.0, 1.0: 2.0})
    assert p1.a.sum() == pytest.approx(1.0)
    assert p2.a.sum() == pytest.approx(1.0)


def test_set_death_rates_negative_rate_grows_population(two_pair_growth_model):
    model, p1, _ = two_pair_growth_model
    before = p1._prior_growth.copy()
    model.set_death_rates({0.0: -0.5, 1.0: 0.0})
    assert (p1._prior_growth > before).all()


def test_set_death_rates_zero_is_identity(two_pair_growth_model):
    model, p1, p2 = two_pair_growth_model
    before1, before2 = p1._prior_growth.copy(), p2._prior_growth.copy()
    model.set_death_rates({0.0: 0.0, 1.0: 0.0})
    np.testing.assert_allclose(p1._prior_growth, before1)
    np.testing.assert_allclose(p2._prior_growth, before2)


def test_set_death_rates_missing_timepoint_raises(two_pair_growth_model):
    model, _, _ = two_pair_growth_model
    with pytest.raises(KeyError, match='1.0'):
        model.set_death_rates({0.0: 0.5})


# --- estimate_population_sizes_by_group -------------------------------------

class _FakeSolution:
    def __init__(self, transport_matrix):
        self.transport_matrix = transport_matrix


class _FakeGrowthProblem(_FakeProblem):
    """_FakeProblem plus the coupling that push_forward needs."""

    def __init__(self, adata_src, adata_tgt, prior_growth, transport_matrix):
        super().__init__(adata_src, adata_tgt, prior_growth)
        self.solution = _FakeSolution(np.asarray(transport_matrix, dtype=float))


def _group_model(problems, adata, group_key='compartment'):
    model = MoscotModel.__new__(MoscotModel)

    class _MM(dict):
        pass

    mm = _MM(problems)
    mm.adata = adata
    model.moscot_model = mm
    model.compartment_key = group_key
    return model


def _two_group_adata(n_per_group=2, groups=('A', 'B')):
    labels = [g for g in groups for _ in range(n_per_group)]
    adata = ad.AnnData(np.zeros((len(labels), 1)))
    adata.obs_names = [f'{g}{i}' for g in groups for i in range(n_per_group)]
    adata.obs['compartment'] = pd.Categorical(labels)
    return adata


def _identity_coupling(n):
    return np.eye(n)


def _swap_coupling(n_per_group):
    """Every A cell maps entirely to B cells and vice versa."""
    n = 2 * n_per_group
    T = np.zeros((n, n))
    T[:n_per_group, n_per_group:] = 1.0 / n_per_group
    T[n_per_group:, :n_per_group] = 1.0 / n_per_group
    return T


def test_by_group_identity_coupling_unit_growth_preserves_sizes():
    adata = _two_group_adata()
    prob = _FakeGrowthProblem(adata, adata.copy(), np.ones(4), _identity_coupling(4))
    model = _group_model({(0.0, 1.0): prob}, adata)
    sizes = model.estimate_population_sizes_by_group('compartment', {'A': 100.0, 'B': 25.0})
    assert sizes[1.0]['A'] == pytest.approx(100.0)
    assert sizes[1.0]['B'] == pytest.approx(25.0)


def test_by_group_swap_coupling_exchanges_sizes():
    adata = _two_group_adata()
    prob = _FakeGrowthProblem(adata, adata.copy(), np.ones(4), _swap_coupling(2))
    model = _group_model({(0.0, 1.0): prob}, adata)
    sizes = model.estimate_population_sizes_by_group('compartment', {'A': 100.0, 'B': 25.0})
    assert sizes[1.0]['A'] == pytest.approx(25.0)
    assert sizes[1.0]['B'] == pytest.approx(100.0)


def test_by_group_growth_scales_by_mean_prior_growth():
    """_prior_growth is already g ** dt, so a group scales by its mean."""
    adata = _two_group_adata()
    growth = np.array([2.0, 4.0, 1.0, 1.0])       # A cells grow, B cells do not
    prob = _FakeGrowthProblem(adata, adata.copy(), growth, _identity_coupling(4))
    model = _group_model({(0.0, 1.0): prob}, adata)
    sizes = model.estimate_population_sizes_by_group('compartment', {'A': 100.0, 'B': 40.0})
    assert sizes[1.0]['A'] == pytest.approx(100.0 * 3.0)   # mean(2, 4) = 3
    assert sizes[1.0]['B'] == pytest.approx(40.0)


def test_by_group_records_initial_sizes_at_init_day():
    adata = _two_group_adata()
    prob = _FakeGrowthProblem(adata, adata.copy(), np.ones(4), _identity_coupling(4))
    model = _group_model({(0.0, 1.0): prob}, adata)
    sizes = model.estimate_population_sizes_by_group('compartment', {'A': 7.0, 'B': 3.0})
    assert sizes[0.0] == pytest.approx({'A': 7.0, 'B': 3.0})


def test_by_group_absent_group_carries_no_mass():
    """A group with no cells at the source contributes nothing, without erroring."""
    adata = _two_group_adata()
    prob = _FakeGrowthProblem(adata, adata.copy(), np.ones(4), _identity_coupling(4))
    model = _group_model({(0.0, 1.0): prob}, adata)
    sizes = model.estimate_population_sizes_by_group('compartment', {'A': 100.0})
    assert sizes[1.0]['A'] == pytest.approx(100.0)
    assert sizes[1.0]['B'] == pytest.approx(0.0)


def test_by_group_chains_across_day_pairs():
    adata = _two_group_adata()
    growth = np.array([2.0, 2.0, 1.0, 1.0])
    problems = {
        (0.0, 1.0): _FakeGrowthProblem(adata, adata.copy(), growth, _identity_coupling(4)),
        (1.0, 2.0): _FakeGrowthProblem(adata.copy(), adata.copy(), growth,
                                       _identity_coupling(4)),
    }
    model = _group_model(problems, adata)
    sizes = model.estimate_population_sizes_by_group('compartment', {'A': 10.0, 'B': 10.0})
    assert sizes[2.0]['A'] == pytest.approx(40.0)   # doubled twice
    assert sizes[2.0]['B'] == pytest.approx(10.0)


def test_by_group_unknown_group_key_raises():
    adata = _two_group_adata()
    prob = _FakeGrowthProblem(adata, adata.copy(), np.ones(4), _identity_coupling(4))
    model = _group_model({(0.0, 1.0): prob}, adata)
    with pytest.raises(KeyError, match='tissue'):
        model.estimate_population_sizes_by_group('tissue', {'A': 1.0})


def test_by_group_unknown_group_in_init_sizes_raises():
    adata = _two_group_adata()
    prob = _FakeGrowthProblem(adata, adata.copy(), np.ones(4), _identity_coupling(4))
    model = _group_model({(0.0, 1.0): prob}, adata)
    with pytest.raises(KeyError, match='C'):
        model.estimate_population_sizes_by_group('compartment', {'A': 1.0, 'C': 1.0})
