import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from scotty.plotting.sankey import Sankey, _calculate_entropy


def _flow_matrix(flow_df, col, groups):
    return flow_df.pivot(
        index='source', columns='target', values=col
    ).loc[groups, groups].values


def _snapshot_couplings(ot_model):
    """Copy every coupling up front, before anything has a chance to mutate them."""
    return {pair: ot_model.get_coupling(*pair).X.copy() for pair in ot_model.day_pairs}


def _reference_flows(ot_model, couplings, subsets, t0, t1, groups):
    """Flows computed directly from the raw couplings, no push_forward involved.

    `couplings` must be snapshotted *before* the code under test runs, otherwise
    a mutating implementation would be compared against its own corrupted state
    and the check would pass vacuously.
    """
    meta = ot_model.meta
    dummies = pd.get_dummies(subsets).astype(float)
    S = {
        t: dummies.loc[meta.index[meta[ot_model.time_var] == t], groups].values
        for t in (t0, t1)
    }
    fwd, bwd = None, None
    tps = np.array(ot_model.timepoints)
    span = tps[(tps >= t0) & (tps <= t1)]
    for a, b in zip(span[:-1], span[1:]):
        K = couplings[(a, b)]
        R = K / K.sum(1, keepdims=True)
        C = K / K.sum(0, keepdims=True)
        fwd = R if fwd is None else fwd @ R
        bwd = C if bwd is None else bwd @ C
    out = S[t0].T @ fwd @ S[t1]
    inn = S[t0].T @ bwd @ S[t1]
    return out / out.sum(), inn / inn.sum()


def test_sankey_init(ot_model, all_cell_subsets):
    sankey = Sankey(ot_model, all_cell_subsets)
    assert sankey is not None


def test_calculate_flows_columns(ot_model, all_cell_subsets):
    sankey = Sankey(ot_model, all_cell_subsets)
    flow_df = sankey.calculate_flows(1.0, 2.0)
    assert {'source', 'target', 'outflow', 'inflow'}.issubset(set(flow_df.columns))


def test_calculate_flows_outflow_nonneg(ot_model, all_cell_subsets):
    sankey = Sankey(ot_model, all_cell_subsets)
    flow_df = sankey.calculate_flows(1.0, 2.0)
    assert (flow_df['outflow'] >= 0).all()


def test_calculate_flows_caching(ot_model, all_cell_subsets):
    # calculate_flows always computes; the cache is populated for plot_sankey to reuse.
    sankey = Sankey(ot_model, all_cell_subsets, cache_flow_dfs=True)
    flow_df = sankey.calculate_flows(1.0, 2.0)
    assert (1.0, 2.0) in sankey.flow_dfs
    assert sankey.flow_dfs[(1.0, 2.0)] is flow_df


def test_plot_sankey_returns_axes(ot_model, all_cell_subsets):
    sankey = Sankey(ot_model, all_cell_subsets)
    ax = sankey.plot_sankey(1.0, 2.0)
    assert isinstance(ax, plt.Axes)
    plt.close('all')


def test_plot_all_transitions_returns_figure(ot_model, all_cell_subsets):
    sankey = Sankey(ot_model, all_cell_subsets)
    fig = sankey.plot_all_transitions()
    assert isinstance(fig, plt.Figure)
    plt.close('all')


GROUPS = ['A', 'B', 'C']


def test_outflow_matches_forward_coupling(unbalanced_ot_model, all_cell_subsets):
    couplings = _snapshot_couplings(unbalanced_ot_model)
    flow_df = Sankey(unbalanced_ot_model, all_cell_subsets).calculate_flows(1.0, 2.0)
    out_ref, _ = _reference_flows(
        unbalanced_ot_model, couplings, all_cell_subsets, 1.0, 2.0, GROUPS
    )
    assert np.allclose(_flow_matrix(flow_df, 'outflow', GROUPS), out_ref)


def test_inflow_matches_backward_coupling(unbalanced_ot_model, all_cell_subsets):
    """Inflow must use the column-normalized raw coupling, not a rescaled one."""
    couplings = _snapshot_couplings(unbalanced_ot_model)
    flow_df = Sankey(unbalanced_ot_model, all_cell_subsets).calculate_flows(1.0, 2.0)
    _, in_ref = _reference_flows(
        unbalanced_ot_model, couplings, all_cell_subsets, 1.0, 2.0, GROUPS
    )
    assert np.allclose(_flow_matrix(flow_df, 'inflow', GROUPS), in_ref)


def test_multistep_flows_match_composed_couplings(unbalanced_ot_model, all_cell_subsets):
    couplings = _snapshot_couplings(unbalanced_ot_model)
    flow_df = Sankey(unbalanced_ot_model, all_cell_subsets).calculate_flows(1.0, 3.0)
    out_ref, in_ref = _reference_flows(
        unbalanced_ot_model, couplings, all_cell_subsets, 1.0, 3.0, GROUPS
    )
    assert np.allclose(_flow_matrix(flow_df, 'outflow', GROUPS), out_ref)
    assert np.allclose(_flow_matrix(flow_df, 'inflow', GROUPS), in_ref)


def test_calculate_flows_is_repeatable(unbalanced_ot_model, all_cell_subsets):
    sankey = Sankey(unbalanced_ot_model, all_cell_subsets, cache_flow_dfs=False)
    first = sankey.calculate_flows(1.0, 2.0)
    second = sankey.calculate_flows(1.0, 2.0)
    assert np.allclose(first['outflow'], second['outflow'])
    assert np.allclose(first['inflow'], second['inflow'])


def test_calculate_flows_does_not_mutate_model(unbalanced_ot_model, all_cell_subsets):
    before = unbalanced_ot_model.get_coupling(1.0, 2.0).X.copy()
    Sankey(unbalanced_ot_model, all_cell_subsets).calculate_flows(1.0, 2.0)
    assert np.allclose(before, unbalanced_ot_model.get_coupling(1.0, 2.0).X)


def test_outflow_source_marginal_is_subset_size(unbalanced_ot_model, all_cell_subsets):
    """Row-stochastic forward push conserves each source subset's mass."""
    flow_df = Sankey(unbalanced_ot_model, all_cell_subsets).calculate_flows(1.0, 2.0)
    meta = unbalanced_ot_model.meta
    ix = meta.index[meta['day'] == 1.0]
    expected = all_cell_subsets[ix].value_counts(normalize=True).loc[GROUPS].values
    got = flow_df.groupby('source')['outflow'].sum().loc[GROUPS].values
    assert np.allclose(got, expected)


def test_inflow_target_marginal_is_subset_size(unbalanced_ot_model, all_cell_subsets):
    flow_df = Sankey(unbalanced_ot_model, all_cell_subsets).calculate_flows(1.0, 2.0)
    meta = unbalanced_ot_model.meta
    ix = meta.index[meta['day'] == 2.0]
    expected = all_cell_subsets[ix].value_counts(normalize=True).loc[GROUPS].values
    got = flow_df.groupby('target')['inflow'].sum().loc[GROUPS].values
    assert np.allclose(got, expected)


def test_calculate_entropy_handles_zero_flows():
    assert _calculate_entropy(pd.Series([0.5, 0.5, 0.0])) == pytest.approx(np.log(2))


def test_calculate_entropy_does_not_mutate_input():
    x = pd.Series([1.0, 3.0])
    _calculate_entropy(x)
    assert list(x) == [1.0, 3.0]
