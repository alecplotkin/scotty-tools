import matplotlib.pyplot as plt

from scotty.plotting.sankey import Sankey


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
