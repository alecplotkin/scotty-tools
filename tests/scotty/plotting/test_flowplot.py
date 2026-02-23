import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from scotty.plotting._flowplot import plot_flows


@pytest.fixture
def flow_df():
    sources = 3 * ['A'] + 3 * ['B'] + 3 * ['C']
    targets = 3 * ['A', 'B', 'C']
    outflows = np.arange(1, 10, dtype=float)
    inflows = np.arange(9, 0, -1, dtype=float)
    return pd.DataFrame({
        'source': sources,
        'target': targets,
        'outflow': outflows,
        'inflow': inflows,
    })


def test_plot_flows_returns_axes(flow_df):
    ax = plot_flows(
        sources=flow_df['source'],
        targets=flow_df['target'],
        outflows=flow_df['outflow'],
        inflows=flow_df['inflow'],
    )
    assert isinstance(ax, plt.Axes)
    plt.close('all')


def test_plot_flows_with_group_order(flow_df):
    ax = plot_flows(
        sources=flow_df['source'],
        targets=flow_df['target'],
        outflows=flow_df['outflow'],
        inflows=flow_df['inflow'],
        group_order=['C', 'A', 'B'],
    )
    assert isinstance(ax, plt.Axes)
    plt.close('all')


def test_plot_flows_with_threshold(flow_df):
    ax = plot_flows(
        sources=flow_df['source'],
        targets=flow_df['target'],
        outflows=flow_df['outflow'],
        inflows=flow_df['inflow'],
        min_flow_threshold=5.0,
    )
    assert isinstance(ax, plt.Axes)
    plt.close('all')
