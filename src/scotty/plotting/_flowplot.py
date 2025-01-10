import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.axes import Axes
from typing import Dict, List, Union
from collections import defaultdict


# TODO: plot text.
def plot_flows(
        sources,
        targets,
        outflows,
        inflows,
        palette: Union[str, List, Dict] = 'husl',
        group_order: List = None,
        show_source_labels: bool = True,
        show_target_labels: bool = True,
        show_text: bool = True,
        endpoint_width: float = 0.05,
        endpoint_linewidth: float = 1,
        endpoint_edgecolor: str = 'w',
        textpad: float = 0.025,
        fontsize: float = 14,
        flow_alpha: float = 0.6,
        min_flow_alpha: float = 0.05,
        min_flow_threshold: float = None,
        n_steps: int = 50,
        kernel_width: float = 0.5,
        ax: Axes = None,
):

    if ax is None:
        _, ax = plt.subplots(1, 1)
    ax.axis('off')
    # Assume for now that sources, targets, outflows, inflows are the same
    # length and nonempty.

    # TODO: allow user to pass a single flow, in case that outflows == inflows.

    groups = np.unique((sources, targets))
    if group_order is None:
        group_order = groups
    elif len(group_order) < len(groups):
        raise ValueError('group_order does not contain all groups.')

    if isinstance(palette, dict):
        color_dict = palette
    else:
        if isinstance(palette, str):
            palette = sns.color_palette(palette, len(group_order))
        color_dict = dict(zip(group_order, palette))

    flow_df = pd.DataFrame({
        'source': pd.Series(pd.Categorical(sources, categories=group_order)),
        'target': pd.Series(pd.Categorical(targets, categories=group_order)),
        'outflow': outflows,
        'inflow': inflows,
    })
    if (flow_df[['source', 'target']].value_counts() > 1).any():
        raise ValueError(
            'Duplicate flows detected; '
            'make sure all (source, target) pairs are unique'
        )

    # If total_outflow != total_inflow, center the flow with lower mass.
    y0_source, y0_target = _calculate_endpoint_offsets(outflows, inflows)

    # Make dicts of source and target endpoint ylims
    source_ranges = _calculate_endpoint_ranges(
        flow_df, group_var='source', value_var='outflow', init_offset=y0_source
    )
    target_ranges = _calculate_endpoint_ranges(
        flow_df, group_var='target', value_var='inflow', init_offset=y0_target
    )

    # Get exact y ranges for each (source, target) pair.
    flow_df = _calculate_flow_ranges(flow_df, source_ranges, target_ranges)

    # Sort flow_df by max flow prior to plotting, to prioritize larger flows
    # (which will be plotted on top).
    flow_df['max_flow'] = flow_df[['inflow', 'outflow']].max(1)
    flow_df = flow_df.sort_values('max_flow')

    # TODO: check boundary conditions on kernel_width.
    kernel_steps = int(kernel_width * n_steps * 2)
    z = np.linspace(-3, 3, kernel_steps)
    kernel = np.exp(-z**2)
    kernel /= kernel.sum()

    for ix, row in flow_df.iterrows():
        if _check_min_flow_conditions(row, min_flow_threshold):
            alpha = min_flow_alpha
        else:
            alpha = flow_alpha
        y0_source = row['y0_source']
        y1_source = row['y1_source']
        y0_target = row['y0_target']
        y1_target = row['y1_target']
        y_low = np.array(n_steps * [y0_source] + n_steps * [y0_target])
        y_low = np.convolve(y_low, kernel, mode='valid')
        y_high = np.array(n_steps * [y1_source] + n_steps * [y1_target])
        y_high = np.convolve(y_high, kernel, mode='valid')

        x = np.linspace(0.01, 0.99, 2*n_steps - kernel_steps + 1)
        ax.fill_between(
            x, y_low, y_high,
            alpha=alpha,
            color=color_dict[row['source']],
        )

    # Plot endpoints.
    for source, (y0, y1) in source_ranges.items():
        rect = patches.Rectangle(
            (-0.01, y0), endpoint_width + 0.01, y1 - y0,
            facecolor=color_dict[source],
            edgecolor=endpoint_edgecolor,
            linewidth=endpoint_linewidth,
        )
        ax.add_patch(rect)

    for target, (y0, y1) in target_ranges.items():
        rect = patches.Rectangle(
            (1 - endpoint_width, y0), endpoint_width + 0.01, y1 - y0,
            facecolor=color_dict[target],
            edgecolor=endpoint_edgecolor,
            linewidth=endpoint_linewidth,
        )
        ax.add_patch(rect)

    return ax


def _calculate_endpoint_offsets(outflows, inflows):
    total_outflow = np.sum(outflows)
    total_inflow = np.sum(inflows)
    if total_outflow < total_inflow:
        y0_source = (total_inflow - total_outflow) / 2
        y0_target = 0
    elif total_outflow > total_inflow:
        y0_source = 0
        y0_target = (total_outflow - total_inflow) / 2
    else:
        y0_source = y0_target = 0
    return y0_source, y0_target


def _calculate_endpoint_ranges(flow_df, group_var, value_var, init_offset=0):
    ranges = dict()
    y0 = init_offset
    for group, df in flow_df.groupby(group_var, observed=True):
        y1 = df[value_var].sum() + y0
        ranges[group] = (y0, y1)
        y0 = y1
    return ranges


def _calculate_flow_ranges(flow_df, source_ranges, target_ranges):
    target_offsets = defaultdict(float)
    for source, df in flow_df.groupby('source', observed=True):
        source_offset = 0
        for target, r in df.groupby('target', observed=True):
            outflow = r.at[r.index[0], 'outflow']
            y0_source = source_ranges[source][0] + source_offset
            y1_source = y0_source + outflow
            flow_df.loc[r.index, 'y0_source'] = y0_source
            flow_df.loc[r.index, 'y1_source'] = y1_source
            source_offset += outflow

            inflow = r.at[r.index[0], 'inflow']
            y0_target = target_ranges[target][0] + target_offsets[target]
            y1_target = y0_target + inflow
            flow_df.loc[r.index, 'y0_target'] = y0_target
            flow_df.loc[r.index, 'y1_target'] = y1_target
            target_offsets[target] += inflow
    return flow_df


def _check_min_flow_conditions(row, thresh):
    if thresh is None:
        return False
    else:
        return (row['inflow'] < thresh) & (row['outflow'] < thresh)
