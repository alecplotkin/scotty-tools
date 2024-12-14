import anndata as ad
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Literal, Dict, List, Tuple
from src.sctrat.models.trajectory import OTModel
from src.sctrat.utils import window
from ._flowplot import plot_flows
# from src.pysankey.sankey import sankey


class Sankey:
    """Class to plot sankey diagrams."""

    def __init__(
            self,
            ot_model: OTModel,
            subsets: Union[pd.DataFrame, pd.Series],
            color_dict: Dict = None,
            group_order: List = None,
            palette: str = None,
            endpoint_width: float = 0.05,
            cache_flow_dfs: bool = True,
    ):
        self.ot_model = ot_model
        if isinstance(subsets, pd.Series):
            subsets = pd.get_dummies(subsets).astype(float)
        else:
            subsets = subsets.astype(float)
        subsets.columns = subsets.columns.astype(str)
        self.subsets = subsets
        if group_order is None:
            group_order = subsets.columns.tolist()
        self.group_order = group_order
        if color_dict is None:
            pal = sns.color_palette(palette=palette, n_colors=len(group_order))
            color_dict = dict(zip(group_order, pal))
        self.color_dict = color_dict
        self.endpoint_width = endpoint_width
        self.cache_flow_dfs = cache_flow_dfs
        if cache_flow_dfs:
            self.flow_dfs = dict()

    def plot_all_transitions(
            self,
            show_labels: bool = False,
            timepoints: List = None,
            figsize: Tuple = None,
            endpoint_linewidth: float = None,
    ):
        """Plot all consecutive transitions in a single fig."""

        day_pairs = self.ot_model.day_pairs
        if timepoints is None:
            timepoints = self.ot_model.timepoints
        timepoints = list(sorted(timepoints))
        n_timepoints = len(timepoints)
        timepoints = dict(zip(timepoints, range(n_timepoints)))

        fig = plt.figure(figsize=figsize)
        for i, day_pair in enumerate(day_pairs):
            start = timepoints[day_pair[0]]
            stop = timepoints[day_pair[1]]

            if i == 0 and start != 0:
                ax0 = plt.subplot(1, n_timepoints - 1, (1, start))
                ax0.axis('off')

            ax = plt.subplot(1, n_timepoints - 1, (start + 1, stop))
            endpoint_width = self.endpoint_width / (stop - start)
            self.plot_sankey(
                *day_pair, ax=ax,
                show_source_labels=show_labels,
                show_target_labels=show_labels,
                endpoint_width=endpoint_width,
                endpoint_linewidth=endpoint_linewidth,
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        if stop != n_timepoints - 1:
            axn = plt.subplot(1, n_timepoints - 1, (stop, n_timepoints - 1))
            axn.axis('off')

        plt.subplots_adjust(wspace=0, hspace=0)
        return fig

    def plot_sankey(
            self,
            t0: float,
            t1: float,
            ax: plt.Axes = None,
            show_source_labels: bool = True,
            show_target_labels: bool = True,
            endpoint_width: float = None,
            endpoint_linewidth: float = None,
    ) -> plt.Axes:
        """Plot a single sankey facet"""

        if ax is None:
            ax = plt.gca()
        if self.cache_flow_dfs and (t0, t1) in self.flow_dfs:
            flow_df = self.flow_dfs[(t0, t1)]
        else:
            flow_df = self.calculate_flows(t0, t1)
        # Get rid of populations with zero members to control visual clutter.
        ix_null = (flow_df['outflow'] == 0) | (flow_df['inflow'] == 0)
        flow_df = flow_df.loc[~ix_null, :].reset_index()

        # # Need to put labels in specified order.
        # source_groups = []
        # target_groups = []
        # for group in self.group_order:
        #     if group in flow_df['source'].unique():
        #         source_groups.append(group)
        #     if group in flow_df['target'].unique():
        #         target_groups.append(group)

        plot_flows(
            sources=flow_df['source'],
            targets=flow_df['target'],
            outflows=flow_df['outflow'],
            inflows=flow_df['inflow'],
            palette=self.color_dict,
            group_order=self.group_order,
            endpoint_width=endpoint_width,
            endpoint_linewidth=endpoint_linewidth,
            fontsize=10,
            ax=ax,
        )
        return ax

    def calculate_flows(
            self,
            t0: float,
            t1: float,
    ) -> pd.DataFrame:
        """Calculate flow df for subsets between t0 and t1."""

        meta = self.ot_model.meta
        ix_t0 = meta.index[meta[self.ot_model.time_var] == t0]
        ix_t1 = meta.index[meta[self.ot_model.time_var] == t1]
        s0 = self.subsets.loc[ix_t0, :]
        s1 = self.subsets.loc[ix_t1, :]
        s0 = s0 / s0.values.sum()
        s1 = s1 / s1.values.sum()
        tps = np.array(self.ot_model.timepoints)
        day_pairs = window(tps[(tps >= t0) & (tps <= t1)])
        outflow = ad.AnnData(s0)
        inflow = ad.AnnData(s0)
        for day_pair in day_pairs:
            outflow = self.ot_model.push_forward(
                outflow, *day_pair, normalize=True, norm_axis=1
            )
            inflow = self.ot_model.push_forward(
                inflow, *day_pair, normalize=True, norm_axis=0
            )
        outflow = outflow.X.T @ s1.values
        outflow = outflow / outflow.sum()
        inflow = inflow.X.T @ s1.values
        inflow = inflow / inflow.sum()
        outflow = self._format_flow(outflow, s0, s1, name='outflow')
        inflow = self._format_flow(inflow, s0, s1, name='inflow')
        flow_df = pd.merge(outflow, inflow, on=['source', 'target'])

        if self.cache_flow_dfs:
            self.flow_dfs[(t0, t1)] = flow_df
        return flow_df

    @staticmethod
    def _format_flow(
            flow: npt.NDArray,
            p0: pd.DataFrame,
            p1: pd.DataFrame,
            name: Literal['inflow', 'outflow'],
    ) -> pd.DataFrame:
        df = pd.DataFrame(
            flow, index=p0.columns, columns=p1.columns
        ).reset_index(names=['source']).melt(
            id_vars=['source'], var_name='target', value_name=name
        )
        return df
