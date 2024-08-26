import anndata as ad
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Literal, Dict, List
from src.sctrat.models.trajectory import OTModel
from src.sctrat.utils import window
from src.pysankey.sankey import sankey


class Sankey:
    """Class to plot sankey diagrams."""

    def __init__(
            self,
            ot_model: OTModel,
            subsets: Union[pd.DataFrame, pd.Series],
            color_dict: Dict = None,
            label_order: List = None,
            palette: str = None,
    ):
        self.ot_model = ot_model
        if isinstance(subsets, pd.Series):
            subsets = pd.get_dummies(subsets).astype(float)
        else:
            subsets = subsets.astype(float)
        subsets.columns = subsets.columns.astype(str)
        self.subsets = subsets
        if label_order is None:
            label_order = subsets.columns.tolist()
        self.label_order = label_order
        if color_dict is None:
            pal = sns.color_palette(palette=palette, n_colors=len(label_order))
            color_dict = dict(zip(label_order, pal))
        self.color_dict = color_dict

    def plot_all_transitions(
            self,
            show_labels: bool = False,
            aspect_ratio: float = 0.4,
    ):
        """Plot all consecutive transitions in a single fig."""

        day_pairs = self.ot_model.day_pairs
        n_plots = len(day_pairs)
        fig, axs = plt.subplots(
            1, n_plots, figsize=(4*aspect_ratio*n_plots, 4*n_plots)
        )
        for i, day_pair in enumerate(day_pairs):
            self.plot_sankey(
                *day_pair, ax=axs[i],
                show_left_labels=show_labels,
                show_right_labels=show_labels,
            )
        return fig, axs

    def plot_sankey(
            self,
            t0: float,
            t1: float,
            ax: plt.Axes = None,
            show_left_labels: bool = True,
            show_right_labels: bool = True,
    ) -> plt.Axes:
        """Plot a single sankey facet"""

        if ax is None:
            ax = plt.gca()
        flow_df = self.calculate_flows(t0, t1)
        # Get rid of populations with zero members to control visual clutter.
        ix_null = (flow_df['outflow'] == 0) | (flow_df['inflow'] == 0)
        flow_df = flow_df.loc[~ix_null, :].reset_index()
        # Need to put labels in specified order.
        left_labels = []
        right_labels = []
        for lab in self.label_order:
            if lab in flow_df['source'].unique():
                left_labels.append(lab)
            if lab in flow_df['target'].unique():
                right_labels.append(lab)

        sankey(
            left=flow_df['source'],
            right=flow_df['target'],
            leftWeight=flow_df['outflow'],
            rightWeight=flow_df['inflow'],
            colorDict=self.color_dict,
            leftLabels=left_labels,
            rightLabels=right_labels,
            showLeftLabels=show_left_labels,
            showRightLabels=show_right_labels,
            stripWidth=0.1,
            stripPad=0,
            fontsize=10,
            wrapText=True,
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
