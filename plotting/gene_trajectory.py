import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from src.sctrat.tools.trajectories import GeneTrajectory


# TODO: refactor so that we can plot trajectories with different groups of
# time points.
def plot_gene_trajectory(
        traj: GeneTrajectory,
        gene: str,
        color_dict: Dict = None,
        ax: plt.Axes = None
) -> plt.Axes:
    if ax is None:
        ax = plt.subplot()
    for subset, df in traj.obs.groupby(traj.subset_var):
        traj_sub = traj[df.index, gene]
        mean = traj_sub.X[:, 0]
        std = traj_sub.layers['std'][:, 0]
        se = std / np.sqrt(traj_sub.obs['nobs'])
        time = np.arange(traj_sub.shape[0])
        c = None
        if color_dict is not None:
            c = color_dict[subset]
        ax.plot(time, mean, label=subset, c=c)
        ax.fill_between(time, mean - se, mean + se, color=c, alpha=0.2)
        ax.set_xticks(ticks=time, labels=traj_sub.obs[traj.time_var])
    return ax
