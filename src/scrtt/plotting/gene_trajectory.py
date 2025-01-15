import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from scrtt.tools.trajectories import GeneTrajectory


# TODO: refactor so that we can plot trajectories with different groups of
# time points.
def plot_gene_trajectory(
        traj: GeneTrajectory,
        gene: str,
        color_dict: Dict = None,
        show_ref_line: bool = True,
        ax: plt.Axes = None
) -> plt.Axes:

    if ax is None:
        ax = plt.subplot()
    times = np.unique(traj.obs[traj.time_var])
    ix_times = np.arange(times.shape[0])
    ax.set_xticks(ticks=ix_times, labels=times)
    if show_ref_line:
        # Get numeric index for the ref_time
        ix_ref_time = np.argmax(times == traj.ref_time)
        ax.axvline(ix_ref_time, c='k', linestyle=':')

    for subset, df in traj.obs.groupby(traj.subset_var):
        traj_sub = traj[df.index, gene]
        mean = traj_sub.X[:, 0]
        std = traj_sub.layers['std'][:, 0]
        se = std / np.sqrt(traj_sub.obs['nobs'])
        times_sub = traj_sub.obs[traj.time_var]
        ix_times_sub = np.where(np.isin(times, times_sub))[0]
        c = None
        if color_dict is not None:
            c = color_dict[subset]
        ax.plot(ix_times_sub, mean, label=subset, c=c)
        ax.fill_between(ix_times_sub, mean - se, mean + se, color=c, alpha=0.2)
    return ax
