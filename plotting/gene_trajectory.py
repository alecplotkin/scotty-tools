import numpy as np
import matplotlib.pyplot as plt
from src.sctrat.tools.trajectories import GeneTrajectory


def plot_gene_trajectory(
        traj: GeneTrajectory,
        gene: str,
) -> plt.Axes:
    for subset, df in traj.obs.groupby(traj.subset_var):
        traj_sub = traj[df.index, gene]
        mean = traj_sub.X[:, 0]
        std = traj_sub.layers['std'][:, 0]
        se = std / np.sqrt(traj_sub.obs['nobs'])
        time = np.arange(traj_sub.shape[0])
        plt.plot(time, mean, label=subset)
        plt.fill_between(time, mean - se, mean + se, alpha=0.2)
        plt.xticks(ticks=time, labels=traj_sub.obs[traj.time_var])
    plt.legend()
    ax = plt.gca()
    return ax
