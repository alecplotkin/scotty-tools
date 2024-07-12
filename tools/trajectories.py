import anndata as ad
from statsmodels.stats.weightstats import DescrStatsW


class TrajectoyMixIn(ad.AnnData):
    def __init__(
        self,
        trajectory: ad.AnnData,
        time_var: str = 'day',
    ):
        super().__init__(trajectory)
        self.time_var = time_var


class CellBySubsetTrajectory(TrajectoyMixIn):
    def __init__(
        self,
        trajectory: ad.AnnData,
        time_var: str = 'day',
    ):
        super().__init__(trajectory, time_var)


class SubsetByGeneTrajectory(TrajectoyMixIn):
    def __init__(
        self,
        trajectory: ad.AnnData,
        time_var: str = 'day',
        subset_var: str = 'subset',
    ):
        super().__init__(trajectory, time_var)
        self.subset_var = subset_var

    @staticmethod
    def compute_gene_trajectories(
        trajectory: CellBySubsetTrajectory,
        features: ad.AnnData,
        layer: str = None,
        subset_var: str = 'subset',
    ):
        # Make sure feature and trajectory indices are in same order
        features = features[trajectory.obs_names]
        if layer is not None:
            features.X = features.layers[layer]
        weights = trajectory.to_df()
        time_var = trajectory.time_var
        weights[time_var] = trajectory.obs[time_var]
        weights = weights.melt(
            id_vars=time_var, var_name=subset_var, value_name='stats',
            ignore_index=False,
        )
        # obs = weights.groupby([subset_var, time_var])['weight'].sum().reset_index()
        obs = weights.groupby([subset_var, time_var])['stats'].apply(
            lambda x: DescrStatsW(features[x.index].X, weights=x, ddof=0)
        ).reset_index()
        obs['nobs'] = obs['stats'].apply(lambda x: x.nobs)
        print(obs)
        means = obs['stats'].apply(lambda x: x.mean).to_numpy()
        print(means)
        quit()
        stds = obs['stats'].apply(lambda x: x.std).to_numpy()
        # ses = obs['stats'].apply(lambda x: x.std_mean).to_numpy()
        gene_traj = ad.AnnData(
            obs=obs, var=features.var,
            layers={'mean': means, 'std': stds, 'se': ses}
        )
        gene_traj = SubsetByGeneTrajectory(
            gene_traj, time_var=time_var, subset_var=subset_var
        )
        return gene_traj
