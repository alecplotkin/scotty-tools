import scanpy as sc
import pandas as pd
import scrtt as sct


if __name__ == '__main__':
    path_adata = 'data/experiments/GSE131847_spleen/gex.h5ad'
    path_tmaps = 'data/experiments/GSE131847_spleen/tmaps/lambda1_5_growth_iters_1/'
    path_subsets = 'data/experiments/GSE131847_spleen/subsets/GSE173512_GSE157072/subsets.csv'
    adata = sc.read_h5ad(path_adata)
    model = sct.models.trajectory.WOTModel.load(path_tmaps)
    df_subsets = pd.read_csv(path_subsets, index_col=0)
    traj = model.compute_trajectories(df_subsets['subset'], ref_time=7)
    print(traj)
    for day, df in traj.obs.groupby(traj.time_var):
        print(traj[df.index].X.sum())
