import numpy as np
import scanpy as sc
import pandas as pd
import scotty as sct
import sys


ot_model = sct.models.trajectory.GenericOTModel(
    tmaps=dict(),
    meta=pd.DataFrame,
)


def test_compute_trajectory_expectation():
    ...


def main():
    test_compute_trajectory_expectation()
    return


if __name__ == '__main__':
    sys.exit(main())
    # path_adata = 'data/experiments/GSE131847_spleen/gex.h5ad'
    # path_subsets = 'data/experiments/GSE131847_spleen/subsets/EEC_fates/GSE173512_GSE157072-lambda1_5_growth_iters_1/subsets.csv'
    # path_tmaps = 'data/experiments/GSE131847_spleen/tmaps/lambda1_5_growth_iters_1/'
    # adata = sc.read_h5ad(path_adata)
    # df_subsets = pd.read_csv(path_subsets, index_col=0)
    # model = sct.models.trajectory.WOTModel.load(path_tmaps)
    # traj = model.compute_trajectories(df_subsets['subset'], ref_time=7)
    # gene_traj = sct.tools.trajectories.GeneTrajectory.from_subset_trajectory(
    #         traj, adata[:, adata.var['highly_variable']]
    # )
    # print(repr(gene_traj))

    # # test whether alternative trajectories add up.
    # traj_alt = traj.compute_alternative()
    # assert np.allclose(traj.X, traj.X.sum(1, keepdims=True) - traj_alt.X)
