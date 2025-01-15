import scanpy as sc
import pandas as pd
import scrtt as sct
import matplotlib.pyplot as plt


if __name__ == '__main__':
    path_adata = 'data/experiments/GSE131847_spleen/gex.h5ad'
    path_subsets = 'data/experiments/GSE131847_spleen/subsets/trajectory_clusters/20240717/subsets.csv'
    path_tmaps = 'data/experiments/GSE131847_spleen/tmaps/lambda1_5_growth_iters_1/'
    adata = sc.read_h5ad(path_adata)
    df_subsets = pd.read_csv(path_subsets, index_col=0)
    ot_model = sct.models.trajectory.WOTModel.load(path_tmaps)

    sankey = sct.plotting.Sankey(ot_model, df_subsets['trajectory_cluster'])
    sankey.plot_all_transitions()
    plt.tight_layout()
    plt.show()
