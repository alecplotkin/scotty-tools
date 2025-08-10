import numpy as np
import pandas as pd
import scanpy as sc
import scotty as sct
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    path_adata = 'data/experiments/GSE131847_spleen/gex.h5ad'
    path_tmaps = 'data/experiments/GSE131847_spleen/tmaps/lambda1_5_growth_iters_1/'
    path_sub1 = 'data/experiments/GSE131847_spleen/subsets/trajectory_clusters/20240717/subsets.csv'
    path_sub2 = 'data/experiments/GSE131847_spleen/subsets/EEC_fates/GSE173512_GSE157072-lambda1_5_growth_iters_1/subsets.csv'
    adata = sc.read_h5ad(path_adata)
    ot_model = sct.models.trajectory.WOTModel.load(path_tmaps)
    df_sub1 = pd.read_csv(path_sub1, index_col=0)
    df_sub2 = pd.read_csv(path_sub2, index_col=0)
    adata.obs = adata.obs.merge(df_sub1, how='left', left_index=True, right_index=True)
    adata.obs = adata.obs.merge(df_sub2, how='left', left_index=True, right_index=True)
    sc.tl.leiden(adata, resolution=1, key_added='leiden')

    # groupings = [None, 'subset', 'trajectory_cluster', 'leiden']
    groupings = ['subset', 'trajectory_cluster', 'leiden']
    results = dict()
    for grouping in groupings:
        print(f'Calculating trajectory entropy for {grouping}')
        if grouping is not None:
            subsets = adata.obs[grouping]
        else:
            subsets = None
        entropy_fwd = sct.tools.metrics.compute_trajectory_entropy(
            ot_model, subsets=subsets, direction='forward'
        ).rename('entropy_fwd')
        entropy_bwd = sct.tools.metrics.compute_trajectory_entropy(
            ot_model, subsets=subsets, direction='backward'
        ).rename('entropy_bwd')
        results[str(grouping)] = pd.merge(
            entropy_fwd, entropy_bwd,
            how='outer', left_index=True, right_index=True,
        )

    df = pd.concat(
        results.values(), axis=0, keys=results.keys(), names=['grouping']
    ).reset_index(level='grouping').merge(
        adata.obs['day'], how='left', left_index=True, right_index=True
    ).set_index('grouping', append=True)
    sns.boxplot(df, x='day', hue='grouping', y='entropy_fwd')
    plt.show()
    sns.boxplot(df, x='day', hue='grouping', y='entropy_bwd')
    plt.show()
