import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from scotty.models.trajectory import OTModel
from scotty.tools.metrics import compute_trajectory_entropy
from typing import List


def compare_trajectory_entropy(
        adata: ad.AnnData,
        ot_model: OTModel,
        groupings: List[str],
        compute_ratio: bool = True,
        showfliers: bool = True,
):
    results = dict()
    for grouping in groupings:
        print(f'Calculating trajectory entropy for {grouping}')
        if grouping is not None:
            subsets = adata.obs[grouping]
        else:
            subsets = None
        entropy_fwd = compute_trajectory_entropy(
            ot_model, subsets=subsets, direction='forward',
            compute_ratio=compute_ratio
        ).rename('entropy_fwd')
        entropy_bwd = compute_trajectory_entropy(
            ot_model, subsets=subsets, direction='backward',
            compute_ratio=compute_ratio
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
    sns.boxplot(df, x='day', hue='grouping', y='entropy_fwd', showfliers=showfliers)
    plt.show()
    sns.boxplot(df, x='day', hue='grouping', y='entropy_bwd', showfliers=showfliers)
    plt.show()
    return None
