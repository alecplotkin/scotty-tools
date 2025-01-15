import pandas as pd
import scanpy as sc
import scrtt as sct
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':
    path_adata = 'data/experiments/GSE131847_spleen/gex.h5ad'
    path_tmaps = 'data/experiments/GSE131847_spleen/tmaps/lambda1_5_growth_iters_1/'
    adata = sc.read_h5ad(path_adata)
    ot_model = sct.models.trajectory.WOTModel.load(path_tmaps)
    tracter = sct.models.cluster.TRACTER(
        rep_dims=10, n_clusters=8, random_state=585
    )
    tracter.fit(ot_model, adata)

    # emb_dict = tracter.embed_data(adata)
    # emb_dict = {tp: df.idxmax(axis=1) for tp, df in emb_dict.items()}
    # emb = pd.concat(emb_dict.values(), axis=0)
    # emb = emb.str.split('_', expand=True)[0]
    # adata.obs.loc[emb.index, 'landmark'] = emb
    # sc.pl.umap(adata, color='landmark')

    # X = StandardScaler().fit_transform(tracter.trajectory_matrix.X)
    X = tracter.trajectory_matrix.X
    sns.clustermap(X, )
    plt.show()
    plt.close()
    clusters = tracter.predict(adata).astype(str)
    adata.obs.loc[clusters.index, 'trajectory_cluster'] = clusters
    sc.pl.umap(adata, color='trajectory_cluster')

    sct.plotting.plot_subset_frequencies(
        adata.obs, subset_field='trajectory_cluster'
    )
    plt.show()
    plt.close()
