import scrtt as sct
import scanpy as sc


def test_kh(ot_model, adata):
    tracter = sct.models.cluster.tracter_kh.TRACTER_kh(
        n_clusters=10, n_subsamples=500, random_state=585, gamma=0.5
    )
    tracter.fit(ot_model, adata)
    clusters = tracter.predict(adata)
    adata.obs.loc[clusters.index, 'tracter_kh'] = clusters.astype(str)
    sc.pl.umap(adata, color='tracter_kh')
    return


if __name__ == '__main__':
    path_adata = 'data/experiments/GSE131847_spleen/gex.h5ad'
    path_tmaps = 'data/experiments/GSE131847_spleen/tmaps/lambda1_5_growth_iters_1/'
    adata = sc.read_h5ad(path_adata)
    ot_model = sct.models.trajectory.WOTModel.load(path_tmaps)
    test_kh(ot_model, adata)
