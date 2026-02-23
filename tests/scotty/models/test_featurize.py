import anndata as ad

from scotty.models.featurize import TrajectoryKMEFeaturizer


def test_fit_transform_returns_anndata(ot_model, gene_adata):
    featurizer = TrajectoryKMEFeaturizer(embedding_size=4, random_state=0)
    result = featurizer.fit_transform(ot_model, gene_adata)
    assert isinstance(result, ad.AnnData)


def test_fit_transform_obs_matches_cells(ot_model, gene_adata):
    featurizer = TrajectoryKMEFeaturizer(embedding_size=4, random_state=0)
    result = featurizer.fit_transform(ot_model, gene_adata)
    assert set(result.obs_names) == set(ot_model.meta.index)


def test_fit_transform_auto_gamma(ot_model, gene_adata):
    featurizer = TrajectoryKMEFeaturizer(embedding_size=4, gamma=None, random_state=0)
    result = featurizer.fit_transform(ot_model, gene_adata)
    assert isinstance(result, ad.AnnData)


def test_fit_transform_rbf_sampler(ot_model, gene_adata):
    featurizer = TrajectoryKMEFeaturizer(
        embedding_size=4, embedding_model='RBFSampler', random_state=0
    )
    result = featurizer.fit_transform(ot_model, gene_adata)
    assert isinstance(result, ad.AnnData)


def test_fit_transform_embedding_size(ot_model, gene_adata):
    embedding_size = 5
    featurizer = TrajectoryKMEFeaturizer(embedding_size=embedding_size, random_state=0)
    result = featurizer.fit_transform(ot_model, gene_adata)
    n_timepoints = len(ot_model.timepoints)
    assert result.n_vars == n_timepoints * embedding_size
