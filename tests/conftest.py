import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import anndata as ad
import pytest

from scotty.models.trajectory.ot import GenericOTModel
from scotty.tools.trajectories import compute_trajectories


@pytest.fixture(scope='session')
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope='session')
def cell_ids():
    return {
        1.0: [f'cell_{i}' for i in range(15)],
        2.0: [f'cell_{i}' for i in range(15, 30)],
        3.0: [f'cell_{i}' for i in range(30, 45)],
    }


@pytest.fixture(scope='session')
def meta(cell_ids):
    records = []
    for day, ids in cell_ids.items():
        for cell_id in ids:
            records.append({'cell_id': cell_id, 'day': day})
    return pd.DataFrame(records).set_index('cell_id')


@pytest.fixture(scope='session')
def ot_model(rng, cell_ids, meta):
    tmaps = {}
    for t0, t1 in [(1.0, 2.0), (2.0, 3.0)]:
        src = cell_ids[t0]
        tgt = cell_ids[t1]
        tmap_x = rng.dirichlet(np.ones(len(tgt)), size=len(src))
        tmap = ad.AnnData(tmap_x)
        tmap.obs_names = src
        tmap.var_names = tgt
        tmaps[(t0, t1)] = tmap
    return GenericOTModel(tmaps=tmaps, meta=meta, time_var='day')


@pytest.fixture(scope='session')
def all_cell_subsets(cell_ids, rng):
    all_cells = [c for ids in cell_ids.values() for c in ids]
    labels = rng.choice(['A', 'B', 'C'], size=len(all_cells))
    return pd.Series(labels, index=all_cells, name='subset')


@pytest.fixture(scope='session')
def subset_trajectory(ot_model, all_cell_subsets):
    return compute_trajectories(ot_model, all_cell_subsets, ref_time=2.0)


@pytest.fixture(scope='session')
def gene_adata(cell_ids, meta, rng):
    all_cells = [c for ids in cell_ids.values() for c in ids]
    n_cells = len(all_cells)
    n_genes = 5
    X = rng.random((n_cells, n_genes))
    adata = ad.AnnData(X)
    adata.obs_names = all_cells
    adata.var_names = [f'gene_{i}' for i in range(n_genes)]
    adata.obsm['X_pca'] = rng.random((n_cells, 10))
    adata.obs['day'] = meta.loc[all_cells, 'day'].values
    return adata
