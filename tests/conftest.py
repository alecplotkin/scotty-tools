import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import anndata as ad
import pytest

from scotty.models.trajectory.ot import GenericOTModel
from scotty.tools.trajectories import compute_trajectories

try:
    from scotty.simulate.latents import (
        Clone,
        CloneForest,
        CloneTrajectory,
        standard_normal_init,
    )
except ImportError:
    # The simulate module is absent on some branches (reverted in a740b2b).
    # Skip its tests rather than failing collection for the whole suite.
    Clone = CloneForest = CloneTrajectory = standard_normal_init = None
    collect_ignore_glob = ['scotty/simulate/*']


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


@pytest.fixture
def unbalanced_ot_model(cell_ids, meta):
    """OT model whose couplings have non-uniform row and column sums.

    The `ot_model` fixture draws dirichlet rows, so every row sums to 1 and
    row- vs column-normalization coincide up to a constant. Growth makes them
    genuinely different, which is what most normalization bugs need to show up.
    Function-scoped so mutation of the couplings cannot leak between tests.
    """
    rng = np.random.default_rng(7)
    tmaps = {}
    for t0, t1 in [(1.0, 2.0), (2.0, 3.0)]:
        src, tgt = cell_ids[t0], cell_ids[t1]
        tmap_x = rng.dirichlet(np.ones(len(tgt)), size=len(src))
        growth = np.linspace(0.2, 3.0, len(src))[:, None]
        tmap_x = tmap_x * growth
        tmap_x = tmap_x / tmap_x.sum()
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


@pytest.fixture
def clone():
    return Clone(parent=None, birth_time=0)


@pytest.fixture
def clone_forest():
    return CloneForest(5)


@pytest.fixture
def clone_trajectory():
    return CloneTrajectory(init_size=5, ndim=2, init_fun=standard_normal_init)
