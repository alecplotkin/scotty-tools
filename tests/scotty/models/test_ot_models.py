import numpy as np
import anndata as ad
import pytest

from scotty.models.trajectory.ot import coarsen_ot_model


def test_timepoints_sorted(ot_model):
    assert ot_model.timepoints == [1.0, 2.0, 3.0]


def test_day_pairs(ot_model):
    assert list(ot_model.day_pairs) == [(1.0, 2.0), (2.0, 3.0)]


def test_get_coupling_shape(ot_model):
    tmap = ot_model.get_coupling(1.0, 2.0)
    assert tmap.shape == (15, 15)


def test_get_coupling_obs_names(ot_model, cell_ids):
    tmap = ot_model.get_coupling(1.0, 2.0)
    assert list(tmap.obs_names) == cell_ids[1.0]


def test_get_coupling_var_names(ot_model, cell_ids):
    tmap = ot_model.get_coupling(1.0, 2.0)
    assert list(tmap.var_names) == cell_ids[2.0]


def _make_p(obs_names, var_name='x'):
    p = ad.AnnData(np.ones((len(obs_names), 1)))
    p.obs_names = obs_names
    p.var_names = [var_name]
    return p


def test_push_forward_output_shape(ot_model, cell_ids):
    p = _make_p(cell_ids[1.0])
    result = ot_model.push_forward(p, 1.0, 2.0)
    assert result.n_obs == 15


def test_push_forward_preserves_var_names(ot_model, cell_ids):
    p = _make_p(cell_ids[1.0])
    result = ot_model.push_forward(p, 1.0, 2.0)
    assert list(result.var_names) == ['x']


def test_pull_back_output_shape(ot_model, cell_ids):
    p = _make_p(cell_ids[2.0])
    result = ot_model.pull_back(p, 1.0, 2.0)
    assert result.n_obs == 15


def test_pull_back_preserves_var_names(ot_model, cell_ids):
    p = _make_p(cell_ids[2.0])
    result = ot_model.pull_back(p, 1.0, 2.0)
    assert list(result.var_names) == ['x']


def test_push_forward_non_negative(ot_model, cell_ids):
    p = _make_p(cell_ids[1.0])
    result = ot_model.push_forward(p, 1.0, 2.0)
    assert (result.X >= 0).all()


def test_pull_back_non_negative(ot_model, cell_ids):
    p = _make_p(cell_ids[2.0])
    result = ot_model.pull_back(p, 1.0, 2.0)
    assert (result.X >= 0).all()


@pytest.mark.xfail(
    strict=True,
    reason="known bug: coarsen_ot_model passes unexpected kwargs to GenericOTModel",
)
def test_coarsen_ot_model(ot_model, cell_ids):
    import pandas as pd

    timepoint_mixtures = {}
    for tp, ids in cell_ids.items():
        n = len(ids)
        timepoint_mixtures[tp] = pd.DataFrame(
            np.eye(n),
            index=ids,
            columns=[f'cluster_{i}' for i in range(n)],
        )
    coarsen_ot_model(ot_model, timepoint_mixtures)
