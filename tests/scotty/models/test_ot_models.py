import numpy as np
import anndata as ad
import pytest

from scotty.models.trajectory.ot import coarsen_ot_model, GenericOTModel


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


def test_save_creates_expected_files(ot_model, tmp_path):
    ot_model.save(str(tmp_path))
    assert (tmp_path / "meta.csv").exists()
    assert (tmp_path / "metadata.json").exists()
    assert (tmp_path / "coupling_1.0_2.0.h5ad").exists()
    assert (tmp_path / "coupling_2.0_3.0.h5ad").exists()


def test_save_metadata_json_contents(ot_model, tmp_path):
    import json
    ot_model.save(str(tmp_path))
    with open(tmp_path / "metadata.json") as f:
        metadata = json.load(f)
    assert metadata["time_var"] == "day"
    assert metadata["timepoints"] == [1.0, 2.0, 3.0]
    assert metadata["day_pairs"] == [[1.0, 2.0], [2.0, 3.0]]


def test_generic_load_roundtrip_timepoints(ot_model, tmp_path):
    ot_model.save(str(tmp_path))
    loaded = GenericOTModel.load(str(tmp_path))
    assert loaded.timepoints == ot_model.timepoints


def test_generic_load_roundtrip_day_pairs(ot_model, tmp_path):
    ot_model.save(str(tmp_path))
    loaded = GenericOTModel.load(str(tmp_path))
    assert loaded.day_pairs == ot_model.day_pairs


def test_generic_load_roundtrip_time_var(ot_model, tmp_path):
    ot_model.save(str(tmp_path))
    loaded = GenericOTModel.load(str(tmp_path))
    assert loaded.time_var == ot_model.time_var


def test_generic_load_roundtrip_coupling_values(ot_model, tmp_path):
    ot_model.save(str(tmp_path))
    loaded = GenericOTModel.load(str(tmp_path))
    for t0, t1 in ot_model.day_pairs:
        orig = ot_model.get_coupling(t0, t1)
        reloaded = loaded.get_coupling(t0, t1)
        assert np.allclose(orig.X, reloaded.X)


def test_generic_load_roundtrip_coupling_obs_names(ot_model, tmp_path, cell_ids):
    ot_model.save(str(tmp_path))
    loaded = GenericOTModel.load(str(tmp_path))
    tmap = loaded.get_coupling(1.0, 2.0)
    assert list(tmap.obs_names) == cell_ids[1.0]


def test_generic_load_roundtrip_coupling_var_names(ot_model, tmp_path, cell_ids):
    ot_model.save(str(tmp_path))
    loaded = GenericOTModel.load(str(tmp_path))
    tmap = loaded.get_coupling(1.0, 2.0)
    assert list(tmap.var_names) == cell_ids[2.0]


def test_generic_load_roundtrip_meta(ot_model, tmp_path):
    ot_model.save(str(tmp_path))
    loaded = GenericOTModel.load(str(tmp_path))
    assert list(loaded.meta.index) == list(ot_model.meta.index)
    assert list(loaded.meta["day"]) == list(ot_model.meta["day"])


def test_base_load_raises(ot_model, tmp_path):
    from scotty.models.trajectory.ot import BaseOTModel
    ot_model.save(str(tmp_path))
    with pytest.raises(NotImplementedError):
        BaseOTModel.load(str(tmp_path))


def test_save_creates_directory_if_missing(ot_model, tmp_path):
    dest = tmp_path / "nested" / "subdir"
    ot_model.save(str(dest))
    assert dest.is_dir()
    assert (dest / "metadata.json").exists()
