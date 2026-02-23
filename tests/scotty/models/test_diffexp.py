import numpy as np
import pandas as pd
import pytest

from scotty.models.diffexp import TemporalDifferentialExpression, calculate_temporal_diff_exp


@pytest.fixture
def diffexp_df():
    rng = np.random.default_rng(0)
    groups = ['G1', 'G2']
    times = [1, 2, 3]
    records = []
    for g in groups:
        for t in times:
            for _ in range(20):
                records.append({'expr': rng.standard_normal(), 'group': g, 'time': t})
    return pd.DataFrame(records)


@pytest.fixture
def fitted_tde(diffexp_df):
    tde = TemporalDifferentialExpression(diffexp_df, 'expr', 'group', 'time')
    tde.fit()
    return tde


def test_fit_sets_is_fitted(diffexp_df):
    tde = TemporalDifferentialExpression(diffexp_df, 'expr', 'group', 'time')
    tde.fit()
    assert tde._is_fitted is True


def test_test_all_groups_returns_dataframe(fitted_tde):
    result = fitted_tde.test_all_groups()
    assert isinstance(result, pd.DataFrame)


def test_test_all_groups_has_all_groups(fitted_tde, diffexp_df):
    result = fitted_tde.test_all_groups()
    expected_groups = set(diffexp_df['group'].unique())
    assert set(result.index) == expected_groups


def test_test_all_groups_columns(fitted_tde):
    result = fitted_tde.test_all_groups()
    expected = ['coef', 'std err', 't', 'P>|t|', 'Conf. Int. Low', 'Conf. Int. Upp.']
    assert list(result.columns) == expected


def test_time_averaged_t_test_before_fit_raises(diffexp_df):
    tde = TemporalDifferentialExpression(diffexp_df, 'expr', 'group', 'time')
    with pytest.raises(RuntimeError):
        tde.time_averaged_t_test('G1')


def test_calculate_temporal_diff_exp_returns_dataframe(diffexp_df):
    result = calculate_temporal_diff_exp(
        diffexp_df, ['expr'], 'group', 'time', max_workers=1
    )
    assert isinstance(result, pd.DataFrame)
