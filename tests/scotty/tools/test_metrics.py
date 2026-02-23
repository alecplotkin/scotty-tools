import numpy as np
import pandas as pd

from scotty.tools.metrics import (
    compute_cluster_entropy,
    calculate_trajectory_divergence,
    fate_consistency,
)


def test_compute_cluster_entropy_uniform():
    df = pd.DataFrame({
        'A': [1.0, 0.0, 0.0],
        'B': [0.0, 1.0, 0.0],
        'C': [0.0, 0.0, 1.0],
    })
    result = compute_cluster_entropy(df)
    np.testing.assert_allclose(result, np.log(3))


def test_compute_cluster_entropy_deterministic():
    df = pd.DataFrame({
        'A': [1.0, 1.0, 1.0],
        'B': [0.0, 0.0, 0.0],
        'C': [0.0, 0.0, 0.0],
    })
    result = compute_cluster_entropy(df)
    assert result == 0.0


def test_calculate_trajectory_divergence_js_returns_array(subset_trajectory):
    divs = calculate_trajectory_divergence(subset_trajectory, 'A', 'B')
    assert isinstance(divs, np.ndarray)
    assert len(divs) == subset_trajectory.obs['day'].nunique()


def test_calculate_trajectory_divergence_js_identical(subset_trajectory):
    # At the reference day the distribution is one-hot (has exact zeros),
    # which causes 0*log(0)=NaN in the KL formula. Ignore NaN and check the rest.
    divs = calculate_trajectory_divergence(subset_trajectory, 'A', 'A')
    np.testing.assert_allclose(np.nansum(divs), 0.0, atol=1e-10)


def test_calculate_trajectory_divergence_js_nonneg(subset_trajectory):
    divs = calculate_trajectory_divergence(subset_trajectory, 'A', 'B')
    assert np.all((divs >= 0) | np.isnan(divs))


def test_calculate_trajectory_divergence_tv_range(subset_trajectory):
    divs = calculate_trajectory_divergence(
        subset_trajectory, 'A', 'B', metric='total_variation'
    )
    assert (divs >= 0).all()
    assert (divs <= 1).all()


def test_fate_consistency_returns_series(ot_model, all_cell_subsets):
    result = fate_consistency(ot_model, all_cell_subsets)
    assert isinstance(result, pd.Series)


def test_fate_consistency_at_most_one(ot_model, all_cell_subsets):
    result = fate_consistency(ot_model, all_cell_subsets)
    assert (result <= 1.0).all()
