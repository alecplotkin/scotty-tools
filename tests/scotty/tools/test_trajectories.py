import numpy as np
import pandas as pd
import pytest

from scotty.tools.trajectories import (
    SubsetTrajectory,
    GeneTrajectory,
    TrajectoryExpectation,
    compute_trajectories,
    compute_trajectory_expectation,
    compute_trajectory_entropy,
    calculate_feature_correlation,
)


def test_returns_subset_trajectory(subset_trajectory):
    assert isinstance(subset_trajectory, SubsetTrajectory)


def test_ref_time_stored(subset_trajectory):
    assert subset_trajectory.ref_time == 2.0


def test_time_var_stored(subset_trajectory):
    assert subset_trajectory.time_var == 'day'


def test_all_timepoints_in_obs(subset_trajectory):
    tps = {float(t) for t in subset_trajectory.obs['day'].unique()}
    assert tps == {1.0, 2.0, 3.0}


def test_norm_strategy_expected_value(subset_trajectory):
    assert subset_trajectory.norm_strategy == 'expected_value'


def test_norm_strategy_joint(ot_model, all_cell_subsets):
    traj = compute_trajectories(
        ot_model, all_cell_subsets, ref_time=2.0, normalize_to_population_size=False
    )
    assert traj.norm_strategy == 'joint'


def test_compute_alternative_sum(subset_trajectory):
    traj_x = subset_trajectory.X
    alt_x = subset_trajectory.compute_alternative().X
    combined = traj_x + alt_x
    row_sums = traj_x.sum(axis=1)
    # Every column of combined should equal the row sums (broadcasts correctly)
    for col in range(combined.shape[1]):
        np.testing.assert_allclose(combined[:, col], row_sums)


def test_copy_preserves_attrs(subset_trajectory):
    traj_copy = subset_trajectory.copy()
    assert traj_copy.ref_time == subset_trajectory.ref_time
    assert traj_copy.norm_strategy == subset_trajectory.norm_strategy
    assert traj_copy.time_var == subset_trajectory.time_var


def test_gene_trajectory_from_subset_trajectory_returns_gene_trajectory(
    subset_trajectory, gene_adata
):
    gene_traj = GeneTrajectory.from_subset_trajectory(subset_trajectory, gene_adata)
    assert isinstance(gene_traj, GeneTrajectory)


def test_gene_trajectory_shape(subset_trajectory, gene_adata):
    gene_traj = GeneTrajectory.from_subset_trajectory(subset_trajectory, gene_adata)
    n_subsets = subset_trajectory.n_vars
    n_timepoints = subset_trajectory.obs['day'].nunique()
    n_genes = gene_adata.n_vars
    assert gene_traj.n_obs == n_subsets * n_timepoints
    assert gene_traj.n_vars == n_genes


def test_gene_trajectory_std_layer_exists(subset_trajectory, gene_adata):
    gene_traj = GeneTrajectory.from_subset_trajectory(subset_trajectory, gene_adata)
    assert 'std' in gene_traj.layers


def test_gene_trajectory_std_shape(subset_trajectory, gene_adata):
    gene_traj = GeneTrajectory.from_subset_trajectory(subset_trajectory, gene_adata)
    assert gene_traj.layers['std'].shape == gene_traj.X.shape


def test_compute_trajectory_expectation_type(ot_model, gene_adata):
    result = compute_trajectory_expectation(ot_model, gene_adata.to_df(), ref_time=2.0)
    assert isinstance(result, TrajectoryExpectation)


def test_compute_trajectory_expectation_all_timepoints(ot_model, gene_adata):
    result = compute_trajectory_expectation(ot_model, gene_adata.to_df(), ref_time=2.0)
    tps = {float(t) for t in result.obs['day'].unique()}
    assert tps == {1.0, 2.0, 3.0}


def test_compute_trajectory_entropy_returns_dataframe(ot_model, all_cell_subsets):
    result = compute_trajectory_entropy(ot_model, all_cell_subsets)
    assert isinstance(result, pd.DataFrame)


def test_compute_trajectory_entropy_columns(ot_model, all_cell_subsets):
    result = compute_trajectory_entropy(ot_model, all_cell_subsets)
    assert set(result.columns) == {'source_day', 'target_day', 'entropy'}


def test_compute_trajectory_entropy_nonnegative(ot_model, all_cell_subsets):
    result = compute_trajectory_entropy(ot_model, all_cell_subsets)
    assert (result['entropy'] >= 0).all()


def test_calculate_feature_correlation_shape(ot_model, gene_adata):
    corr = calculate_feature_correlation(ot_model, gene_adata, 1.0, 2.0)
    assert corr.shape == (gene_adata.n_vars, 2)


def test_calculate_feature_correlation_columns(ot_model, gene_adata):
    corr = calculate_feature_correlation(ot_model, gene_adata, 1.0, 2.0)
    assert list(corr.columns) == ['corr', 'pval']


def test_calculate_feature_correlation_invalid_method_raises(ot_model, gene_adata):
    with pytest.raises(NotImplementedError):
        calculate_feature_correlation(ot_model, gene_adata, 1.0, 2.0, method='invalid')
