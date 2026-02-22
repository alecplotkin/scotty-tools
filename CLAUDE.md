# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**scotty-tools** is a Python library for downstream analysis of single-cell optimal transport (OT) trajectory models. The name is an acronym for **S**ingle **C**ell **O**ptimal-**T**ransport **T**rajector**Y** analysis tools. It interfaces with OT trajectory models (Moscot, WOT) to compute and visualize cell fate flows over time.

## Commands

**Install (editable):**
```
pip install -e .
pip install -e ".[examples]"  # includes moscot, jupyter, leidenalg
```

**Run tests:**
```
pytest tests/
pytest tests/scotty/tools/test_trajectory.py  # single file
pytest tests/scotty/tools/test_trajectory.py::TestClassName::test_method  # single test
```

## Architecture

The package lives in `src/scotty/` and is organized into three layers:

### 1. Models (`models/`)

Abstractions over OT libraries, providing a unified interface for transport maps:

- **`models/trajectory/ot.py`** — Core OT model classes. `BaseOTModel` is the abstract base with `push_forward()` and `pull_back()` methods. `MoscotModel`, `WOTModel`, and `GenericOTModel` wrap different backends. Transport maps are stored as `AnnData` objects (obs = source cells, var = target cells). The `coarsen_ot_model()` function embeds transport maps into a lower-dimensional (e.g. cluster-level) space.
- **`models/featurize/`** — `TrajectoryKMEFeaturizer` computes Trajectory Kernel Mean Embeddings using Nystroem/RBFSampler approximation.
- **`models/diffexp/`** — `TemporalDifferentialExpression` fits OLS models (`gene ~ group * time`) for differential expression across cell groups over time.

### 2. Tools (`tools/`)

Analysis functions built on top of the model layer:

- **`tools/trajectories.py`** — The central module. Three trajectory classes all subclass `anndata.AnnData` via `TrajectoryMixIn`:
  - `SubsetTrajectory`: tracks discrete cell subset fate flows. Normalization can be `'joint'` (fraction of all cells) or `'expected_value'` (conditional on source subset).
  - `GeneTrajectory`: gene expression weighted by trajectory probabilities; created via `GeneTrajectory.from_subset_trajectory()`.
  - `TrajectoryExpectation`: continuous feature expectations propagated forward/backward.
  - Key functions: `compute_trajectories()` (main entry point), `compute_trajectory_expectation()`, `compute_subset_frequency_table()`, `compare_trajectory_means()`, `calculate_feature_correlation()`, `compute_trajectory_entropy()`.
- **`tools/metrics.py`** — `fate_consistency()`, `compute_cluster_entropy()`, `calculate_trajectory_divergence()` (Jensen-Shannon, Total Variation, MMD).
- **`tools/kh.py`** — Kernel herding subsampling (`sketch()`) using `sketchKH`.

### 3. Plotting (`plotting/`)

- **`plotting/sankey.py`** — `Sankey` class for visualizing fate flows between timepoints via `plot_all_transitions()`. Caches intermediate computations.
- **`plotting/subsets.py`** — `plot_subset_frequencies()` and `plot_subset_frequencies_trajectory()` for line plots of subset frequencies over time.
- **`plotting/_flowplot.py`** — Low-level flow network rendering utilities.
- Other modules handle gene trajectory plots and entropy comparison plots.

### Data Flow Pattern

The typical analysis pipeline:
1. Load an OT model (e.g. `MoscotModel.load(path)`)
2. Call `compute_trajectories(ot_model, subsets=..., ref_time=...)` → `SubsetTrajectory`
3. Optionally derive `GeneTrajectory.from_subset_trajectory(...)` for gene-level analysis
4. Visualize with `Sankey(ot_model, subsets).plot_all_transitions()` or subset frequency plots

### Key Design Choices

- **AnnData-centric**: All trajectory classes inherit from `anndata.AnnData`. Transport maps are stored as AnnData (obs = source cells, var = target cells).
- **Push/pull semantics**: `push_forward` propagates a distribution forward in time (`tmap.T @ p`); `pull_back` propagates backward (`tmap @ p`). A small regularization term is added to prevent zero-probability collapse.
- **Time handling**: Trajectories store a reference time (`ref_time`) and time variable name (`time_var`). The time axis in trajectory AnnData uses `obs` for timepoints and `var` for cell subsets.
