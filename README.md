# scotty-tools

**S**ingle **C**ell **O**ptimal-**T**ransport **T**rajector**Y** analysis tools.

scotty-tools is a Python library for downstream analysis of single-cell optimal transport (OT) trajectory models. It interfaces with OT trajectory models (Moscot, WOT) to compute and visualize cell fate flows over time.

---

![Sankey diagram showing fate flows between hematopoietic cell types across timepoints](docs/images/sankey_cell_type.png)

*Fate-flow Sankey diagram: OT-predicted cell type transitions in a differentiating hematopoietic stem cell dataset.*

---

## Features

- **Fate-flow Sankey diagrams** — Visualize differentiation relationships between cell types (or any discrete labels) across consecutive timepoints, with flow widths proportional to transition probabilities and population size changes reflecting predicted growth.
- **Trajectory Kernel Mean Embedding (TKME)** — Integrate cells across timepoints into a shared embedding space using the OT model, enabling time-invariant lineage clustering and dimensionality reduction.
- **Fate probability computation** — Compute `SubsetTrajectory` and `GeneTrajectory` objects that track how discrete cell subsets or gene expression programs evolve over time under the OT model.
- **Fate entropy** — Quantify per-cell fate uncertainty with respect to a set of discrete labels; high-entropy cells have ambiguous outcomes while low-entropy cells have deterministic fates.
- **Differential expression over trajectories** — `TemporalDifferentialExpression` fits OLS models (`gene ~ group * time`) to identify genes that are differentially expressed between lineages over time.

## Quick start

```python
import scotty as sct

# 1. Wrap a fitted Moscot model
ot_model = sct.models.MoscotModel.load(path)

# 2. Compute trajectory fate flows
trajectory = sct.tools.compute_trajectories(
    ot_model, subsets=adata.obs['cell_type'], ref_time=4.0
)

# 3. Visualize fate flows as a Sankey diagram
sankey = sct.plotting.Sankey(ot_model, adata.obs['cell_type'])
sankey.plot_all_transitions(min_flow_threshold=0.01)
```

## Installation

Install from GitHub:

```
pip install git+https://github.com/alecplotkin/scotty-tools
```

Install with optional extras (moscot, jupyter, leidenalg):

```
pip install "git+https://github.com/alecplotkin/scotty-tools#egg=scotty-tools[examples]"
```

Editable install for development:

```
pip install -e .
pip install -e ".[examples]"
```

## Tutorial

See [`examples/quickstart.ipynb`](examples/quickstart.ipynb) for a full walkthrough using a hematopoietic stem cell differentiation dataset, covering OT model fitting, Sankey visualization, fate entropy, and trajectory clustering with TKME.
