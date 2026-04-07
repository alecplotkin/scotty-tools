# %%
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scotty.simulate as scsim
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from jax import random

# %%
# %load_ext autoreload
# %autoreload 2

# %% [md]
"""
# Overview

This notebook will outline how to construct diffusion-drift simulation of single cell clonal dynamics using the `scotty.tools.simulate` module.
"""

# %% [md]
"""
We'll use a classic double-well potential for simulating trajectories.
"""


# %%
xmax = 4
x = y = np.linspace(-xmax, xmax, 201)
xs, ys = np.meshgrid(x, y, indexing='xy')
X_grid = np.stack((xs, ys), axis=-1)
potential_grid = scsim.utils.double_well_2d_potential(X_grid)
plt.imshow(potential_grid, extent=[-xmax, xmax, -xmax, xmax], origin='lower')
plt.title('Potential function')
plt.colorbar()

# %%
X_grid[::4, ::4, :].shape

# %%
# x = y = np.linspace(-xmax, xmax, 21)
# xs, ys = np.meshgrid(x, y, indexing='xy')
# X_grid = np.stack((xs, ys), axis=-1)
U = -scsim.utils.double_well_2d_drift(X_grid[::10, ::10, :])
us = U[..., 0]
vs = U[..., 1]

plt.quiver(xs[::10, ::10], ys[::10, ::10], us, vs)
plt.title('Vector field (drift)')

# %% [md]
"""
We can convert these latent dimensions into gene expression using a count simulation. This first converts the latent state into gene modules, and then into expected gene expression. Finally, gene expression counts are sampled using a multi-step observation model that incorporates per-cell library sizes, technical batch effects, and technical dropout.
"""

# %%
key = random.PRNGKey(585)

config = scsim.counts.SimulatorConfig(
    latent_dim=2,
    hidden_dims=(64, 64),
    n_modules=32,
    n_genes=2048,
    module_bias_bound=1.0,
    genes_per_module=100,
    overlap=0.05,
    weight_scale=2.0,
    theta_low=0.3,
    theta_high=5.0,
    mean_log_library=8.0,
)

init_key, sim_key = random.split(key, 2)

params = scsim.counts.init_simulator(init_key, config)
z_grid = jnp.array(X_grid[::5, ::5, :])
out_grid = scsim.counts.simulate_counts(
    params,
    sim_key,
    z_grid,
    config,
    add_technical_dropout=True,
    dropout_strength=1.0,
    add_batches=False,
)

scsim.counts.qc_counts(out_grid)

# %% [md]
"""
We can visualize the module activity on the 2d latent grid:
"""

# %%
# TODO: Add plot modules / genes to a top-level wrapper.

module_grid = out_grid['module_activity']
n_modules = module_grid.shape[-1]

res = 2
n_cols = 4
n_rows = -(n_modules // -n_cols)  # Equivalent to ceiling function.

fig, axs = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows), sharex=True, sharey=True)
for m in range(n_modules):
    i = m // n_cols
    j = m % n_cols
    im = axs[i, j].imshow(module_grid[..., m], extent=[-4, 4, -4, 4], origin='lower')
    axs[i, j].set_title(f"Module {m}")
    plt.colorbar(im, ax=axs[i, j], shrink=0.8)
plt.tight_layout()
plt.show()


# %% [md]
"""
We can also look at a few randomly selected genes to visualize how the module scores are converted into counts by the observation model:
"""

# %%
gene_grid = out_grid["gene_mean_base"]  # Output of module -> gene layer.
n_genes = 8

res = 2
n_cols = 4
n_rows = n_genes // n_cols

key = random.PRNGKey(585)
example_genes = random.choice(key, gene_grid.shape[-1], shape=(n_genes,), replace=False)

fig, axs = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows), sharex=True, sharey=True)
for k, g in enumerate(example_genes):
    i = k // n_cols
    j = k % n_cols
    im = axs[i, j].imshow(gene_grid[..., g], extent=[-4, 4, -4, 4])
    axs[i, j].set_title(f"Gene {g}")
    plt.colorbar(im, ax=axs[i, j], shrink=0.8)
plt.tight_layout()
plt.show()

# %%
gene_grid = out_grid["mu"]  # Library size scaled means.
n_genes = 8

res = 2
n_cols = 4
n_rows = n_genes // n_cols

key = random.PRNGKey(585)
example_genes = random.choice(key, gene_grid.shape[-1], shape=(n_genes,), replace=False)

fig, axs = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows), sharex=True, sharey=True)
for k, g in enumerate(example_genes):
    i = k // n_cols
    j = k % n_cols
    im = axs[i, j].imshow(gene_grid[..., g], extent=[-4, 4, -4, 4])
    axs[i, j].set_title(f"Gene {g}")
    plt.colorbar(im, ax=axs[i, j], shrink=0.8)
plt.tight_layout()
plt.show()

# %%
gene_grid = out_grid["counts"]  # Sampled counts.
n_genes = 8

res = 2
n_cols = 4
n_rows = n_genes // n_cols

key = random.PRNGKey(585)
example_genes = random.choice(key, gene_grid.shape[-1], shape=(n_genes,), replace=False)

fig, axs = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows), sharex=True, sharey=True)
for k, g in enumerate(example_genes):
    i = k // n_cols
    j = k % n_cols
    im = axs[i, j].imshow(gene_grid[..., g], extent=[-4, 4, -4, 4])
    axs[i, j].set_title(f"Gene {g}")
    plt.colorbar(im, ax=axs[i, j], shrink=0.8)
plt.tight_layout()
plt.show()

# %% [md]
"""
Next we will simulate diffusion-drift dynamics with birth and death in the latent space.

We first need functions to define the birth and death rates. We can define these directly in the latent space, but we can also choose one of the modules for interpretability.
"""

# %%
# TODO: Add growth and death tuning to a top-level simulation wrapper.

# %%
proliferation_module = 24
apoptosis_module = 11

max_proliferation_rate = 3.0
max_apoptosis_rate = 3.0


def extract_module(z, module_idx, scale=1.0):
    module_activity = scsim.counts.mlp_forward(z, params["module_decoder"])
    return scale * module_activity[..., module_idx]


proliferation_scores = extract_module(z_grid, proliferation_module)
proliferation_scale = max_proliferation_rate / proliferation_scores.max()
proliferation_genes = np.argwhere(params['gene_program']['W'][proliferation_module, :] > 1.0)

apoptosis_scores = extract_module(z_grid, apoptosis_module)
apoptosis_scale = max_apoptosis_rate / apoptosis_scores.max()
apoptosis_genes = np.argwhere(params['gene_program']['W'][apoptosis_module, :] > 1.0)


def beta(z):
    return extract_module(z, proliferation_module, scale=proliferation_scale)


def delta(z):
    return extract_module(z, apoptosis_module, scale=apoptosis_scale)


# %%
beta_grid = beta(X_grid)
plt.imshow(beta_grid, extent=[-4, 4, -4, 4], origin='lower')
plt.title('Birth rate')
plt.colorbar()
plt.show()

delta_grid = delta(X_grid)
plt.imshow(delta_grid, extent=[-4, 4, -4, 4], origin='lower')
plt.title('Death rate')
plt.colorbar()
plt.show()

growth_grid = beta_grid - delta_grid
plt.imshow(beta_grid - delta_grid, extent=[-4, 4, -4, 4], cmap='bwr', norm=mcolors.CenteredNorm(), origin='lower')
plt.title('Cumulative growth')
plt.colorbar()
plt.show()

event_grid = beta_grid + delta_grid
plt.imshow(beta_grid + delta_grid, extent=[-4, 4, -4, 4], origin='lower')
plt.title('Number of events per unit time')
plt.colorbar()
plt.show()

# %%
print(f"Max birth rate: {beta_grid.max():.1f}")
print(f"Max death rate: {delta_grid.max():.1f}")
print(f"Max growth rate: {growth_grid.max():.1f}")
print(f"Max event rate: {event_grid.max():.1f}")

# %% [md]
"""
This last plot will help inform how small our `dt` needs to be so that the birth / death probabilities don't explode during the simulation.
"""

# %% [md]
"""
Now that we have functions for drift, birth rate, and death rate, we can simulate trajectories in the latent space.
"""

# %%
sim = scsim.latents.Simulation(
    diffusivity=1e-2,
    ndim=2,
    drift=scsim.utils.double_well_2d_drift,
    birth=beta,
    death=delta,
    init_fun=scsim.utils.double_well_2d_init,
)
traj = sim.simulate(dt=0.01, n_steps=100, init_size=400, update_every=10, seed=42)
traj

# %%
print(sim)
print(traj)

# %%
# How many births did we get?
len(traj.lineages)

# %% [md]
"""
Let's look at how the populations evolve over time:
"""

# %%
dfs = dict()
for i, time in enumerate(traj.times):
    dfs[time] = pd.DataFrame(traj.X[i], index=traj.indices[i], columns=['x', 'y'])
df = pd.concat(dfs.values(), keys=dfs.keys(), names=['time', 'index'])
sns.scatterplot(df, x='x', y='y', hue='time', palette='turbo')

# %% [md]
"""
What are the birth, death, and cumulative growth rates at each sampled point?
"""

# %%
df['birth_rate'] = beta(df[['x', 'y']].values)
df['death_rate'] = delta(df[['x', 'y']].values)
df['cum_growth'] = df['birth_rate'] - df['death_rate']

sns.scatterplot(df, x='x', y='y', hue='birth_rate', palette='bwr')
plt.show()
sns.scatterplot(df, x='x', y='y', hue='death_rate', palette='bwr')
plt.show()
sns.scatterplot(df, x='x', y='y', hue='cum_growth', palette='bwr')
plt.show()

# %% [md]
"""
How do the population densities evolve over time?
"""

# %%
g = sns.FacetGrid(df.reset_index(), col='time', col_wrap=4)
g.map_dataframe(sns.kdeplot, x='x', y='y')

# %% [md]
"""
We can see that one of the wells outcompetes the other.
"""

# %% [md]
"""
Trace trajectories for each clone:
"""

# %%
for ix, df_traj in df.groupby('index'):
    plt.plot(df_traj['x'], df_traj['y'], alpha=0.3)
plt.show()

# %%
times = [np.array([t] * len(X)) for t, X in zip(traj.times, traj.X)]
times = np.concat(times)
clones = np.concat(traj.indices)
Z = np.concat(traj.X)

# %%
key = random.PRNGKey(585)
init_key, sim_key = random.split(key, 2)

Z = jnp.array(Z)
out = scsim.counts.simulate_counts(
    params,
    sim_key,
    Z,
    config,
    add_technical_dropout=True,
    dropout_strength=1.0,
    add_batches=False,
)

X = out["counts"]
modules = out["module_activity"]

scsim.counts.qc_counts(out)

# %%
from scipy.sparse import csr_matrix

# %%
meta = pd.DataFrame({'time': times, 'clone': clones})
adata = ad.AnnData(csr_matrix(X), obs=meta, obsm={'X_latent': np.array(Z), 'X_modules': np.array(modules)})
adata.var.loc[:, 'proliferation_gene'] = adata.var_names.isin(proliferation_genes.flatten().astype(str))
adata.var.loc[:, 'apoptosis_gene'] = adata.var_names.isin(apoptosis_genes.flatten().astype(str))
adata

# %%
sc.pl.embedding(adata, basis='X_latent', color='time', cmap='turbo')

# %%
sc.pp.filter_genes(adata, min_cells=1)
adata

# %%
sc.experimental.pp.highly_variable_genes(adata, flavor='pearson_residuals', n_top_genes=1024)

# %%
hvgs = adata.var['highly_variable']

fig, ax = plt.subplots()
ax.scatter(adata.var["means"], adata.var["residual_variances"], s=3, edgecolor="none")
ax.scatter(
    adata.var["means"][hvgs],
    adata.var["residual_variances"][hvgs],
    c="tab:red",
    label="selected genes",
    s=3,
    edgecolor="none",
)
ax.set_xscale("log")
ax.set_xlabel("mean expression")
ax.set_yscale("log")
ax.set_ylabel("residual variance")

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.yaxis.set_ticks_position("left")
ax.xaxis.set_ticks_position("bottom")

# %%
adata = adata[:, adata.var['highly_variable']]
adata

# %%
adata.layers['counts'] = adata.X.copy()
sc.experimental.pp.normalize_pearson_residuals(adata)

# %%
sc.pp.pca(adata)
sc.pp.neighbors(adata, n_neighbors=15)
sc.tl.umap(adata)

# %%
sc.pl.umap(adata, color='time', cmap='turbo')

# %%
ix_fate_A = adata.obsm['X_latent'][:, 0] < 0.0
adata.obs['fate'] = 'B'
adata.obs.loc[ix_fate_A, 'fate'] = 'A'
sc.pl.umap(adata, color='fate')

# %%
sns.kdeplot(x=adata.obsm['X_umap'][:, 0], y=adata.obsm['X_umap'][:, 1])

# %%
clone_tags = traj.get_tags()
for time, tags in clone_tags.items():
    ix_time = adata.obs['time'] >= time
    adata.obs.loc[ix_time, f"tag_{time:0.1f}"] = adata.obs["clone"].map(tags)

# %%
adata.obs.isna().sum()

# %%
adata.obs['time'].value_counts(sort=False).cumsum()

# %%
adata.write_h5ad('../data/lineage_sim.h5ad')

