# %%
import numpy as np
import seaborn as sns
import scanpy as sc
import moscot as mt
import scotty as sct

# %%
# %load_ext autoreload
# %autoreload 2

# %% [md]
"""
# Overview

This notebook will highlight some of the trajectory analysis tools included in `scotty-tools`, including 
:

* "Fate flow" visualization
* Fate entropy analysis (TODO)
* Trajectory integration and clustering
"""

# %% [md]
"""
# Dataset

For this tutorial, we will use a dataset of differentiating hematopoietic stem cells collected at several timepoints from a single donor. Each cell has been labeled with its cell type based on prior biological knowledge (i.e. marker gene expression), giving us a ground-truth against which to compare our trajectory analysis.
"""

# %%
adata = mt.datasets.hspc()
adata

# %% [md]
"""
Although the dataset includes both single-cell RNA and ATAC sequencing, we will only use the RNA gene expression (GEX) data for simplicity.
"""

# %%
sc.pl.embedding(adata, basis='umap_GEX', color=['cell_type', 'day'])

# %% [md]
"""
We see that the cells form clusters based on cell type, and that each cell type is stratified by timepoint as well.
"""

# %% [md]
"""
## Fit trajectories using `moscot`

First, we will fit optimal transport (OT) trajectories using the computational machinery built into `moscot`. In addition, we will check for outliers in predicted growth rates, and clip them to prevent trajectory collapse.

We compute the prior growth rates from proliferation and apoptosis gene set scores and apply a scaling factor, $\sigma$, that determines how heavily the raw gene set scores impact the prior growth rates, via the formula:

$$g = \exp\left[\frac{\left(y_\text{prolif.} - y_\text{apop.}\right)\left(t_\text{target} - t_\text{source}\right)}{\sigma}\right]$$
"""

# %%
ot_model = sct.models.trajectory.MoscotModel.from_adata(adata)
ot_model.score_genes_for_marginals(gene_set_proliferation='human', gene_set_apoptosis='human')
ot_model.prepare(time_key='day', marginal_kwargs={'scaling': 2})

# %% [md]
"""
Let's look at the prior growth rates more closely:
"""

# %%
adata.obs['prior_growth'] = ot_model.moscot_model.prior_growth_rates
df_growth = adata.obs[['day', 'prior_growth']]
df_growth = df_growth.loc[~df_growth['prior_growth'].isna()]
df_growth['day'] = df_growth['day'].cat.remove_unused_categories()

g = sns.FacetGrid(df_growth, col='day', hue='day', sharex=False, sharey=False)
g.map_dataframe(sns.histplot, x='prior_growth')
g.set(yscale='log')

# %% [md]
"""
There are a few outliers at each source day with high prior growth rates. Due to the exponential population model, these can exert a high amount of influence on the trajectory when there are large time gaps. In the extreme, this may collapse the entirety of the trajectory onto a single cell. To avoid this undesirable behavior, we can clip the growth rates for each timepoint at an upper threshold of our choosing.
"""

# %%
ot_model.clip_growth_rates(upper_quantile=0.95)

adata.obs['prior_growth_clipped'] = ot_model.moscot_model.prior_growth_rates
df_growth = adata.obs[['day', 'prior_growth_clipped']]
df_growth = df_growth.loc[~df_growth['prior_growth_clipped'].isna()]
df_growth['day'] = df_growth['day'].cat.remove_unused_categories()

g = sns.FacetGrid(df_growth, col='day', hue='day', sharex=False, sharey=False)
g.map_dataframe(sns.histplot, x='prior_growth_clipped')
g.set(yscale='log')

# %% [md]
"""
Clipping the growth rates effectively creates a larger pool of ancestors at each source timepoint.
"""

# %% [md]
"""
Now, we can solve the OT problem.
"""

# %%
ot_model.solve(epsilon=0.01, tau_a=0.95, tau_b=0.9995, scale_cost='mean')

# %% [md]
"""
Let's visualize how the (clipped) prior growth rates compare to the posterior growth rates after fitting:
"""

# %%
adata.obs['posterior_growth'] = ot_model.moscot_model.posterior_growth_rates
adata.obs['posterior_vs_prior_growth'] = np.log2(adata.obs['posterior_growth'] / adata.obs['prior_growth_clipped'])

sc.pl.embedding(adata, basis='umap_GEX', color=['prior_growth_clipped', 'posterior_growth', 'posterior_vs_prior_growth'])

df_growth = adata.obs[['day', 'prior_growth_clipped', 'posterior_growth']]
df_growth = df_growth.loc[~df_growth['prior_growth_clipped'].isna()]
df_growth['day'] = df_growth['day'].cat.remove_unused_categories()

g = sns.FacetGrid(df_growth, col='day', hue='day', sharey=False, sharex=False)
g.map_dataframe(sns.scatterplot, x='prior_growth_clipped', y='posterior_growth')
g.set(xscale='log', yscale='log')

# %% [md]
"""
We see good agreement between the prior and posterior growth rates.
"""


# %% [md]
"""
## Plot fate-flow Sankeys using `scotty-tools`

We can visualize the differentiation relationships between cell types over time by using a Sankey diagram to summarize "fate flows." This visualization represents the probabilities of cell transitions at consecutive timepoints through the sizes of the flows that connect clusters. We can further represent population expansion and contraction resulting from the aggregate effects of proliferation and apoptosis through the changing sizes of flows between timepoints.

Let's look at fate flows between labeled cell types under OT:
"""

# %%
sankey_cell_type = sct.plotting.Sankey(ot_model, adata.obs['cell_type'])
_ = sankey_cell_type.plot_all_transitions(min_flow_threshold=0.01)
sc.pl.embedding(adata, basis='umap_GEX', color='cell_type')

# %% [md]
"""
## Integrate timepoints using `scotty-tools`

For timecourse datasets, we may be interested in finding time-invariant features of lineages. For example, we may want to find genes that stably identify lineages over time, rather than those which are transiently expressed. Or, we may want to cluster cells into distinct lineages based on their predicted trajectories.

We can leverage our OT model to integrate across timepoints using a trajectory kernel mean embedding (TKME). This effectively integrates cells from distinct timepoints according to the OT model, by projecting cells into an embedding space shared across timepoints.
"""

# %%
tkme = sct.models.featurize.TrajectoryKMEFeaturizer()
X_tkme = tkme.fit_transform(ot_model, adata)

# %% [md]
"""
We can treat the TKME features like any other transformation, adding them to our `adata` and performing dimensionality reduction.
"""

# %%
adata.obsm['X_tkme'] = X_tkme[adata.obs_names].X
sc.pp.pca(adata, obsm='X_tkme', key_added='X_pca_tkme')
sc.pp.neighbors(adata, use_rep='X_pca_tkme', key_added='neighbors_tkme', n_neighbors=30)
sc.tl.umap(adata, neighbors_key='neighbors_tkme', key_added='X_umap_TKME')

# %%
sc.pl.embedding(adata, basis='umap_TKME', color=['day', 'cell_type'])

# %% [md]
"""
We can see that the TKME featurization effectively integrated cells across timepoints, while keeping distinct cell type lineages intact.
"""

# %% [md]
"""
## Cluster trajectories using `scotty-tools`

One use for whole-trajectory embeddings is to perform clustering to identify major cell lineages. We will compare the results of Leiden clustering on both gene expression (GEX) as well as TKME to see how their trajectories differ.
"""

# %%
sc.tl.leiden(adata, resolution=1, neighbors_key='neighbors', key_added='leiden_GEX')
sc.tl.leiden(adata, resolution=1, neighbors_key='neighbors_tkme', key_added='leiden_TKME')

print('Clusters visualized in GEX UMAP space:')
sc.pl.embedding(adata, basis='umap_GEX', color=['cell_type', 'leiden_GEX', 'leiden_TKME'])
print('Clusters visualized in TKME UMAP space:')
sc.pl.embedding(adata, basis='umap_TKME', color=['cell_type', 'leiden_GEX', 'leiden_TKME'])

# %% [md]
"""
Both sets of clusters correspond quite well to the ground-truth cell types. This makes sense, since the gene expression information alone already does a good job of encoding lineage in this dataset without being confounded by time. 

However, we can still see that the trajectory clusters provide us with a smoother picture of the differentiation process by comparing fate flows for both sets of clusters:
"""

# %%
print('GEX clusters:')
sankey_leiden_gex = sct.plotting.Sankey(ot_model, adata.obs['leiden_GEX'])
_ = sankey_leiden_gex.plot_all_transitions(min_flow_threshold=0.01)
sc.pl.embedding(adata, basis='umap_GEX', color='leiden_GEX')

# %%
print('TKME clusters:')
sankey_leiden_tkme = sct.plotting.Sankey(ot_model, adata.obs['leiden_TKME'])
_ = sankey_leiden_tkme.plot_all_transitions(min_flow_threshold=0.01)
sc.pl.embedding(adata, basis='umap_TKME', color='leiden_TKME')

# %% [md]
"""
The trajectory clusters have much smoother flows, forming streamlines that cross-section the trajectory into distinct lineages.
"""
