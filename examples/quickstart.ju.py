# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import moscot as mt
import scotty as sct

from scipy.spatial import distance_matrix
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import adjusted_mutual_info_score

# %%
# %load_ext autoreload
# %autoreload 2

# %% [md]
"""
# Overview

This notebook will highlight some of the trajectory analysis tools included in `scotty-tools`, including 
:

* "Fate flow" visualization
* Trajectory integration and clustering
* Fate entropy analysis
"""

# %% [md]
"""
# Dataset


"""

# %%
adata = mt.datasets.hspc()
adata

# %%
sc.pl.embedding(adata, basis='umap_GEX', color='cell_type')

# %%
sc.pl.embedding(adata, basis='umap_GEX', color='day')

# %% [md]
"""
## Fit trajectories using `moscot`

First, we will fit optimal transport trajectories using the computational machinery built into `moscot`. In addition, we will check for outliers in predicted growth rates, and clip them to prevent trajectory collapse.

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

We can visualize the differentiation relationships between cell types over time by using a Sankey diagram to summarize "fate flows":
"""

# %%
sankey_cell_type = sct.plotting.Sankey(ot_model, adata.obs['cell_type'])
_ = sankey_cell_type.plot_all_transitions(min_flow_threshold=0.01)
sc.pl.embedding(adata, basis='umap_GEX', color='cell_type')

# %% [md]
"""
## Cluster trajectories using `scotty-tools`
"""

# %%
gammas = dict()
for day, df in adata.obs.groupby('day'):
    X = adata[df.index].obsm['X_pca']
    D = distance_matrix(X, X)
    sig = np.median(D[np.triu_indices_from(D, k=1)])
    gammas[day] = 1 / sig**2

print(gammas)


# %%
feats = dict()
for day, df in adata.obs.groupby('day'):
    phi = Nystroem(n_components=50, gamma=gammas[day], random_state=585)
    X = phi.fit_transform(adata[df.index].obsm['X_pca'])
    feats[day] = pd.DataFrame(X, index=df.index)

X_tkme = sct.models.featurize.featurize_trajectories(feats, ot_model)

# %%
adata.obsm['X_tkme'] = X_tkme[adata.obs_names].X
sc.pp.pca(adata, obsm='X_tkme', key_added='X_pca_tkme')
sc.pp.neighbors(adata, use_rep='X_pca_tkme', key_added='neighbors_tkme', n_neighbors=30)
sc.tl.umap(adata, neighbors_key='neighbors_tkme', key_added='X_umap_TKME')

# %%
sc.pl.embedding(adata, basis='umap_TKME', color=['day', 'cell_type'])

# %%
sc.tl.leiden(adata, resolution=1, neighbors_key='neighbors', key_added='leiden_GEX')
sc.tl.leiden(adata, resolution=1, neighbors_key='neighbors_tkme', key_added='leiden_TKME')

sc.pl.embedding(adata, basis='umap_GEX', color=['cell_type', 'leiden_GEX', 'leiden_TKME'])
sc.pl.embedding(adata, basis='umap_TKME', color=['cell_type', 'leiden_GEX', 'leiden_TKME'])

# %%
print(adjusted_mutual_info_score(adata.obs['cell_type'], adata.obs['leiden_GEX']))
print(adjusted_mutual_info_score(adata.obs['cell_type'], adata.obs['leiden_TKME']))

# %%
sankey_leiden_gex = sct.plotting.Sankey(ot_model, adata.obs['leiden_GEX'])
_ = sankey_leiden_gex.plot_all_transitions(min_flow_threshold=0.01)

# %%
sankey_leiden_tkme = sct.plotting.Sankey(ot_model, adata.obs['leiden_TKME'])
_ = sankey_leiden_tkme.plot_all_transitions(min_flow_threshold=0.01)

# %% [md]
"""
## Quantify fate entropy using `scotty-tools`
"""

# %%
df = sankey_cell_type.compute_flow_entropy()
1 - df['expected'] / df['prior']

# %%
df = sankey_leiden_tkme.compute_flow_entropy()
1 - df['expected'] / df['prior']

# %%
df = sankey_leiden_gex.compute_flow_entropy()
1 - df['expected'] / df['prior']

# %%
subsettings = ['cell_type', 'leiden_GEX', 'leiden_TKME']
consistency = dict()
for sub in subsettings:
    consistency[sub] = sct.tools.metrics.fate_consistency(ot_model, adata.obs[sub])

consistency = pd.DataFrame(consistency).melt(var_name='subsetting', value_name='consistency')
sns.violinplot(consistency, x='subsetting', y='consistency', hue='subsetting')
plt.show()
sns.boxplot(consistency, x='subsetting', y='consistency', hue='subsetting')
plt.show()
