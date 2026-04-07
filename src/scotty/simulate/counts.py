from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

import jax
import jax.numpy as jnp
from jax import random


# --------------------------------------------------
# Config
# --------------------------------------------------

@dataclass
class SimulatorConfig:
    latent_dim: int = 2
    hidden_dims: tuple[int, ...] = (64, 64)
    n_modules: int = 10
    n_genes: int = 1000
    module_bias_bound: float = 0.1
    module_scale_factor: float = 3.0
    genes_per_module: int = 120
    overlap: float = 0.05
    weight_scale: float = 0.35
    bias_init: float = 0.0
    mean_log_library: float = 9.0
    sd_log_library: float = 0.35
    theta_low: float = 1.0
    theta_high: float = 20.0


# --------------------------------------------------
# Small NN utilities
# --------------------------------------------------

def glorot_init(key, in_dim, out_dim, scale=1.0, dtype=jnp.float32):
    lim = scale * jnp.sqrt(6.0 / (in_dim + out_dim))
    return random.uniform(key, (in_dim, out_dim), minval=-lim, maxval=lim, dtype=dtype)


def zeros(shape, dtype=jnp.float32):
    return jnp.zeros(shape, dtype=dtype)


def mlp_forward(z, params):
    x = z
    for layer in params[:-1]:
        x = jnp.tanh(x @ layer["W"] + layer["b"])
    x = x @ params[-1]["W"] + params[-1]["b"]
    return jax.nn.softplus(x)


# --------------------------------------------------
# Sparse gene program initialization
# --------------------------------------------------

def make_sparse_mask(key, n_modules, n_genes, genes_per_module, overlap):
    keys = random.split(key, n_modules + 1)
    overlap_key = keys[0]
    module_keys = keys[1:]

    mask = jnp.zeros((n_modules, n_genes), dtype=jnp.float32)

    for m in range(n_modules):
        idx = random.permutation(module_keys[m], n_genes)[:genes_per_module]
        mask = mask.at[m, idx].set(1.0)

    if overlap > 0:
        extra = random.bernoulli(
            overlap_key, p=overlap, shape=(n_modules, n_genes)
        ).astype(jnp.float32)
        mask = jnp.clip(mask + extra, 0.0, 1.0)

    return mask


def make_sparse_weights(key, mask, weight_scale):
    raw = weight_scale * random.normal(key, mask.shape, dtype=jnp.float32) + 1.0
    return mask * raw


def make_gene_bias(n_genes, bias_init):
    return jnp.full((n_genes,), bias_init, dtype=jnp.float32)


def gene_program_forward(module_activity, gene_params):
    gene_logits = module_activity @ gene_params["W"] + gene_params["b"]
    return jax.nn.softplus(gene_logits)


# --------------------------------------------------
# Observation model
# --------------------------------------------------

def sample_library_size(key, shape, mean_log=9.0, sd_log=0.35):
    eps = random.normal(key, (*shape, 1))
    return jnp.exp(mean_log + sd_log * eps)


def scale_to_library(gene_mean_base, library_size, eps=1e-8):
    gene_freq = gene_mean_base / (jnp.sum(gene_mean_base, axis=-1, keepdims=True) + eps)
    return library_size * gene_freq


def make_gene_dispersion(key, shape, low=1.0, high=20.0):
    log_theta = random.uniform(
        key,
        shape=shape,
        minval=jnp.log(low),
        maxval=jnp.log(high),
    )
    return jnp.exp(log_theta)


def sample_negative_binomial(key, mu, theta, eps=1e-8):
    theta = jnp.broadcast_to(theta, mu.shape)
    key_gamma, key_poisson = random.split(key)

    gamma_scale = mu / (theta + eps)
    lam = random.gamma(key_gamma, theta, shape=mu.shape) * gamma_scale
    counts = random.poisson(key_poisson, lam)
    return counts


def add_dropout(key, counts, mu, dropout_strength=1.0):
    """
    dropout_strength=1.0: moderate (realistic)
    dropout_strength=2.0: droplet-like
    dropout_strength=3.0+: very sparse protocols
    """

    p_drop = jnp.exp(-jnp.log1p(mu) / dropout_strength)
    keep = random.bernoulli(key, p=1.0 - p_drop, shape=counts.shape)
    return counts * keep.astype(counts.dtype)


def add_batch_effect(key, z, n_batches=2, scale=0.2):
    n_cells, latent_dim = z.shape
    key_id, key_offset = random.split(key)

    batch_id = random.randint(key_id, (n_cells,), minval=0, maxval=n_batches)
    batch_offsets = random.normal(key_offset, (n_batches, latent_dim)) * scale
    z_shifted = z + batch_offsets[batch_id]
    return z_shifted, batch_id


# --------------------------------------------------
# Init
# --------------------------------------------------

def init_bias_uniform(key, shape, bound=0.1):
    return random.uniform(key, shape, minval=-bound, maxval=bound)


def init_module_decoder(key, config: SimulatorConfig):
    dims = (config.latent_dim,) + config.hidden_dims + (config.n_modules,)
    keys = random.split(key, len(dims) - 1)
    layers = []
    for k, d_in, d_out in zip(keys, dims[:-1], dims[1:]):
        layers.append({
            "W": glorot_init(k, d_in, d_out),
            "b": init_bias_uniform(random.fold_in(k, 123), (d_out,), bound=config.module_bias_bound)
        })
    return layers


def init_gene_program_layer(key, config: SimulatorConfig):
    k_mask, k_weight = random.split(key)
    mask = make_sparse_mask(
        k_mask,
        n_modules=config.n_modules,
        n_genes=config.n_genes,
        genes_per_module=config.genes_per_module,
        overlap=config.overlap,
    )
    W = make_sparse_weights(k_weight, mask, config.weight_scale)
    b = make_gene_bias(config.n_genes, config.bias_init)
    return {
        "mask": mask,
        "W": W,
        "b": b,
    }


def init_simulator(key, config: SimulatorConfig):
    k_dec, k_gene, k_theta = random.split(key, 3)
    params = {
        "module_decoder": init_module_decoder(k_dec, config),
        "gene_program": init_gene_program_layer(k_gene, config),
        "theta": make_gene_dispersion(
            k_theta,
            config.n_genes,
            low=config.theta_low,
            high=config.theta_high,
        ),
    }
    return params


# --------------------------------------------------
# Main simulation
# --------------------------------------------------

def simulate_counts(
    params: Dict[str, Any],
    key,
    z,
    config: SimulatorConfig,
    add_technical_dropout: bool = False,
    dropout_strength: float = 1.0,
    add_batches: bool = False,
    n_batches: int = 2,
    batch_scale: float = 0.2,
):
    k_batch, k_lib, k_nb, k_drop = random.split(key, 4)

    if add_batches:
        z_eff, batch_id = add_batch_effect(
            k_batch, z, n_batches=n_batches, scale=batch_scale
        )
    else:
        z_eff = z
        batch_id = None

    module_activity = mlp_forward(z_eff, params["module_decoder"]) * config.module_scale_factor
    gene_mean_base = gene_program_forward(module_activity, params["gene_program"])

    library_size = sample_library_size(
        k_lib,
        shape=z.shape[:-1],
        mean_log=config.mean_log_library,
        sd_log=config.sd_log_library,
    )

    mu = scale_to_library(gene_mean_base, library_size)
    counts = sample_negative_binomial(k_nb, mu, params["theta"])

    if add_technical_dropout:
        counts = add_dropout(k_drop, counts, mu, dropout_strength=dropout_strength)

    return {
        "z": z,
        "z_effective": z_eff,
        "batch_id": batch_id,
        "module_activity": module_activity,
        "gene_mean_base": gene_mean_base,
        "mu": mu,
        "theta": params["theta"],
        "counts": counts,
        "library_size": library_size,
        "mask": params["gene_program"]["mask"],
        "W": params["gene_program"]["W"],
    }


def qc_counts(out):
    print("=== PIPELINE BREAKDOWN ===")
    print(f"Module activity: min={out['module_activity'].min():.3f}, max={out['module_activity'].max():.3f}")
    print(f"Gene mean base: min={out['gene_mean_base'].min():.3f}, max={out['gene_mean_base'].max():.3f}")
    print(f"Mu: min={out['mu'].min():.3f}, mean={out['mu'].mean():.3f}, max={out['mu'].max():.3f}")
    print(f"Library size: min={out['library_size'].min():.0f}, mean={out['library_size'].mean():.0f}, max={out['library_size'].max():.0f}")
    print(f"Counts: max={out['counts'].max()}, 99% quantile={jnp.quantile(out['counts'], 0.99):.0f}")
    print(f"Gene sparsity: {jnp.mean(out['counts'] == 0) * 100:.1f}%")
    print(f"W nonzero frac: {jnp.mean(out['W'] != 0):.1%}")


