"""Legitimate (non-oracle) importance-weighted training: reweight the loss
by an inverse DENSITY ESTIMATE built from a large pool of unlabeled x0
samples, instead of the known true mixture weights.

CLAUDE.md's plateau finding (doubling iterations 30k->60k on the improved
architecture barely moved yellow's shortfall, 29% -> 26% low) argues the
bottleneck is structural: every training batch draws yellow-region examples
at a fixed 1:3:3 ratio relative to red/green, forever -- more training
reduces noise but can't change that ratio. This reweights each training
example by 1/density(x0), where density is estimated purely empirically
from a big pool of samples drawn from the same (unlabeled) source used for
training. Unlike the earlier-rejected idea of reweighting by the *known*
analytic component weights 1/pi_k, this uses no ground truth about modes or
their true weights -- exactly the kind of information available on real,
unlabeled data (e.g. a k-means or KDE density estimate over training images).

Built on top of train_multistep_improved.py's architecture (sinusoidal time
embedding, hidden=192) so the comparison isolates just the reweighting's
effect: same architecture, same N_ITERS=30000, only the loss weighting
differs from that script's unweighted run (yellow was 0.090, 29% low, there).
"""

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hue_gmm import gmm_pdf, sample_gmm, x_to_rgb
from mlp import TinyMLP
from schedule import ALPHA_BARS, ALPHAS, BETAS, T
from train_multistep_improved import HIDDEN, INPUT_DIM, LR, basin_fracs, hue_strip, make_inputs, reverse_sample, REGIONS

OUT_DIR = "output"
X_GRID = np.linspace(-1.0, 1.0, 1000)
N_ITERS = 30000
BATCH_SIZE = 512

DENSITY_POOL_SIZE = 200000
DENSITY_BINS = 100
MAX_WEIGHT = 15.0  # cap so rare tail values can't destabilize training


def estimate_density(rng):
    """Empirical density of x0 from a large pool of samples drawn from the
    same (unlabeled) source used for training -- no knowledge of true
    component identity or weights, just observed data."""
    pool = sample_gmm(DENSITY_POOL_SIZE, rng)
    counts, edges = np.histogram(pool, bins=DENSITY_BINS, range=(-1.0, 1.0), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts


def make_weight_fn(centers, counts):
    def weight_fn(x):
        density = np.interp(x, centers, counts, left=counts[0], right=counts[-1])
        density = np.maximum(density, 1e-8)
        return np.minimum(1.0 / density, MAX_WEIGHT)
    return weight_fn


def train(rng):
    centers, counts = estimate_density(rng)
    weight_fn = make_weight_fn(centers, counts)

    model = TinyMLP(input_dim=INPUT_DIM, hidden=HIDDEN, rng=rng, lr=LR)
    losses = []
    for _ in range(N_ITERS):
        x0 = sample_gmm(BATCH_SIZE, rng)
        t_idx = rng.integers(1, T + 1, size=BATCH_SIZE)
        alpha_bar_t = ALPHA_BARS[t_idx - 1]
        eps = rng.normal(size=BATCH_SIZE)
        x_t = np.sqrt(alpha_bar_t) * x0 + np.sqrt(1.0 - alpha_bar_t) * eps

        w = weight_fn(x0)  # importance weight from x0's own estimated rarity
        inputs = make_inputs(x_t, t_idx / T)
        eps_hat = model.forward(inputs)[:, 0]
        residual = eps_hat - eps

        w_sum = w.sum()
        losses.append(float((w * residual**2).sum() / w_sum))
        grad_out = (2.0 * w * residual / w_sum)[:, None]
        model.backward(grad_out)
    return model, losses


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(0)

    t0 = time.time()
    model, losses = train(rng)
    print(f"training took {time.time() - t0:.1f}s, "
          f"final loss (mean of last 200 iters): {np.mean(losses[-200:]):.4f}")

    n_samples = 20000
    t0 = time.time()
    x0_generated = reverse_sample(model, n_samples, rng)
    print(f"reverse sampling took {time.time() - t0:.1f}s")

    x0_true = sample_gmm(n_samples, rng)
    gen_frac = basin_fracs(x0_generated)
    true_frac = basin_fracs(x0_true)
    print(f"{'region':8s} {'true':>8s} {'gen':>8s}")
    for name, _, _ in REGIONS:
        print(f"{name:8s} {true_frac[name]:8.4f} {gen_frac[name]:8.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    ax.plot(losses, linewidth=0.5, color="tab:green")
    ax.set_yscale("log")
    ax.set_xlabel("training iteration")
    ax.set_ylabel("weighted MSE(eps_hat, eps)")
    ax.set_title("Density-reweighted training loss")

    ax = axes[1]
    bins = np.linspace(-1.0, 1.0, 150)
    ax.hist(x0_true, bins=bins, density=True, alpha=0.5, label="original p(x0)", color="tab:blue")
    ax.hist(x0_generated, bins=bins, density=True, alpha=0.5,
            label="reverse-sampled (density-reweighted training)", color="tab:orange")
    ymax = max(gmm_pdf(X_GRID).max(), 1.0) * 1.1
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.08 * ymax, ymax)
    hue_strip(ax, -0.08 * ymax, 0.0)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("x0")
    ax.set_title(f"Original vs. reverse-sampled ({T}-step, density-reweighted)")
    ax.legend(fontsize=8)

    fig.suptitle("Self-estimated (non-oracle) importance-weighted training", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(OUT_DIR, "train_multistep_reweighted.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
