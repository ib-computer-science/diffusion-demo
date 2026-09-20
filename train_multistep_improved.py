"""Improved multi-step denoiser: richer sinusoidal time embedding + more
capacity/training, testing whether these (legitimate, non-oracle) changes
close more of the yellow-bump gap than train_multistep.py's baseline.

Two changes relative to train_multistep.py:
  1. Time is now embedded via multi-frequency sin/cos features instead of a
     single raw scalar t/T, giving the network more expressive power to
     represent very different behavior across the schedule (sharp local
     corrections near t=1 vs. the near-constant "return to the prior mean"
     near t=T -- see learned_curve_on_joint_multistep.png).
  2. More hidden units and more training iterations than the baseline
     (train_multistep.py used hidden=128, N_ITERS=20000; doubling iterations
     there alone previously shrank yellow's shortfall from ~44% low to
     ~16% low -- see CLAUDE.md finding #2).
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

OUT_DIR = "output"
X_GRID = np.linspace(-1.0, 1.0, 1000)

HIDDEN = 192
N_ITERS = 60000
BATCH_SIZE = 512
LR = 2e-3
TIME_FREQS = (1, 2, 4, 8, 16, 32)
INPUT_DIM = 1 + 2 * len(TIME_FREQS)

REGIONS = [("red", -0.5, -0.18), ("yellow", -0.08, 0.08), ("green", 0.18, 0.5)]


def time_embedding(t_norm):
    """t_norm: (N,) in (0, 1] -> (N, 2*len(TIME_FREQS)) sinusoidal features."""
    t_norm = np.asarray(t_norm)[:, None]
    freqs = np.array(TIME_FREQS)[None, :]
    angles = 2.0 * np.pi * freqs * t_norm
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)


def make_inputs(x, t_norm):
    return np.concatenate([x[:, None], time_embedding(t_norm)], axis=1)


def train(rng):
    model = TinyMLP(input_dim=INPUT_DIM, hidden=HIDDEN, rng=rng, lr=LR)
    losses = []
    for _ in range(N_ITERS):
        x0 = sample_gmm(BATCH_SIZE, rng)
        t_idx = rng.integers(1, T + 1, size=BATCH_SIZE)
        alpha_bar_t = ALPHA_BARS[t_idx - 1]
        eps = rng.normal(size=BATCH_SIZE)
        x_t = np.sqrt(alpha_bar_t) * x0 + np.sqrt(1.0 - alpha_bar_t) * eps

        inputs = make_inputs(x_t, t_idx / T)
        eps_hat = model.forward(inputs)[:, 0]
        residual = eps_hat - eps
        losses.append(np.mean(residual**2))

        grad_out = (2.0 / BATCH_SIZE) * residual[:, None]
        model.backward(grad_out)
    return model, losses


def reverse_sample(model, n_samples, rng, x_init=None):
    """x_init lets a caller supply (and thus display/track) the exact x_T
    the reverse process starts from, instead of it being drawn internally."""
    x = rng.normal(size=n_samples) if x_init is None else np.asarray(x_init, dtype=float)
    for t in range(T, 0, -1):
        t_norm = np.full(n_samples, t / T)
        eps_hat = model.predict(make_inputs(x, t_norm))[:, 0]

        alpha_t = ALPHAS[t - 1]
        beta_t = BETAS[t - 1]
        alpha_bar_t = ALPHA_BARS[t - 1]

        mean = (x - (beta_t / np.sqrt(1.0 - alpha_bar_t)) * eps_hat) / np.sqrt(alpha_t)
        if t > 1:
            z = rng.normal(size=n_samples)
            x = mean + np.sqrt(beta_t) * z
        else:
            x = mean
    return x


def basin_fracs(x0):
    return {name: float(np.mean((x0 > lo) & (x0 < hi))) for name, lo, hi in REGIONS}


def hue_strip(ax, y0, y1, n=600):
    xs = np.linspace(*ax.get_xlim(), n)
    colors = np.array([x_to_rgb(v) for v in xs])[None, :, :]
    ax.imshow(colors, extent=[xs[0], xs[-1], y0, y1], aspect="auto", origin="lower", zorder=0)


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
    ax.set_ylabel("MSE(eps_hat, eps)")
    ax.set_title(f"Improved model training loss (hidden={HIDDEN}, sinusoidal t embedding)")

    ax = axes[1]
    bins = np.linspace(-1.0, 1.0, 150)
    ax.hist(x0_true, bins=bins, density=True, alpha=0.5, label="original p(x0)", color="tab:blue")
    ax.hist(x0_generated, bins=bins, density=True, alpha=0.5,
            label="reverse-sampled (improved model)", color="tab:orange")
    ymax = max(gmm_pdf(X_GRID).max(), 1.0) * 1.1
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.08 * ymax, ymax)
    hue_strip(ax, -0.08 * ymax, 0.0)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("x0")
    ax.set_title(f"Original vs. reverse-sampled ({T}-step, improved model)")
    ax.legend(fontsize=8)

    fig.suptitle("Improved multi-step DDPM: sinusoidal time embedding + more capacity/training", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(OUT_DIR, "train_multistep_improved.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
