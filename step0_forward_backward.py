"""Demo: one forward diffusion step on a 1D hue distribution, and why it can
be reversed exactly when the data distribution is known.

Produces output/step0_forward_backward.png with four panels:
  1. p(x0)          -- the true hue distribution (red / yellow / green bumps)
  2. q(x1)          -- after one forward noising step
  3. q(x0 | x1=v)   -- exact reverse posterior, for several example v
  4. round trip     -- x0 -> x1 -> resampled x0, checked against the original
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ddpm_step import forward_step, marginal_x1_params, posterior_pdf, sample_posterior
from hue_gmm import MEANS, NAMES, WEIGHTS, gmm_pdf, sample_gmm, x_to_rgb

BETA1 = 0.005
N_SAMPLES = 20000
X_GRID = np.linspace(-1.0, 1.0, 1000)
OUT_DIR = "output"


def hue_strip(ax, y0, y1, n=600):
    xs = np.linspace(*ax.get_xlim(), n)
    colors = np.array([x_to_rgb(x) for x in xs])[None, :, :]
    ax.imshow(colors, extent=[xs[0], xs[-1], y0, y1], aspect="auto", origin="lower", zorder=0)


def setup_axis(ax, title, ymax):
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.08 * ymax, ymax)
    hue_strip(ax, -0.08 * ymax, 0.0)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("x")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # --- panel 1: p(x0) ---
    ax = axes[0, 0]
    p0 = gmm_pdf(X_GRID)
    ax.plot(X_GRID, p0, color="black", linewidth=1.5)
    setup_axis(ax, "p(x0): true hue distribution", 1.05 * p0.max())
    for mu, name in zip(MEANS, NAMES):
        ax.annotate(name, (mu, gmm_pdf(np.array([mu]))[0]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)

    # --- panel 2: q(x1) after one forward step ---
    ax = axes[0, 1]
    w1, m1, s1 = marginal_x1_params(BETA1)
    from hue_gmm import normal_pdf  # local import to avoid cluttering top-level namespace
    q1 = (w1 * normal_pdf(X_GRID[..., None], m1, s1)).sum(-1)
    ax.plot(X_GRID, q1, color="black", linewidth=1.5)
    ax.plot(X_GRID, p0, color="gray", linewidth=1.0, linestyle="--", label="p(x0) (reference)")
    setup_axis(ax, f"q(x1): after one forward step (beta1={BETA1})", 1.05 * max(p0.max(), q1.max()))
    ax.legend(fontsize=8, loc="upper right")

    # --- panel 3: exact reverse posterior q(x0 | x1=v) for several v ---
    ax = axes[1, 0]
    example_vs = [-1 / 3, -1 / 6, 0.0, 1 / 6, 1 / 3]
    ymax3 = p0.max()
    for v in example_vs:
        post = posterior_pdf(X_GRID, v, BETA1)
        ymax3 = max(ymax3, post.max())
        line, = ax.plot(X_GRID, post, linewidth=1.3, label=f"v={v:+.3f}")
        ax.axvline(v, color=line.get_color(), linewidth=0.8, linestyle=":")
    ax.plot(X_GRID, p0, color="gray", linewidth=1.0, linestyle="--", label="p(x0) (reference)")
    setup_axis(ax, "q(x0 | x1=v): exact reverse posterior", 1.05 * ymax3)
    ax.legend(fontsize=7, loc="upper right", ncol=2)

    # --- panel 4: round-trip sanity check ---
    ax = axes[1, 1]
    x0_true = sample_gmm(N_SAMPLES, rng)
    x1 = forward_step(x0_true, BETA1, rng)
    x0_reconstructed = sample_posterior(x1, BETA1, rng)
    bins = np.linspace(-1.0, 1.0, 120)
    ax.hist(x0_true, bins=bins, density=True, alpha=0.5, label="x0 (original samples)", color="tab:blue")
    ax.hist(x0_reconstructed, bins=bins, density=True, alpha=0.5,
            label="x0 reconstructed via x0->x1->q(x0|x1)", color="tab:orange")
    setup_axis(ax, "Round trip: forward then exact reverse", 1.05 * p0.max())
    ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("One DDPM step on 1D hue data: forward noising and exact Bayesian reversal", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(OUT_DIR, "step0_forward_backward.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
