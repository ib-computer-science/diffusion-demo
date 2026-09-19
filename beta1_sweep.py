"""How beta1 shapes the exact reverse posterior q(x0 | x1=v).

Produces output/beta1_sweep.png with two panels:
  1. q(x0 | x1=v) for a fixed, deliberately ambiguous v (between red and
     yellow), overlaid for several beta1 values -- from a near-delta spike
     to something close to the full prior p(x0).
  2. Two quantitative summaries vs beta1: the per-component posterior std,
     and the responsibility entropy at that same v (both saturate: std ->
     the prior's own std, entropy -> the entropy of the prior mixture
     weights, as beta1 -> 1).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ddpm_step import posterior_given_x1, posterior_pdf
from hue_gmm import STDS, WEIGHTS, gmm_pdf

OUT_DIR = "output"
X_GRID = np.linspace(-1.0, 1.0, 1000)
V = -1 / 6  # ambiguous point, midway between the red and yellow bumps
BETAS = [0.0005, 0.005, 0.02, 0.1, 0.4, 0.9]

PRIOR_STD = STDS[0]  # all components share the same std in this demo
PRIOR_ENTROPY = -(WEIGHTS * np.log(WEIGHTS)).sum()


def responsibility_entropy(v, beta1):
    resp, _, _ = posterior_given_x1(v, beta1)
    return -(resp * np.log(resp)).sum()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- panel 1: posterior shape across beta1, at fixed ambiguous v ---
    ax = axes[0]
    p0 = gmm_pdf(X_GRID)
    ax.plot(X_GRID, p0, color="gray", linestyle="--", linewidth=1.2, label="p(x0) (beta1->1 limit)")
    for beta1 in BETAS:
        post = posterior_pdf(X_GRID, V, beta1)
        ax.plot(X_GRID, post, linewidth=1.5, label=f"beta1={beta1}")
    ax.axvline(V, color="black", linewidth=0.8, linestyle=":", label=f"v={V:.3f}")
    ax.set_xlim(-0.8, 0.8)
    ax.set_title(f"q(x0 | x1={V:.3f}) as beta1 varies")
    ax.set_xlabel("x0")
    ax.legend(fontsize=7, loc="upper left")

    # --- panel 2: quantitative trend vs beta1 ---
    beta_grid = np.geomspace(1e-4, 0.999, 200)
    post_stds = np.array([
        np.sqrt(1.0 / ((1 - b) / b + 1.0 / PRIOR_STD**2)) for b in beta_grid
    ])
    entropies = np.array([responsibility_entropy(V, b) for b in beta_grid])

    ax2 = axes[1]
    ax2.plot(beta_grid, post_stds, color="tab:blue", label="posterior std (per component)")
    ax2.axhline(PRIOR_STD, color="tab:blue", linestyle=":", linewidth=1,
                label="prior std (beta1->1 limit)")
    ax2.set_xscale("log")
    ax2.set_xlabel("beta1 (log scale)")
    ax2.set_ylabel("posterior std", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    ax3 = ax2.twinx()
    ax3.plot(beta_grid, entropies, color="tab:red", label=f"responsibility entropy at v={V:.3f}")
    ax3.axhline(PRIOR_ENTROPY, color="tab:red", linestyle=":", linewidth=1,
                label="entropy of prior weights (beta1->1 limit)")
    ax3.set_ylabel("responsibility entropy (nats)", color="tab:red")
    ax3.tick_params(axis="y", labelcolor="tab:red")

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="lower right")
    ax2.set_title("Posterior sharpness vs beta1: spike -> prior")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "beta1_sweep.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
