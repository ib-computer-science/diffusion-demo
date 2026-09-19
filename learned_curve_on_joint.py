"""What does the MLP actually learn? Answer: a single curve through the
joint distribution p(x0, x1).

Squared-error training converges to f(v) = E[eps | x1=v], which is just a
linear reparameterization of E[x0 | x1=v] -- the density-weighted centroid
of each horizontal slice through the joint distribution (see
ddpm_step.posterior_mean_curve). This script overlays that exact curve, and
the curve implied by the trained MLP, directly on the joint-density heatmap
from step0_forward_backward.py, to show precisely what training recovers
and where it's forced to cut through low-density gaps between hue bumps.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ddpm_step import joint_pdf, posterior_mean_curve
from hue_gmm import x_to_rgb
from train_denoiser import A1, BETA1, train

OUT_DIR = "output"


def hue_strip(ax, y0, y1, n=600):
    xs = np.linspace(*ax.get_xlim(), n)
    colors = np.array([x_to_rgb(x) for x in xs])[None, :, :]
    ax.imshow(colors, extent=[xs[0], xs[-1], y0, y1], aspect="auto", origin="lower", zorder=0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(0)
    model, losses = train(rng)
    print(f"final training loss (mean of last 100 iters): {np.mean(losses[-100:]):.4f}")

    lim = 0.6
    plot_grid = np.linspace(-lim, lim, 400)
    x0_grid, x1_grid = np.meshgrid(plot_grid, plot_grid, indexing="ij")
    joint = joint_pdf(x0_grid, x1_grid, BETA1)

    v_grid = np.linspace(-lim, lim, 300)
    exact_curve = posterior_mean_curve(v_grid, BETA1)

    eps_hat = model.predict(v_grid[:, None])[:, 0]
    learned_curve = (v_grid - np.sqrt(BETA1) * eps_hat) / np.sqrt(A1)

    fig, ax = plt.subplots(figsize=(7.5, 7))
    strip_h = 0.08 * 2 * lim
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim - strip_h, lim)
    im = ax.imshow(joint.T, origin="lower", extent=[-lim, lim, -lim, lim], aspect="auto", cmap="viridis")
    hue_strip(ax, -lim - strip_h, -lim)

    ax.plot(exact_curve, v_grid, color="white", linewidth=2.2, label="exact E[x0 | x1=v]")
    ax.plot(learned_curve, v_grid, color="tab:orange", linewidth=2.0, linestyle="--",
            label="learned (from trained MLP)")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="density")
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_title("What the MLP learns: a curve through p(x0, x1)")
    ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "learned_curve_on_joint.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
