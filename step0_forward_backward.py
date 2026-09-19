"""Demo: one forward diffusion step on a 1D hue distribution, and why it can
be reversed exactly when the data distribution is known.

Produces output/step0_forward_backward.png with four panels:
  1. p(x0)             -- the true hue distribution (red / yellow / green bumps)
  2. q(x1)             -- after one forward noising step
  3. q(x0 | x1=v)      -- exact reverse posterior, for several example v
  4. p(x0, x1)         -- their joint distribution: a horizontal slice at
                          x1=v (dashed lines, colors matching panel 3) is,
                          once renormalized, exactly the posterior curve
                          shown in panel 3 for that v.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ddpm_step import joint_pdf, marginal_x1_params, posterior_pdf
from hue_gmm import MEANS, NAMES, gmm_pdf, normal_pdf, x_to_rgb

BETA1 = 0.005
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
    q1 = (w1 * normal_pdf(X_GRID[..., None], m1, s1)).sum(-1)
    ax.plot(X_GRID, q1, color="black", linewidth=1.5)
    ax.plot(X_GRID, p0, color="gray", linewidth=1.0, linestyle="--", label="p(x0) (reference)")
    setup_axis(ax, f"q(x1): after one forward step (beta1={BETA1})", 1.05 * max(p0.max(), q1.max()))
    ax.legend(fontsize=8, loc="upper right")

    # --- panel 3: exact reverse posterior q(x0 | x1=v) for several v ---
    ax = axes[1, 0]
    example_vs = [-1 / 3, -1 / 6, 0.0, 1 / 6, 1 / 3]
    example_colors = []
    ymax3 = p0.max()
    for v in example_vs:
        post = posterior_pdf(X_GRID, v, BETA1)
        ymax3 = max(ymax3, post.max())
        line, = ax.plot(X_GRID, post, linewidth=1.3, label=f"v={v:+.3f}")
        ax.axvline(v, color=line.get_color(), linewidth=0.8, linestyle=":")
        example_colors.append(line.get_color())
    ax.plot(X_GRID, p0, color="gray", linewidth=1.0, linestyle="--", label="p(x0) (reference)")
    setup_axis(ax, "q(x0 | x1=v): exact reverse posterior", 1.05 * ymax3)
    ax.legend(fontsize=7, loc="upper right", ncol=2)

    # --- panel 4: joint distribution p(x0, x1) ---
    ax = axes[1, 1]
    lim = 0.6
    plot_grid = np.linspace(-lim, lim, 400)
    x0_grid, x1_grid = np.meshgrid(plot_grid, plot_grid, indexing="ij")
    joint = joint_pdf(x0_grid, x1_grid, BETA1)

    strip_frac = 0.08
    y0_strip = -lim - strip_frac * 2 * lim
    ax.set_xlim(-lim, lim)
    ax.set_ylim(y0_strip, lim)
    im = ax.imshow(joint.T, origin="lower", extent=[-lim, lim, -lim, lim], aspect="auto", cmap="viridis")
    hue_strip(ax, y0_strip, -lim)
    for v, color in zip(example_vs, example_colors):
        ax.axhline(v, color=color, linewidth=1.2, linestyle="--")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="density")
    ax.set_title("p(x0, x1): joint distribution")
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")

    fig.suptitle("One DDPM step on 1D hue data: forward noising and exact Bayesian reversal", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(OUT_DIR, "step0_forward_backward.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
