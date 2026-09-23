"""Demo: one forward diffusion step on a 1D hue distribution, and why it can
be reversed exactly when the data distribution is known.

Produces output/step0_forward_backward.png with four panels:
  1. p(x0)             -- the true hue distribution (red / green bumps)
  2. q(x1)             -- after one forward noising step
  3. p(x0, x1)         -- their joint distribution: a horizontal slice at
                          x1=v (dashed lines, colors matching panel 4) is,
                          once renormalized, exactly the posterior curve
                          shown in panel 4 for that v.
  4. q(x0 | x1=v)      -- exact reverse posterior, for several example v

main() takes beta1 as a parameter (default matches the original demo
value) so the same figure can be regenerated at other noise levels, e.g.
step0_forward_backward_large_noise.py calls it with a deliberately too
large beta1. The plotted x-range widens automatically with beta1, since a
large beta1 spreads q(x1) far beyond the original data's range.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ddpm_step import joint_pdf, marginal_x1_params, posterior_pdf
from hue_gmm import MEANS, NAMES, STDS, gmm_pdf, normal_pdf, x_to_rgb

BETA1 = 0.005
OUT_DIR = "output"


def hue_strip(ax, y0, y1, n=600):
    xs = np.linspace(*ax.get_xlim(), n)
    colors = np.array([x_to_rgb(x) for x in xs])[None, :, :]
    ax.imshow(colors, extent=[xs[0], xs[-1], y0, y1], aspect="auto", origin="lower", zorder=0)


def hue_strip_vertical(ax, x0, x1, n=600):
    ys = np.linspace(*ax.get_ylim(), n)
    colors = np.array([x_to_rgb(v) for v in ys])[:, None, :]
    ax.imshow(colors, extent=[x0, x1, ys[0], ys[-1]], aspect="auto", origin="lower", zorder=0)


def setup_axis(ax, title, ymax, xlim):
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-0.08 * ymax, ymax)
    hue_strip(ax, -0.08 * ymax, 0.0)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("x")


def main(beta1=BETA1, out_path=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    a1 = 1.0 - beta1

    # Plot range for the 1D curve panels (p(x0), q(x1), reverse posterior):
    # wide enough that q(x1) isn't clipped even when beta1 is large.
    x1_std_est = np.sqrt(a1 * STDS.max() ** 2 + beta1)
    plot_range = max(1.0, np.abs(MEANS).max() * np.sqrt(a1) + 4.5 * x1_std_est)
    x_grid = np.linspace(-plot_range, plot_range, 1000)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # --- panel 1: p(x0) ---
    ax = axes[0, 0]
    p0 = gmm_pdf(x_grid)
    ax.plot(x_grid, p0, color="black", linewidth=1.5)
    setup_axis(ax, "p(x0): true hue distribution", 1.05 * p0.max(), plot_range)
    for mu, name in zip(MEANS, NAMES):
        ax.annotate(name, (mu, gmm_pdf(np.array([mu]))[0]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)

    # --- panel 2: q(x1) after one forward step ---
    ax = axes[0, 1]
    w1, m1, s1 = marginal_x1_params(beta1)
    q1 = (w1 * normal_pdf(x_grid[..., None], m1, s1)).sum(-1)
    ax.plot(x_grid, q1, color="black", linewidth=1.5)
    ax.plot(x_grid, p0, color="gray", linewidth=1.0, linestyle="--", label="p(x0) (reference)")
    setup_axis(ax, f"q(x1): after one forward step (beta1={beta1:g})", 1.05 * max(p0.max(), q1.max()), plot_range)
    ax.legend(fontsize=8, loc="upper right")

    # --- compute exact reverse posterior curves (used by panels 3 and 4) ---
    example_vs = [-1 / 3, -1 / 6, 1 / 6, 1 / 3]
    example_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:len(example_vs)]

    # --- panel 3: joint distribution p(x0, x1) ---
    ax = axes[1, 0]
    lim0 = 0.6  # x0's own range never changes with beta1
    lim1 = plot_range  # x1's range does
    grid0 = np.linspace(-lim0, lim0, 400)
    grid1 = np.linspace(-lim1, lim1, 400)
    x0_grid, x1_grid = np.meshgrid(grid0, grid1, indexing="ij")
    joint = joint_pdf(x0_grid, x1_grid, beta1)

    strip_w = 0.08 * 2 * lim1
    ax.set_xlim(-lim1 - strip_w, lim1)
    ax.set_ylim(-lim0, lim0)
    im = ax.imshow(joint, origin="lower", extent=[-lim1, lim1, -lim0, lim0], aspect="auto", cmap="viridis")
    hue_strip_vertical(ax, -lim1 - strip_w, -lim1)
    ax.set_aspect("equal")
    for v, color in zip(example_vs, example_colors):
        ax.axvline(v, color=color, linewidth=1.2, linestyle="--")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="density")
    ax.set_title("p(x0, x1): joint distribution")
    ax.set_xlabel("x1")
    ax.set_ylabel("x0")

    # --- panel 4: exact reverse posterior q(x0 | x1=v) for several v ---
    ax = axes[1, 1]
    ymax4 = p0.max()
    for v, color in zip(example_vs, example_colors):
        post = posterior_pdf(x_grid, v, beta1)
        ymax4 = max(ymax4, post.max())
        ax.plot(x_grid, post, color=color, linewidth=1.3, label=f"v={v:+.3f}")
        ax.axvline(v, color=color, linewidth=0.8, linestyle=":")
    ax.plot(x_grid, p0, color="gray", linewidth=1.0, linestyle="--", label="p(x0) (reference)")
    setup_axis(ax, "q(x0 | x1=v): exact reverse posterior", 1.05 * ymax4, plot_range)
    ax.legend(fontsize=7, loc="upper right", ncol=2)

    fig.suptitle(f"One DDPM step on 1D hue data (beta1={beta1:g}): forward noising and exact Bayesian reversal",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if out_path is None:
        suffix = "" if beta1 == BETA1 else f"_beta{beta1:g}"
        out_path = os.path.join(OUT_DIR, f"step0_forward_backward{suffix}.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
