"""Companion to step0_forward_backward.py: same top two panels (p(x0),
q(x1)), but the bottom panels now describe x1 directly in terms of x0
instead of inverting via Bayes' rule.

Panel 3 plots the joint distribution p(x0, x1), sliced VERTICALLY at fixed
x0=u (dashed lines). Panel 4 plots that same vertical slice directly, as a
curve over x1 -- q(x0=u, x1) = p(x0=u) * q(x1 | x0=u), the raw joint
density along the slice, left unnormalized rather than divided down into a
proper conditional density in x1. Dividing panel 4's curves by p(x0=u)
would recover the forward conditional q(x1 | x0=u) (still a single
Gaussian for every choice of u, unlike the reverse posterior's genuine
bimodality at ambiguous points -- mirroring how a horizontal slice in
step0_forward_backward.py's panel 3 becomes its reverse posterior panel 4
once renormalized).

main() takes beta1 as a parameter (default matches step0_forward_backward.py)
so the same figure can be regenerated at other noise levels.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ddpm_step import joint_pdf, marginal_x1_params
from hue_gmm import MEANS, NAMES, STDS, gmm_pdf, normal_pdf, x_to_rgb

BETA1 = 0.005
OUT_DIR = "output"


def hue_strip(ax, y0, y1, n=600):
    xs = np.linspace(*ax.get_xlim(), n)
    colors = np.array([x_to_rgb(x) for x in xs])[None, :, :]
    ax.imshow(colors, extent=[xs[0], xs[-1], y0, y1], aspect="auto", origin="lower", zorder=0)


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

    # Plot range for the 1D curve panels (p(x0), q(x1), forward conditional):
    # wide enough that q(x1) isn't clipped even when beta1 is large.
    x1_std_est = np.sqrt(a1 * STDS.max() ** 2 + beta1)
    plot_range = max(1.0, np.abs(MEANS).max() * np.sqrt(a1) + 4.5 * x1_std_est)
    x_grid = np.linspace(-plot_range, plot_range, 1000)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # --- panel 1: p(x0) ---
    ax = axes[0, 0]
    p0 = gmm_pdf(x_grid)
    ax.plot(x_grid, p0, color="black", linewidth=1.5)
    setup_axis(ax, "p(x0): true hue distribution", 1.15 * p0.max(), plot_range)
    for mu, name in zip(MEANS, NAMES):
        ax.annotate(name, (mu, gmm_pdf(np.array([mu]))[0]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)

    # --- panel 2: q(x1) after one forward step ---
    ax = axes[0, 1]
    w1, m1, s1 = marginal_x1_params(beta1)
    q1 = (w1 * normal_pdf(x_grid[..., None], m1, s1)).sum(-1)
    ax.plot(x_grid, q1, color="black", linewidth=1.5)
    ax.plot(x_grid, p0, color="gray", linewidth=1.0, linestyle="--", label="p(x0) (reference)")
    setup_axis(ax, f"q(x1): after one forward step (beta1={beta1:g})", 1.15 * max(p0.max(), q1.max()), plot_range)
    ax.legend(fontsize=8, loc="upper right")

    # --- example x0 values used by panels 3 and 4 ---
    example_us = [-1 / 3, 0.25]
    example_colors = [x_to_rgb(u) for u in example_us]

    # --- panel 3: joint distribution p(x0, x1), sliced vertically (by x0) ---
    ax = axes[1, 1]
    lim0 = 0.6  # x0's own range never changes with beta1
    lim1 = plot_range  # x1's range does
    grid0 = np.linspace(-lim0, lim0, 400)
    grid1 = np.linspace(-lim1, lim1, 400)
    x0_grid, x1_grid = np.meshgrid(grid0, grid1, indexing="ij")
    joint = joint_pdf(x0_grid, x1_grid, beta1)

    strip_h = 0.08 * 2 * lim1
    ax.set_xlim(-lim0, lim0)
    ax.set_ylim(-lim1 - strip_h, lim1)
    im = ax.imshow(joint.T, origin="lower", extent=[-lim0, lim0, -lim1, lim1], aspect="auto", cmap="viridis")
    hue_strip(ax, -lim1 - strip_h, -lim1)
    ax.set_aspect("equal")
    for u, color in zip(example_us, example_colors):
        ax.axvline(u, color=color, linewidth=1.2, linestyle="--")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="density")
    ax.set_anchor("C")  # fig.colorbar() re-anchors ax flush against the colorbar; recenter it
    ax.set_title("p(x0, x1): joint distribution (sliced by x0)")
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")

    # --- panel 4: joint density along the vertical slice x0=u, q(x0=u, x1) ---
    ax = axes[1, 0]
    ymax4 = 0.0
    for u, color in zip(example_us, example_colors):
        slice_ = joint_pdf(u, x_grid, beta1)
        ymax4 = max(ymax4, slice_.max())
        ax.plot(x_grid, slice_, color=color, linewidth=1.3, label=f"x0={u:+.3f}")
        ax.axvline(u, color=color, linewidth=0.8, linestyle=":")
    setup_axis(ax, "q(x0=u, x1): joint density along the vertical slice (unnormalized)", 1.05 * ymax4, plot_range)
    ax.legend(fontsize=7, loc="upper right", ncol=2)

    fig.suptitle(f"One DDPM step on 1D hue data (beta1={beta1:g}): x1 expressed directly in terms of x0",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if out_path is None:
        suffix = "" if beta1 == BETA1 else f"_beta{beta1:g}"
        out_path = os.path.join(OUT_DIR, f"step0_x1_given_x0{suffix}.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
