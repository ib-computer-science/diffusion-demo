"""Constant beta vs. an increasing beta schedule, over the same 4 steps.

Real DDPM schedules increase beta_t across steps (small near t=0, larger
near T) rather than reusing the same beta every time. This compares the
constant-beta chain from multi_step_forward.py against a linearly
increasing schedule, to see how the shape of the schedule -- not just the
number of steps -- changes how quickly structure washes out.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ddpm_step import cumulative_alpha_bar, marginal_xt_params
from hue_gmm import gmm_pdf, normal_pdf, x_to_rgb

OUT_DIR = "output"
X_GRID = np.linspace(-1.0, 1.0, 1000)
T = 4

CONSTANT_BETAS = np.full(T, 0.005)
INCREASING_BETAS = np.linspace(0.001, 0.02, T)


def hue_strip(ax, y0, y1, n=600):
    xs = np.linspace(*ax.get_xlim(), n)
    colors = np.array([x_to_rgb(x) for x in xs])[None, :, :]
    ax.imshow(colors, extent=[xs[0], xs[-1], y0, y1], aspect="auto", origin="lower", zorder=0)


def q_xt_curve(alpha_bar_t):
    w, m, s = marginal_xt_params(alpha_bar_t)
    return (w * normal_pdf(X_GRID[..., None], m, s)).sum(-1)


def plot_schedule(ax, betas, title):
    alpha_bars = cumulative_alpha_bar(betas)
    p0 = gmm_pdf(X_GRID)
    ymax = p0.max()
    curves = [(0, p0)]
    for t in range(1, len(betas) + 1):
        pdf = q_xt_curve(alpha_bars[t - 1])
        ymax = max(ymax, pdf.max())
        curves.append((t, pdf))

    for t, pdf in curves:
        label = r"$p(x_0)$" if t == 0 else rf"$q(x_{t})$"
        ax.plot(X_GRID, pdf, linewidth=1.8, label=label)

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.08 * ymax, 1.05 * ymax)
    hue_strip(ax, -0.08 * ymax, 0.0)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("x")
    ax.set_title(title)
    ax.legend(fontsize=8)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    plot_schedule(axes[0], CONSTANT_BETAS, rf"Constant $\beta$ = {CONSTANT_BETAS[0]} at every step")
    plot_schedule(axes[1], INCREASING_BETAS,
                  rf"Increasing $\beta$: {INCREASING_BETAS[0]:.3f} -> {INCREASING_BETAS[-1]:.3f}")

    fig.suptitle(r"Forward process over 4 steps: constant vs. increasing $\beta$ schedule", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(OUT_DIR, "schedule_comparison.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
