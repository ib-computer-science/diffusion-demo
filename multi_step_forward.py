"""Forward process after chaining multiple steps of the same beta1.

Shows q(x_t) for t = 0 (the data), 1, 2, 3, 4, using the closed-form
direct-jump formula for a linear-Gaussian chain (ddpm_step.marginal_xt_params).

All steps here reuse the same beta1 as the rest of the demo (not a real,
increasing schedule) -- this isolates the effect of "how many steps" from
"how the schedule is shaped". A real DDPM schedule increases beta_t across
steps, so it would wash out structure faster than this constant-beta chain
does.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ddpm_step import marginal_xt_params
from hue_gmm import gmm_pdf, normal_pdf, x_to_rgb

BETA1 = 0.005
OUT_DIR = "output"
X_GRID = np.linspace(-1.0, 1.0, 1000)
STEPS = [0, 1, 2, 3, 4]


def hue_strip(ax, y0, y1, n=600):
    xs = np.linspace(*ax.get_xlim(), n)
    colors = np.array([x_to_rgb(x) for x in xs])[None, :, :]
    ax.imshow(colors, extent=[xs[0], xs[-1], y0, y1], aspect="auto", origin="lower", zorder=0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))

    curves = []
    ymax = 0.0
    for t in STEPS:
        if t == 0:
            pdf = gmm_pdf(X_GRID)
        else:
            w, m, s = marginal_xt_params((1.0 - BETA1) ** t)
            pdf = (w * normal_pdf(X_GRID[..., None], m, s)).sum(-1)
        ymax = max(ymax, pdf.max())
        curves.append((t, pdf))

    for t, pdf in curves:
        label = r"$p(x_0)$: data" if t == 0 else rf"$q(x_{t})$: after {t} step{'s' if t > 1 else ''}"
        ax.plot(X_GRID, pdf, linewidth=1.8, label=label)

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.08 * ymax, 1.05 * ymax)
    hue_strip(ax, -0.08 * ymax, 0.0)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("x")
    ax.set_title(rf"Forward process: $q(x_t)$ for $t$=0..4 ($\beta_1$={BETA1} per step)")
    ax.legend(fontsize=9)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "multi_step_forward.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
