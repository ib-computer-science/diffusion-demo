"""Forward process under an increasing beta schedule, pushed out to T=100 --
far enough that alpha_bar drops close to 0, i.e. close to pure noise.
Extends schedule_comparison.py's increasing-schedule idea past the small
t=1..4 range shown there.
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
T = 100
BETAS = np.linspace(0.001, 0.08, T)
STEPS_TO_SHOW = [0, 5, 10, 20, 40, 60, 80, 100]


def hue_strip(ax, y0, y1, n=600):
    xs = np.linspace(*ax.get_xlim(), n)
    colors = np.array([x_to_rgb(x) for x in xs])[None, :, :]
    ax.imshow(colors, extent=[xs[0], xs[-1], y0, y1], aspect="auto", origin="lower", zorder=0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    alpha_bars = cumulative_alpha_bar(BETAS)
    print("alpha_bar at shown steps:", {t: round(float(alpha_bars[t - 1]), 4) for t in STEPS_TO_SHOW if t > 0})

    fig, ax = plt.subplots(figsize=(9, 6))
    p0 = gmm_pdf(X_GRID)
    ymax = p0.max()
    curves = [(0, p0)]
    for t in STEPS_TO_SHOW:
        if t == 0:
            continue
        w, m, s = marginal_xt_params(alpha_bars[t - 1])
        pdf = (w * normal_pdf(X_GRID[..., None], m, s)).sum(-1)
        ymax = max(ymax, pdf.max())
        curves.append((t, pdf))

    for t, pdf in curves:
        label = "p(x0)" if t == 0 else f"q(x{t})  (alpha_bar={alpha_bars[t - 1]:.3f})"
        ax.plot(X_GRID, pdf, linewidth=1.8, label=label)

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.08 * ymax, 1.05 * ymax)
    hue_strip(ax, -0.08 * ymax, 0.0)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("x")
    ax.set_title(f"Forward process, increasing schedule (beta: {BETAS[0]:.3f} -> {BETAS[-1]:.3f}, T={T})")
    ax.legend(fontsize=9)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "long_schedule_forward.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
