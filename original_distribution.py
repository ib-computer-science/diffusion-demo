"""Just the original data distribution p(x0), on its own -- the starting
point for everything else in this demo.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hue_gmm import MEANS, NAMES, gmm_pdf, x_to_rgb

OUT_DIR = "output"
X_GRID = np.linspace(-1.0, 1.0, 1000)


def hue_strip(ax, y0, y1, n=600):
    xs = np.linspace(*ax.get_xlim(), n)
    colors = np.array([x_to_rgb(x) for x in xs])[None, :, :]
    ax.imshow(colors, extent=[xs[0], xs[-1], y0, y1], aspect="auto", origin="lower", zorder=0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    p0 = gmm_pdf(X_GRID)
    ax.plot(X_GRID, p0, color="black", linewidth=1.5)

    ymax = 1.15 * p0.max()
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.08 * ymax, ymax)
    hue_strip(ax, -0.08 * ymax, 0.0)
    ax.axhline(0.0, color="black", linewidth=0.8)

    for mu, name in zip(MEANS, NAMES):
        ax.annotate(name, (mu, gmm_pdf(np.array([mu]))[0]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)

    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_title(r"$p(x_0)$: the original data distribution")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "original_distribution.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
