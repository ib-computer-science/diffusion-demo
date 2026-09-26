"""Draws a batch of samples from the true hue distribution p(x0) and lays
them out as a grid of colored cells in a single saved image.

A direct, hands-on way to *feel* the distribution rather than just look at
its density curve: red shows up about twice as often as green (weights
2/3, 1/3) -- the same GMM used throughout the rest of this demo
(hue_gmm.py).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from hue_gmm import sample_gmm, x_to_rgb

OUT_DIR = "output"
GRID_WIDTH = 5
GRID_HEIGHT = 10


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng()
    n = GRID_WIDTH * GRID_HEIGHT
    xs = sample_gmm(n, rng)

    fig, ax = plt.subplots(figsize=(GRID_WIDTH, GRID_HEIGHT))
    ax.set_xlim(0, GRID_WIDTH)
    ax.set_ylim(0, GRID_HEIGHT)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i, x in enumerate(xs):
        col, row = i % GRID_WIDTH, i // GRID_WIDTH
        ax.add_patch(Rectangle((col, row), 1, 1, facecolor=x_to_rgb(float(x)),
                                edgecolor="black", linewidth=0.5))

    fig.suptitle(rf"{n} samples from $p(x_0)$", fontsize=10)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "sample_grid.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
