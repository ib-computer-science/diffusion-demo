"""One reverse-diffusion trajectory, drawn through the evolving density
landscape it moves through: a direct visualization of "guided random walk
toward regions of high density."

The background is p(x_t) at every t (via ddpm_step.marginal_xt_params) --
starting as one wide, undifferentiated blob near t=T and gradually
splitting into two separated lobes as t->0, exactly the schedule behavior
already measured in the conversation (mode separation only becomes
resolvable in the last ~20-30 steps). A single sampled trajectory is drawn
on top as a solid line, so you can watch where in that landscape it
happens to be sitting when the two lobes actually separate, and see it
commit to whichever one it's nearer to.

x-axis is "steps completed" (k=0 at x_T, k=T at x0) rather than t itself,
so the plot reads left-to-right in the same order the reverse sampler
actually runs.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ddpm_step import marginal_xt_params
from hue_gmm import MEANS, gmm_pdf, normal_pdf, x_to_rgb
from multistep_model import T, load_trained, reverse_sample_trajectory
from schedule import ALPHA_BARS

OUT_DIR = "output"
SEED = 3  # picked for a trajectory that visibly wanders before committing


def hue_strip_vertical(ax, x0, x1, n=600):
    ys = np.linspace(*ax.get_ylim(), n)
    colors = np.array([x_to_rgb(v) for v in ys])[:, None, :]
    ax.imshow(colors, extent=[x0, x1, ys[0], ys[-1]], aspect="auto", origin="lower", zorder=0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    model = load_trained()

    trajectory = reverse_sample_trajectory(model, 1, rng)[:, 0]  # (T+1,)

    lim = max(0.7, np.abs(trajectory).max() * 1.05)  # wide enough to show the whole path, no clipping
    x_grid = np.linspace(-lim, lim, 400)
    # alpha_bar at step k (k=0 -> t=T, k=T -> t=0): reverse of ALPHA_BARS, then append 1.0 for t=0
    alpha_bar_for_k = np.concatenate([ALPHA_BARS[::-1], [1.0]])

    background = np.empty((T + 1, len(x_grid)))
    for k, ab in enumerate(alpha_bar_for_k):
        w, m, s = marginal_xt_params(ab)
        background[k] = (w * normal_pdf(x_grid[:, None], m, s)).sum(-1)
    # Normalize each time-slice (each k) to its own max: absolute density
    # magnitude varies hugely across the schedule (peak ~5 at t=0 vs ~0.4
    # near t=T), which would otherwise wash out the shape at high t under
    # one shared linear color scale. We only care about *where* the mass
    # concentrates at each t, not comparing magnitudes across different t.
    background = background / background.max(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    strip_w = 0.04 * T
    ax.set_xlim(-strip_w, T)
    ax.set_ylim(-lim, lim)
    ax.imshow(background.T, origin="lower", extent=[0, T, -lim, lim], aspect="auto", cmap="viridis")
    hue_strip_vertical(ax, -strip_w, 0)
    for mean in MEANS:
        ax.axhline(mean, color="white", linewidth=0.6, linestyle=":", alpha=0.6)

    k_grid = np.arange(T + 1)
    ax.plot(k_grid, trajectory, color="white", linewidth=1.6, zorder=3)
    ax.scatter([0], [trajectory[0]], color="white", edgecolor="black", s=50, zorder=4, label=r"$x_T$ (start)")
    ax.scatter([T], [trajectory[-1]], color=x_to_rgb(trajectory[-1]), edgecolor="black", s=90, zorder=4,
               label=r"$x_0$ (generated)")

    ax.set_xlabel(r"reverse step $k$  ($k$=0: $x_T$ noise  ->  $k$=T: $x_0$ generated)")
    ax.set_ylabel("x")
    ax.set_title("One reverse-diffusion trajectory through the evolving density landscape")
    ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "single_trajectory_landscape.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
