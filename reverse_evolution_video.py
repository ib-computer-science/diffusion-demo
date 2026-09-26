"""Animate how the reverse process's sample distribution evolves from pure
noise back to the true data distribution, frame by frame across all T
reverse steps -- the "multistep_from_checkpoint.png" comparison, but shown
evolving over t instead of only at the final result.

Loads the trained checkpoint (multistep_model.load_trained) and the full
per-step trajectories (multistep_model.reverse_sample_trajectory) for a
large batch of samples, then renders one histogram frame per timestep
(from x_T down to x_0) and saves it as an .mp4.

Note: this is the one script in the demo that needs more than NumPy +
Matplotlib -- saving the animation requires ffmpeg (a system binary, not a
pip package) to be installed and on PATH.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from hue_gmm import gmm_pdf, sample_gmm, x_to_rgb
from multistep_model import T, load_trained, reverse_sample_trajectory

OUT_DIR = "output"
N_SAMPLES = 20000
X_GRID = np.linspace(-1.0, 1.0, 1000)
# x_T ~ N(0, 1), so +-1.0 (the data's own range) clips almost all of the
# starting noise distribution; +-2.5 std covers most of it (and everything
# in between) while keeping a fixed axis across the whole animation.
X_RANGE = 2.5
BINS = np.linspace(-X_RANGE, X_RANGE, 375)
FPS = 12


def hue_strip(ax, y0, y1, n=600):
    xs = np.linspace(*ax.get_xlim(), n)
    colors = np.array([x_to_rgb(v) for v in xs])[None, :, :]
    ax.imshow(colors, extent=[xs[0], xs[-1], y0, y1], aspect="auto", origin="lower", zorder=0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(0)
    model = load_trained()

    print(f"generating {N_SAMPLES} trajectories over {T} steps...")
    trajectory = reverse_sample_trajectory(model, N_SAMPLES, rng)  # (T+1, N_SAMPLES)
    x0_true = sample_gmm(N_SAMPLES, rng)

    ymax = max(gmm_pdf(X_GRID).max(), 1.0) * 1.1
    fig, ax = plt.subplots(figsize=(8, 5.5))

    def draw_frame(k):
        ax.clear()
        t = T - k  # trajectory[k] corresponds to diffusion time T-k
        ax.hist(x0_true, bins=BINS, density=True, alpha=0.4, color="tab:blue", label=r"true $p(x_0)$")
        ax.hist(trajectory[k], bins=BINS, density=True, alpha=0.6, color="tab:orange",
                label=rf"reverse samples ($t$={t})")
        ax.set_xlim(-X_RANGE, X_RANGE)
        ax.set_ylim(-0.08 * ymax, ymax)
        hue_strip(ax, -0.08 * ymax, 0.0)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("x")
        ax.set_title(rf"Reverse process evolution: $t$={t} / {T}")
        ax.legend(fontsize=8, loc="upper right")

    print(f"rendering {T + 1} frames...")
    anim = animation.FuncAnimation(fig, draw_frame, frames=T + 1, interval=1000 / FPS)

    out_path = os.path.join(OUT_DIR, "reverse_evolution.mp4")
    anim.save(out_path, writer=animation.FFMpegWriter(fps=FPS), dpi=110)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
