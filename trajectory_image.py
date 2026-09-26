"""One image, many trajectories: each pixel column is one sample's full
reverse-diffusion path, from random noise at the bottom to a generated
sample at the top. Every pixel is colored by the hue that value maps to
(hue_gmm.x_to_rgb), so you can watch a wide band of essentially random
colors at the bottom sharpen into solid red/green columns by the top,
directly visualizing the reverse process across many samples in parallel.

Columns are sorted by their final (x0) value, so trajectories that end up
red are grouped together and separate from ones that end up green -- this
turns the bottom rows into a visible "basin of attraction" shape for each
color, instead of the two outcomes being interleaved randomly across x.

Loads the trained checkpoint (multistep_model.load_trained) rather than
training -- run train_multistep_model.py first if it doesn't exist yet.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hue_gmm import x_to_rgb
from multistep_model import T, load_trained, reverse_sample_trajectory

OUT_DIR = "output"
N_SAMPLES = 400


def trajectories_to_rgb(trajectory):
    """trajectory: (T+1, n_samples) -> (T+1, n_samples, 3) RGB image array."""
    flat = trajectory.reshape(-1)
    rgb = np.array([x_to_rgb(v) for v in flat])
    return rgb.reshape(trajectory.shape[0], trajectory.shape[1], 3)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(0)
    model = load_trained()

    trajectory = reverse_sample_trajectory(model, N_SAMPLES, rng)  # (T+1, N_SAMPLES)
    sort_order = np.argsort(trajectory[-1])
    trajectory = trajectory[:, sort_order]
    image = trajectories_to_rgb(trajectory)

    fig, ax = plt.subplots(figsize=(12, 12 * T / N_SAMPLES + 1.2))
    # Row 0 of the array (x_T, noise) is the first reverse step taken, drawn
    # at the bottom via origin="lower"; row T (x_0, generated) is the last,
    # drawn at the top. So the y-axis is the reverse-step count, increasing
    # bottom-to-top, not the diffusion timestep t (which runs the other way).
    ax.imshow(image, origin="lower", aspect="equal", interpolation="nearest",
              extent=[0, N_SAMPLES, 0, T])
    ax.set_xlabel(r"sample index (sorted by final $x_0$)")
    ax.set_ylabel(r"reverse step  (bottom=0, $x_T$ noise -> top=T, generated)")
    ax.set_title(f"{N_SAMPLES} reverse-diffusion trajectories, colored by hue, sorted by outcome")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "trajectory_image.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
