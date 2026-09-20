"""Model logic for the multi-step DDPM denoiser: sinusoidal time embedding,
training, and reverse (ancestral) sampling.

Kept free of matplotlib on purpose. train_multistep_improved.py (which
saves static PNGs) forces the non-interactive Agg backend via
matplotlib.use("Agg"); if an interactive script imported that module
directly, it would inherit that forced backend and plt.show() would
silently do nothing. Putting the reusable model logic here lets both
static-figure scripts and interactive viewers (reverse_process_viewer.py)
import it without one dictating the other's matplotlib backend.
"""

import os

import numpy as np

from hue_gmm import sample_gmm
from mlp import TinyMLP
from schedule import ALPHA_BARS, ALPHAS, BETAS, T

HIDDEN = 192
N_ITERS = 60000
BATCH_SIZE = 512
LR = 2e-3
TIME_FREQS = (1, 2, 4, 8, 16, 32)
INPUT_DIM = 1 + 2 * len(TIME_FREQS)

CHECKPOINT_PATH = "checkpoints/multistep_model.npz"

REGIONS = [("red", -0.5, -0.05), ("green", 0.05, 0.5)]  # no yellow on this branch


def time_embedding(t_norm):
    """t_norm: (N,) in (0, 1] -> (N, 2*len(TIME_FREQS)) sinusoidal features."""
    t_norm = np.asarray(t_norm)[:, None]
    freqs = np.array(TIME_FREQS)[None, :]
    angles = 2.0 * np.pi * freqs * t_norm
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)


def make_inputs(x, t_norm):
    return np.concatenate([x[:, None], time_embedding(t_norm)], axis=1)


def train(rng, n_iters=None):
    n_iters = N_ITERS if n_iters is None else n_iters
    model = TinyMLP(input_dim=INPUT_DIM, hidden=HIDDEN, rng=rng, lr=LR)
    losses = []
    for _ in range(n_iters):
        x0 = sample_gmm(BATCH_SIZE, rng)
        t_idx = rng.integers(1, T + 1, size=BATCH_SIZE)
        alpha_bar_t = ALPHA_BARS[t_idx - 1]
        eps = rng.normal(size=BATCH_SIZE)
        x_t = np.sqrt(alpha_bar_t) * x0 + np.sqrt(1.0 - alpha_bar_t) * eps

        inputs = make_inputs(x_t, t_idx / T)
        eps_hat = model.forward(inputs)[:, 0]
        residual = eps_hat - eps
        losses.append(np.mean(residual**2))

        grad_out = (2.0 / BATCH_SIZE) * residual[:, None]
        model.backward(grad_out)
    return model, losses


def reverse_sample(model, n_samples, rng, x_init=None):
    """x_init lets a caller supply (and thus display/track) the exact x_T
    the reverse process starts from, instead of it being drawn internally."""
    x = rng.normal(size=n_samples) if x_init is None else np.asarray(x_init, dtype=float)
    for t in range(T, 0, -1):
        t_norm = np.full(n_samples, t / T)
        eps_hat = model.predict(make_inputs(x, t_norm))[:, 0]

        alpha_t = ALPHAS[t - 1]
        beta_t = BETAS[t - 1]
        alpha_bar_t = ALPHA_BARS[t - 1]

        mean = (x - (beta_t / np.sqrt(1.0 - alpha_bar_t)) * eps_hat) / np.sqrt(alpha_t)
        if t > 1:
            z = rng.normal(size=n_samples)
            x = mean + np.sqrt(beta_t) * z
        else:
            x = mean
    return x


def load_trained(path=CHECKPOINT_PATH):
    """Load a model saved by train_multistep_model.py, instead of training
    one from scratch -- for consumers (e.g. reverse_process_viewer.py) that
    just need a working model fast, not to demonstrate training itself."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained model found at '{path}'. Run `python train_multistep_model.py` first."
        )
    return TinyMLP.load(path)


def basin_fracs(x0):
    return {name: float(np.mean((x0 > lo) & (x0 < hi))) for name, lo, hi in REGIONS}
