"""Train a noise-predictor across the full T-step schedule (schedule.py),
then run the reverse (ancestral sampling) process end-to-end: start from
pure noise and denoise back down to x0, comparing the resulting
distribution against the true data distribution p(x0).

Unlike train_denoiser.py (a single fixed beta1), this network also takes
the timestep t as input, since beta_t now varies across the schedule.
Training follows the standard DDPM objective (Ho et al. 2020, Algorithm 1):
sample x0, a random t, and noise eps; jump directly to x_t via the
closed-form alpha_bar_t (no need to simulate all t intermediate steps);
train the network to predict eps from (x_t, t).

Sampling follows the standard ancestral procedure (Algorithm 2): start at
x_T ~ N(0, 1) and repeatedly apply the learned reverse step down to x_0,
injecting fresh Gaussian noise at every step except the last.
"""

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hue_gmm import gmm_pdf, sample_gmm, x_to_rgb
from mlp import TinyMLP
from schedule import ALPHA_BARS, ALPHAS, BETAS, T

OUT_DIR = "output"
X_GRID = np.linspace(-1.0, 1.0, 1000)

HIDDEN = 128
N_ITERS = 20000
BATCH_SIZE = 512
LR = 2e-3


def train(rng):
    model = TinyMLP(input_dim=2, hidden=HIDDEN, rng=rng, lr=LR)
    losses = []
    for _ in range(N_ITERS):
        x0 = sample_gmm(BATCH_SIZE, rng)
        t_idx = rng.integers(1, T + 1, size=BATCH_SIZE)  # 1..T inclusive
        alpha_bar_t = ALPHA_BARS[t_idx - 1]
        eps = rng.normal(size=BATCH_SIZE)
        x_t = np.sqrt(alpha_bar_t) * x0 + np.sqrt(1.0 - alpha_bar_t) * eps

        inputs = np.stack([x_t, t_idx / T], axis=1)
        eps_hat = model.forward(inputs)[:, 0]
        residual = eps_hat - eps
        losses.append(np.mean(residual**2))

        grad_out = (2.0 / BATCH_SIZE) * residual[:, None]
        model.backward(grad_out)
    return model, losses


def reverse_sample(model, n_samples, rng):
    x = rng.normal(size=n_samples)  # x_T ~ N(0, 1)
    for t in range(T, 0, -1):
        t_norm = np.full(n_samples, t / T)
        eps_hat = model.predict(np.stack([x, t_norm], axis=1))[:, 0]

        alpha_t = ALPHAS[t - 1]
        beta_t = BETAS[t - 1]
        alpha_bar_t = ALPHA_BARS[t - 1]

        mean = (x - (beta_t / np.sqrt(1.0 - alpha_bar_t)) * eps_hat) / np.sqrt(alpha_t)
        if t > 1:
            z = rng.normal(size=n_samples)
            x = mean + np.sqrt(beta_t) * z
        else:
            x = mean  # no noise injected on the final step (standard practice)
    return x


def hue_strip(ax, y0, y1, n=600):
    xs = np.linspace(*ax.get_xlim(), n)
    colors = np.array([x_to_rgb(v) for v in xs])[None, :, :]
    ax.imshow(colors, extent=[xs[0], xs[-1], y0, y1], aspect="auto", origin="lower", zorder=0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(0)

    t0 = time.time()
    model, losses = train(rng)
    print(f"training took {time.time() - t0:.1f}s, "
          f"final loss (mean of last 200 iters): {np.mean(losses[-200:]):.4f}")

    n_samples = 20000
    t0 = time.time()
    x0_generated = reverse_sample(model, n_samples, rng)
    print(f"reverse sampling took {time.time() - t0:.1f}s")

    x0_true = sample_gmm(n_samples, rng)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    ax.plot(losses, linewidth=0.5, color="tab:green")
    ax.set_yscale("log")
    ax.set_xlabel("training iteration")
    ax.set_ylabel(r"MSE($\hat\epsilon$, $\epsilon$)")
    ax.set_title("Multi-step training loss (random t each iteration)")

    ax = axes[1]
    bins = np.linspace(-1.0, 1.0, 150)
    ax.hist(x0_true, bins=bins, density=True, alpha=0.5, label=r"original $p(x_0)$", color="tab:blue")
    ax.hist(x0_generated, bins=bins, density=True, alpha=0.5,
            label="reverse-sampled (learned model)", color="tab:orange")
    ymax = max(gmm_pdf(X_GRID).max(), 1.0) * 1.1
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.08 * ymax, ymax)
    hue_strip(ax, -0.08 * ymax, 0.0)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel(r"$x_0$")
    ax.set_title(f"Original vs. reverse-sampled ({T}-step ancestral sampling)")
    ax.legend(fontsize=8)

    fig.suptitle(f"Full {T}-step DDPM: trained denoiser, sampled end-to-end", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(OUT_DIR, "multistep_train_and_sample.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
