"""Train a tiny noise-predictor for the single DDPM step (fixed beta1), and
compare what it implies about the reverse posterior against the exact,
closed-form posterior from ddpm_step.py.

This is where "learning" enters the demo. The network only ever sees
sampled (x1, eps) pairs -- it never gets the GMM formula for p(x0) -- and
has to recover, purely from data, an approximation to the reverse step that
ddpm_step.py computes analytically. It's trained with the standard DDPM
objective: predict the noise eps that was added, given the noisy value x1.

A network trained with squared-error loss converges to E[eps | x1=v], the
*mean* of the true (possibly multimodal) posterior. Converting that back to
an x0 estimate and wrapping it in a fixed-variance Gaussian (the standard
DDPM reverse-step assumption) gives a single unimodal bump. For v deep
inside one hue bump this matches the true posterior well; for an ambiguous
v between two bumps, the learned Gaussian collapses onto the average of the
two modes -- missing the true bimodal shape entirely. That gap is the
visible cost of using a single large step, and part of why real DDPM uses
many small ones (see beta1_sweep.py) instead of asking one Gaussian step to
resolve genuine multimodal ambiguity.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ddpm_step import posterior_pdf
from hue_gmm import gmm_pdf, normal_pdf, sample_gmm, x_to_rgb
from mlp import TinyMLP

BETA1 = 0.005
A1 = 1.0 - BETA1
OUT_DIR = "output"
X_GRID = np.linspace(-1.0, 1.0, 1000)
EXAMPLE_VS = [-1 / 3, -1 / 6, 0.0, 1 / 6, 1 / 3]

HIDDEN = 192
N_ITERS = 20000
BATCH_SIZE = 512


def train(rng):
    model = TinyMLP(hidden=HIDDEN, rng=rng, lr=2e-3)
    losses = []
    for _ in range(N_ITERS):
        x0 = sample_gmm(BATCH_SIZE, rng)
        eps = rng.normal(size=BATCH_SIZE)
        x1 = np.sqrt(A1) * x0 + np.sqrt(BETA1) * eps

        eps_hat = model.forward(x1[:, None])[:, 0]
        residual = eps_hat - eps
        losses.append(np.mean(residual**2))

        grad_out = (2.0 / BATCH_SIZE) * residual[:, None]  # d(mean squared error)/d(output)
        model.backward(grad_out)
    return model, losses


def learned_posterior_gaussian(model, v):
    eps_hat = model.predict(np.array([[v]]))[0, 0]
    x0_hat = (v - np.sqrt(BETA1) * eps_hat) / np.sqrt(A1)
    return x0_hat, np.sqrt(BETA1)  # fixed variance = beta1, the standard DDPM choice


def hue_strip(ax, y0, y1, n=400):
    xs = np.linspace(*ax.get_xlim(), n)
    colors = np.array([x_to_rgb(x) for x in xs])[None, :, :]
    ax.imshow(colors, extent=[xs[0], xs[-1], y0, y1], aspect="auto", origin="lower", zorder=0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(0)
    model, losses = train(rng)
    print(f"final training loss (mean of last 100 iters): {np.mean(losses[-100:]):.4f}")

    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(2, len(EXAMPLE_VS), height_ratios=[1, 1.4])

    ax_loss = fig.add_subplot(gs[0, :])
    ax_loss.plot(losses, linewidth=0.8, color="tab:green")
    ax_loss.axhline(1.0, color="gray", linestyle="--", linewidth=1,
                     label="loss=1.0 (blindly predicting eps=0 everywhere)")
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("training iteration")
    ax_loss.set_ylabel("MSE(eps_hat, eps)")
    ax_loss.set_title("Learning curve: noise-prediction loss (fresh samples each iteration)")
    ax_loss.legend(fontsize=8)

    p0 = gmm_pdf(X_GRID)
    for i, v in enumerate(EXAMPLE_VS):
        ax = fig.add_subplot(gs[1, i])
        true_post = posterior_pdf(X_GRID, v, BETA1)
        x0_hat, sigma = learned_posterior_gaussian(model, v)
        learned_post = normal_pdf(X_GRID, x0_hat, sigma)
        ymax = max(true_post.max(), learned_post.max(), p0.max()) * 1.15

        ax.plot(X_GRID, p0, color="gray", linestyle=":", linewidth=1, label="p(x0)")
        ax.plot(X_GRID, true_post, color="tab:blue", linewidth=1.8, label="exact q(x0|x1=v)")
        ax.plot(X_GRID, learned_post, color="tab:orange", linewidth=1.8, linestyle="--",
                label="learned q_theta(x0|x1=v)")
        ax.axvline(v, color="black", linewidth=0.7, linestyle=":")
        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(-0.08 * ymax, ymax)
        hue_strip(ax, -0.08 * ymax, 0.0)
        ax.set_title(f"v={v:+.3f}", fontsize=10)
        if i == 0:
            ax.legend(fontsize=6, loc="upper left")

    fig.suptitle("Learned single-step reverse model vs. exact posterior", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(OUT_DIR, "learned_vs_exact_posterior.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
