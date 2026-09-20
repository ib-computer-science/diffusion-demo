"""Same exact-vs-learned curve comparison as learned_curve_on_joint.py, but
with a bigger single-step model: hidden=192 (matching the final multi-step
model's capacity) and 5x the training iterations.

The multi-step model's "final" architecture also uses a sinusoidal time
embedding, but that doesn't transfer here -- this single-step model has no
`t` input at all (beta1 is fixed, there's only one step), so the only
applicable changes are more capacity and more training.

Two gaps are layered in learned_curve_on_joint.png: (1) exact E[x0|x1] vs.
the true multimodal shape -- unfixable, a mean is a mean regardless of
model size; (2) the learned curve vs. that exact mean -- a genuine
approximation error that more capacity/training can shrink. This script
tests how much of (2) actually closes.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ddpm_step import joint_pdf, posterior_mean_curve
from hue_gmm import sample_gmm, x_to_rgb
from mlp import TinyMLP

BETA1 = 0.005
A1 = 1.0 - BETA1
OUT_DIR = "output"

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

        grad_out = (2.0 / BATCH_SIZE) * residual[:, None]
        model.backward(grad_out)
    return model, losses


def hue_strip(ax, y0, y1, n=600):
    xs = np.linspace(*ax.get_xlim(), n)
    colors = np.array([x_to_rgb(x) for x in xs])[None, :, :]
    ax.imshow(colors, extent=[xs[0], xs[-1], y0, y1], aspect="auto", origin="lower", zorder=0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(0)
    model, losses = train(rng)
    final_loss = np.mean(losses[-100:])
    print(f"final training loss (mean of last 100 iters): {final_loss:.4f} "
          f"(original hidden=64/4000-iter model: 0.6488; Bayes-optimal floor: ~0.582)")

    lim = 0.6
    plot_grid = np.linspace(-lim, lim, 400)
    x0_grid, x1_grid = np.meshgrid(plot_grid, plot_grid, indexing="ij")
    joint = joint_pdf(x0_grid, x1_grid, BETA1)

    v_grid = np.linspace(-lim, lim, 300)
    exact_curve = posterior_mean_curve(v_grid, BETA1)

    eps_hat = model.predict(v_grid[:, None])[:, 0]
    learned_curve = (v_grid - np.sqrt(BETA1) * eps_hat) / np.sqrt(A1)

    fig, ax = plt.subplots(figsize=(7.5, 7))
    strip_h = 0.08 * 2 * lim
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim - strip_h, lim)
    im = ax.imshow(joint.T, origin="lower", extent=[-lim, lim, -lim, lim], aspect="auto", cmap="viridis")
    hue_strip(ax, -lim - strip_h, -lim)

    ax.plot(exact_curve, v_grid, color="white", linewidth=2.2, label="exact E[x0 | x1=v]")
    ax.plot(learned_curve, v_grid, color="tab:orange", linewidth=2.0, linestyle="--",
            label=f"learned (hidden={HIDDEN}, {N_ITERS} iters)")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="density")
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_title("Bigger single-step model: exact vs. learned curve")
    ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "learned_curve_on_joint_bigger.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
