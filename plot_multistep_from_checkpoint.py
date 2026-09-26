"""Plot the multi-step original-vs-reverse-sampled comparison using the
already-trained checkpoint (multistep_model.load_trained), instead of
retraining from scratch the way train_multistep_improved.py does.

Reuses the training run already paid for by train_multistep_model.py --
useful when you just want the comparison figure for the current checkpoint
(e.g. after switching data distributions on a branch) without paying for
another ~11-20 minute training run.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hue_gmm import gmm_pdf, sample_gmm, x_to_rgb
from multistep_model import REGIONS, T, basin_fracs, load_trained, reverse_sample

OUT_DIR = "output"
X_GRID = np.linspace(-1.0, 1.0, 1000)


def hue_strip(ax, y0, y1, n=600):
    xs = np.linspace(*ax.get_xlim(), n)
    colors = np.array([x_to_rgb(v) for v in xs])[None, :, :]
    ax.imshow(colors, extent=[xs[0], xs[-1], y0, y1], aspect="auto", origin="lower", zorder=0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(0)
    model = load_trained()

    n_samples = 20000
    x0_generated = reverse_sample(model, n_samples, rng)
    x0_true = sample_gmm(n_samples, rng)

    gen_frac = basin_fracs(x0_generated)
    true_frac = basin_fracs(x0_true)
    print(f"{'region':8s} {'true':>8s} {'gen':>8s}")
    for name, _, _ in REGIONS:
        print(f"{name:8s} {true_frac[name]:8.4f} {gen_frac[name]:8.4f}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bins = np.linspace(-1.0, 1.0, 150)
    ax.hist(x0_true, bins=bins, density=True, alpha=0.5, label=r"original $p(x_0)$", color="tab:blue")
    ax.hist(x0_generated, bins=bins, density=True, alpha=0.5,
            label="reverse-sampled (from checkpoint)", color="tab:orange")
    ymax = max(gmm_pdf(X_GRID).max(), 1.0) * 1.1
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.08 * ymax, ymax)
    hue_strip(ax, -0.08 * ymax, 0.0)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel(r"$x_0$")
    ax.set_title(f"Original vs. reverse-sampled ({T}-step, from saved checkpoint)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "multistep_from_checkpoint.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
