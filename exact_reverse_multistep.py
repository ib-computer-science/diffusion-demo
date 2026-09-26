"""Isolate whether the fixed reverse-step variance (sigma_t^2 = beta_t)
causes the yellow-bump leakage seen in train_multistep.py's *learned*
reverse sampler -- independent of any network training error.

Uses the closed-form exact conditional mean E[x0 | x_t=v]
(ddpm_step.exact_conditional_mean_x0) as a zero-training-error "oracle" in
place of the learned model's prediction, but keeps everything else about
the ancestral sampling loop identical to train_multistep.py: same
schedule, same fixed variance sigma_t^2 = beta_t, same number of steps.

If yellow is still underrepresented here, the fixed-variance/point-estimate
sampling scheme itself is the cause. If yellow comes out correct, the
leakage in train_multistep.py traces back to the learned model (undertraining
and/or training-data imbalance), not the sampling scheme.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ddpm_step import exact_conditional_mean_x0
from hue_gmm import gmm_pdf, sample_gmm, x_to_rgb
from schedule import ALPHA_BARS, ALPHAS, BETAS, T

OUT_DIR = "output"
X_GRID = np.linspace(-1.0, 1.0, 1000)
REGIONS = [("red", -0.5, -0.05), ("green", 0.05, 0.5)]  # no yellow on this branch


def exact_eps_hat(x, alpha_bar_t):
    x0_mean = exact_conditional_mean_x0(x, alpha_bar_t)
    return (x - np.sqrt(alpha_bar_t) * x0_mean) / np.sqrt(1.0 - alpha_bar_t)


def reverse_sample_oracle(n_samples, rng):
    x = rng.normal(size=n_samples)  # x_T ~ N(0, 1)
    for t in range(T, 0, -1):
        alpha_bar_t = ALPHA_BARS[t - 1]
        eps_hat = exact_eps_hat(x, alpha_bar_t)

        alpha_t = ALPHAS[t - 1]
        beta_t = BETAS[t - 1]
        mean = (x - (beta_t / np.sqrt(1.0 - alpha_bar_t)) * eps_hat) / np.sqrt(alpha_t)
        if t > 1:
            z = rng.normal(size=n_samples)
            x = mean + np.sqrt(beta_t) * z
        else:
            x = mean
    return x


def print_basin_table(x0_true, x0_gen):
    print(f"{'region':8s} {'true':>8s} {'oracle-gen':>12s}")
    for name, lo, hi in REGIONS:
        t_frac = np.mean((x0_true > lo) & (x0_true < hi))
        g_frac = np.mean((x0_gen > lo) & (x0_gen < hi))
        print(f"{name:8s} {t_frac:8.4f} {g_frac:12.4f}")


def hue_strip(ax, y0, y1, n=600):
    xs = np.linspace(*ax.get_xlim(), n)
    colors = np.array([x_to_rgb(v) for v in xs])[None, :, :]
    ax.imshow(colors, extent=[xs[0], xs[-1], y0, y1], aspect="auto", origin="lower", zorder=0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(0)
    n_samples = 20000

    x0_gen = reverse_sample_oracle(n_samples, rng)
    x0_true = sample_gmm(n_samples, rng)
    print_basin_table(x0_true, x0_gen)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bins = np.linspace(-1.0, 1.0, 150)
    ax.hist(x0_true, bins=bins, density=True, alpha=0.5, label=r"original $p(x_0)$", color="tab:blue")
    ax.hist(x0_gen, bins=bins, density=True, alpha=0.5,
            label=r"reverse-sampled (exact mean, fixed variance=$\beta_t$)", color="tab:orange")
    ymax = max(gmm_pdf(X_GRID).max(), 1.0) * 1.1
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.08 * ymax, ymax)
    hue_strip(ax, -0.08 * ymax, 0.0)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel(r"$x_0$")
    ax.set_title(rf"Oracle-mean ancestral sampling ({T} steps, fixed variance = $\beta_t$)")
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, "exact_reverse_multistep.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
