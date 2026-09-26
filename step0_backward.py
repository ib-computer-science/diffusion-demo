"""Demo: the exact reverse step on a 1D hue distribution, and what a trained
model actually learns of it.

Produces output/step0_backward.png with four panels:
  1. p(x0, x1)         -- the joint distribution, sliced at x1=v (dashed
                          line, color matching panel 2); once renormalized,
                          that slice is exactly the posterior curve shown
                          in panel 2 for that v.
  2. q(x0 | x1=v)      -- exact reverse posterior, for a single example v
  3. p(x0, x1)         -- same joint as panel 1, but overlaid with the
                          exact E[x0|x1=v] curve and the curve implied by a
                          freshly trained single-step MLP (learned_curve_on_joint.py)
  4. p(x_t, x_{t+1})   -- joint for a pair of ADJACENT intermediate steps
                          (t=10 -> 11) deep in the full T=100 schedule,
                          overlaid with the exact and learned E[x_t|x_{t+1}]
                          curves (learned_curve_on_joint_adjacent.py)

Panels 3 and 4 are independent of the beta1 parameter below -- they always
retrain/reload their own models at their own fixed noise level -- only
panels 1 and 2 change shape with beta1.

main() takes beta1 as a parameter (default matches the original demo
value) so panels 1-2 can be regenerated at other noise levels; the plotted
x1-range widens automatically with beta1, since a large beta1 spreads
q(x1) far beyond the original data's range. (step0_forward_backward_large_noise.py
no longer reuses this main() -- it kept its own copy of this script's
pre-learned-curve 4-panel layout, since panels 3-4 here are unrelated to
its large-beta1 point.)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import train_denoiser
from ddpm_step import (
    exact_conditional_mean_adjacent,
    joint_pdf,
    joint_pdf_adjacent,
    posterior_mean_curve,
    posterior_pdf,
)
from hue_gmm import MEANS, STDS, gmm_pdf, x_to_rgb
from multistep_model import load_trained, make_inputs
from schedule import ALPHA_BARS, ALPHAS, BETAS, T

BETA1 = 0.005
OUT_DIR = "output"
T_STEP = 10  # examine the adjacent transition x_{T_STEP} -> x_{T_STEP+1}


def hue_strip(ax, y0, y1, n=600):
    xs = np.linspace(*ax.get_xlim(), n)
    colors = np.array([x_to_rgb(x) for x in xs])[None, :, :]
    ax.imshow(colors, extent=[xs[0], xs[-1], y0, y1], aspect="auto", origin="lower", zorder=0)


def hue_strip_vertical(ax, x0, x1, n=600):
    ys = np.linspace(*ax.get_ylim(), n)
    colors = np.array([x_to_rgb(v) for v in ys])[:, None, :]
    ax.imshow(colors, extent=[x0, x1, ys[0], ys[-1]], aspect="auto", origin="lower", zorder=0)


def setup_axis(ax, title, ymax, xlim):
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-0.08 * ymax, ymax)
    hue_strip(ax, -0.08 * ymax, 0.0)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("x")


def main(beta1=BETA1, out_path=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    a1 = 1.0 - beta1

    # Plot range for the reverse-posterior curve panel: wide enough that
    # q(x1) isn't clipped even when beta1 is large.
    x1_std_est = np.sqrt(a1 * STDS.max() ** 2 + beta1)
    plot_range = max(1.0, np.abs(MEANS).max() * np.sqrt(a1) + 4.5 * x1_std_est)
    x_grid = np.linspace(-plot_range, plot_range, 1000)
    p0 = gmm_pdf(x_grid)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    example_vs = [-0.25]
    example_colors = [x_to_rgb(v) for v in example_vs]

    # --- panel 1: joint distribution p(x0, x1) ---
    ax = axes[0, 0]
    lim0 = 0.6  # x0's own range never changes with beta1
    lim1 = plot_range  # x1's range does
    grid0 = np.linspace(-lim0, lim0, 400)
    grid1 = np.linspace(-lim1, lim1, 400)
    x0_grid, x1_grid = np.meshgrid(grid0, grid1, indexing="ij")
    joint = joint_pdf(x0_grid, x1_grid, beta1)

    strip_w = 0.08 * 2 * lim1
    ax.set_xlim(-lim1 - strip_w, lim1)
    ax.set_ylim(-lim0, lim0)
    im = ax.imshow(joint, origin="lower", extent=[-lim1, lim1, -lim0, lim0], aspect="auto", cmap="viridis")
    hue_strip_vertical(ax, -lim1 - strip_w, -lim1)
    ax.set_aspect("equal")
    for v, color in zip(example_vs, example_colors):
        ax.axvline(v, color=color, linewidth=1.2, linestyle="--")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="density")
    ax.set_title(r"$p(x_0, x_1)$: joint distribution")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_0$")

    # --- panel 2: exact reverse posterior q(x0 | x1=v) ---
    ax = axes[0, 1]
    ymax2 = p0.max()
    for v, color in zip(example_vs, example_colors):
        post = posterior_pdf(x_grid, v, beta1)
        ymax2 = max(ymax2, post.max())
        ax.plot(x_grid, post, color=color, linewidth=1.3, label=rf"$v$={v:+.3f}")
        ax.axvline(v, color=color, linewidth=0.8, linestyle=":")
    setup_axis(ax, r"$q(x_0 \mid x_1=v)$: exact reverse posterior", 1.05 * ymax2, plot_range)
    ax.legend(fontsize=7, loc="upper right", ncol=2)

    # --- panel 3: learned_curve_on_joint -- single-step model vs. exact curve ---
    ax = axes[1, 0]
    rng = np.random.default_rng(0)
    denoiser_model, denoiser_losses = train_denoiser.train(rng)
    print(f"[panel 3] single-step model final training loss (mean of last 100 iters): "
          f"{np.mean(denoiser_losses[-100:]):.4f}")

    lc_grid = np.linspace(-lim0, lim0, 400)
    lc_x0, lc_x1 = np.meshgrid(lc_grid, lc_grid, indexing="ij")
    lc_joint = joint_pdf(lc_x0, lc_x1, train_denoiser.BETA1)
    lc_v = np.linspace(-lim0, lim0, 300)
    lc_exact = posterior_mean_curve(lc_v, train_denoiser.BETA1)
    lc_eps_hat = denoiser_model.predict(lc_v[:, None])[:, 0]
    lc_learned = (lc_v - np.sqrt(train_denoiser.BETA1) * lc_eps_hat) / np.sqrt(train_denoiser.A1)

    strip_w3 = 0.08 * 2 * lim0
    ax.set_xlim(-lim0 - strip_w3, lim0)
    ax.set_ylim(-lim0, lim0)
    im3 = ax.imshow(lc_joint, origin="lower", extent=[-lim0, lim0, -lim0, lim0], aspect="auto", cmap="viridis")
    hue_strip_vertical(ax, -lim0 - strip_w3, -lim0)
    ax.set_aspect("equal")
    ax.plot(lc_v, lc_exact, color="white", linewidth=2.2, label=r"exact $E[x_0 \mid x_1=v]$")
    ax.plot(lc_v, lc_learned, color="tab:orange", linewidth=2.0, linestyle="--",
            label="learned (from trained MLP)")
    fig.colorbar(im3, ax=ax, fraction=0.046, pad=0.04, label="density")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_0$")
    ax.set_title(rf"What the MLP learns ($\beta_1$={train_denoiser.BETA1:g}, fixed)")
    ax.legend(fontsize=8, loc="upper left")

    # --- panel 4: learned_curve_on_joint_adjacent -- multi-step model vs. exact curve ---
    ax = axes[1, 1]
    multistep = load_trained()

    alpha_bar_t = ALPHA_BARS[T_STEP - 1]      # x_t's own cumulative signal fraction
    alpha_tp1 = ALPHAS[T_STEP]                # single-step signal fraction, t -> t+1
    beta_tp1 = BETAS[T_STEP]
    alpha_bar_tp1 = ALPHA_BARS[T_STEP]        # x_{t+1}'s cumulative signal fraction

    adj_grid = np.linspace(-lim0, lim0, 400)
    x_t_grid, x_tp1_grid = np.meshgrid(adj_grid, adj_grid, indexing="ij")
    adj_joint = joint_pdf_adjacent(x_t_grid, x_tp1_grid, alpha_bar_t, alpha_tp1)

    adj_v = np.linspace(-lim0, lim0, 300)
    adj_exact = exact_conditional_mean_adjacent(adj_v, alpha_bar_t, alpha_tp1)
    t_norm = np.full_like(adj_v, (T_STEP + 1) / T)
    adj_eps_hat = multistep.predict(make_inputs(adj_v, t_norm))[:, 0]
    adj_learned = (adj_v - (beta_tp1 / np.sqrt(1.0 - alpha_bar_tp1)) * adj_eps_hat) / np.sqrt(alpha_tp1)

    strip_w4 = 0.08 * 2 * lim0
    ax.set_xlim(-lim0 - strip_w4, lim0)
    ax.set_ylim(-lim0, lim0)
    im4 = ax.imshow(adj_joint, origin="lower", extent=[-lim0, lim0, -lim0, lim0], aspect="auto", cmap="viridis")
    hue_strip_vertical(ax, -lim0 - strip_w4, -lim0)
    ax.set_aspect("equal")
    ax.plot(adj_v, adj_exact, color="white", linewidth=2.2,
            label=rf"exact $E[x_{{{T_STEP}}} \mid x_{{{T_STEP + 1}}}=v]$")
    ax.plot(adj_v, adj_learned, color="tab:orange", linewidth=2.0, linestyle="--",
            label="learned (from checkpoint)")
    fig.colorbar(im4, ax=ax, fraction=0.046, pad=0.04, label="density")
    ax.set_xlabel(rf"$x_{{{T_STEP + 1}}}$")
    ax.set_ylabel(rf"$x_{{{T_STEP}}}$")
    ax.set_title(rf"Adjacent-step reverse: $x_{{{T_STEP + 1}}} \to x_{{{T_STEP}}}$ ($\bar\alpha_{{{T_STEP}}}$={alpha_bar_t:.3f})")
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(rf"DDPM reverse step on 1D hue data ($\beta_1$={beta1:g}): exact math vs. what a trained model learns",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if out_path is None:
        if beta1 == BETA1:
            out_path = os.path.join(OUT_DIR, "step0_backward.png")
        else:
            out_path = os.path.join(OUT_DIR, f"step0_forward_backward_beta{beta1:g}.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
