"""Same idea as learned_curve_on_joint.py -- exact vs. learned E[x_t|x_{t+1}]
overlaid on the joint-density heatmap -- but for a single pair of ADJACENT
intermediate timesteps in the middle of the schedule, instead of a direct
jump back to x0.

x_t's own marginal distribution (itself a blurred GMM at this point in the
schedule, via ddpm_step.marginal_xt_params) plays the role p(x0) played in
the single-step demo, and the "forward step" relating x_t to x_{t+1} uses
the single-step alpha_{t+1}/beta_{t+1}, not a cumulative alpha_bar. The
learned curve is exactly the deterministic mean half of the same per-step
reverse formula already used inside multistep_model.reverse_sample -- just
evaluated across a grid of x_{t+1} instead of propagated through a batch.

x_{t+1} is plotted horizontally (the input you'd observe) and x_t
vertically (the output you'd read off the curve), matching
learned_curve_on_joint.py's convention.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ddpm_step import exact_conditional_mean_adjacent, joint_pdf_adjacent
from hue_gmm import x_to_rgb
from multistep_model import load_trained, make_inputs
from schedule import ALPHA_BARS, ALPHAS, BETAS, T

OUT_DIR = "output"
T_STEP = 10  # examine the transition x_{T_STEP} -> x_{T_STEP+1}


def hue_strip_vertical(ax, x0, x1, n=600):
    ys = np.linspace(*ax.get_ylim(), n)
    colors = np.array([x_to_rgb(v) for v in ys])[:, None, :]
    ax.imshow(colors, extent=[x0, x1, ys[0], ys[-1]], aspect="auto", origin="lower", zorder=0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = load_trained()

    alpha_bar_t = ALPHA_BARS[T_STEP - 1]      # x_t's own cumulative signal fraction
    alpha_tp1 = ALPHAS[T_STEP]                # single-step signal fraction, t -> t+1
    beta_tp1 = BETAS[T_STEP]
    alpha_bar_tp1 = ALPHA_BARS[T_STEP]        # x_{t+1}'s cumulative signal fraction
    print(f"t={T_STEP}: alpha_bar_t={alpha_bar_t:.4f}, alpha_(t+1)={alpha_tp1:.4f}, beta_(t+1)={beta_tp1:.4f}")

    lim = 0.6
    plot_grid = np.linspace(-lim, lim, 400)
    x_t_grid, x_tp1_grid = np.meshgrid(plot_grid, plot_grid, indexing="ij")
    joint = joint_pdf_adjacent(x_t_grid, x_tp1_grid, alpha_bar_t, alpha_tp1)

    v_grid = np.linspace(-lim, lim, 300)
    exact_curve = exact_conditional_mean_adjacent(v_grid, alpha_bar_t, alpha_tp1)

    t_norm = np.full_like(v_grid, (T_STEP + 1) / T)
    eps_hat = model.predict(make_inputs(v_grid, t_norm))[:, 0]
    learned_curve = (v_grid - (beta_tp1 / np.sqrt(1.0 - alpha_bar_tp1)) * eps_hat) / np.sqrt(alpha_tp1)

    fig, ax = plt.subplots(figsize=(7.5, 7))
    strip_w = 0.08 * 2 * lim
    ax.set_xlim(-lim - strip_w, lim)
    ax.set_ylim(-lim, lim)
    im = ax.imshow(joint, origin="lower", extent=[-lim, lim, -lim, lim], aspect="auto", cmap="viridis")
    hue_strip_vertical(ax, -lim - strip_w, -lim)
    ax.set_aspect("equal")

    ax.plot(v_grid, exact_curve, color="white", linewidth=2.2,
            label=rf"exact $E[x_{{{T_STEP}}} \mid x_{{{T_STEP + 1}}}=v]$")
    ax.plot(v_grid, learned_curve, color="tab:orange", linewidth=2.0, linestyle="--",
            label="learned (from checkpoint)")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="density")
    ax.set_xlabel(rf"$x_{{{T_STEP + 1}}}$")
    ax.set_ylabel(rf"$x_{{{T_STEP}}}$")
    ax.set_title(rf"Adjacent-step reverse: $x_{{{T_STEP + 1}}} \to x_{{{T_STEP}}}$ ($\bar\alpha_{{{T_STEP}}}$={alpha_bar_t:.3f})")
    ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "learned_curve_on_joint_adjacent.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
