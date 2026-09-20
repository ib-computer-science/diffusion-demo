"""Extend learned_curve_on_joint.py to the full multi-step model: for
several representative t, compare the exact E[x0|x_t=v] curve against the
trained multi-step model's implied x0 estimate, overlaid on the joint
distribution p(x0, x_t) at that t.

This checks directly whether the wiggle-smoothing bias seen in the
single-step demo (learned_curve_on_joint.py) -- the learned curve cutting
through the low-density gap next to the small, flanked yellow bump instead
of detouring through its peak -- is present per-timestep in the actual
multi-step model, and how it evolves across the schedule. That bias is
exactly what CLAUDE.md's "yellow bump" investigation traced the
underrepresentation back to, after the oracle-mean experiment
(exact_reverse_multistep.py) ruled out the fixed-variance sampling scheme
itself.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ddpm_step import exact_conditional_mean_x0, joint_pdf_xt
from hue_gmm import x_to_rgb
from schedule import ALPHA_BARS, T
from train_multistep import train

OUT_DIR = "output"
T_VALUES = [1, 10, 30, 60, 100]
LIM = 0.6


def hue_strip(ax, y0, y1, n=600):
    xs = np.linspace(*ax.get_xlim(), n)
    colors = np.array([x_to_rgb(v) for v in xs])[None, :, :]
    ax.imshow(colors, extent=[xs[0], xs[-1], y0, y1], aspect="auto", origin="lower", zorder=0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(0)
    model, losses = train(rng)
    print(f"final training loss (mean of last 200 iters): {np.mean(losses[-200:]):.4f}")

    plot_grid = np.linspace(-LIM, LIM, 300)
    v_grid = np.linspace(-LIM, LIM, 300)
    x0_grid, xt_grid = np.meshgrid(plot_grid, plot_grid, indexing="ij")

    fig, axes = plt.subplots(1, len(T_VALUES), figsize=(4 * len(T_VALUES), 5.2))

    for ax, t in zip(axes, T_VALUES):
        alpha_bar_t = ALPHA_BARS[t - 1]
        joint = joint_pdf_xt(x0_grid, xt_grid, alpha_bar_t)

        exact_curve = exact_conditional_mean_x0(v_grid, alpha_bar_t)

        t_norm = np.full_like(v_grid, t / T)
        eps_hat = model.predict(np.stack([v_grid, t_norm], axis=1))[:, 0]
        learned_curve = (v_grid - np.sqrt(1.0 - alpha_bar_t) * eps_hat) / np.sqrt(alpha_bar_t)

        strip_h = 0.08 * 2 * LIM
        ax.set_xlim(-LIM, LIM)
        ax.set_ylim(-LIM - strip_h, LIM)
        ax.imshow(joint.T, origin="lower", extent=[-LIM, LIM, -LIM, LIM], aspect="auto", cmap="viridis")
        hue_strip(ax, -LIM - strip_h, -LIM)

        ax.plot(exact_curve, v_grid, color="white", linewidth=2.0, label="exact E[x0|x_t]")
        ax.plot(learned_curve, v_grid, color="tab:orange", linewidth=1.8, linestyle="--", label="learned")

        ax.set_title(f"t={t}  (alpha_bar={alpha_bar_t:.3f})", fontsize=10)
        ax.set_xlabel("x0")
        if ax is axes[0]:
            ax.set_ylabel("x_t")
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Multi-step model: exact vs. learned E[x0|x_t], across timesteps", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(OUT_DIR, "learned_curve_on_joint_multistep.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
