"""Interactive viewer: press Enter to draw a fresh x_T ~ N(0, 1) noise
sample, run it back through the full trained 100-step reverse process, and
see both the noise input and the resulting denoised x0 as colored squares
side by side.

Trains the multi-step model once at startup -- the only slow part (a few
minutes). Every Enter press after that is a single reverse pass through the
already-trained model for one sample, which takes a fraction of a second.

Uses multistep_model's architecture (sinusoidal time embedding, hidden=192
-- see CLAUDE.md's yellow-bump investigation for why this one and not the
original train_multistep.py's), but with fewer iterations than that
model's canonical N_ITERS=60000, trading some sample quality for a startup
time closer to a couple of minutes instead of ten.

Deliberately imports multistep_model (not train_multistep_improved) since
the latter calls matplotlib.use("Agg") for its own non-interactive
PNG-saving -- importing it here would force that same non-interactive
backend on this script too, silently breaking plt.show().
"""

import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import multistep_model as mm
from hue_gmm import x_to_hue, x_to_rgb

N_ITERS = 20000  # faster startup than mm's canonical 60000; still has the sinusoidal-t fix

rng = np.random.default_rng()


def train_model():
    print(f"Training the {mm.HIDDEN}-hidden-unit multi-step model "
          f"({N_ITERS} iterations) -- this takes a few minutes...")
    t0 = time.time()
    model, losses = mm.train(rng, n_iters=N_ITERS)
    print(f"done in {time.time() - t0:.0f}s (final loss {np.mean(losses[-200:]):.4f})")
    return model


def new_pair(model):
    x_T = float(rng.normal())
    x0 = float(mm.reverse_sample(model, 1, rng, x_init=np.array([x_T]))[0])
    return x_T, x0


def style_axis(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    model = train_model()

    fig, (ax_noise, ax_denoised) = plt.subplots(1, 2, figsize=(9, 4.7))
    style_axis(ax_noise)
    style_axis(ax_denoised)

    x_T, x0 = new_pair(model)
    sq_noise = Rectangle((0.1, 0.1), 0.8, 0.8, facecolor=x_to_rgb(x_T), edgecolor="black", linewidth=1.5)
    sq_denoised = Rectangle((0.1, 0.1), 0.8, 0.8, facecolor=x_to_rgb(x0), edgecolor="black", linewidth=1.5)
    ax_noise.add_patch(sq_noise)
    ax_denoised.add_patch(sq_denoised)
    title_noise = ax_noise.set_title(f"x_T={x_T:+.3f}  (noise)")
    title_denoised = ax_denoised.set_title(f"x_0={x0:+.3f}  hue={x_to_hue(x0):.1f} deg")

    def on_key(event):
        if event.key == "enter":
            x_T, x0 = new_pair(model)
            sq_noise.set_facecolor(x_to_rgb(x_T))
            sq_denoised.set_facecolor(x_to_rgb(x0))
            title_noise.set_text(f"x_T={x_T:+.3f}  (noise)")
            title_denoised.set_text(f"x_0={x0:+.3f}  hue={x_to_hue(x0):.1f} deg")
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.suptitle("Press Enter: new noise sample -> reverse process -> denoised result", fontsize=9)
    plt.show()


if __name__ == "__main__":
    main()
