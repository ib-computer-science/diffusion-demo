"""Interactive viewer: press Enter to draw a new sample from the true hue
distribution p(x0) and see it displayed as a colored square.

A direct, hands-on way to *feel* the distribution rather than just look at
its density curve: run it for a couple dozen presses and notice red and
green come up about equally often while yellow is much rarer (weights
3/7, 1/7, 3/7) -- the same GMM used throughout the rest of this demo
(hue_gmm.py).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from hue_gmm import sample_gmm, x_to_hue, x_to_rgb

rng = np.random.default_rng()


def new_sample():
    x = float(sample_gmm(1, rng)[0])
    return x, x_to_hue(x), x_to_rgb(x)


def main():
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    x, hue, rgb = new_sample()
    square = Rectangle((0.1, 0.1), 0.8, 0.8, facecolor=rgb, edgecolor="black", linewidth=1.5)
    ax.add_patch(square)
    title = ax.set_title(f"x={x:+.3f}    hue={hue:.1f} deg")

    def on_key(event):
        if event.key == "enter":
            x, hue, rgb = new_sample()
            square.set_facecolor(rgb)
            title.set_text(f"x={x:+.3f}    hue={hue:.1f} deg")
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.suptitle("Press Enter for a new sample from p(x0)  --  close window to quit", fontsize=9)
    plt.show()


if __name__ == "__main__":
    main()
