"""1D toy data distribution: hue of an HSV color, modeled as a Gaussian mixture.

Hue is circular (0deg and 360deg are the same point), but our three bumps
(red=0deg, yellow=60deg, green=120deg) all sit far from the blue region
(~240deg), so we cut the circle open there and treat hue as an ordinary
linear variable with no wraparound artifacts.

We then rescale that cut-open hue axis into a normalized coordinate x
(roughly [-1, 1]), matching the convention DDPM papers use for pixel data.
"""

import colorsys

import numpy as np

# --- coordinate transform: hue (degrees, cut at 240deg) <-> normalized x ---

HUE_CUT = 240.0   # cut point of the circle, placed in the empty blue region
HUE_SHIFT = 60.0  # center of mass of our three bumps
HUE_SCALE = 180.0 # scale so the cut point maps to +-1


def hue_to_x(hue_deg):
    hue_deg = np.where(hue_deg >= HUE_CUT, hue_deg - 360.0, hue_deg)
    return (hue_deg - HUE_SHIFT) / HUE_SCALE


def x_to_hue(x):
    return (x * HUE_SCALE + HUE_SHIFT) % 360.0


def x_to_rgb(x):
    hue_deg = x_to_hue(x)
    return colorsys.hsv_to_rgb(hue_deg / 360.0, 1.0, 1.0)


# --- the target distribution p(x0): three hue bumps ---

_COMPONENTS = [
    dict(name="red", hue_deg=0.0, weight=3 / 7, std_deg=10.0),
    dict(name="yellow", hue_deg=60.0, weight=1 / 7, std_deg=10.0),
    dict(name="green", hue_deg=120.0, weight=3 / 7, std_deg=10.0),
]

WEIGHTS = np.array([c["weight"] for c in _COMPONENTS])
MEANS = np.array([hue_to_x(c["hue_deg"]) for c in _COMPONENTS])
STDS = np.array([c["std_deg"] / HUE_SCALE for c in _COMPONENTS])
NAMES = [c["name"] for c in _COMPONENTS]


def normal_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def gmm_pdf(x, weights=WEIGHTS, means=MEANS, stds=STDS):
    x = np.asarray(x)[..., None]
    return (weights * normal_pdf(x, means, stds)).sum(-1)


def sample_gmm(n, rng, weights=WEIGHTS, means=MEANS, stds=STDS):
    k = rng.choice(len(weights), size=n, p=weights)
    return rng.normal(means[k], stds[k])
