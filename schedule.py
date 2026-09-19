"""Shared increasing beta schedule for the full multi-step (T=100) demos.

Same schedule used in long_schedule_forward.py, factored out so the
training script can reuse the exact alpha_bar values instead of
re-deriving them.
"""

import numpy as np

from ddpm_step import cumulative_alpha_bar

T = 100
BETAS = np.linspace(0.001, 0.08, T)
ALPHAS = 1.0 - BETAS
ALPHA_BARS = cumulative_alpha_bar(BETAS)
