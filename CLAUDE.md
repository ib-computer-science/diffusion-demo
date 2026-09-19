# diffusion-demo

A from-scratch, dependency-light (NumPy + Matplotlib only) walkthrough of DDPM
mechanics using a 1D toy distribution, built incrementally to make each piece
of the theory visible before moving to the next.

## The data

`hue_gmm.py` defines the toy distribution: the **hue** channel of an HSV
color, modeled as a 3-component Gaussian mixture (red, yellow, green), with
weights 3/7, 1/7, 3/7. Hue is circular (0deg and 360deg coincide), so the
circle is cut open in the empty blue region (~240deg) to avoid wraparound
artifacts, then rescaled into a normalized coordinate `x` roughly in
`[-1, 1]` (matching how DDPM normalizes pixel data). `x_to_rgb(x)` converts
back for the colored "hue strip" shown under most plots.

## Files, in the order they were built

- **`hue_gmm.py`** — the GMM data distribution and hue/x/RGB conversions.
- **`ddpm_step.py`** — closed-form DDPM math: forward step, marginal
  `q(x_t)` for arbitrary `alpha_bar_t` (constant or scheduled beta),
  the exact reverse posterior `q(x0|x1=v)` via Bayes' rule, the joint
  distribution `p(x0, x1)`, and the exact conditional-mean curve
  `E[x0|x1=v]`. This all works in closed form only because `p(x0)` is a
  known GMM — a real DDPM needs a learned approximation precisely because
  real data distributions aren't known analytically.
- **`step0_forward_backward.py`** — 4-panel figure: `p(x0)`, `q(x1)` after
  one forward step, the exact reverse posterior for several example `x1`
  values (showing genuine bimodality at ambiguous points), and the joint
  distribution `p(x0,x1)` with the same example points marked — making
  visible that the posterior panel is just a renormalized horizontal slice
  of the joint panel.
- **`beta1_sweep.py`** — how beta1 controls the reverse posterior's shape:
  interpolates from a near-delta spike (beta1->0) to the full prior
  `p(x0)` (beta1->1), with posterior std and responsibility entropy
  plotted quantitatively.
- **`mlp.py`** — a tiny hand-rolled 2-hidden-layer MLP (manual forward/
  backward/Adam, no autograd) so training is fully transparent. Supports
  arbitrary input dimension (originally just `x_t`; later `[x_t, t]`).
- **`train_denoiser.py`** — trains a single-step (fixed beta1=0.005) noise
  predictor via the standard MSE objective, then compares the learned
  reverse posterior against the exact one. Demonstrates that MSE training
  recovers only `E[x0|x1=v]` (a mean), so at ambiguous `v` the learned
  model collapses true bimodality into one Gaussian sitting between the
  modes ("mode averaging"). Verified the training loss plateau against the
  theoretical Bayes-optimal floor computed from the exact posterior
  variance.
- **`learned_curve_on_joint.py`** — plots the exact `E[x0|x1=v]` curve and
  the trained MLP's implied curve directly on the joint-density heatmap.
  The exact curve wiggles through the low-density gaps between hue bumps
  (encoding real multimodal ambiguity); the learned curve smooths that
  wiggle out.
- **`multi_step_forward.py`** — chains `t=0..4` forward steps at a
  *constant* beta1, using the closed-form direct-jump formula.
- **`schedule_comparison.py`** — same 4 steps, constant vs. a linearly
  increasing beta schedule: increasing starts gentler and ends more
  aggressive than constant, for the same step budget.
- **`schedule.py`** — single source of truth for the "full" increasing
  schedule: `T=100`, `beta` linear from 0.001 to 0.08, plus
  `ALPHAS`/`ALPHA_BARS`. Reused by `long_schedule_forward.py` and
  `train_multistep.py`.
- **`long_schedule_forward.py`** — forward marginal at t=5,10,20,40,60,80,100
  under the full schedule; by t=100, `alpha_bar~0.016` and the distribution
  is visually indistinguishable from isotropic Gaussian noise.
- **`train_multistep.py`** — trains a noise predictor across the *entire*
  schedule (input is now `[x_t, t/T]`), using the standard DDPM training
  algorithm (random `x0`, random `t`, direct jump to `x_t` via
  `alpha_bar_t`). Then runs full ancestral sampling (Ho et al. Algorithm 2:
  start at `x_T ~ N(0,1)`, denoise down to `x_0`, injecting fresh noise at
  every step except the last) and compares the generated distribution
  against the true `p(x0)`. Red/green recovered well; yellow (the minor
  mode) is visibly underrepresented — see below.

## Open investigation: the yellow bump is underrepresented

`train_multistep.py`'s reverse-sampled output systematically shorts the
yellow component. Measured (20k samples, narrow window around each mode):

| region | true weight | generated |
|---|---|---|
| red    | 0.435 | 0.395 |
| yellow | 0.116 | 0.065 (~44% low) |
| green  | 0.425 | 0.449 |

Leaked mass shows up mostly in green, not spread evenly.

**Candidate causes, most to least likely:**

1. **Training data imbalance.** Batches sample `x0` proportional to the true
   weights, so yellow (1/7) contributes ~3x fewer training examples near
   its basin than red or green (3/7 each) — weaker gradient signal there.
2. **Yellow is flanked on both sides**, unlike red/green which each face
   only one (lighter) competitor. `learned_curve_on_joint.py` already
   showed the learned conditional-mean curve smooths out the wiggle that
   encodes a mode's identity near gaps; yellow gets this pressure from two
   directions at once.
3. **Compounding of small per-step biases over 100 sequential steps.** A
   single-shot exact posterior sample (as in `step0_forward_backward.py`'s
   joint-distribution panel) reproduces every mode's weight exactly, by a
   marginalization identity, regardless of how small the mode is. Ancestral
   sampling chains 100 imperfect steps instead, so small directional biases
   don't have to cancel.
4. **Fixed reverse-step variance** (`sigma_t^2 = beta_t`, the standard
   simplified DDPM choice) injects more noise per step than the true,
   tighter posterior variance (established back in `train_denoiser.py`'s
   comparison at the unambiguous example points) — giving trajectories more
   opportunity to drift out of a narrow minor-mode basin than a real
   posterior-variance step would.

**Next steps to isolate which mechanism dominates:**

- Retrain with more capacity/iterations (bigger hidden size, more steps)
  and re-measure the same basin-occupancy table. If the gap shrinks
  substantially, the issue is mainly a training/capacity limitation. If it
  persists, it points to something structural (schedule/variance choice).
- Swap the network's predicted mean into the ancestral sampler but replace
  its *variance* with the true per-component posterior variance
  (computable from `ddpm_step.py`) instead of `beta_t`, to isolate whether
  the fixed-variance simplification alone (independent of network error)
  causes the leakage.
- Extend the `learned_curve_on_joint.py`-style diagnostic to the
  multi-step model: plot the learned vs. exact conditional-mean curve for
  several fixed `t` values, to see whether the bias near yellow grows or
  shrinks as `t` increases.
- Track individual sample trajectories through the reverse chain (store
  `x_t` at every step for a batch of samples) to see roughly which `t`
  range is where a trajectory's fate near yellow gets decided, i.e. where
  it commits to or drifts away from that basin.
