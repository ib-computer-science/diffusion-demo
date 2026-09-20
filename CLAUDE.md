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

**Findings so far, in the order they were established:**

1. **The fixed-variance sampling scheme is exonerated.** `exact_reverse_multistep.py`
   reruns the identical 100-step ancestral sampler but replaces the
   network's prediction with the closed-form, zero-training-error exact
   conditional mean `E[x0|x_t]` (`ddpm_step.exact_conditional_mean_x0`) at
   every step, keeping the same fixed `sigma_t^2=beta_t` noise injection.
   Result: yellow comes out correct (0.129 vs. true 0.120). So candidates
   #3 (compounding over steps) and #4 (fixed variance) above are **not**
   the cause — the leakage is entirely attributable to the learned model.
2. **More training partially helps, but reveals a second, separate bias.**
   Doubling training iterations (20k -> 40k, same architecture) shrank
   yellow's shortfall from ~44% low to ~16% low -- real evidence for
   training/data-imbalance (#1). But red *also* got worse (more
   underrepresented) while green overshot further, even though the true
   distribution is symmetric between red and green (both 3/7) -- this
   asymmetry can't be structural, so at this budget the model still hasn't
   converged and red/green's split is partly run-to-run noise.
3. **The multi-t exact-vs-learned curve comparison**
   (`learned_curve_on_joint_multistep.py`) shows the wiggle-smoothing
   mechanism (#2) is real but only matters at low-to-moderate `t` (t=1,10
   track the exact curve closely). By t=30 and beyond, even the *exact*
   curve stops showing a yellow-specific wiggle -- noise dilutes yellow's
   thin evidence faster than red's/green's, so there's nothing left to
   smooth. Instead, a **previously unseen systematic red-vs-green bias**
   appears in the learned curve at high `t` (t=60, t=100, near-pure noise),
   pulling toward green. Since reverse sampling starts at t=T and works
   down, an early (high-t) bias has the most steps left to compound through.
4. **A theoretical framing that ties #1-#3 together:** DDPM's "reverse step
   is approximately Gaussian" assumption is only justified when beta_t is
   small *relative to local mode spacing and weight*, not small in an
   absolute sense (`beta1_sweep.py` demonstrates this quantitatively; see
   `step0_forward_backward.py`'s panel 3 for a direct example -- the same
   beta1 gives unimodal posteriors at the three bump centers but bimodal
   posteriors at the two gaps between them). Yellow's "safe" threshold is
   stricter than red's or green's (flanked on both sides, minority weight),
   but a real schedule beta_t(t) is a single global function of t, so it
   cannot be simultaneously safe everywhere.
5. **Tested and ruled out: finer per-step noise does not fix it, at matched
   compute.** If (4) were the dominant mechanism, using smaller beta_t with
   more steps (same total noise budget) should help, since it pushes the
   *exact* posterior toward unimodal almost everywhere. Tested T=200
   (beta: 0.0005->0.04, alpha_bar_T=0.0165, matching the original
   schedule's endpoint) with N_ITERS=40000 (matching the compute budget of
   the T=100/40k-iteration comparison run in finding #2). Result: yellow
   came out at 0.101 -- statistically indistinguishable from the T=100/40k
   result (0.100). No additional benefit from finer granularity. Likely
   explanation: finer steps spread the *same* fixed per-batch training
   imbalance (yellow is still only 1/7 of every batch) across twice as many
   distinct `t` values for the network to learn, trading one shape of the
   data-scarcity problem for another rather than resolving it. This
   suggests training-data imbalance and/or the high-t red/green bias from
   finding #3 -- not per-step schedule coarseness -- are the dominant
   levers.

6. **Multi-seed check (`multiseed_check.py`): both effects are systematic,
   not run-specific noise.** Retrained the default T=100 configuration from
   3 independent seeds. Yellow's shortfall is consistent and large in every
   run (seeds 0/1/2: generated 0.065 / 0.046 / 0.065 vs. exact true 0.121 --
   46-62% low every time). More surprisingly, the red/green asymmetry
   flagged as "probably just optimization noise" in finding #2 turned out
   to be **also consistent across all 3 seeds**: red always came out below
   its true 0.427 (0.395 / 0.389 / 0.412) and green always above (0.449 /
   0.480 / 0.442) -- despite red and green being exactly symmetric in the
   true distribution (same weight, spacing, and std). So there are
   apparently *two* distinct systematic effects layered on top of each
   other: (a) yellow's minority/flanked-mode shortfall, consistent with
   findings #1-#5, and (b) a separate, reproducible red-vs-green bias whose
   origin isn't yet explained -- it isn't predicted by anything in the data
   (which is symmetric), so it likely comes from some asymmetry in the
   architecture, the (x_t, t) input encoding, or the training/optimization
   procedure itself.

**Open next steps:**

- Investigate the newly-confirmed systematic red/green bias directly: it
  is unexplained by the (symmetric) data, so the search should focus on
  the network/training pipeline itself -- e.g. whether initialization,
  the tanh activation's odd symmetry interacting with the extra `t` input
  dimension, or the Adam optimizer's update dynamics break the x -> -x
  symmetry consistently rather than randomly per seed.
- Track individual sample trajectories through the reverse chain (store
  `x_t` at every step for a batch of samples) to see which `t` range is
  where a trajectory's fate near yellow -- and separately, near red vs.
  green -- actually gets decided.
- If training-data imbalance is confirmed as the dominant factor behind
  yellow specifically, look for a legitimate (non-oracle) mitigation --
  i.e. one that doesn't require knowing the true component weights, since
  real data has no such ground truth (a self-estimated density correction
  from the training data itself would qualify; reweighting by the known
  true weights would not -- it changes what distribution is being fit
  rather than correcting an estimation error, see conversation history for
  why this was rejected).
