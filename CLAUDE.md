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
  predictor (hidden=192, 20000 iterations -- large enough to get within
  ~3% of the theoretical Bayes-optimal loss floor, see below) via the
  standard MSE objective, then compares the learned reverse posterior
  against the exact one. Demonstrates that MSE training recovers only
  `E[x0|x1=v]` (a mean), so at ambiguous `v` the learned model collapses
  true bimodality into one Gaussian sitting between the modes ("mode
  averaging") -- and this happens even at near-Bayes-optimal training, so
  it isn't a capacity/undertraining artifact, it's what squared-error loss
  necessarily produces.
- **`learned_curve_on_joint.py`** — plots the exact `E[x0|x1=v]` curve and
  the trained MLP's implied curve directly on the joint-density heatmap.
  With the original small model (hidden=64, 4000 iterations) the learned
  curve visibly smoothed out the wiggle that encodes real multimodal
  ambiguity; with the bigger model above, the learned curve now tracks the
  exact wiggle almost exactly (loss ~0.598 vs. a Bayes-optimal floor of
  ~0.582), isolating that the wiggle-smoothing seen at small scale was a
  fixable approximation error, not the fundamental limitation. The wiggle
  itself never goes away, at any model size -- it's the shape of the exact
  conditional mean, not something a bigger network could smooth *into*
  existence or *out of* existence.
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

## Interactive tools and reusable training

- **`multistep_model.py`** — the multi-step model's logic (sinusoidal time
  embedding, `train`, `reverse_sample`, `load_trained`), factored out of
  `train_multistep_improved.py` and kept free of matplotlib. Needed because
  that script calls `matplotlib.use("Agg")` for its own PNG-saving; an
  interactive script importing it directly would silently inherit that
  non-interactive backend and `plt.show()` would do nothing.
- **`train_multistep_model.py`** — trains the canonical model
  (hidden=192, N_ITERS=60000) once and saves its weights to
  `checkpoints/multistep_model.npz` (gitignored, like `output/`) via
  `mlp.TinyMLP.save`/`load`. Run this once; other programs then call
  `multistep_model.load_trained()` instead of retraining from scratch.
  Scripts whose point *is* to demonstrate training
  (`train_multistep_improved.py`, `train_multistep_reweighted.py`, both of
  which plot the loss curve) deliberately keep training their own model
  fresh rather than loading this checkpoint.
- **`sample_viewer.py`** — press Enter to draw a fresh sample from `p(x0)`
  and see it as a colored square; a hands-on way to feel the true weights
  (red/green common, yellow rare) instead of just reading a density curve.
- **`reverse_process_viewer.py`** — press Enter to draw a fresh
  `x_T ~ N(0,1)`, run it through the full trained reverse process, and see
  the noise input and denoised result side by side. Loads the saved
  checkpoint above, so startup is near-instant instead of a multi-minute
  training wait.

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

7. **Mirror-symmetry test on the red/green bias: probably not a hard bug.**
   Mirrored the data (negated all component means, so red and green swap
   positions -- a meaningless change mathematically, since they're
   statistically identical apart from label) and retrained from 2 seeds.
   A deterministic sign-based bug should keep favoring the same geometric
   side regardless of which color sits there; instead one seed flipped to
   favor the other side and one didn't. Combined with the fact that 3/3
   seeds agreeing in the un-mirrored test has a ~25% chance of happening by
   pure luck even in an unbiased process, this walks back finding #6's
   framing -- a manual code review of `mlp.py` and `train_multistep.py`
   found no actual defect either. Best current read: seed-and-data-specific
   optimization dynamics, not a deterministic implementation bug.
8. **`train_multistep_improved.py` (sinusoidal multi-frequency time
   embedding instead of raw scalar `t/T`, plus hidden=192) fixed the
   red/green asymmetry outright**, and the fix held robustly across both
   30k and 60k training iterations (red/green: 0.414/0.404 at 30k,
   0.410/0.408 at 60k -- both close to the true 0.427/0.427). This was a
   welcome side effect, not the intended target. **Yellow, however,
   plateaued**: 0.090 at both 30k (29% low) and 60k (26% low) iterations,
   with the loss curve visibly flat over that doubling. Since more
   training only bought large gains the *first* time we tried it (finding
   #2, starting from a badly undertrained state), plateauing here despite
   still having room to improve elsewhere (fixing red/green) points to a
   persistent, structural bottleneck: every training batch draws
   yellow-region examples at a fixed 1:3:3 ratio relative to red/green,
   forever -- more iterations reduce noise but can never change that ratio.
9. **First attempt at a legitimate (non-oracle) fix backfired.**
   `train_multistep_reweighted.py` reweights the training loss by
   `1/density(x0)`, where density is estimated from a large pool of
   samples drawn from the same unlabeled source used for training (no
   ground-truth component weights involved, unlike the earlier-rejected
   `1/pi_k` idea). Result: yellow flipped to *over*-represented (0.161 vs.
   true 0.114), but at real cost -- final loss got worse, and the
   generated histogram became broad and lumpy (several spurious bumps,
   only ~74% of mass landing in any of the three defined regions vs.
   ~93-97% in every prior run). Likely cause: the weight is based on x0's
   rarity but applied uniformly at every `t`, including large `t` where
   `x_t` is nearly pure noise and barely depends on x0 at all --
   upweighting "rare x0" there just injects large, inappropriate gradient
   scale where the rarity signal is no longer meaningful. A refined version
   would need to decay the reweighting toward 1 as `t` grows, or weight by
   an estimate of `x_t`'s own local density instead of x0's.

**Current status:** the investigation is well-characterized but not fully
resolved. Ruled out: fixed-variance sampling, compounding over steps,
schedule coarseness, and (most likely) a hard code bug. Confirmed
structural: yellow's shortfall survives more training, more capacity, and
a richer time embedding, and tracks a fixed per-batch sampling ratio rather
than an easily-fixed approximation error. One legitimate mitigation
attempt (naive density reweighting) made things worse rather than better.

**Open next steps:**

- Refine the reweighting idea to be `t`-dependent (decay weight strength
  toward 1 as `t` increases, or base it on estimated `x_t` density rather
  than x0's) rather than applying a single x0-based weight uniformly
  across the whole schedule.
- Track individual sample trajectories through the reverse chain (store
  `x_t` at every step for a batch of samples) to see which `t` range is
  where a trajectory's fate near yellow actually gets decided.
