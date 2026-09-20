"""Multi-seed check on the yellow-bump investigation: is the shortfall a
systematic effect that survives across independent training runs, or is it
(like the red/green split found earlier) partly optimization noise specific
to one seed?

Retrains train_multistep.py's default T=100 configuration from several
different random seeds, runs full ancestral sampling for each, and reports
basin occupancy per seed against the exact (numerically integrated) true
fractions.
"""

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import train_multistep as tm
from hue_gmm import gmm_pdf, sample_gmm

SEEDS = [0, 1, 2]
REGIONS = [("red", -0.5, -0.18), ("yellow", -0.08, 0.08), ("green", 0.18, 0.5)]
OUT_DIR = "output"


def basin_fracs(x0):
    return {name: float(np.mean((x0 > lo) & (x0 < hi))) for name, lo, hi in REGIONS}


def exact_basin_fracs(n=200000):
    grid = np.linspace(-1.0, 1.0, n)
    pdf = gmm_pdf(grid)
    out = {}
    for name, lo, hi in REGIONS:
        mask = (grid > lo) & (grid < hi)
        out[name] = float(np.trapezoid(pdf[mask], grid[mask]))
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    true_frac = exact_basin_fracs()
    print("exact true fractions:", {k: round(v, 4) for k, v in true_frac.items()})

    results = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        t0 = time.time()
        model, losses = tm.train(rng)
        x0_gen = tm.reverse_sample(model, 20000, rng)
        elapsed = time.time() - t0
        gen_frac = basin_fracs(x0_gen)
        results.append((seed, gen_frac))
        print(f"seed={seed} ({elapsed:.0f}s, final loss={np.mean(losses[-200:]):.4f}) gen={ {k: round(v,4) for k,v in gen_frac.items()} }")

    print(f"\n{'seed':6s} {'red':>8s} {'yellow':>8s} {'green':>8s}")
    print(f"{'true':6s} {true_frac['red']:8.4f} {true_frac['yellow']:8.4f} {true_frac['green']:8.4f}")
    for seed, g_f in results:
        print(f"{seed:<6d} {g_f['red']:8.4f} {g_f['yellow']:8.4f} {g_f['green']:8.4f}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    width = 0.25
    x = np.arange(len(SEEDS))
    colors = {"red": "tab:red", "yellow": "goldenrod", "green": "tab:green"}
    for i, region in enumerate(["red", "yellow", "green"]):
        gens = [g_f[region] for _, g_f in results]
        ax.bar(x + (i - 1) * width, gens, width, label=f"{region} (generated)", color=colors[region], alpha=0.75)
        ax.hlines(true_frac[region], x[0] + (i - 1) * width - width / 2, x[-1] + (i - 1) * width + width / 2,
                  colors=colors[region], linestyles="dashed", linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in SEEDS])
    ax.set_ylabel("basin occupancy fraction")
    ax.set_title("Multi-seed check: generated occupancy (bars) vs. exact true weight (dashed)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "multiseed_check.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
