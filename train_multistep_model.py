"""Train the canonical multi-step denoiser once and save its weights, so
other programs can load a ready-to-use model instantly instead of
retraining from scratch every time they run.

Run this once (takes several minutes -- hidden=192, N_ITERS=60000):

    python train_multistep_model.py

Then, anywhere else:

    from multistep_model import load_trained
    model = load_trained()

reverse_process_viewer.py uses exactly this. Scripts whose whole point is
to *demonstrate* training (train_multistep_improved.py,
train_multistep_reweighted.py, both of which plot the loss curve) still
train their own model fresh each run instead of loading this checkpoint --
this script is for consumers that just need a working model, not to watch
one being trained.
"""

import os
import time

import numpy as np

from multistep_model import CHECKPOINT_PATH, HIDDEN, N_ITERS, train


def main():
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    rng = np.random.default_rng(0)

    print(f"Training multi-step model (hidden={HIDDEN}, N_ITERS={N_ITERS})...")
    t0 = time.time()
    model, losses = train(rng)
    print(f"done in {time.time() - t0:.0f}s, "
          f"final loss (mean of last 200 iters): {np.mean(losses[-200:]):.4f}")

    model.save(CHECKPOINT_PATH)
    print(f"saved model to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
