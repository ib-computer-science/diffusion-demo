"""Same figure as step0_backward.py, but with a deliberately too
large beta1 -- showing what happens once the single-step-is-Gaussian
assumption clearly breaks: q(x1) merges into one broad hump, and the exact
reverse posterior stops depending much on v at all, collapsing toward the
unconditional prior p(x0) (per the theoretical beta1->1 limit demonstrated
quantitatively in beta1_sweep.py).
"""

from step0_backward import main

LARGE_BETA1 = 0.3

if __name__ == "__main__":
    main(beta1=LARGE_BETA1)
