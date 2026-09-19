"""Math for a single DDPM forward/reverse step on a known Gaussian-mixture prior.

Forward step:      x1 = sqrt(1 - beta1) * x0 + sqrt(beta1) * eps,  eps ~ N(0, 1)

Because x0 ~ GMM and the forward step is linear-Gaussian, everything about the
reverse step is available in closed form via Bayes' rule -- no neural network
needed. This is the "why it can be reversed" answer for the toy case: a real
DDPM needs a learned approximation only because the true data distribution
p(x0) (e.g. over natural images) isn't known analytically like it is here.
"""

import numpy as np

from hue_gmm import MEANS, STDS, WEIGHTS, gmm_pdf, normal_pdf


def forward_step(x0, beta1, rng):
    a1 = 1.0 - beta1
    eps = rng.normal(size=np.shape(x0))
    return np.sqrt(a1) * x0 + np.sqrt(beta1) * eps


def marginal_x1_params(beta1, weights=WEIGHTS, means=MEANS, stds=STDS):
    """Params of q(x1), itself a GMM with the same weights as p(x0)."""
    a1 = 1.0 - beta1
    means1 = np.sqrt(a1) * means
    stds1 = np.sqrt(a1 * stds**2 + beta1)
    return weights, means1, stds1


def posterior_given_x1(v, beta1, weights=WEIGHTS, means=MEANS, stds=STDS):
    """Exact q(x0 | x1=v): a mixture of per-component Gaussian posteriors.

    Returns (responsibilities, posterior_means, posterior_stds), each of
    shape (n_components,). The posterior std is the same for every v (a
    property of linear-Gaussian models); only the mean and the mixing
    responsibilities depend on v.
    """
    a1 = 1.0 - beta1
    prior_var = stds**2

    post_var = 1.0 / (a1 / beta1 + 1.0 / prior_var)
    post_std = np.sqrt(post_var)
    post_mean = post_var * (np.sqrt(a1) * v / beta1 + means / prior_var)

    marg_weights, marg_means, marg_stds = marginal_x1_params(beta1, weights, means, stds)
    likelihood = marg_weights * normal_pdf(v, marg_means, marg_stds)
    responsibilities = likelihood / likelihood.sum()

    return responsibilities, post_mean, post_std


def posterior_pdf(x_grid, v, beta1, weights=WEIGHTS, means=MEANS, stds=STDS):
    resp, post_mean, post_std = posterior_given_x1(v, beta1, weights, means, stds)
    x_grid = np.asarray(x_grid)[..., None]
    return (resp * normal_pdf(x_grid, post_mean, post_std)).sum(-1)


def joint_pdf(x0, x1, beta1, weights=WEIGHTS, means=MEANS, stds=STDS):
    """p(x0, x1) = p(x0) * q(x1 | x0). A horizontal slice at fixed x1=v,
    renormalized, is exactly posterior_pdf(..., v, beta1)."""
    a1 = 1.0 - beta1
    return gmm_pdf(x0, weights, means, stds) * normal_pdf(x1, np.sqrt(a1) * x0, np.sqrt(beta1))


def posterior_mean_curve(x1_grid, beta1, weights=WEIGHTS, means=MEANS, stds=STDS):
    """E[x0 | x1=v] for each v in x1_grid: the density-weighted centroid of
    each horizontal slice through the joint distribution."""
    out = np.empty_like(x1_grid, dtype=float)
    for i, v in enumerate(x1_grid):
        resp, post_mean, _ = posterior_given_x1(v, beta1, weights, means, stds)
        out[i] = (resp * post_mean).sum()
    return out


def sample_posterior(v_array, beta1, rng, weights=WEIGHTS, means=MEANS, stds=STDS):
    """Ancestral sampling from q(x0 | x1=v) for an array of v's (the reverse step)."""
    out = np.empty_like(v_array, dtype=float)
    for i, v in enumerate(v_array):
        resp, post_mean, post_std = posterior_given_x1(v, beta1, weights, means, stds)
        k = rng.choice(len(weights), p=resp)
        out[i] = rng.normal(post_mean[k], post_std[k])
    return out
