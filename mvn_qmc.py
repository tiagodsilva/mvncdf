import numpy as np
from scipy.stats import qmc

_sample_cache = {}


def sample_mvn(n, d, cholesky_L=None):
    # cholesky_L is added so that different caches are used for the function that
    # uses and doesn't use pre-computed cholesky decomposition
    if (n, d, cholesky_L) not in _sample_cache:
        sampler = qmc.MultivariateNormalQMC(np.zeros(d), np.eye(d), rng=42)
        _sample_cache[(n, d, cholesky_L)] = sampler.random(n)

    return _sample_cache[(n, d, cholesky_L)]


def qmc_mvn_cdf(a, mu, L, n_samples: int = 2**12):
    sample = sample_mvn(n_samples, len(mu), cholesky_L=True)
    sample = sample @ L.T + mu

    sample_is_less_than_a = np.all(sample <= a, axis=1)
    return np.mean(sample_is_less_than_a)


def qmc_mvn_cdf_without_cholesky(a, mu, cov, n_samples: int = 2**12):
    sample = sample_mvn(n_samples, len(mu), cholesky_L=False)

    sample = sample @ np.linalg.cholesky(cov).T + mu
    sample_is_less_than_a = np.all(sample <= a, axis=1)
    return np.mean(sample_is_less_than_a)
