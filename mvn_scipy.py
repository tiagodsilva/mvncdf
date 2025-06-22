import numpy as np
from scipy.stats import multivariate_normal


def scipy_mvn_cdf(a: np.ndarray, mu: np.ndarray, cov: np.ndarray):
    return multivariate_normal.cdf(a, mean=mu, cov=cov)
