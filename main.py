import time
import numpy as np
import jax.numpy as jnp
import tensorflow as tf
from mvn_scipy import scipy_mvn_cdf
from mvn_qmc import qmc_mvn_cdf, qmc_mvn_cdf_without_cholesky
from mvn_tf import tf_mvn_cdf
from mvn_jax import mvn_cdf_jax


def time_function(f, repeat=100):
    def wrapper(*args, **kwargs):
        start_time = time.monotonic_ns()
        result = 0
        for _ in range(repeat):
            result += f(*args, **kwargs)
        end_time = time.monotonic_ns()
        elapsed_time_in_ms = (end_time - start_time) / 1e6
        print(f"Elapsed time for {f.__name__}: {elapsed_time_in_ms}ms")
        print(f"Result for {f.__name__}: {result}")
        print("--------------------------------")
        return elapsed_time_in_ms

    return wrapper


def main(repeat_per_function=500):
    dim = 6
    a = np.random.rand(dim)
    mu = np.random.rand(dim)
    cov = np.random.rand(dim, dim)
    cov = cov @ cov.T + np.eye(dim) * 1e-3

    L = np.linalg.cholesky(cov)

    time_function(qmc_mvn_cdf, repeat_per_function)(a, mu, L)
    time_function(qmc_mvn_cdf_without_cholesky, repeat_per_function)(a, mu, cov)
    time_function(scipy_mvn_cdf, repeat_per_function)(a, mu, cov)
    time_function(tf_mvn_cdf, repeat_per_function)(
        tf.convert_to_tensor(a), tf.convert_to_tensor(mu), tf.convert_to_tensor(L)
    )
    time_function(mvn_cdf_jax, repeat_per_function)(
        jnp.asarray(a), jnp.asarray(mu), jnp.asarray(L)
    )


if __name__ == "__main__":
    main()
