import os
import time
import json
import tqdm

import numpy as np
import jax.numpy as jnp
import tensorflow as tf
import torch
from mvn_scipy import scipy_mvn_cdf
from mvn_qmc import qmc_mvn_cdf, qmc_mvn_cdf_without_cholesky
from mvn_tf import tf_mvn_cdf
from mvn_jax import mvn_cdf_jax
from mvn_torch import mvn_cdf_torch, mvn_cdf_torch_jit

from functools import reduce


def time_function(f, repeat=100, verbose=True):
    def wrapper(*args, **kwargs):
        start_time = time.monotonic_ns()
        result = 0
        for _ in range(repeat):
            result += f(*args, **kwargs)
        end_time = time.monotonic_ns()
        elapsed_time_in_ms = (end_time - start_time) / 1e6
        if verbose:
            print(f"Elapsed time for {f.__name__}: {elapsed_time_in_ms}ms")
            print(f"Result for {f.__name__}: {result}")
            print("--------------------------------")

        # Convert non-float results to float (negligible time)
        if isinstance(result, tf.Tensor):
            result = result.numpy().item()

        if torch.is_tensor(result):
            result = result.item()

        if isinstance(result, jnp.ndarray):
            result = result.item()

        if isinstance(result, np.ndarray):
            result = result.item()

        return {"elapsed_time_in_ms": elapsed_time_in_ms, "result": result}

    return wrapper


def compare_functions_on_dim(dim: int, repeat_per_function: int = 500, verbose=True):
    a = np.random.rand(dim)
    mu = np.random.rand(dim)
    cov = np.random.rand(dim, dim)
    cov = cov @ cov.T + np.eye(dim) * 1e-3

    L = np.linalg.cholesky(cov)

    elapsed_times_in_ms_and_results = {}

    # Evaluate the time required and the resulting CDF for each method

    elapsed_times_in_ms_and_results["qmc_mvn_cdf"] = time_function(
        qmc_mvn_cdf, repeat_per_function, verbose=verbose
    )(a, mu, L)
    elapsed_times_in_ms_and_results["qmc_mvn_cdf_without_cholesky"] = time_function(
        qmc_mvn_cdf_without_cholesky, repeat_per_function, verbose=verbose
    )(a, mu, cov)
    elapsed_times_in_ms_and_results["scipy_mvn_cdf"] = time_function(
        scipy_mvn_cdf, repeat_per_function, verbose=verbose
    )(a, mu, cov)
    elapsed_times_in_ms_and_results["tf_mvn_cdf"] = time_function(
        tf_mvn_cdf, repeat_per_function, verbose=verbose
    )(tf.convert_to_tensor(a), tf.convert_to_tensor(mu), tf.convert_to_tensor(L))
    elapsed_times_in_ms_and_results["mvn_cdf_jax"] = time_function(
        mvn_cdf_jax, repeat_per_function, verbose=verbose
    )(jnp.asarray(a), jnp.asarray(mu), jnp.asarray(L))
    elapsed_times_in_ms_and_results["mvn_cdf_torch"] = time_function(
        mvn_cdf_torch, repeat_per_function, verbose=verbose
    )(
        torch.as_tensor(a, dtype=torch.float32),
        torch.as_tensor(mu, dtype=torch.float32),
        torch.as_tensor(L, dtype=torch.float32),
    )
    elapsed_times_in_ms_and_results["mvn_cdf_torch_jit"] = time_function(
        mvn_cdf_torch_jit, repeat_per_function, verbose=verbose
    )(
        torch.as_tensor(a, dtype=torch.float32),
        torch.as_tensor(mu, dtype=torch.float32),
        torch.as_tensor(L, dtype=torch.float32),
    )

    return elapsed_times_in_ms_and_results


def main(
    min_dim: int = 1,
    max_dim: int = 8,
    repeat_per_dim: int = 10,
    repeat_per_function: int = 10,
    verbose: bool = False,
    data_dir: str = "data",
):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    evaluations_per_dim = {}

    elapsed_times_per_dim_in_ms = {}
    results_per_dim = {}

    pbar = tqdm.tqdm(total=(max_dim - min_dim + 1) * repeat_per_dim)

    for dim in range(min_dim, max_dim + 1):
        if dim not in evaluations_per_dim:
            evaluations_per_dim[dim] = []

        for _ in range(repeat_per_dim):
            evaluations_per_dim[dim].append(
                compare_functions_on_dim(dim, repeat_per_function, verbose=verbose)
            )
            pbar.update(1)
            pbar.set_description(f"Dim: {dim}")

        # Compute mean elapsed time and result for each key
        elapsed_times_per_dim_in_ms[dim] = reduce(
            lambda acc, x: {
                k: (acc[k] + x[k]["elapsed_time_in_ms"]) / repeat_per_dim for k in acc
            },
            evaluations_per_dim[dim],
            {k: 0 for k in evaluations_per_dim[dim][0].keys()},
        )

        results_per_dim[dim] = reduce(
            lambda acc, x: {k: (acc[k] + x[k]["result"]) / repeat_per_dim for k in acc},
            evaluations_per_dim[dim],
            {k: 0 for k in evaluations_per_dim[dim][0].keys()},
        )

    with open(os.path.join(data_dir, "elapsed_times.json"), "w") as f:
        json.dump(elapsed_times_per_dim_in_ms, f)

    with open(os.path.join(data_dir, "results.json"), "w") as f:
        json.dump(results_per_dim, f)


if __name__ == "__main__":
    main(verbose=False, data_dir="data")
