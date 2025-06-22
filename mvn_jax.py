import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(42)
_sample_cache = {}


def sample_mvn(n, d):
    if (n, d) not in _sample_cache:
        _sample_cache[(n, d)] = jax.random.normal(key, shape=(n, d))

    return _sample_cache[(n, d)]


@jax.jit
def mvn_cdf_jax(a, mu, L):
    samples = sample_mvn(2**12, L.shape[0])

    samples = mu + samples @ L.T
    indicators = jnp.all(samples <= a, axis=1)
    return jnp.mean(indicators)
