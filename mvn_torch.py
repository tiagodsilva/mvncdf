import torch
from botorch.sampling import NormalQMCEngine

_sample_cache = {}


def sample_mvn(n: int, d: int) -> torch.Tensor:
    if (n, d) in _sample_cache:
        return _sample_cache[(n, d)]

    engine = NormalQMCEngine(d)
    samples = engine.draw(n)
    _sample_cache[(n, d)] = samples
    return _sample_cache[(n, d)]


def mvn_cdf_torch(a, mu, L):
    samples = sample_mvn(2**12, L.shape[0])
    samples = mu + samples @ L.T
    indicators = torch.all(samples <= a, dim=1)
    indicators = indicators.type(torch.float32)

    return torch.mean(indicators)


@torch.jit.script
def _mvn_cdf_torch_jit(samples, a, mu, L):
    samples = mu + samples @ L.T
    indicators = torch.all(samples <= a, dim=1)
    indicators = indicators.type(torch.float32)

    return torch.mean(indicators)


def mvn_cdf_torch_jit(a, mu, L):
    samples = sample_mvn(2**12, L.shape[0])
    return _mvn_cdf_torch_jit(samples, a, mu, L)
