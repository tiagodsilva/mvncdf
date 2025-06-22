## MVN CDF

Often, a useful decomposition of the covariance matrix (e.g., Cholesky) is available. However, most MVN CDF implementations, e.g., scipy, can only receive the full covariance matrix, which is internally (and redundantly) decomposed.

This is not an issue when the MVN CDF is evaluated just a couple of times during an experiment. Nonetheless, when probing the MVN CDF thousands of times per minute, this becomes a bottleneck.

```
dim: 6
repeat_per_function: 100

Elapsed time for qmc_mvn_cdf: 14.725292ms
Result for qmc_mvn_cdf: 26.6357421875
--------------------------------
Elapsed time for qmc_mvn_cdf_without_cholesky: 10.870375ms
Result for qmc_mvn_cdf_without_cholesky: 26.6357421875
--------------------------------
Elapsed time for scipy_mvn_cdf: 109176.309583ms
Result for scipy_mvn_cdf: 26.606595025402907
--------------------------------
Elapsed time for tf_mvn_cdf: 388.040208ms
Result for tf_mvn_cdf: 26.36699676513672
--------------------------------
Elapsed time for mvn_cdf_jax: 175.17275ms
Result for mvn_cdf_jax: 27.1484375
--------------------------------
Elapsed time for mvn_cdf_torch: 11.176ms
Result for mvn_cdf_torch: 26.7333984375
--------------------------------
Elapsed time for mvn_cdf_torch_jit: 27.214709ms
Result for mvn_cdf_torch_jit: 26.7
```

`scipy`'s implementation is blazingly fast for bivariate distributions, but scales very poorly with the dimension. Changes on the parameters (e.g., number of points for the numerical integration) could reduce its runtime, but this would require careful tuning of a parameter with unpredictable effects.
