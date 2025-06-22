## MVN CDF

Often, a useful decomposition of the covariance matrix (e.g., Cholesky) is available. However, most MVN CDF implementations, e.g., scipy, can only receive the full covariance matrix, which is internally (and redundantly) decomposed.

This is not an issue when the MVN CDF is evaluated just a couple of times during an experiment. Nonetheless, when probing the MVN CDF thousands of times per minute, this becomes a bottleneck.

```
Elapsed time for qmc_mvn_cdf: 54.136833ms
Result for qmc_mvn_cdf: 133.7890625
--------------------------------
Elapsed time for scipy_mvn_cdf: 168809.674916ms
Result for scipy_mvn_cdf: 135.1777601695129
--------------------------------
Elapsed time for tf_mvn_cdf: 2058.386834ms
Result for tf_mvn_cdf: 135.1470489501953
--------------------------------
Elapsed time for mvn_cdf_jax: 308.542958ms
Result for mvn_cdf_jax: 134.27734375
--------------------------------
Elapsed time for qmc_mvn_cdf_without_cholesky: 54.143ms
Result for qmc_mvn_cdf_without_cholesky: 133.7890625
--------------------------------
```

`scipy`'s implementation is blazingly fast for bivariate distributions, but scales very poorly with the dimension. Changes on the parameters (e.g., number of points for the numerical integration) could reduce its runtime, but this would require careful tuning of a parameter with unpredictable effects.
