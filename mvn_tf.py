import tensorflow as tf
import tensorflow_probability as tfp


def tf_mvn_cdf(a, mu, L, n_samples=1000):
    dist = tfp.distributions.MultivariateNormalTriL(loc=mu, scale_tril=L)
    samples = dist.sample(n_samples)
    return tf.reduce_mean(tf.cast(tf.reduce_all(samples <= a, axis=1), tf.float32))
