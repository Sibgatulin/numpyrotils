"""Tests for the Ricean distribution.

Validates sampling shapes, positivity, sample statistics against
scipy.stats.rice, log_prob correctness, and autodiff compatibility.
"""

import jax.numpy as jnp
from jax import random, grad
from scipy.stats import rice as scipy_rice

from numpyrotils.distributions import Ricean


class TestRiceanSampling:
    """Verify that Ricean.sample produces correctly shaped, positive arrays."""

    def test_sample_shape_scalar(self):
        """Scalar parameters produce a flat sample array."""
        d = Ricean(loc=2.0, scale=1.0)
        samples = d.sample(random.PRNGKey(0), sample_shape=(1000,))
        assert samples.shape == (1000,)

    def test_sample_shape_batched(self):
        """Batched loc broadcasts correctly into sample shape."""
        loc = jnp.array([1.0, 2.0, 3.0])
        d = Ricean(loc=loc, scale=1.0)
        samples = d.sample(random.PRNGKey(0), sample_shape=(500,))
        assert samples.shape == (500, 3)

    def test_samples_are_positive(self):
        """Ricean is supported on (0, inf); all samples must be positive."""
        d = Ricean(loc=2.0, scale=1.0)
        samples = d.sample(random.PRNGKey(0), sample_shape=(1000,))
        assert (samples > 0).all()

    def test_sample_mean_close_to_expected(self):
        """Empirical mean should match scipy.stats.rice analytical mean."""
        loc, scale = 3.0, 1.0
        d = Ricean(loc=loc, scale=scale)
        samples = d.sample(random.PRNGKey(1), sample_shape=(50_000,))
        # scipy rice: b = loc / scale, scale = scale
        expected_mean = scipy_rice.mean(loc / scale, scale=scale)
        assert jnp.abs(samples.mean() - expected_mean) < 0.1


class TestRiceanLogProb:
    """Verify Ricean.log_prob against scipy and through autodiff."""

    def test_matches_scipy(self):
        """log_prob values should match scipy.stats.rice.logpdf."""
        loc, scale = 2.5, 0.8
        x = jnp.array([0.5, 1.0, 2.0, 3.0, 5.0])
        d = Ricean(loc=loc, scale=scale)
        lp = d.log_prob(x)
        # scipy parameterisation: rice(b, scale=scale) where b = loc / scale
        lp_ref = scipy_rice.logpdf(x, loc / scale, scale=scale)
        assert jnp.allclose(lp, jnp.array(lp_ref), atol=1e-5)

    def test_gradient_exists(self):
        """Gradient of log_prob w.r.t. loc should be finite (no NaN from i0e)."""
        def neg_log_prob(loc):
            return -Ricean(loc=loc, scale=1.0).log_prob(jnp.array(2.0))

        g = grad(neg_log_prob)(jnp.array(1.5))
        assert jnp.isfinite(g)

    def test_batched_log_prob(self):
        """Batched parameters should produce element-wise log_prob."""
        loc = jnp.array([1.0, 2.0])
        scale = jnp.array([0.5, 1.5])
        d = Ricean(loc=loc, scale=scale)
        x = jnp.array([1.5, 2.5])
        lp = d.log_prob(x)
        assert lp.shape == (2,)
        assert jnp.isfinite(lp).all()
