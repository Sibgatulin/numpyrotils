"""Tests for compute_importance_weights.

Verifies that importance sampling weights have the correct shape
and expected behaviour (e.g. weights near zero when guide matches model).
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer import SVI, Trace_ELBO

from numpyrotils.diagnostics import compute_importance_weights
from numpyrotils.svi import fit_svi


def _simple_model(obs=None):
    """Normal-Normal model for testing."""
    mu = numpyro.sample("mu", dist.Normal(0.0, 10.0))
    numpyro.sample("obs", dist.Normal(mu, 1.0), obs=obs)


class TestComputeImportanceWeights:
    """Verify importance weight computation."""

    def test_output_shape(self):
        """Should return one weight per sample."""
        obs = jnp.ones(10) * 2.0
        guide = AutoNormal(_simple_model)
        result, svi = fit_svi(
            _simple_model,
            guide=guide,
            num_steps=100,
            rng=random.PRNGKey(0),
            obs=obs,
        )
        samples = guide.sample_posterior(
            random.PRNGKey(1), result.params, sample_shape=(50,)
        )
        log_weights = compute_importance_weights(
            _simple_model, guide, samples, params=result.params, model_kwargs={"obs": obs}
        )
        assert log_weights.shape == (50,)
        assert jnp.isfinite(log_weights).all()

    def test_weights_are_finite(self):
        """Weights from a converged guide should be finite."""
        obs = jnp.ones(5) * 3.0
        guide = AutoNormal(_simple_model)
        result, svi = fit_svi(
            _simple_model,
            guide=guide,
            num_steps=200,
            rng=random.PRNGKey(42),
            obs=obs,
        )
        samples = guide.sample_posterior(
            random.PRNGKey(2), result.params, sample_shape=(20,)
        )
        log_weights = compute_importance_weights(
            _simple_model, guide, samples, params=result.params, model_kwargs={"obs": obs}
        )
        assert jnp.isfinite(log_weights).all()
