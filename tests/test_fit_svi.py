"""Tests for fit_svi, run_svi_train_loop, and NaN policies.

Covers the main SVI entry points: construction + training (fit_svi),
the standalone training loop (run_svi_train_loop), the deprecated
run_svi wrapper, and NaN handling behaviour.
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import optax
import pytest
from jax import random
from numpyro.infer import Trace_ELBO, SVI
from numpyro.infer.autoguide import AutoDelta

from numpyrotils.svi import fit_svi, run_svi, run_svi_train_loop


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _trivial_model(obs=None):
    """Normal-Normal model with a single latent."""
    mu = numpyro.sample("mu", dist.Normal(0.0, 10.0))
    numpyro.sample("obs", dist.Normal(mu, 1.0), obs=obs)


@pytest.fixture
def obs_data():
    """20 observations at mu=3."""
    return jnp.ones(20) * 3.0


# ---------------------------------------------------------------------------
# fit_svi
# ---------------------------------------------------------------------------

class TestFitSvi:
    """Tests for the high-level fit_svi convenience function."""

    def test_loss_decreases(self, obs_data):
        """ELBO loss should decrease over 100 steps on a trivial model."""
        result, svi = fit_svi(
            _trivial_model,
            guide=AutoDelta(_trivial_model),
            num_steps=100,
            rng=random.PRNGKey(0),
            obs=obs_data,
        )
        assert result.losses.shape == (100,)
        assert result.losses[-1] < result.losses[0]

    def test_returns_params(self, obs_data):
        """Result should contain the expected AutoDelta parameter."""
        result, svi = fit_svi(
            _trivial_model,
            guide=AutoDelta(_trivial_model),
            num_steps=50,
            rng=random.PRNGKey(0),
            obs=obs_data,
        )
        assert "mu_auto_loc" in result.params

    def test_default_guide_is_autodelta(self, obs_data):
        """Omitting guide should default to AutoDelta."""
        result, svi = fit_svi(
            _trivial_model,
            num_steps=50,
            rng=random.PRNGKey(0),
            obs=obs_data,
        )
        assert "mu_auto_loc" in result.params

    def test_warns_on_missing_rng(self, obs_data):
        """Omitting rng should emit a warning about PRNGKey(0)."""
        with pytest.warns(match="No rng key provided"):
            fit_svi(_trivial_model, num_steps=10, obs=obs_data)

    def test_deprecated_run_svi(self, obs_data):
        """run_svi should still work but emit a DeprecationWarning."""
        with pytest.deprecated_call():
            result, svi = run_svi(
                _trivial_model,
                num_steps=10,
                rng=random.PRNGKey(0),
                obs=obs_data,
            )
        assert result.losses.shape == (10,)

    def test_callbacks_are_called(self, obs_data):
        """A callback with period=10 should fire at steps 0, 10, 20, 30, 40."""
        call_log = []
        callback = (10, lambda svi, state, loss: call_log.append(loss))
        result, svi = fit_svi(
            _trivial_model,
            num_steps=50,
            rng=random.PRNGKey(0),
            callbacks=[callback],
            obs=obs_data,
        )
        # steps 0, 10, 20, 30, 40
        assert len(call_log) == 5

    def test_teardown_is_called(self, obs_data):
        """Teardown callback should fire exactly once after the loop."""
        teardown_log = []
        teardown = lambda svi, state: teardown_log.append(True)
        fit_svi(
            _trivial_model,
            num_steps=10,
            rng=random.PRNGKey(0),
            callback_teardown=[teardown],
            obs=obs_data,
        )
        assert len(teardown_log) == 1

    def test_per_param_learning_rate(self, obs_data):
        """Per-parameter learning rates via dict should not crash."""
        result, svi = fit_svi(
            _trivial_model,
            guide=AutoDelta(_trivial_model),
            num_steps=50,
            learning_rate={"default": 1e-1, "mu_auto_loc": 5e-2},
            rng=random.PRNGKey(0),
            obs=obs_data,
        )
        assert result.losses.shape == (50,)

    def test_store_grads_false(self, obs_data):
        """Disabling gradient storage should not affect training."""
        result, svi = fit_svi(
            _trivial_model,
            num_steps=20,
            rng=random.PRNGKey(0),
            store_grads=False,
            obs=obs_data,
        )
        assert result.losses.shape == (20,)


# ---------------------------------------------------------------------------
# run_svi_train_loop
# ---------------------------------------------------------------------------

class TestRunSviTrainLoop:
    """Tests for the standalone training loop."""

    def test_basic_loop(self, obs_data):
        """Should run the requested number of steps and return losses."""
        guide = AutoDelta(_trivial_model)
        svi = SVI(_trivial_model, guide, optax.adam(1e-2), Trace_ELBO())
        rng = random.PRNGKey(42)
        state = svi.init(rng, obs=obs_data)
        result = run_svi_train_loop(svi, state, num_steps=30, obs=obs_data)
        assert result.losses.shape == (30,)

    def test_can_resume_from_state(self, obs_data):
        """Passing the returned state into a second run should resume training."""
        guide = AutoDelta(_trivial_model)
        svi = SVI(_trivial_model, guide, optax.adam(1e-2), Trace_ELBO())
        rng = random.PRNGKey(42)
        state = svi.init(rng, obs=obs_data)

        result1 = run_svi_train_loop(svi, state, num_steps=30, obs=obs_data)
        result2 = run_svi_train_loop(svi, result1.state, num_steps=30, obs=obs_data)

        # second run should start from where the first left off
        assert result2.losses[0] <= result1.losses[-1] + 1.0  # allow some jitter


# ---------------------------------------------------------------------------
# NaN policy
# ---------------------------------------------------------------------------

def _nan_model(obs=None):
    """Model that will produce NaNs with a high learning rate."""
    mu = numpyro.sample("mu", dist.Normal(0.0, 1e-6))
    numpyro.sample("obs", dist.Normal(mu, 1e-6), obs=obs)


class TestNanPolicy:
    """Verify NaN detection and policy behaviour."""

    def test_exit_stops_early(self):
        """nan_policy='exit' should stop before exhausting all steps."""
        obs = jnp.array([1.0])
        guide = AutoDelta(_nan_model)
        svi = SVI(_nan_model, guide, optax.sgd(1e6), Trace_ELBO())
        state = svi.init(random.PRNGKey(0), obs=obs)
        result = run_svi_train_loop(
            svi, state, num_steps=500, nan_policy="exit", obs=obs,
        )
        # should have stopped before 500 steps
        assert result.losses.shape[0] < 500

    def test_propagate_runs_full(self):
        """nan_policy='propagate' should run all requested steps."""
        obs = jnp.array([1.0])
        guide = AutoDelta(_nan_model)
        svi = SVI(_nan_model, guide, optax.sgd(1e6), Trace_ELBO())
        state = svi.init(random.PRNGKey(0), obs=obs)
        result = run_svi_train_loop(
            svi, state, num_steps=50, nan_policy="propagate", obs=obs,
        )
        assert result.losses.shape[0] == 50
