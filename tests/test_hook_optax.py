"""Tests for hook_optax gradient capture.

Verifies that the optimizer wrapper correctly intercepts gradients
and stores them in the expected format (latest-only vs. append mode),
including aggregation of high-dimensional arrays.
"""

import jax.numpy as jnp
import optax

from numpyrotils.svi import hook_optax


def _step(opt, params, grads):
    """Run one optimizer step and return updated params."""
    state = opt.init(params)
    updates, _ = opt.update(grads, state, params)
    return optax.apply_updates(params, updates)


class TestHookOptax:
    """Tests for the basic gradient capture modes."""

    def test_latest_only(self):
        """append=False should store the latest gradient as a plain value."""
        opt, store = hook_optax(optax.adam(1e-2), append=False)
        params = {"a": jnp.array(1.0), "b": jnp.array([1.0, 2.0])}
        grads = {"a": jnp.array(0.5), "b": jnp.array([0.1, 0.2])}
        _step(opt, params, grads)

        assert "∇a" in store
        assert "∇b" in store
        # append=False -> values are arrays, not lists
        assert not isinstance(store["∇a"], list)

    def test_append_mode(self):
        """append=True should accumulate gradients in lists."""
        opt, store = hook_optax(optax.adam(1e-2), append=True)
        params = {"x": jnp.array(1.0)}
        grads = {"x": jnp.array(0.5)}
        state = opt.init(params)
        for _ in range(3):
            _, state = opt.update(grads, state, params)

        assert "∇x" in store
        assert isinstance(store["∇x"], list)
        assert len(store["∇x"]) == 3

    def test_custom_estimator(self):
        """A custom estimator (jnp.max) should be accepted without error."""
        opt, store = hook_optax(
            optax.adam(1e-2), append=False, estimator=jnp.max
        )
        params = {"w": jnp.array([1.0, 2.0, 3.0])}
        grads = {"w": jnp.array([0.1, 0.5, 0.3])}
        _step(opt, params, grads)
        # small array (dim <= 5) -> no aggregation, estimator not applied
        assert "∇w" in store


class TestHookOptaxHighDim:
    """Tests for gradient aggregation on arrays with large dimensions."""

    def test_aggregates_large_dims(self):
        """Dims > 5 should be aggregated by mean, yielding a scalar."""
        opt, store = hook_optax(optax.sgd(1e-2), append=False)
        params = {"w": jnp.ones((10, 10))}
        grads = {"w": jnp.ones((10, 10)) * 0.1}
        _step(opt, params, grads)
        # dims > 5 get aggregated by mean
        val = store["∇w"]
        # should be a scalar (both axes aggregated)
        assert val.shape == ()
