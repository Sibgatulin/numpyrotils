"""Tests for generate_optimiser.

Verifies that generate_optimiser produces a working optax
GradientTransformation for all supported learning rate formats:
scalar, schedule, CycledScheduleSpec, per-parameter dict, and
custom optimizer classes.
"""

import jax.numpy as jnp
import optax

from numpyrotils.optimiser import CycledScheduleSpec, generate_optimiser


def _can_step(opt, params):
    """Verify the optimizer can init and produce a finite update."""
    state = opt.init(params)
    grads = {k: jnp.ones_like(v) for k, v in params.items()}
    updates, new_state = opt.update(grads, state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params


class TestGenerateOptimiser:
    """Test optimizer construction for each learning rate format."""

    def test_scalar_lr(self):
        """A plain float learning rate should produce a working optimizer."""
        opt = generate_optimiser(0.01)
        params = {"x": jnp.array(1.0)}
        new_params = _can_step(opt, params)
        assert jnp.isfinite(new_params["x"])

    def test_schedule_lr(self):
        """An optax schedule should be accepted as a learning rate."""
        schedule = optax.cosine_onecycle_schedule(100, 0.01)
        opt = generate_optimiser(schedule)
        params = {"x": jnp.array(1.0)}
        new_params = _can_step(opt, params)
        assert jnp.isfinite(new_params["x"])

    def test_cycled_spec(self):
        """A CycledScheduleSpec should be converted to a schedule."""
        spec = CycledScheduleSpec(start=0.1, end=1e-3, ncycle=2, ntotal=100)
        opt = generate_optimiser(spec)
        params = {"x": jnp.array(1.0)}
        new_params = _can_step(opt, params)
        assert jnp.isfinite(new_params["x"])

    def test_dict_with_default(self):
        """A dict with 'default' key should route unlisted params correctly."""
        opt = generate_optimiser({"default": 0.01, "alpha": 0.001})
        params = {"alpha": jnp.array(1.0), "beta": jnp.array(2.0)}
        new_params = _can_step(opt, params)
        assert jnp.isfinite(new_params["alpha"])
        assert jnp.isfinite(new_params["beta"])

    def test_custom_opt_cls(self):
        """A custom optimizer class (optax.adam) should be respected."""
        opt = generate_optimiser(0.01, opt_cls=optax.adam)
        params = {"x": jnp.array(1.0)}
        new_params = _can_step(opt, params)
        assert jnp.isfinite(new_params["x"])
