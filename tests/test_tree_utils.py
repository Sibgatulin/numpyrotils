"""Tests for tree_is_finite and frac_of_nans_in_tree.

These are small utility functions used internally by the SVI
training loop for NaN detection and diagnostics.
"""

import jax.numpy as jnp

from numpyrotils.svi import frac_of_nans_in_tree, tree_is_finite


class TestTreeIsFinite:
    """Verify tree_is_finite detects NaN and Inf in pytrees."""

    def test_all_finite(self):
        """A tree with only finite values should return True."""
        tree = {"a": jnp.array([1.0, 2.0]), "b": jnp.array(3.0)}
        assert tree_is_finite(tree)

    def test_has_nan(self):
        """A single NaN in any leaf should make the tree non-finite."""
        tree = {"a": jnp.array([1.0, jnp.nan]), "b": jnp.array(3.0)}
        assert not tree_is_finite(tree)

    def test_has_inf(self):
        """A single Inf in any leaf should make the tree non-finite."""
        tree = {"a": jnp.array([1.0, 2.0]), "b": jnp.array(jnp.inf)}
        assert not tree_is_finite(tree)


class TestFracOfNansInTree:
    """Verify frac_of_nans_in_tree computes per-leaf NaN fractions."""

    def test_no_nans(self):
        """All-finite leaf should report 0% NaN."""
        tree = {"x": jnp.array([1.0, 2.0, 3.0])}
        result = frac_of_nans_in_tree(tree)
        assert result["x"] == 0.0

    def test_some_nans(self):
        """Two NaNs out of four elements should report 50%."""
        tree = {"x": jnp.array([1.0, jnp.nan, 3.0, jnp.nan])}
        result = frac_of_nans_in_tree(tree)
        assert jnp.isclose(result["x"], 0.5)

    def test_all_nans(self):
        """All-NaN leaf should report 100%."""
        tree = {"x": jnp.array([jnp.nan, jnp.nan])}
        result = frac_of_nans_in_tree(tree)
        assert result["x"] == 1.0
