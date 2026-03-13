"""Custom numpyro distributions."""

import jax.numpy as jnp
from jax import lax, random
from jax.scipy.special import i0e
from numpyro.distributions import Distribution, constraints
from numpyro.distributions.util import is_prng_key, promote_shapes, validate_sample

__all__ = ["Ricean"]


class Ricean(Distribution):
    arg_constraints = {"loc": constraints.positive, "scale": constraints.positive}
    support = constraints.positive
    reparametrized_params = ["loc", "scale"]

    def __init__(self, loc=1.25, scale=1.0, *, validate_args=None):
        self.loc, self.scale = promote_shapes(loc, scale)
        batch_shape = lax.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        eps = random.normal(
            key, shape=(2,) + sample_shape + self.batch_shape + self.event_shape
        )
        return jnp.linalg.norm(self.loc / 2**0.5 + eps * self.scale, axis=0)

    @validate_sample
    def log_prob(self, value):
        # Adopted from scipy.stats.rice
        # rice.pdf(x, b) = x * exp(-(x**2+b**2)/2) * I[0](x*b)
        #
        # We use (x**2 + b**2)/2 = ((x-b)**2)/2 + xb.
        # The factor of np.exp(-xb) is then included in the i0e function
        # in place of the modified Bessel function, i0, improving
        # numerical stability for large values of xb.
        s2 = self.scale**2
        return (
            jnp.log(value)
            - jnp.log(s2)
            - 0.5 * (value - self.loc) ** 2 / s2
            + jnp.log(i0e(value * self.loc / s2))
        )
