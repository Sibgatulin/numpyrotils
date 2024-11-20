import os
from collections import defaultdict
from collections.abc import Callable
from pprint import pprint
from typing import Literal, overload

import jax.numpy as jnp
import optax
from jax import jit, lax, pure_callback, random, tree_util
from jax.scipy.special import i0e
from jaxtyping import Array, ArrayLike, Bool
from numpyro.distributions import Distribution, constraints
from numpyro.distributions.util import is_prng_key, promote_shapes, validate_sample
from numpyro.infer import ELBO, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from numpyro.infer.svi import SVIRunResult
from tqdm import trange

from numpyrotils.optimiser import ScalarOrScheduleOrSpec, generate_optimiser

__all__ = ["run_with_callbacks", "run_svi", "Ricean"]

NAN_POLICY = Literal["exit", "propagate", "stable_update"]


def frac_of_nans_in_tree(tree):
    """Used to calculate the percentage of NaN in the variational parameters.

    Helpful to identify the culprits when SVI stumbles over NaNs.
    """
    return tree_util.tree_map(lambda x: jnp.isnan(x).mean(), tree)


def run_with_callbacks(
    svi,
    rng_key,
    num_steps,
    nan_policy="exit",
    callbacks: list[tuple[int, Callable]] = [],
    callback_teardown: list[Callable] = [],
    **kwargs,
) -> SVIRunResult:
    """
    Version of numpyro.infer.SVI.run with callbacks and NaN-handling.

    Specified callbacks are executed periodically in the optimisation loop.
    Callback teardown functions (if any) are called at the end of the loop.
    If any NaNs appear after a single update step, the loop can be stopped
    returning the latest all-finite state (missing in SVI.run).

    Parameters
    ----------
    nan_policy: 'exit', 'propagate', or 'stable_update'
        Learning rate that is too high can introduce NaNs in the course of optimisation.
        By default, numpyro.infer.run(stable_update=False) continues updating,
        effectively drowning everything in NaNs (nan_policy='propagate').
        Setting stable_update=True (nan_policy='stable_update') forces optimisation
        to keep the last non-NaN parameter value. This, while rescues the optimisation
        up to the gradient explosion, halts the subsequent course of optimisation,
        at least when constant learning rate is used, as in that case there is no hope
        to recover the gradient, wasting the rest of the calculation.
        So far I have no explicit* experience with decaying learning rate, thus the
        default NaN policy is to exit once a single NaN is found
        * except for whatever numpyro.optim.ClippedAdam is doing, which neither seems
        apparent for me from the documentation, nor seem to help to recover from NaNs
        in my experience)
    """
    print(f"Provided {len(callbacks)} callbacks and {len(callback_teardown)} teardowns")
    _losses = []

    state = svi.init(rng_key, **kwargs)

    stable_update = nan_policy == "stable_update"

    @jit  # this makes so much difference!
    def body_fn(state):
        if stable_update:  # inside or outside does not matter much
            return svi.stable_update(state, **kwargs)
        else:
            return svi.update(state, **kwargs)

    for i in trange(num_steps):
        _state, loss = body_fn(state)
        if (
            jnp.isfinite(loss)
            and state_is_fully_finite(svi, _state).item()
            or nan_policy == "propagate"
        ):
            state = _state
            _losses.append(loss)
        else:
            print(f"Loss became NaN at iter={i}")
            nan_fracs = {
                k: frac_of_nans_in_tree(v) for k, v in svi.get_params(_state).items()
            }
            nan_fracs = {k: v for k, v in nan_fracs.items() if v > 0}
            if nan_fracs:
                print("Proportion of NaNs found:")
                for k, v in nan_fracs.items():
                    print(f"{k}: {v:.1%}")
            else:
                pprint(svi.get_params(_state))
            print("Keeping the last iteration with finite loss")
            break

        for cb_period, cb in callbacks:
            if i % cb_period == 0:
                cb(svi, state, loss)

    for cb in callback_teardown:
        cb(svi, state)
    return SVIRunResult(svi.get_params(state), state, jnp.stack(_losses))


@jit
def tree_is_finite(tree):
    return tree_util.tree_reduce(
        lambda x, y: x * y,
        tree_util.tree_map(lambda x: jnp.isfinite(x).all(), tree),
    )


def state_is_fully_finite(svi, svi_state) -> Bool[Array, ""]:
    return tree_is_finite(svi.get_params(svi_state))


def flattened_traversal(fn):
    from flax import traverse_util

    def mask(tree):
        flat = traverse_util.flatten_dict(tree)
        return traverse_util.unflatten_dict({k: fn(k, v) for k, v in flat.items()})

    return mask


@overload
def hook_optax(
    optimizer: optax.GradientTransformation,
    append: Literal[True],
    estimator: Callable[[ArrayLike, tuple[int, ...]], ArrayLike] = jnp.mean,
) -> tuple[optax.GradientTransformation, defaultdict]: ...


@overload
def hook_optax(
    optimizer: optax.GradientTransformation,
    append: Literal[False],
    estimator: Callable[[ArrayLike, tuple[int, ...]], ArrayLike] = jnp.mean,
) -> tuple[optax.GradientTransformation, dict]: ...


def hook_optax(
    optimizer: optax.GradientTransformation,
    append=False,
    estimator: Callable[[ArrayLike, tuple[int, ...]], ArrayLike] = jnp.mean,
) -> tuple[optax.GradientTransformation, dict | defaultdict]:
    """Generate optimizer that stores gradient estimates.

    Parameters
    ----------
    optimizer : optax gradient tranformation
    estimator : callable[[ArrayLike, tuple[int,...]], float]
        By default: jnp.mean, can be any other function, that is able to aggregate
        a jax array along specified `axis` (that can be a tuple), e.g. jnp.linalg.norm
    append : bool
        if True, all gradient estimates will be preserved in a defaultdict,
        else only the latest will be stored in a dict
    """
    if append:
        grad_store = defaultdict(list)
    else:
        grad_store = {}

    def agg_fn(array, fn=jnp.mean, max_allowed_dim_size=5):
        agg_axes = tuple(
            idx for idx, sz in enumerate(array.shape) if sz > max_allowed_dim_size
        )
        return fn(array, axis=agg_axes)

    def push_grad(grad):
        for name, g in grad.items():
            val = agg_fn(g, fn=estimator)
            if append:
                grad_store[f"∇{name}"].append(val)
            else:
                grad_store[f"∇{name}"] = val
        return grad

    def update_fn(grads, state, params=None):
        grads = pure_callback(push_grad, grads, grads)
        return optimizer.update(grads, state, params=params)

    return optax.GradientTransformation(optimizer.init, update_fn), grad_store


def run_svi(
    model,
    guide=None,
    num_steps: int = 1_000,
    learning_rate: ScalarOrScheduleOrSpec | dict[str, ScalarOrScheduleOrSpec] = 1e-1,
    rng=None,
    loss: ELBO = Trace_ELBO(),
    nan_policy="exit",
    callbacks: list[tuple[int, Callable]] = [],
    callback_teardown: list[Callable] = [],
    wandb_proj=None,
    **model_kws,
) -> tuple[SVIRunResult, SVI]:
    """Convenience function to init and run SVI.

    Offers a convenience to set up different learning rate for different parameters.
    Additionally, orchestrates Weights and Biases callbacks for convenience.
    """

    opt = generate_optimiser(learning_rate)
    opt, grad_store = hook_optax(opt, append=False)

    if wandb_proj or (wandb_proj := os.getenv("WANDB_PROJECT")):
        try:
            import wandb
        except ImportError as err:
            raise ImportError("wandb_project specified but no wandb found") from err

        from numpyrotils._wandb import wandb_callback, wandb_teardown

        if not callback_teardown:
            callback_teardown = [wandb_teardown]
        if not callbacks:
            callbacks = [
                (
                    max(1, int(num_steps % 100)),
                    lambda *args: wandb_callback(*args, payload=grad_store),
                )
            ]

        wandb.init(
            project=wandb_proj,
            config={
                "learning_rate": learning_rate,
                "num_steps": num_steps,
                "guide": str(guide),
            },
            dir="./.wandb",  # default ./wandb confuses LSP's import resolution
        )

    svi = SVI(model, guide or AutoDelta(model), opt, loss)
    return (
        run_with_callbacks(
            svi,
            rng or random.PRNGKey(0),
            num_steps=num_steps,
            nan_policy=nan_policy,
            callbacks=callbacks,
            callback_teardown=callback_teardown,
            **model_kws,
        ),
        svi,
    )


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
