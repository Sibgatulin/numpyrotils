import os
import warnings
from collections import defaultdict
from collections.abc import Callable
from pprint import pprint
from typing import Literal, overload

import jax.numpy as jnp
import optax
from jax import jit, lax, pure_callback, random, tree_util, vmap
from jax.scipy.special import i0e
from jaxtyping import Array, ArrayLike, Bool
from numpyro.distributions import Distribution, constraints
from numpyro.distributions.util import is_prng_key, promote_shapes, validate_sample
from numpyro.infer import ELBO, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from numpyro.infer.svi import SVIRunResult
from numpyro.infer.util import log_density
from tqdm import trange

from numpyrotils.optimiser import ScalarOrScheduleOrSpec, generate_optimiser

__all__ = ["run_with_callbacks", "run_svi", "Ricean", "compute_importance_weights"]

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
    nan_policy: NAN_POLICY = "exit",
    callbacks: list[tuple[int, Callable]] | None = None,
    callback_teardown: list[Callable] | None = None,
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
    callbacks = callbacks or []
    callback_teardown = callback_teardown or []
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
            nan_policy == "propagate"
            or jnp.isfinite(loss)
            and state_is_fully_finite(svi, _state).item()
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
        jnp.logical_and,
        tree_util.tree_map(lambda x: jnp.isfinite(x).all(), tree),
    )


def state_is_fully_finite(svi, svi_state) -> Bool[Array, ""]:
    return tree_is_finite(svi.get_params(svi_state))


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
    nan_policy: NAN_POLICY = "exit",
    callbacks: list[tuple[int, Callable]] | None = None,
    callback_teardown: list[Callable] | None = None,
    wandb_proj=None,
    store_grads=True,
    **model_kws,
) -> tuple[SVIRunResult, SVI]:
    """Convenience function to init and run SVI.

    Offers a convenience to set up different learning rate for different parameters.
    Additionally, orchestrates Weights and Biases callbacks for convenience.
    """
    callbacks = list(callbacks) if callbacks else []
    callback_teardown = list(callback_teardown) if callback_teardown else []

    if rng is None:
        warnings.warn("No rng key provided, using PRNGKey(0).", stacklevel=2)
        rng = random.PRNGKey(0)

    opt = generate_optimiser(learning_rate)
    if store_grads:
        opt, grad_store = hook_optax(opt, append=False)
    else:
        grad_store = {}

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
                    max(1, int(num_steps // 100)),
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
            rng,
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


def compute_importance_weights(
    model,
    guide,
    samples: dict,
    params: dict = {},
    model_args=(),
    model_kwargs={},
):
    """Given the model and the guide, return the log of the IS weights.

    For each sample θ_s from the guide (proposal distribution), estimates
    the log of the joint probability log p(θ_s, y), where θ are the samples
    and y is the data, along side with log q(θ_s), then returns `log p - log q`.

    Parameters
    ----------
    model: callable
        Python callable with `numpyro.sample` primitives
    guide: callable
        Python callable with `numpyro.sample` and `numpyro.param` primitives.
        If an autoguide object, must be seeded.
    samples: dict
        Samples *from the guide*. For example, for autoguides, that expose
        `sample_posterior` method, samples can be generated by calling
        `guide.sample_posterior(rng_key, params, sample_shape=(num_sample,))`.
    params: dict, optional
        Values of the guide parameters to be substituted when evaluating the
        guide's probability density.
    model_args, model_kwargs: tuple and dict respectively
        Arguments to be passed to both the model and the guide when evaluating
        the probability density.

    Returns
    -------
    log_weights: Float[Array, " num_sample"]
        Array of importance sampling weights for each passed sample.
        Can be further passed to `arviz.stats.stats.psislw`, but must be converted
        to a numpy array first. (Then optionally to an `xr.DataArray` with an axis
        named `__sample__`.)

    """
    log_density_model = vmap(
        lambda sample: log_density(model, model_args, model_kwargs, params=sample)[0],
        in_axes=0,  # map over samples
    )(samples)

    # consider checking if the guide is seeded and if not,
    # at least notify the user if not seed it yourself
    log_density_guide = vmap(
        lambda sample: log_density(
            guide, model_args, model_kwargs, params=sample | params
        )[0],
        in_axes=0,  # map over samples
    )(samples)
    return log_density_model - log_density_guide


# def _gpdfit(ary: jnp.ndarray) -> tuple:
#     """Estimate the GPD shape (k) and scale (sigma) parameters."""
#     prior_bs = 3.0
#     prior_k = 10.0
#     n = ary.shape[0]
#     m_est = 30 + int(np.sqrt(n))

#     # build b_ary
#     i = jnp.arange(1, m_est + 1, dtype=float)
#     b_ary = 1 - jnp.sqrt(m_est / (i - 0.5))
#     b_ary = b_ary / (prior_bs * ary[jnp.floor(n / 4 + 0.5).astype(int) - 1])
#     b_ary = b_ary + 1 / ary[-1]

#     # compute k_ary
#     # b_ary[:, None] * ary -> broadcast shape (m_est, n)
#     k_ary = jnp.log1p(-b_ary[:, None] * ary).mean(axis=1)
#     len_scale = n * (jnp.log(-(b_ary / k_ary)) - k_ary - 1)
#     # weights
#     weights = jnp.exp(-(len_scale - len_scale[:, None]))
#     weights = weights.sum(axis=1)
#     weights = 1 / weights

#     # prune near-zero weights
#     eps = jnp.finfo(float).eps
#     mask = weights >= 10 * eps
#     weights = weights[mask]
#     b_ary = b_ary[mask]

#     # normalize
#     weights = weights / jnp.sum(weights)

#     # posterior mean for b
#     b_post = jnp.sum(b_ary * weights)
#     k_post = jnp.log1p(-b_post * ary).mean()
#     sigma = -k_post / b_post
#     k_post = (n * k_post + prior_k * 0.5) / (n + prior_k)
#     return k_post, sigma


# def _gpinv(probs: jnp.ndarray, kappa: float, sigma: float) -> jnp.ndarray:
#     """Inverse CDF of the Generalized Pareto distribution."""
#     x = jnp.full_like(probs, jnp.nan)
#     if sigma <= 0:
#         return x
#     # valid slice
#     ok = (probs > 0) & (probs < 1)

#     def _compute(val):
#         p = val
#         return lax.cond(
#             jnp.abs(kappa) < jnp.finfo(float).eps,
#             lambda pr: -jnp.log1p(-pr),
#             lambda pr: jnp.expm1(-kappa * jnp.log1p(-pr)) / kappa,
#             p,
#         )
#     # compute for ok
#     x = x.at[ok].set(_compute(probs[ok]) * sigma)
#     # handle boundaries
#     x = x.at[probs == 0].set(0.0)
#     x = x.at[probs == 1].set(jnp.where(kappa >= 0, jnp.inf, -sigma / kappa))
#     return x


# def _psislw(log_weights: jnp.ndarray,
#             cutoff_ind: int,
#             cutoffmin: float = -jnp.inf,
#             normalize: bool = True) -> tuple:
#     """Pareto smoothed importance sampling for 1D log-weights."""
#     x = log_weights.astype(float)
#     max_x = jnp.max(x)
#     x = x - max_x

#     # sort
#     ind = jnp.argsort(x)
#     # cutoff
#     xcutoff = jnp.maximum(x[ind[cutoff_ind]], cutoffmin)
#     exp_xcut = jnp.exp(xcutoff)

#     tail_mask = x > xcutoff
#     tailinds = jnp.where(tail_mask)[0]
#     tail_len = tailinds.shape[0]

#     def no_smooth():
#         return jnp.inf, x

#     def do_smooth():
#         x_tail = x[tailinds]
#         si = jnp.argsort(x_tail)
#         # fit
#         k, sigma = _gpdfit(jnp.exp(x_tail) - exp_xcut)
#         def smooth_branch(args):
#             k_val, sigma_val = args
#             # ordered probs
#             sti = jnp.arange(0.5, tail_len) / tail_len
#             st = _gpinv(sti, k_val, sigma_val)
#             st = jnp.log(st + exp_xcut)
#             # insert
#             x_new = x.at[tailinds[si]].set(st)
#             x_new = jnp.where(x_new > 0, 0.0, x_new)
#             return k_val, x_new
# n
#         return lax.cond(jnp.isfinite(k), smooth_branch, lambda _: (k, x), (k, sigma))

#     k, x = lax.cond(tail_len <= 4, lambda: no_smooth(), do_smooth, operand=None)

#     # renormalize or re-add max
#     if normalize:
#         # logsumexp
#         mx = jnp.max(x)
#         lse = mx + jnp.log(jnp.sum(jnp.exp(x - mx)))
#         x = x - lse
#     else:
#         x = x + max_x
#     return x, k
