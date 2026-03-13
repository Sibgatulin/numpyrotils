"""SVI training loop, convenience wrapper, and optimizer hooking."""

import os
import warnings
from collections import defaultdict
from collections.abc import Callable
from pprint import pprint
from typing import Literal, overload

import jax.numpy as jnp
import optax
from jax import jit, pure_callback, random, tree_util
from jaxtyping import Array, ArrayLike, Bool
from numpyro.infer import ELBO, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from numpyro.infer.svi import SVIRunResult
from tqdm import trange

from numpyrotils.optimiser import ScalarOrScheduleOrSpec, generate_optimiser

__all__ = [
    "run_svi_train_loop",
    "fit_svi",
    "run_svi",
    "hook_optax",
    "tree_is_finite",
    "frac_of_nans_in_tree",
]

NAN_POLICY = Literal["exit", "propagate", "stable_update"]


def frac_of_nans_in_tree(tree):
    """Used to calculate the percentage of NaN in the variational parameters.

    Helpful to identify the culprits when SVI stumbles over NaNs.
    """
    return tree_util.tree_map(lambda x: jnp.isnan(x).mean(), tree)


def run_svi_train_loop(
    svi,
    state,
    num_steps,
    nan_policy: NAN_POLICY = "exit",
    callbacks: list[tuple[int, Callable]] | None = None,
    callback_teardown: list[Callable] | None = None,
    **kwargs,
) -> SVIRunResult:
    """Run the SVI training loop with NaN handling and periodic callbacks.

    This is a pure execution function: it takes an already-initialised SVI
    state and steps it ``num_steps`` times.  Because the caller owns
    initialisation, the same function can be used to resume training from
    a checkpoint.

    Parameters
    ----------
    svi : numpyro.infer.SVI
        An already-constructed SVI object.
    state : SVIState
        Initial state, typically from ``svi.init(rng_key, ...)``.
    num_steps : int
        Number of optimisation steps.
    nan_policy : 'exit', 'propagate', or 'stable_update'
        What to do when NaNs appear in the loss or parameters.
        - ``'exit'`` (default): stop and return the last finite state.
        - ``'propagate'``: keep updating (numpyro default behaviour).
        - ``'stable_update'``: keep the last non-NaN parameters via
          ``svi.stable_update``.
    callbacks : list of (period, callable), optional
        Each callable is invoked every ``period`` steps with
        ``(svi, state, loss)``.
    callback_teardown : list of callable, optional
        Called once after the loop with ``(svi, state)``.
    **kwargs
        Forwarded to ``svi.update`` / ``svi.stable_update``.
    """
    callbacks = callbacks or []
    callback_teardown = callback_teardown or []
    print(f"Provided {len(callbacks)} callbacks and {len(callback_teardown)} teardowns")
    _losses = []

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
            and _state_is_fully_finite(svi, _state).item()
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


def _state_is_fully_finite(svi, svi_state) -> Bool[Array, ""]:
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


def fit_svi(
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
    """Construct an SVI object, initialise it, and run the training loop.

    Handles optimizer construction (with optional per-parameter learning
    rates), gradient hooking, Weights & Biases integration, and delegates
    the actual training to ``run_svi_train_loop``.

    Parameters
    ----------
    model : callable
        Numpyro model.
    guide : callable, optional
        Numpyro guide.  Defaults to ``AutoDelta(model)``.
    num_steps : int
        Number of optimisation steps.
    learning_rate : float, schedule, or dict
        A scalar, an optax schedule, a ``CycledScheduleSpec``, or a dict
        mapping parameter names to any of those for per-parameter rates.
    rng : jax PRNGKey, optional
        Random key for initialisation.  If ``None``, uses ``PRNGKey(0)``
        and emits a warning.
    loss : ELBO
        Loss function.  Defaults to ``Trace_ELBO()``.
    nan_policy : NAN_POLICY
        Forwarded to ``run_svi_train_loop``.
    callbacks, callback_teardown
        Forwarded to ``run_svi_train_loop``.
    wandb_proj : str, optional
        Weights & Biases project name.  Also picked up from the
        ``WANDB_PROJECT`` environment variable.
    store_grads : bool
        Whether to hook the optimizer to capture gradient estimates.
    **model_kws
        Keyword arguments forwarded to the model (and through to
        ``svi.init`` / ``svi.update``).
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
    state = svi.init(rng, **model_kws)
    return (
        run_svi_train_loop(
            svi,
            state,
            num_steps=num_steps,
            nan_policy=nan_policy,
            callbacks=callbacks,
            callback_teardown=callback_teardown,
            **model_kws,
        ),
        svi,
    )


def run_svi(*args, **kwargs) -> tuple[SVIRunResult, SVI]:
    """Deprecated: use ``fit_svi`` instead."""
    warnings.warn(
        "run_svi is deprecated, use fit_svi instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return fit_svi(*args, **kwargs)
