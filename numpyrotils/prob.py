import os
from collections.abc import Callable
from typing import Literal

import jax.numpy as jnp
import optax
from jax import jit, random, tree_util
from loguru import logger
from numpyro.infer import ELBO, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from numpyro.infer.svi import SVIRunResult
from numpyro.optim import Adam
from tqdm import trange

__all__ = ["run_with_callbacks", "svi"]

NAN_POLICY = Literal["exit", "propagate", "stable_update"]


def percentage_of_nans_in_tree(tree):
    """Used to calculate the percentage of NaN in the variational parameters.

    Helpful to identify the culprits when SVI stumbles over NaNs.
    """
    return tree_util.tree_map(lambda x: (jnp.isnan(x).mean() * 100).round(), tree)


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
        if jnp.isfinite(loss) or nan_policy == "propagate":
            state = _state
            _losses.append(loss)
        else:
            print(f"Loss became NaN at iter={i}. Proportion of NaNs found:")
            for k, v in svi.get_params(_state).items():
                print(f"{k}: {percentage_of_nans_in_tree(v)}%")
            print("Keeping the last iteration with finite loss")
            break

        for cb_period, cb in callbacks:
            if i % cb_period == 0:
                cb(svi, state, loss)

    for cb in callback_teardown:
        cb(svi, state)
    return SVIRunResult(svi.get_params(state), state, jnp.stack(_losses))


def flattened_traversal(fn):
    from flax import traverse_util

    def mask(tree):
        flat = traverse_util.flatten_dict(tree)
        return traverse_util.unflatten_dict({k: fn(k, v) for k, v in flat.items()})

    return mask


def svi(
    model,
    guide=None,
    num_steps: int = 1_000,
    learning_rate: float | dict[str, float] = 1e-1,
    rng=None,
    loss: ELBO = Trace_ELBO(),
    nan_policy="exit",
    callbacks: list[tuple[int, Callable]] = [],
    callback_teardown: list[Callable] = [],
    wandb_proj=None,
    **model_kws,
) -> SVIRunResult:
    """Convenience function to init and run SVI.

    Offers a convenience to set up different learning rate for different parameters.
    Additionally, orchestrates Weights and Biases callbacks for convenience.
    """
    if wandb_proj or (wandb_proj := os.getenv("WANDB_PROJECT")):
        if callbacks and callback_teardown:
            raise NotImplementedError(
                "Passing wandb_proj and callbacks simultaneously is not supported yet. "
                "Call wandb.init() externally"
            )
        elif not callbacks and not callback_teardown:
            try:
                import wandb
            except ImportError as err:
                raise ImportError("wandb_project specified but no wandb found") from err

            from numpyrotils._wandb import wandb_callback, wandb_teardown

            wandb.init(
                project=wandb_proj,
                config={
                    "learning_rate": learning_rate,
                    "num_steps": num_steps,
                    "guide": str(guide),
                },
            )
            callbacks = [(max(1, int(num_steps % 100)), wandb_callback)]
            callback_teardown = [wandb_teardown]
        else:
            raise ValueError(f"Inconsistent {callbacks=} and {callback_teardown=}")

    if isinstance(learning_rate, dict):
        logger.info("Recieved learning_rate={}", learning_rate)
        if "default" not in learning_rate:
            logger.warning(
                "When learning_rate dict does not contain 'default' key, "
                "it must contain ALL parameter names explicitly"
            )
            # TODO: trace the guide and compare the parameters to the specified
            # dictionary more intelligently and completely
        label_fn = flattened_traversal(
            lambda path, _: path[0] if path[0] in learning_rate else "default"
        )
        opt = optax.multi_transform(
            {k: optax.adabelief(v) for k, v in learning_rate.items()},
            label_fn,
        )
    else:
        opt = Adam(learning_rate)

    return run_with_callbacks(
        SVI(model, guide or AutoDelta(model), opt, loss),
        rng or random.PRNGKey(0),
        num_steps=num_steps,
        nan_policy=nan_policy,
        callbacks=callbacks,
        callback_teardown=callback_teardown,
        **model_kws,
    )
