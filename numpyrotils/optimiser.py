from collections import namedtuple

import optax
import jax.numpy as jnp
from flax import traverse_util
from loguru import logger
from jax.tree_util import tree_map

CycledScheduleSpec = namedtuple("ScheduleSpec", ["start", "end", "ncycle", "ntotal"])
ScalarOrScheduleOrSpec = CycledScheduleSpec | optax.ScalarOrSchedule


def flattened_traversal(fn):
    def mask(tree):
        flat = traverse_util.flatten_dict(tree)
        return traverse_util.unflatten_dict({k: fn(k, v) for k, v in flat.items()})

    return mask


def generate_cycled_schedule(spec: CycledScheduleSpec) -> optax.Schedule:
    trans_steps = int(spec.ntotal / spec.ncycle)
    max_learning_rates = jnp.logspace(
        jnp.log10(spec.start), jnp.log10(spec.end), spec.ncycle
    )
    return optax.join_schedules(
        [optax.cosine_onecycle_schedule(trans_steps, lr) for lr in max_learning_rates],
        jnp.arange(1, spec.ncycle) * trans_steps,
    )


def schedule_if_spec(lr: ScalarOrScheduleOrSpec) -> optax.ScalarOrSchedule:
    """For a single allowed input, setup a Schedule if specs given else return as is."""
    if isinstance(lr, CycledScheduleSpec):
        return generate_cycled_schedule(lr)
    return lr


def schedule_if_specs(
    learning_rate: ScalarOrScheduleOrSpec | dict[str, ScalarOrScheduleOrSpec]
) -> optax.ScalarOrSchedule | dict[str, optax.ScalarOrSchedule]:
    """For all allowed inputs, setup schedules in place of specs."""
    return tree_map(
        schedule_if_spec,
        learning_rate,
        is_leaf=lambda x: isinstance(x, (CycledScheduleSpec, float)) or callable(x),
    )


def generate_optimiser(
    learning_rate: ScalarOrScheduleOrSpec | dict[str, ScalarOrScheduleOrSpec]
):
    learning_rate = schedule_if_specs(learning_rate)
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
        return optax.multi_transform(
            {k: optax.adabelief(v) for k, v in learning_rate.items()},
            label_fn,
        )
    return optax.adabelief(learning_rate)
