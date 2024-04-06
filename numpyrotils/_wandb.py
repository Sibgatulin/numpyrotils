import jax.numpy as jnp
import wandb

from prob_mri.utils.prob import clean_up_param_name


def prepare_to_log(svi, state) -> dict:
    artifacts = {}
    for k, v in svi.get_params(state).items():
        if jnp.size(v) == 1:
            artifacts[clean_up_param_name(k)] = v
        elif jnp.ndim(v) == 1:
            artifacts[clean_up_param_name(k)] = dict(enumerate(v.tolist()))
        elif jnp.ndim(v) > 1 and jnp.size(v) < 20:
            artifacts[clean_up_param_name(k)] = dict(enumerate(v.flatten().tolist()))
    return artifacts


def wandb_callback(svi, state, loss):
    wandb.log({"loss": loss} | prepare_to_log(svi, state))


def wandb_teardown(*_):
    wandb.finish()
