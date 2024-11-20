import re

import jax.numpy as jnp
import wandb


def clean_up_param_name(name: str) -> str:
    return re.sub(r"^(∇)?p_", lambda m: m.groups()[0] or "", name).replace("_auto", "")


def prepare_to_log(values: dict) -> dict:
    artifacts = {}
    for k, v in values.items():
        if jnp.size(v) == 1:
            artifacts[clean_up_param_name(k)] = v
        elif jnp.ndim(v) == 1:
            artifacts[clean_up_param_name(k)] = dict(enumerate(v.tolist()))
        elif jnp.ndim(v) > 1 and jnp.size(v) < 20:
            artifacts[clean_up_param_name(k)] = dict(enumerate(v.flatten().tolist()))
        else:
            artifacts[f"||{clean_up_param_name(k)}||_2"] = jnp.linalg.norm(v).item()
            artifacts[f"⟨{clean_up_param_name(k)}⟩"] = jnp.mean(v).item()
    return artifacts


def wandb_callback(svi, state, loss, payload={}):
    wandb.log({"loss": loss} | prepare_to_log(svi.get_params(state) | payload))


def wandb_teardown(*_):
    wandb.finish()
