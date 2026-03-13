# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Ricean vs Normal likelihood for magnitude data
#
# Demonstrates inference on a decaying exponential observed through
# Ricean noise (magnitude of complex Gaussian).  Two models are compared:
#
# 1. **Ricean likelihood** -- uses `numpyrotils.Ricean` directly.
# 2. **Normal likelihood** -- models real and imaginary parts as independent
#    Gaussians and takes the norm, recovering the Rice distribution implicitly.
#
# Both are fit with SVI (AutoDelta guide, 100k steps) and the MAP
# estimates are plotted against the ground truth.

# %% [markdown]
# ## Data generation

# %%
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random

from numpyrotils.distributions import Ricean


def predict(x, args):
    """Decaying exponential: a * exp(-b * x)."""
    return args[0] * jnp.exp(-x * args[1])


s = 0.1
x = jnp.linspace(1, 10)
p = jnp.array([2.0, 1.0 / 2.0])

pred_magn = predict(x, p)
noise = random.normal(random.PRNGKey(0), (2,) + x.shape) * s
signal_complex = pred_magn + noise[0] + 1j * noise[1]
signal_magn = abs(signal_complex)

# %%
# For a test, mask out a few data points and see how numpyro complains about 0.0
# violating Ricean.support == constraints.positive
#
# idx = random.randint(
#     random.PRNGKey(0),
#     shape=(int(len(x) / 7),),
#     minval=int(len(x) * 0.6),
#     maxval=len(x),
# )
# signal_magn_masked = signal_magn.at[idx].set(0.0)

# %%
plt.plot(x, pred_magn, label="true signal")
plt.scatter(x, signal_magn, label="observed magnitude")
plt.title("Synthetic data: decaying exponential with Ricean noise")
plt.xlabel("x")
plt.ylabel("magnitude")
plt.legend()
plt.grid()
plt.show()

# %% [markdown]
# ## Ricean likelihood

# %%
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer import autoguide as ag
from numpyro.optim import Adam


def model_rice(x):
    """Decaying exponential with Ricean observation noise."""
    s = numpyro.sample("s", dist.Exponential(20.0))
    a = numpyro.sample("a", dist.TruncatedNormal(1.0, 0.3, low=0.0))
    b = numpyro.sample("b", dist.TruncatedNormal(1.0, 0.5, low=0.0))
    loc = a * jnp.exp(-x * b)
    numpyro.sample("obs", Ricean(loc, s).to_event(1))


# %%
samples_prior = Predictive(model_rice, num_samples=75)(random.PRNGKey(2), x)

# %%
fig, axes_dict = plt.subplot_mosaic(
    """ABS
       XXX
    """,
    figsize=(12, 8),
)
for name in ["a", "b", "s"]:
    axes_dict[name.upper()].hist(samples_prior[name])
    axes_dict[name.upper()].set_title(f"Prior: {name}")
axes_dict["X"].plot(x, samples_prior["obs"].T, color="grey", alpha=0.5, zorder=1)
axes_dict["X"].scatter(x, signal_magn, c="k", zorder=2)
axes_dict["X"].set_title("Ricean model: prior predictive")
axes_dict["X"].set_xlabel("x")
axes_dict["X"].set_ylabel("magnitude")
fig.suptitle("Ricean likelihood -- prior predictive check")
fig.tight_layout()
plt.show()

# %%
model_observed = handlers.condition(model_rice, {"obs": signal_magn})
svi = SVI(model_observed, ag.AutoDelta(model_observed), Adam(1e-2), Trace_ELBO())
result = svi.run(random.PRNGKey(1), num_steps=100_000, x=x)
print(result.params)

# %%
posterior_map = result.params["a_auto_loc"] * jnp.exp(-x * result.params["b_auto_loc"])

# %%
plt.scatter(x, signal_magn, zorder=2, label="observed")
plt.plot(x, posterior_map, c="k", zorder=3, label="MAP estimate")
plt.plot(x, pred_magn, c="C0", zorder=3, label="ground truth")
plt.title("Ricean likelihood -- MAP fit vs ground truth")
plt.xlabel("x")
plt.ylabel("magnitude")
plt.legend()
plt.show()

# %% [markdown]
# ## Normal likelihood

# %%
def model_normal(x):
    """Decaying exponential with 2D Gaussian noise (implicit Rice via ||obs||)."""
    s = numpyro.sample("s", dist.Exponential(20.0))
    a = numpyro.sample("a", dist.TruncatedNormal(1.0, 0.3, low=0.0))
    b = numpyro.sample("b", dist.TruncatedNormal(1.0, 0.5, low=0.0))
    loc = a * jnp.exp(-x * b)
    loc_2d = jnp.stack([loc, jnp.zeros_like(loc)], axis=-2)

    with numpyro.plate("part", size=2, dim=-2):
        with numpyro.plate("x", size=len(x), dim=-1):
            numpyro.sample("obs", dist.Normal(loc_2d, s))


# %%
samples_prior = Predictive(model_normal, num_samples=75)(random.PRNGKey(2), x)
samples_prior_obs = jnp.linalg.norm(samples_prior["obs"], axis=-2)

# %%
fig, axes_dict = plt.subplot_mosaic(
    """ABS
       XXX
    """,
    figsize=(12, 8),
)
for name in ["a", "b", "s"]:
    axes_dict[name.upper()].hist(samples_prior[name])
    axes_dict[name.upper()].set_title(f"Prior: {name}")
axes_dict["X"].plot(x, samples_prior_obs.T, color="grey", alpha=0.5, zorder=1)
axes_dict["X"].scatter(x, signal_magn, c="k", zorder=2)
axes_dict["X"].set_title("Normal model: prior predictive (||obs||)")
axes_dict["X"].set_xlabel("x")
axes_dict["X"].set_ylabel("magnitude")
fig.suptitle("Normal likelihood -- prior predictive check")
fig.tight_layout()
plt.show()

# %%
model_observed = handlers.condition(model_normal, {"obs": signal_magn})
svi = SVI(model_observed, ag.AutoDelta(model_observed), Adam(1e-2), Trace_ELBO())
result = svi.run(random.PRNGKey(1), num_steps=100_000, x=x)
print(result.params)

# %%
posterior_map = result.params["a_auto_loc"] * jnp.exp(-x * result.params["b_auto_loc"])
posterior = Predictive(model_normal, num_samples=25, guide=svi.guide)(
    random.PRNGKey(3), x
)["obs"]

# %%
plt.scatter(x, signal_magn, zorder=2, label="observed")
plt.plot(x, posterior_map, c="k", zorder=3, label="MAP estimate")
plt.plot(x, pred_magn, c="C0", zorder=3, label="ground truth")
plt.title("Normal likelihood -- MAP fit vs ground truth")
plt.xlabel("x")
plt.ylabel("magnitude")
plt.legend()
plt.show()
