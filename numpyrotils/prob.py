"""Backward-compatible re-export of all public symbols.

The implementation has moved to dedicated submodules:
- ``numpyrotils.svi`` — training loop, fit_svi, hook_optax
- ``numpyrotils.distributions`` — Ricean
- ``numpyrotils.diagnostics`` — compute_importance_weights

Importing from ``numpyrotils.prob`` still works but emits a
DeprecationWarning.
"""

import warnings as _warnings

_warnings.warn(
    "Importing from numpyrotils.prob is deprecated. "
    "Use numpyrotils.svi, numpyrotils.distributions, "
    "or numpyrotils.diagnostics instead.",
    DeprecationWarning,
    stacklevel=2,
)

from numpyrotils.diagnostics import *  # noqa: F401,F403
from numpyrotils.diagnostics import __all__ as _diag_all
from numpyrotils.distributions import *  # noqa: F401,F403
from numpyrotils.distributions import __all__ as _dist_all
from numpyrotils.svi import *  # noqa: F401,F403
from numpyrotils.svi import __all__ as _svi_all

__all__ = _svi_all + _dist_all + _diag_all
