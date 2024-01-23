import pytest
from jax.tree_util import tree_map, tree_all
from numpyrotils.optimiser import (
    CycledScheduleSpec,
    schedule_if_specs,
)

ALLOWED_LEARNING_RATES = [
    0.1,
    CycledScheduleSpec(start=0.1, end=1e-2, ncycle=3, ntotal=1500),
    {"default": 0.1, "alpha": 1e-2},
    {
        "default": CycledScheduleSpec(start=0.1, end=1e-2, ncycle=3, ntotal=1500),
        "alpha": CycledScheduleSpec(start=1e-2, end=1e-3, ncycle=3, ntotal=1500),
    },
]


@pytest.mark.parametrize("learning_rate_inp", ALLOWED_LEARNING_RATES)
def test_scheduling(learning_rate_inp):
    result = schedule_if_specs(learning_rate_inp)
    passed = tree_map(lambda x: isinstance(x, float) or callable(x), result)
    assert tree_all(passed)
