"""Every trajmem module imports clean."""
import importlib

import pytest

MODULES = [
    "trajmem",
    "trajmem.data",
    "trajmem.trajectories",
    "trajmem.simulate",
    "trajmem.frontend",
    "trajmem.model",
    "trajmem.baseline",
    "trajmem.metrics",
    "trajmem.experiment",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)
