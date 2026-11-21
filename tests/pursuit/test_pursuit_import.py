"""The pursuit package imports clean, alongside trajmem."""
import importlib


def test_pursuit_imports():
    importlib.import_module("pursuit")
    importlib.import_module("trajmem.experiment")
