"""qwals — WALS-style linguistic distance calculator."""
from .calculator import QwalsCalculator
from ._presets import TASK_FEATURES, TASKS

__all__ = ["QwalsCalculator", "TASK_FEATURES", "TASKS"]
__version__ = "0.8.0"
