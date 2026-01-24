"""CLI模块"""

from onnxsplit.cli.parser import CliOptions, app
from onnxsplit.cli.runner import run_analyze, run_split

__all__ = [
    "app",
    "CliOptions",
    "run_split",
    "run_analyze",
]
