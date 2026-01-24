"""ONNX模型切分工具

通过算子复制方式切分ONNX模型，降低内存峰值。
"""

__version__ = "0.1.0"

from onnxsplit.analyzer import ModelAnalyzer, OperatorInfo, TensorMetadata
from onnxsplit.config import SplitConfig, load_config
from onnxsplit.memory import (
    AutoSplitAdjuster,
    dtype_to_bytes,
    estimate_tensor_memory,
)
from onnxsplit.splitter import SplitPlan, SplitPlanner
from onnxsplit.transform import GraphTransformer

__all__ = [
    # Version
    "__version__",
    # Analyzer
    "ModelAnalyzer",
    "OperatorInfo",
    "TensorMetadata",
    # Config
    "load_config",
    "SplitConfig",
    # Memory
    "AutoSplitAdjuster",
    "dtype_to_bytes",
    "estimate_tensor_memory",
    # Splitter
    "SplitPlanner",
    "SplitPlan",
    # Transform
    "GraphTransformer",
]
