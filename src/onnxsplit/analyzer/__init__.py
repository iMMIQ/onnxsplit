"""ONNX模型分析模块

提供ONNX模型解析、形状推断和依赖关系图构建功能。
"""

from onnxsplit.analyzer.dependency import (
    DependencyEdge,
    DependencyGraph,
    DependencyNode,
)
from onnxsplit.analyzer.model import ModelAnalyzer
from onnxsplit.analyzer.operator import OperatorInfo
from onnxsplit.analyzer.tensor import TensorMetadata, dtype_to_bytes

__all__ = [
    # Tensor
    "TensorMetadata",
    "dtype_to_bytes",
    # Operator
    "OperatorInfo",
    # Model
    "ModelAnalyzer",
    # Dependency
    "DependencyGraph",
    "DependencyNode",
    "DependencyEdge",
]
