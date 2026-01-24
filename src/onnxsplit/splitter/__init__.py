"""模型切分模块"""

from onnxsplit.splitter.axis_rules import (
    SplitableAxes,
    AxisAnalyzer,
    get_splitable_axes_for_op,
)

__all__ = [
    "SplitableAxes",
    "AxisAnalyzer",
    "get_splitable_axes_for_op",
]
