"""图变换模块

提供节点克隆、数据流重连和图变换功能。
"""

from onnxsplit.transform.executor import GraphTransformer
from onnxsplit.transform.node_clone import clone_node, generate_split_name
from onnxsplit.transform.reconnect import (
    ReconnectConnection,
    ReconnectPlan,
    ReconnectStrategy,
    generate_reconnect_plan,
)
from onnxsplit.transform.split_concat import (
    create_concat_node,
    create_slice_node,
    create_split_node,
    get_slice_initializers,
)

__all__ = [
    # Node clone
    "clone_node",
    "generate_split_name",
    # Split/Concat
    "create_split_node",
    "create_concat_node",
    "create_slice_node",
    "get_slice_initializers",
    # Reconnect
    "ReconnectStrategy",
    "ReconnectConnection",
    "ReconnectPlan",
    "generate_reconnect_plan",
    # Executor
    "GraphTransformer",
]
