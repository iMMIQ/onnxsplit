"""Split和Concat节点生成"""

import onnx.helper
from onnx import NodeProto, TensorProto

from onnxsplit.utils.naming import sanitize_name_for_node

# 模块级别的字典，用于存储Slice节点的初始器信息
# 键为节点的id，值为初始器张量列表
_slice_node_initializers: dict[int, list] = {}


def create_split_node(
    input_name: str,
    axis: int,
    parts: int,
    output_prefix: str,
    split_sizes: list[int] | None = None,
    node_name: str | None = None,
) -> NodeProto:
    """创建Split算子节点

    Args:
        input_name: 输入张量名称
        axis: 切分轴
        parts: 切分的份数
        output_prefix: 输出张量名称前缀
        split_sizes: 每份的大小（可选，不指定则均分）
        node_name: 节点名称（可选）

    Returns:
        Split节点
    """
    # 生成输出名称
    outputs = [f"{output_prefix}_{i}" for i in range(parts)]

    # 生成节点名称
    if node_name is None:
        sanitized_prefix = sanitize_name_for_node(output_prefix)
        node_name = f"split_{sanitized_prefix}"

    # 创建节点
    kwargs = {
        "inputs": [input_name],
        "outputs": outputs,
        "name": node_name,
    }

    # 添加axis属性
    if split_sizes is None:
        # 使用均分，只需要指定axis
        node = onnx.helper.make_node("Split", axis=axis, **kwargs)
    else:
        # 指定每份大小
        node = onnx.helper.make_node(
            "Split",
            axis=axis,
            split=split_sizes,
            **kwargs,
        )

    return node


def create_concat_node(
    input_names: list[str],
    output_name: str,
    axis: int,
    node_name: str | None = None,
) -> NodeProto:
    """创建Concat算子节点

    Args:
        input_names: 输入张量名称列表
        output_name: 输出张量名称
        axis: 拼接轴
        node_name: 节点名称（可选）

    Returns:
        Concat节点
    """
    # 生成节点名称
    if node_name is None:
        sanitized_name = sanitize_name_for_node(output_name)
        node_name = f"concat_{sanitized_name}"

    node = onnx.helper.make_node(
        "Concat",
        inputs=input_names,
        outputs=[output_name],
        axis=axis,
        name=node_name,
    )

    return node


def create_slice_node(
    input_name: str,
    output_name: str,
    starts: list[int],
    ends: list[int],
    axes: list[int],
    steps: list[int] | None = None,
    node_name: str | None = None,
) -> NodeProto:
    """创建Slice算子节点

    Args:
        input_name: 输入张量名称
        output_name: 输出张量名称
        starts: 每个维度的起始位置
        ends: 每个维度的结束位置
        axes: 切分的轴
        steps: 每个维度的步长（可选，默认为1）
        node_name: 节点名称（可选）

    Returns:
        Slice节点
    """
    # 生成节点名称
    if node_name is None:
        sanitized_name = sanitize_name_for_node(output_name)
        node_name = f"slice_{sanitized_name}"

    # 创建常量张量作为输入
    starts_tensor = onnx.helper.make_tensor(
        "starts", TensorProto.INT64, dims=[len(starts)], vals=starts
    )
    ends_tensor = onnx.helper.make_tensor("ends", TensorProto.INT64, dims=[len(ends)], vals=ends)
    axes_tensor = onnx.helper.make_tensor("axes", TensorProto.INT64, dims=[len(axes)], vals=axes)

    # Slice节点的输入: data, starts, ends, axes
    inputs = [
        input_name,
        starts_tensor.name,
        ends_tensor.name,
        axes_tensor.name,
    ]

    # 如果有steps，添加到输入和初始器列表
    if steps is not None:
        steps_tensor = onnx.helper.make_tensor(
            "steps", TensorProto.INT64, dims=[len(steps)], vals=steps
        )
        inputs.append(steps_tensor.name)

    node = onnx.helper.make_node(
        "Slice",
        inputs=inputs,
        outputs=[output_name],
        name=node_name,
    )

    # 将初始器信息附加到节点上（用于后续处理）
    # 由于protobuf限制，使用模块级字典存储，键为节点id
    initializers = [starts_tensor, ends_tensor, axes_tensor]
    if steps is not None:
        initializers.append(steps_tensor)
    _slice_node_initializers[id(node)] = initializers

    return node


def get_slice_initializers(node: NodeProto) -> list:
    """获取Slice节点的初始器张量

    Args:
        node: Slice节点

    Returns:
        初始器张量列表
    """
    return _slice_node_initializers.get(id(node), [])
