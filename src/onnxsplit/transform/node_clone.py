"""节点克隆功能"""

import re
from onnx import NodeProto


def _sanitize_name_for_node(name: str) -> str:
    """清理名称以用作节点名称

    ONNX节点名称不应包含某些特殊字符（如前导斜杠、空格等）。
    此函数将特殊字符替换为下划线。

    Args:
        name: 原始名称

    Returns:
        清理后的名称
    """
    # 移除前导斜杠并替换其他特殊字符
    cleaned = name.lstrip("/")
    # 替换其他非法字符（除字母、数字、下划线、连字符外的字符）
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", cleaned)
    # 确保不以数字或连字符开头（某些系统不允许）
    if cleaned and cleaned[0] in "0123456789-":
        cleaned = "n_" + cleaned
    # 如果清理后为空，返回默认名称
    return cleaned or "node"


def generate_split_name(original_name: str, part_idx: int, suffix: str = "split") -> str:
    """生成切分后的节点名称

    Args:
        original_name: 原始节点名称
        part_idx: 切分索引
        suffix: 后缀名称

    Returns:
        新的节点名称
    """
    sanitized_name = _sanitize_name_for_node(original_name)
    base_name = sanitized_name if sanitized_name != "node" else f"node_{id(object())}"
    return f"{base_name}_{suffix}_{part_idx}"


def clone_node(
    node: NodeProto,
    suffix: str,
    new_outputs: list[str],
    new_name: str | None = None,
    new_inputs: list[str] | None = None,
) -> NodeProto:
    """克隆ONNX节点

    创建节点的副本，可以修改名称、输出名称和输入名称。

    Args:
        node: 原始节点
        suffix: 名称后缀
        new_outputs: 新的输出名称列表
        new_name: 新的节点名称（如果为None则自动生成）
        new_inputs: 新的输入名称列表（如果为None则使用原输入）

    Returns:
        克隆的节点
    """
    import onnx.helper

    # 生成新名称
    if new_name is None:
        # 清理原始节点名称
        original_name = node.name if node.name else f"{node.op_type}_node"
        sanitized_name = _sanitize_name_for_node(original_name)
        # 如果清理后变为默认值，使用op_type作为基础名称
        base_name = sanitized_name if sanitized_name != "node" else f"{node.op_type}_node"
        new_name = f"{base_name}{suffix}"

    # 使用新输入或原输入
    inputs = new_inputs if new_inputs is not None else list(node.input)

    # 创建新节点，保留所有属性
    new_node = onnx.helper.make_node(
        op_type=node.op_type,
        inputs=inputs,
        outputs=new_outputs,
        name=new_name,
        domain=node.domain,
    )

    # 复制所有属性
    for attr in node.attribute:
        new_node.attribute.append(attr)

    return new_node
