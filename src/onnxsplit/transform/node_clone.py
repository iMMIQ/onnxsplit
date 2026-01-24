"""节点克隆功能"""

from onnx import NodeProto


def generate_split_name(original_name: str, part_idx: int, suffix: str = "split") -> str:
    """生成切分后的节点名称

    Args:
        original_name: 原始节点名称
        part_idx: 切分索引
        suffix: 后缀名称

    Returns:
        新的节点名称
    """
    base_name = original_name or f"node_{id(object())}"
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
        base_name = node.name if node.name else f"{node.op_type}_node"
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
