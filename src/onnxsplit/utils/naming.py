"""Name sanitization utilities for ONNX nodes and tensors."""

import re


def sanitize_name_for_node(name: str, default: str = "tensor") -> str:
    """清理张量名称以用作节点名称

    ONNX节点名称不应包含某些特殊字符（如前导斜杠、空格等）。
    此函数将特殊字符替换为下划线。

    Args:
        name: 原始名称
        default: 清理后为空时的默认名称

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
    return cleaned or default
