"""YAML配置加载器模块。

此模块提供从YAML文件加载配置的功能，包括:
- ConfigError: 配置错误异常类
- load_config: 主入口函数，用于加载配置文件

配置文件结构:
    global:
        default_parts: int (默认1)
        max_memory_mb: int | None (默认None)

    operators:
        "/operator/path":
            parts: int (必需)
            axis: int | None (默认None)

    axis_rules:
        - op_type: str (必需)
          prefer_axis: int | str | None (默认None)

    memory_rules:
        auto_adjust: bool (默认False)
        overflow_strategy: "binary_split" | "linear_split" | None (默认None)
"""

from pathlib import Path

import yaml

from onnxsplit.config.schema import (
    AxisRule,
    GlobalConfig,
    MemoryRule,
    OperatorConfig,
    SplitConfig,
)


class ConfigError(Exception):
    """配置加载或验证错误。"""

    pass


def _validate_int(value: any, field_name: str, context: str = "") -> int:
    """验证值是否为整数。

    Args:
        value: 要验证的值
        field_name: 字段名，用于错误消息
        context: 上下文信息，用于错误消息

    Returns:
        验证后的整数值

    Raises:
        ConfigError: 如果值不是整数
    """
    if isinstance(value, int):
        return value

    # 尝试从字符串转换
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass

    context_msg = f" ({context})" if context else ""
    raise ConfigError(f"Field '{field_name}'{context_msg} must be an integer, got {type(value).__name__}: {value}")


def _load_global_config(data: dict) -> GlobalConfig:
    """加载全局配置。

    Args:
        data: YAML解析后的全局配置字典

    Returns:
        GlobalConfig实例

    Raises:
        ConfigError: 如果配置值无效
    """
    global_data = data.get("global", {})

    # 验证default_parts
    default_parts = global_data.get("default_parts", 1)
    default_parts = _validate_int(default_parts, "default_parts", "global")

    if default_parts < 1:
        raise ConfigError(f"Field 'default_parts' must be >= 1, got {default_parts}")

    # 验证max_memory_mb (可选)
    max_memory_mb = global_data.get("max_memory_mb")
    if max_memory_mb is not None:
        max_memory_mb = _validate_int(max_memory_mb, "max_memory_mb", "global")
        if max_memory_mb < 1:
            raise ConfigError(f"Field 'max_memory_mb' must be >= 1, got {max_memory_mb}")

    return GlobalConfig(
        default_parts=default_parts,
        max_memory_mb=max_memory_mb,
    )


def _load_operator_configs(data: dict) -> dict[str, OperatorConfig]:
    """加载算子配置。

    Args:
        data: YAML解析后的配置字典

    Returns:
        算子路径到OperatorConfig的映射

    Raises:
        ConfigError: 如果配置值无效
    """
    operators_data = data.get("operators", {})
    operators = {}

    for op_path, op_config in operators_data.items():
        if not isinstance(op_config, dict):
            raise ConfigError(f"Operator config for '{op_path}' must be a dict, got {type(op_config).__name__}")

        # 验证parts (必需)
        if "parts" not in op_config:
            raise ConfigError(f"Operator '{op_path}' missing required field 'parts'")

        parts = _validate_int(op_config["parts"], "parts", f"operator '{op_path}'")

        if parts < 1:
            raise ConfigError(f"Field 'parts' for operator '{op_path}' must be >= 1, got {parts}")

        # 验证axis (可选)
        axis = op_config.get("axis")
        if axis is not None:
            axis = _validate_int(axis, "axis", f"operator '{op_path}'")

        operators[op_path] = OperatorConfig(parts=parts, axis=axis)

    return operators


def _load_axis_rules(data: dict) -> list[AxisRule]:
    """加载轴规则配置。

    Args:
        data: YAML解析后的配置字典

    Returns:
        AxisRule对象列表

    Raises:
        ConfigError: 如果配置值无效
    """
    axis_rules_data = data.get("axis_rules", [])
    axis_rules = []

    for i, rule_data in enumerate(axis_rules_data):
        if not isinstance(rule_data, dict):
            raise ConfigError(f"Axis rule at index {i} must be a dict, got {type(rule_data).__name__}")

        # 验证op_type (必需)
        if "op_type" not in rule_data:
            raise ConfigError(f"Axis rule at index {i} missing required field 'op_type'")

        op_type = rule_data["op_type"]
        if not isinstance(op_type, str):
            raise ConfigError(f"Field 'op_type' in axis rule {i} must be a string, got {type(op_type).__name__}")

        # 获取prefer_axis (可选，可以是int, str, 或None)
        prefer_axis = rule_data.get("prefer_axis")

        # 验证prefer_axis的类型
        if prefer_axis is not None and not isinstance(prefer_axis, (int, str)):
            raise ConfigError(
                f"Field 'prefer_axis' in axis rule {i} must be int, str, or None, got {type(prefer_axis).__name__}"
            )

        axis_rules.append(AxisRule(op_type=op_type, prefer_axis=prefer_axis))

    return axis_rules


def _load_memory_rules(data: dict) -> MemoryRule:
    """加载内存规则配置。

    Args:
        data: YAML解析后的配置字典

    Returns:
        MemoryRule对象

    Raises:
        ConfigError: 如果配置值无效
    """
    memory_data = data.get("memory_rules", {})

    # auto_adjust (可选)
    auto_adjust = memory_data.get("auto_adjust", False)
    if not isinstance(auto_adjust, bool):
        raise ConfigError(
            f"Field 'auto_adjust' in memory_rules must be a boolean, got {type(auto_adjust).__name__}"
        )

    # overflow_strategy (可选)
    overflow_strategy = memory_data.get("overflow_strategy")
    if overflow_strategy is not None:
        if not isinstance(overflow_strategy, str):
            raise ConfigError(
                f"Field 'overflow_strategy' in memory_rules must be a string, got {type(overflow_strategy).__name__}"
            )
        if overflow_strategy not in ("binary_split", "linear_split"):
            raise ConfigError(
                f"Field 'overflow_strategy' must be 'binary_split' or 'linear_split', got '{overflow_strategy}'"
            )

    return MemoryRule(auto_adjust=auto_adjust, overflow_strategy=overflow_strategy)


def load_config(path: Path | str) -> SplitConfig:
    """从YAML文件加载配置。

    Args:
        path: YAML配置文件路径

    Returns:
        SplitConfig对象

    Raises:
        ConfigError: 如果文件不存在、YAML解析失败或配置值无效
    """
    # 转换为Path对象
    if isinstance(path, str):
        path = Path(path)

    # 检查文件是否存在
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    # 读取并解析YAML
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML file '{path}': {e}") from e
    except OSError as e:
        raise ConfigError(f"Failed to read file '{path}': {e}") from e

    # 确保数据是字典
    if not isinstance(data, dict):
        raise ConfigError(f"Config file root must be a dict, got {type(data).__name__}")

    # 加载各部分配置
    global_config = _load_global_config(data)
    operators = _load_operator_configs(data)
    axis_rules = _load_axis_rules(data)
    memory_rules = _load_memory_rules(data)

    return SplitConfig(
        global_config=global_config,
        operators=operators,
        axis_rules=axis_rules,
        memory_rules=memory_rules,
    )
