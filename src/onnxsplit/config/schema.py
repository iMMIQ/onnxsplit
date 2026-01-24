"""配置数据结构定义"""
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class GlobalConfig:
    """全局配置"""
    default_parts: int = 1
    max_memory_mb: Optional[int] = None


@dataclass
class OperatorConfig:
    """算子级别的配置"""
    parts: int
    axis: Optional[int] = None


@dataclass
class AxisRule:
    """切分轴规则"""
    op_type: str
    prefer_axis: Optional[int | str] = None


@dataclass
class MemoryRule:
    """内存限制规则"""
    auto_adjust: bool = False
    overflow_strategy: Optional[Literal["binary_split", "linear_split"]] = None


@dataclass
class SplitConfig:
    """完整的切分配置"""
    global_config: GlobalConfig = field(default_factory=GlobalConfig)
    operators: dict[str, OperatorConfig] = field(default_factory=dict)
    axis_rules: list[AxisRule] = field(default_factory=list)
    memory_rules: Optional[MemoryRule] = None
