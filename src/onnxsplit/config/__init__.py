"""配置管理模块

提供YAML配置文件的加载、验证和合并功能。
"""

from onnxsplit.config.loader import ConfigError, load_config
from onnxsplit.config.merger import ConfigMergeError, merge_cli_args
from onnxsplit.config.schema import (
    AxisRule,
    GlobalConfig,
    MemoryRule,
    OperatorConfig,
    SplitConfig,
)

__all__ = [
    # Schema
    "GlobalConfig",
    "OperatorConfig",
    "AxisRule",
    "MemoryRule",
    "SplitConfig",
    # Loader
    "load_config",
    "ConfigError",
    # Merger
    "merge_cli_args",
    "ConfigMergeError",
]
