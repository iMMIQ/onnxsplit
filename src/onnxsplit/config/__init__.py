"""配置管理模块

提供YAML配置文件的加载、验证和合并功能。
"""

from onnxsplit.config.schema import (
    GlobalConfig,
    OperatorConfig,
    AxisRule,
    MemoryRule,
    SplitConfig,
)
from onnxsplit.config.loader import load_config, ConfigError
from onnxsplit.config.merger import merge_cli_args, ConfigMergeError


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
