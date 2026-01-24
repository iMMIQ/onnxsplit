"""测试配置合并逻辑"""

import pytest

from onnxsplit.config.merger import ConfigMergeError, merge_cli_args
from onnxsplit.config.schema import (
    AxisRule,
    GlobalConfig,
    MemoryRule,
    OperatorConfig,
    SplitConfig,
)


def test_merge_cli_parts_only():
    """测试仅合并parts参数"""
    config = SplitConfig(global_config=GlobalConfig(default_parts=2))
    result = merge_cli_args(config, cli_parts=4, cli_max_memory=None)
    assert result.global_config.default_parts == 4
    assert result.global_config.max_memory_mb is None


def test_merge_cli_max_memory_only():
    """测试仅合并max_memory参数"""
    config = SplitConfig(global_config=GlobalConfig(default_parts=2))
    result = merge_cli_args(config, cli_parts=None, cli_max_memory=512)
    assert result.global_config.default_parts == 2
    assert result.global_config.max_memory_mb == 512


def test_merge_both_cli_args():
    """测试同时合并两个CLI参数"""
    config = SplitConfig(global_config=GlobalConfig(default_parts=2))
    result = merge_cli_args(config, cli_parts=8, cli_max_memory=1024)
    assert result.global_config.default_parts == 8
    assert result.global_config.max_memory_mb == 1024


def test_cli_parts_lower_than_config():
    """测试CLI parts小于配置文件值"""
    config = SplitConfig(global_config=GlobalConfig(default_parts=10))
    result = merge_cli_args(config, cli_parts=4, cli_max_memory=None)
    # CLI参数应该覆盖配置文件值，即使更小
    assert result.global_config.default_parts == 4


def test_merge_creates_new_config():
    """测试合并创建新配置对象而非修改原配置"""
    config = SplitConfig(global_config=GlobalConfig(default_parts=2))
    original_id = id(config)
    result = merge_cli_args(config, cli_parts=4, cli_max_memory=None)
    # 应该返回新对象
    assert id(result) != original_id
    # 原配置不应被修改
    assert config.global_config.default_parts == 2


def test_merge_with_no_cli_args():
    """测试无CLI参数时返回原配置"""
    config = SplitConfig(global_config=GlobalConfig(default_parts=2, max_memory_mb=512))
    result = merge_cli_args(config, cli_parts=None, cli_max_memory=None)
    # 应该返回原配置对象
    assert id(result) == id(config)


def test_invalid_cli_parts():
    """测试无效的cli_parts值"""
    config = SplitConfig(global_config=GlobalConfig(default_parts=2))
    with pytest.raises(ConfigMergeError, match="parts.*必须大于0"):
        merge_cli_args(config, cli_parts=0, cli_max_memory=None)

    with pytest.raises(ConfigMergeError, match="parts.*必须大于0"):
        merge_cli_args(config, cli_parts=-1, cli_max_memory=None)


def test_invalid_cli_max_memory():
    """测试无效的cli_max_memory值"""
    config = SplitConfig(global_config=GlobalConfig(default_parts=2))
    with pytest.raises(ConfigMergeError, match="max_memory.*必须大于0"):
        merge_cli_args(config, cli_parts=None, cli_max_memory=0)

    with pytest.raises(ConfigMergeError, match="max_memory.*必须大于0"):
        merge_cli_args(config, cli_parts=None, cli_max_memory=-100)


def test_merge_preserves_operators():
    """测试合并保留operators配置"""
    config = SplitConfig(
        global_config=GlobalConfig(default_parts=2),
        operators={"/model/Conv": OperatorConfig(parts=4, axis=0)},
    )
    result = merge_cli_args(config, cli_parts=8, cli_max_memory=None)
    assert len(result.operators) == 1
    assert result.operators["/model/Conv"].parts == 4
    assert result.operators["/model/Conv"].axis == 0


def test_merge_preserves_axis_rules():
    """测试合并保留axis_rules配置"""
    config = SplitConfig(
        global_config=GlobalConfig(default_parts=2),
        axis_rules=[AxisRule(op_type="Conv", prefer_axis=0)],
    )
    result = merge_cli_args(config, cli_parts=8, cli_max_memory=None)
    assert len(result.axis_rules) == 1
    assert result.axis_rules[0].op_type == "Conv"
    assert result.axis_rules[0].prefer_axis == 0


def test_merge_preserves_memory_rules():
    """测试合并保留memory_rules配置"""
    config = SplitConfig(
        global_config=GlobalConfig(default_parts=2),
        memory_rules=MemoryRule(auto_adjust=True, overflow_strategy="binary_split"),
    )
    result = merge_cli_args(config, cli_parts=8, cli_max_memory=1024)
    assert result.memory_rules is not None
    assert result.memory_rules.auto_adjust is True
    assert result.memory_rules.overflow_strategy == "binary_split"


def test_merge_creates_new_global_config():
    """测试合并创建新的GlobalConfig对象"""
    config = SplitConfig(global_config=GlobalConfig(default_parts=2))
    original_global_id = id(config.global_config)
    result = merge_cli_args(config, cli_parts=4, cli_max_memory=None)
    # 应该创建新的GlobalConfig对象
    assert id(result.global_config) != original_global_id
