"""测试配置数据结构"""

from onnxsplit.config.schema import (
    AxisRule,
    GlobalConfig,
    MemoryRule,
    OperatorConfig,
    SplitConfig,
)


def test_global_config_default_values():
    """测试全局配置的默认值"""
    config = GlobalConfig()
    assert config.default_parts == 1
    assert config.max_memory_mb is None


def test_global_config_with_values():
    """测试创建带值的全局配置"""
    config = GlobalConfig(default_parts=4, max_memory_mb=512)
    assert config.default_parts == 4
    assert config.max_memory_mb == 512


def test_operator_config_creation():
    """测试算子配置创建"""
    config = OperatorConfig(parts=2, axis=0)
    assert config.parts == 2
    assert config.axis == 0


def test_operator_config_without_axis():
    """测试不带axis的算子配置"""
    config = OperatorConfig(parts=2)
    assert config.parts == 2
    assert config.axis is None


def test_axis_rule_creation():
    """测试切分轴规则创建"""
    rule = AxisRule(op_type="Conv", prefer_axis=0)
    assert rule.op_type == "Conv"
    assert rule.prefer_axis == 0


def test_axis_rule_with_null_axis():
    """测试不可切分轴规则"""
    rule = AxisRule(op_type="LayerNorm", prefer_axis=None)
    assert rule.op_type == "LayerNorm"
    assert rule.prefer_axis is None


def test_axis_rule_with_string_axis():
    """测试字符串形式的轴规则"""
    rule = AxisRule(op_type="MatMul", prefer_axis="batch")
    assert rule.op_type == "MatMul"
    assert rule.prefer_axis == "batch"


def test_memory_rule_creation():
    """测试内存规则创建"""
    rule = MemoryRule(auto_adjust=True, overflow_strategy="binary_split")
    assert rule.auto_adjust is True
    assert rule.overflow_strategy == "binary_split"


def test_memory_rule_default_values():
    """测试内存规则默认值"""
    rule = MemoryRule()
    assert rule.auto_adjust is False
    assert rule.overflow_strategy is None


def test_split_config_creation():
    """测试完整配置创建"""
    config = SplitConfig(
        global_config=GlobalConfig(default_parts=2),
        operators={"/model/Conv": OperatorConfig(parts=4)},
        axis_rules=[AxisRule(op_type="Conv", prefer_axis=0)],
        memory_rules=MemoryRule(auto_adjust=True),
    )
    assert config.global_config.default_parts == 2
    assert config.operators["/model/Conv"].parts == 4
    assert len(config.axis_rules) == 1
    assert config.memory_rules.auto_adjust is True


def test_split_config_empty_operators():
    """测试空算子配置"""
    config = SplitConfig(
        global_config=GlobalConfig(), operators={}, axis_rules=[], memory_rules=None
    )
    assert config.operators == {}
    assert config.axis_rules == []
    assert config.memory_rules is None
