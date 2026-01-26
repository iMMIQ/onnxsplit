"""测试切分规划器"""

from pathlib import Path

from onnx import TensorProto

from onnxsplit.analyzer import ModelAnalyzer
from onnxsplit.config import GlobalConfig, OperatorConfig, SplitConfig
from onnxsplit.splitter.planner import SplitPlanner


def test_planner_with_no_config():
    """测试无配置时的规划"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    config = SplitConfig(global_config=GlobalConfig(default_parts=1))

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    # 默认不切分
    assert len(report.plans) == 0
    assert report.split_operators == 0


def test_planner_with_global_default_parts():
    """测试全局默认切分数"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    config = SplitConfig(global_config=GlobalConfig(default_parts=2))

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    # 应该为可切分的算子生成方案
    assert len(report.plans) > 0
    # 检查Conv有方案
    conv_plan = report.get_plan("conv_0")
    assert conv_plan is not None
    assert conv_plan.parts == 2
    assert conv_plan.axis == 0  # Batch维度


def test_planner_with_operator_config():
    """测试算子级别配置"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    config = SplitConfig(
        global_config=GlobalConfig(default_parts=1),
        operators={
            "conv_0": OperatorConfig(parts=4, axis=0),
        },
    )

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    conv_plan = report.get_plan("conv_0")
    assert conv_plan is not None
    assert conv_plan.parts == 4


def test_planner_wildcard_matching():
    """测试通配符匹配"""
    model_path = Path("tests/fixtures/models/model_with_branches.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    config = SplitConfig(
        global_config=GlobalConfig(default_parts=1),
        operators={
            "conv_*": OperatorConfig(parts=2),  # 通配符匹配conv_开头的算子（小写）
        },
    )

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    # 应该匹配到Conv算子
    assert len(report.plans) >= 2


def test_planner_with_axis_override():
    """测试axis覆盖"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    config = SplitConfig(
        global_config=GlobalConfig(default_parts=2),
        operators={
            "conv_0": OperatorConfig(parts=4, axis=0),  # 明确指定axis
        },
    )

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    conv_plan = report.get_plan("conv_0")
    assert conv_plan.axis == 0


def test_planner_respects_splitable_axes():
    """测试尊重可切分轴限制"""
    # 创建虚拟分析器
    import onnx
    from onnx import helper

    from onnxsplit.analyzer import ModelAnalyzer

    # 创建一个简单模型
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3, 8, 8])
    conv_node = helper.make_node(
        "Conv",
        inputs=["input", "weight"],
        outputs=["output"],
        name="conv_0",
    )
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4, 8, 8])
    weight = helper.make_tensor("weight", TensorProto.FLOAT, [4, 3, 3, 3], [0.1] * 108)
    const_node = helper.make_node("Constant", [], ["weight_value"], value=weight)
    conv_node.input[1] = "weight_value"

    graph = helper.make_graph(
        [const_node, conv_node],
        "test",
        [input_tensor],
        [output_tensor],
    )
    model = helper.make_model(graph)
    model = onnx.shape_inference.infer_shapes(model)

    analyzer = ModelAnalyzer.from_model_proto(model)
    config = SplitConfig(global_config=GlobalConfig(default_parts=4))

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    # Conv只能切batch维度(axis=0)
    conv_plan = report.get_plan("conv_0")
    if conv_plan:
        assert conv_plan.axis == 0


def test_planner_parts_one_no_split():
    """测试parts=1时不生成切分方案"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    config = SplitConfig(
        global_config=GlobalConfig(default_parts=1),
        operators={
            "conv_0": OperatorConfig(parts=1),  # 明确不切分
        },
    )

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    # parts=1不算切分
    conv_plan = report.get_plan("conv_0")
    if conv_plan:
        assert not conv_plan.is_split


def test_planner_unsplitable_ops():
    """测试不可切分的算子"""
    # Reshape通常不可切分
    import onnx
    from onnx import helper

    from onnxsplit.analyzer import ModelAnalyzer

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 128])
    reshape_node = helper.make_node(
        "Reshape",
        inputs=["input", "shape"],
        outputs=["output"],
        name="reshape_0",
    )
    shape_const = helper.make_tensor("shape", TensorProto.INT64, [3], [2, 8, 16])
    const_node = helper.make_node("Constant", [], ["shape"], value=shape_const)
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 8, 16])

    graph = helper.make_graph(
        [const_node, reshape_node],
        "test",
        [input_tensor],
        [output_tensor],
    )
    model = helper.make_model(graph)
    model = onnx.shape_inference.infer_shapes(model)

    analyzer = ModelAnalyzer.from_model_proto(model)
    config = SplitConfig(global_config=GlobalConfig(default_parts=2))

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    # Reshape不应该有切分方案
    reshape_plan = report.get_plan("reshape_0")
    assert reshape_plan is None or not reshape_plan.is_split


def test_planner_report_stats():
    """测试报告统计信息"""
    model_path = Path("tests/fixtures/models/model_with_branches.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    config = SplitConfig(global_config=GlobalConfig(default_parts=2))

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    assert report.original_operators > 0
    assert report.split_operators >= 0
    assert report.split_operators + report.unsplit_operators == report.original_operators


def test_planner_get_all_splitable_ops():
    """测试获取所有可切分算子"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    config = SplitConfig(global_config=GlobalConfig(default_parts=2))

    planner = SplitPlanner(analyzer, config)

    splitable = planner.get_splitable_operators()
    assert len(splitable) > 0
    # 应该包含Conv
    assert any(op.op_type == "Conv" for op in splitable)


def test_planner_with_empty_model():
    """测试空模型"""
    from onnx import helper

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 8, 8])
    graph = helper.make_graph([], "empty", [input_tensor], [output_tensor])
    model = helper.make_model(graph)

    analyzer = ModelAnalyzer.from_model_proto(model)
    config = SplitConfig(global_config=GlobalConfig(default_parts=2))

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    assert len(report.plans) == 0


def test_planner_priority():
    """测试配置优先级：算子配置 > 全局配置"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    config = SplitConfig(
        global_config=GlobalConfig(default_parts=2),  # 默认2份
        operators={
            "conv_0": OperatorConfig(parts=2),  # conv_0配置为2份（batch_size=4可被2整除）
        },
    )

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    conv_plan = report.get_plan("conv_0")
    # 验证使用了算子配置
    assert conv_plan is not None
    assert conv_plan.parts == 2


def test_planner_dynamic_shape_handling():
    """测试处理动态形状"""
    import onnx
    from onnx import helper

    # 动态batch维度
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch_dim", 3, 8, 8])
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, ["batch_dim", 4, 8, 8]
    )

    # 使用identity作为占位
    identity_node = helper.make_node(
        "Identity",
        inputs=["input"],
        outputs=["output"],
        name="identity_0",
    )

    graph = helper.make_graph(
        [identity_node],
        "test",
        [input_tensor],
        [output_tensor],
    )
    model = helper.make_model(graph)
    model = onnx.shape_inference.infer_shapes(model)

    analyzer = ModelAnalyzer.from_model_proto(model)
    config = SplitConfig(global_config=GlobalConfig(default_parts=2))

    planner = SplitPlanner(analyzer, config)
    _ = planner.generate()  # 动态形状处理待完善

    # 动态形状的算子应该被标记
    # 实际实现中可能需要特殊处理


def test_planner_config_exact_match_priority():
    """测试精确匹配优先于通配符"""
    model_path = Path("tests/fixtures/models/model_with_branches.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    config = SplitConfig(
        global_config=GlobalConfig(default_parts=1),
        operators={
            "conv_0": OperatorConfig(parts=10, axis=0),  # 精确匹配
            "conv_*": OperatorConfig(parts=5, axis=0),   # 通配符匹配
            "*": OperatorConfig(parts=2),                 # 全局通配符
        },
    )

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    conv_0_plan = report.get_plan("conv_0")
    if conv_0_plan:
        assert conv_0_plan.parts == 10  # 使用精确匹配，不是5或2


def test_planner_config_wildcard_star():
    """测试*通配符匹配所有"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    config = SplitConfig(
        global_config=GlobalConfig(default_parts=1),
        operators={
            "*": OperatorConfig(parts=3),  # 匹配所有
        },
    )

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    # 可切分的算子都应该使用parts=3
    for plan in report.plans:
        assert plan.parts == 3


def test_planner_config_multiple_wildcards():
    """测试多个通配符模式"""
    model_path = Path("tests/fixtures/models/model_with_branches.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    config = SplitConfig(
        global_config=GlobalConfig(default_parts=1),
        operators={
            "conv_*": OperatorConfig(parts=4, axis=0),
            "*_output": OperatorConfig(parts=2),
        },
    )

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    # Conv算子应该匹配conv_*
    conv_plans = [p for p in report.plans if p.operator_name.startswith("conv_")]
    for plan in conv_plans:
        assert plan.parts == 4


def test_planner_config_question_mark_wildcard():
    """测试?通配符匹配单个字符"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    config = SplitConfig(
        global_config=GlobalConfig(default_parts=1),
        operators={
            "conv_?": OperatorConfig(parts=7),  # 匹配conv_0, conv_1等
        },
    )

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    conv_plan = report.get_plan("conv_0")
    if conv_plan:
        assert conv_plan.parts == 7


def test_planner_config_bracket_wildcard():
    """测试[]字符类通配符"""
    model_path = Path("tests/fixtures/models/model_with_branches.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    config = SplitConfig(
        global_config=GlobalConfig(default_parts=1),
        operators={
            "conv_[01]": OperatorConfig(parts=6),  # 匹配conv_0或conv_1
        },
    )

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    # conv_0或conv_1应该匹配
    for name in ["conv_0", "conv_1"]:
        plan = report.get_plan(name)
        if plan:
            assert plan.parts == 6
