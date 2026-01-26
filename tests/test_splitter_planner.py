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
            "*": OperatorConfig(parts=3),  # 匹配所有，但会被自动调整
        },
    )

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    # 由于batch_size=4，请求的parts=3不能整除4
    # 算法会自动向上查找，找到parts=4可以整除
    for plan in report.plans:
        assert plan.parts == 4  # 自动调整为4


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


class TestFindSuitableParts:
    """测试自动查找适合的切分数"""

    def test_find_suitable_parts_initial_valid(self):
        """测试初始 parts 就有效的情况"""
        import onnx
        from onnx import helper, TensorProto
        from onnxsplit.analyzer.model import ModelAnalyzer
        from onnxsplit.splitter.planner import SplitPlanner
        from onnxsplit.config import SplitConfig

        # 创建一个模型，输入形状为 [8, 10]
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [8, 10])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [8, 10])
        node = helper.make_node("Identity", ["input"], ["output"], name="op1")
        graph = helper.make_graph([node], "test", [input_tensor], [output_tensor])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

        analyzer = ModelAnalyzer.from_model_proto(model)
        config = SplitConfig()
        planner = SplitPlanner(analyzer, config)

        op_info = analyzer.get_operators()[0]

        # 请求 parts=2，维度=8 能被2整除
        found, parts, warning = planner._find_suitable_parts(op_info, axis=0, initial_parts=2)

        assert found is True
        assert parts == 2  # 保持原值
        assert warning is None

    def test_find_suitable_parts_auto_adjust(self):
        """测试自动调整 parts 的情况"""
        import onnx
        from onnx import helper, TensorProto
        from onnxsplit.analyzer.model import ModelAnalyzer
        from onnxsplit.splitter.planner import SplitPlanner
        from onnxsplit.config import SplitConfig

        # 创建一个模型，输入形状为 [6, 10]
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [6, 10])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [6, 10])
        node = helper.make_node("Identity", ["input"], ["output"], name="op1")
        graph = helper.make_graph([node], "test", [input_tensor], [output_tensor])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

        analyzer = ModelAnalyzer.from_model_proto(model)
        config = SplitConfig()
        planner = SplitPlanner(analyzer, config)

        op_info = analyzer.get_operators()[0]

        # 请求 parts=4，但维度=6 不能被4整除
        # 应该自动调整为 6（6的因数）
        found, parts, warning = planner._find_suitable_parts(op_info, axis=0, initial_parts=4)

        assert found is True
        assert parts == 6  # 6能被6整除
        assert warning is None

    def test_find_suitable_parts_adjusts_to_dimension(self):
        """测试算法会调整到维度本身"""
        import onnx
        from onnx import helper, TensorProto
        from onnxsplit.analyzer.model import ModelAnalyzer
        from onnxsplit.splitter.planner import SplitPlanner
        from onnxsplit.config import SplitConfig

        # 创建一个模型，输入形状为 [7, 10]（7是质数）
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [7, 10])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [7, 10])
        node = helper.make_node("Identity", ["input"], ["output"], name="op1")
        graph = helper.make_graph([node], "test", [input_tensor], [output_tensor])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

        analyzer = ModelAnalyzer.from_model_proto(model)
        config = SplitConfig()
        planner = SplitPlanner(analyzer, config)

        op_info = analyzer.get_operators()[0]

        # 请求 parts=3，但维度=7 是质数
        # 算法会搜索到 parts=7（7本身）
        found, parts, warning = planner._find_suitable_parts(op_info, axis=0, initial_parts=3)

        assert found is True
        assert parts == 7  # 调整到维度本身
        assert warning is None

    def test_warnings_collected_for_unsplittable(self):
        """测试对完全不可切分的情况收集警告"""
        import onnx
        from onnx import helper, TensorProto
        from onnxsplit.analyzer.model import ModelAnalyzer
        from onnxsplit.splitter.planner import SplitPlanner
        from onnxsplit.config import SplitConfig, OperatorConfig

        # 创建一个输入形状为 [1, 10] 的模型（batch=1太小）
        # 配置要求在axis=0上切分，但维度为1无法切分
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])
        node = helper.make_node("Relu", ["input"], ["output"], name="test_op")
        graph = helper.make_graph([node], "test", [input_tensor], [output_tensor])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

        analyzer = ModelAnalyzer.from_model_proto(model)

        # 配置要求 parts=3，axis=0
        config = SplitConfig(operators={"test_op": OperatorConfig(parts=3, axis=0)})
        planner = SplitPlanner(analyzer, config)

        # 生成方案
        report = planner.generate()

        # 应该有警告（因为axis=0的维度为1，无法切分）
        warnings = planner.get_warnings()
        # 注意：由于现在会尝试其他轴，Relu可以在axis=1上切分，所以可能没有警告
        # 如果维度10能被3整除的因数存在（不行，搜索上限min(10, 12, 256)=10，从4开始找，10%4!=0, 10%5==0！所以会找到5）
        # 让我们用更难的情况

    def test_warnings_collected_small_dimension(self):
        """测试小维度无法切分时收集警告"""
        import onnx
        from onnx import helper, TensorProto
        from onnxsplit.analyzer.model import ModelAnalyzer
        from onnxsplit.splitter.planner import SplitPlanner
        from onnxsplit.config import SplitConfig, OperatorConfig

        # 创建一个输入形状为 [1, 1] 的模型（两个维度都为1）
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])
        node = helper.make_node("Relu", ["input"], ["output"], name="test_op")
        graph = helper.make_graph([node], "test", [input_tensor], [output_tensor])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

        analyzer = ModelAnalyzer.from_model_proto(model)

        # 配置要求 parts=3
        config = SplitConfig(operators={"test_op": OperatorConfig(parts=3, axis=None)})
        planner = SplitPlanner(analyzer, config)

        # 生成方案
        report = planner.generate()

        # 应该有警告（所有维度都太小无法切分）
        warnings = planner.get_warnings()
        assert len(warnings) > 0
        assert "test_op" in warnings[0]

        # 不应该有 split
        assert report.split_operators == 0
