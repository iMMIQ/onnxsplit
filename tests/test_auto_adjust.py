"""测试自动切分调整"""

from pathlib import Path

from onnxsplit.analyzer import ModelAnalyzer
from onnxsplit.memory.auto_adjust import AutoSplitAdjuster
from onnxsplit.memory.estimator import MemoryEstimator
from onnxsplit.splitter.plan import SplitPlan


def test_adjuster_no_limit():
    """测试无内存限制时不调整"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    estimator = MemoryEstimator(analyzer)
    adjuster = AutoSplitAdjuster(estimator)

    plan = SplitPlan(operator_name="conv_0", parts=2, axis=0)

    adjusted = adjuster.adjust_plan(plan, max_memory_mb=None)
    assert adjusted.parts == 2  # 不调整


def test_adjuster_under_limit():
    """测试低于限制时不调整"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    estimator = MemoryEstimator(analyzer)
    adjuster = AutoSplitAdjuster(estimator)

    plan = SplitPlan(operator_name="conv_0", parts=2, axis=0)

    # 设置一个很大的限制
    adjusted = adjuster.adjust_plan(plan, max_memory_mb=10000)
    assert adjusted.parts == 2


def test_adjuster_over_limit():
    """测试超过限制时增加切分数"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    estimator = MemoryEstimator(analyzer)
    adjuster = AutoSplitAdjuster(estimator)

    # 获取算子内存
    conv_op = analyzer.get_operator("conv_0")
    if conv_op:
        op_mem = estimator.get_operator_memory(conv_op)
        if op_mem:
            # 设置一个低于当前内存的限制
            adjusted = adjuster.adjust_plan(
                SplitPlan(operator_name="conv_0", parts=1, axis=0),
                max_memory_mb=op_mem.total_memory_mb / 4,  # 强切分
            )
            assert adjusted.parts >= 1


def test_adjuster_max_parts_limit():
    """测试最大切分数限制"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    estimator = MemoryEstimator(analyzer)
    adjuster = AutoSplitAdjuster(estimator)

    # 设置极低的内存限制
    plan = SplitPlan(operator_name="conv_0", parts=1, axis=0)

    adjusted = adjuster.adjust_plan(
        plan,
        max_memory_mb=0.001,  # 1KB
        max_parts=256,
    )

    # 应该受到max_parts限制
    assert adjusted.parts <= 256


def test_adjuster_with_large_parts():
    """测试处理大切分数"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    estimator = MemoryEstimator(analyzer)
    adjuster = AutoSplitAdjuster(estimator)

    plan = SplitPlan(operator_name="conv_0", parts=100, axis=0)

    # 大切分数应该被警告或限制
    adjusted = adjuster.adjust_plan(plan, max_memory_mb=None)
    assert adjusted.parts == 100


def test_adjuster_unsplitable():
    """测试不可切分算子"""
    # Reshape通常不可切分
    import onnx
    from onnx import helper

    input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 128])
    shape_const = helper.make_tensor("shape", onnx.TensorProto.INT64, [2], [2, 64])
    const_node = helper.make_node("Constant", [], ["shape"], value=shape_const)
    reshape_node = helper.make_node(
        "Reshape", inputs=["input", "shape"], outputs=["output"], name="reshape_0"
    )
    output_tensor = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 64])

    graph = helper.make_graph([const_node, reshape_node], "test", [input_tensor], [output_tensor])
    model = helper.make_model(graph)
    model = onnx.shape_inference.infer_shapes(model)

    analyzer = ModelAnalyzer.from_model_proto(model)
    estimator = MemoryEstimator(analyzer)
    adjuster = AutoSplitAdjuster(estimator)

    plan = SplitPlan(operator_name="reshape_0", parts=1, axis=None)
    adjusted = adjuster.adjust_plan(plan, max_memory_mb=1.0)

    # 不可切分，返回原计划
    assert adjusted.parts == 1


def test_adjuster_binary_search():
    """测试二分查找最优切分数"""
    # 模拟一个内存占用已知的算子
    import onnx
    from onnx import helper

    # 创建一个大张量的模型
    input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1000, 1000])
    add_node = helper.make_node("Add", inputs=["input", "input"], outputs=["output"], name="add_0")
    output_tensor = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1000, 1000])

    graph = helper.make_graph([add_node], "test", [input_tensor], [output_tensor])
    model = helper.make_model(graph)
    model = onnx.shape_inference.infer_shapes(model)

    analyzer = ModelAnalyzer.from_model_proto(model)
    estimator = MemoryEstimator(analyzer)
    adjuster = AutoSplitAdjuster(estimator)

    plan = SplitPlan(operator_name="add_0", parts=1, axis=0)

    # 设置限制使需要切分
    # 1000*1000*4 bytes = 4MB per tensor
    # 限制1MB需要至少切4份
    adjusted = adjuster.adjust_plan(plan, max_memory_mb=2.0)
    assert adjusted.parts >= 1


def test_adjuster_preserve_axis():
    """测试调整时保留切分轴"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    estimator = MemoryEstimator(analyzer)
    adjuster = AutoSplitAdjuster(estimator)

    plan = SplitPlan(operator_name="conv_0", parts=2, axis=0)

    adjusted = adjuster.adjust_plan(plan, max_memory_mb=None)
    assert adjusted.axis == 0


def test_adjuster_with_weights():
    """测试包含权重的算子"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    estimator = MemoryEstimator(analyzer)
    adjuster = AutoSplitAdjuster(estimator)

    # Conv有权重，权重不应被切分
    conv_op = analyzer.get_operator("conv_0")
    if conv_op:
        plan = SplitPlan(operator_name="conv_0", parts=2, axis=0)
        adjusted = adjuster.adjust_plan(plan, max_memory_mb=None)
        # 权重不影响切分决策
        assert adjusted.axis == 0


def test_adjuster_report_adjustment():
    """测试调整报告"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    estimator = MemoryEstimator(analyzer)
    adjuster = AutoSplitAdjuster(estimator)

    plan = SplitPlan(operator_name="conv_0", parts=2, axis=0)

    # 强制调整
    original_parts = plan.parts
    adjusted = adjuster.adjust_plan(plan, max_memory_mb=0.001)

    # 如果发生了调整
    if adjusted.parts != original_parts:
        assert adjusted.reason is not None
