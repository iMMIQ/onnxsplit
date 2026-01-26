"""测试Reshape输出切分时的batch维度验证bug复现

Bug描述:
- Reshape输出batch为18
- 下游算子尝试分成4份
- 18 % 4 != 0，应该失败或调整，但实际被允许切分导致形状推断失败
"""

import onnx
import pytest
from onnx import TensorProto, helper

from onnxsplit.analyzer.model import ModelAnalyzer
from onnxsplit.config import GlobalConfig, OperatorConfig, SplitConfig
from onnxsplit.splitter.planner import SplitPlanner
from onnxsplit.transform.executor import GraphTransformer


def create_reshape_model_with_batch_18() -> onnx.ModelProto:
    """创建一个Reshape后接Add的模型，batch=18不可被4整除

    模型结构:
    input: [18, 64]  -> Reshape -> [18, 8, 8] -> Add -> output: [18, 8, 8]
    """
    # 输入张量: batch=18, 不可被4整除
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [18, 64])

    # Reshape常量: [18, 64] -> [18, 8, 8]
    shape_const = helper.make_tensor("reshape_shape", TensorProto.INT64, [3], [18, 8, 8])
    const_node = helper.make_node("Constant", [], ["reshape_shape"], value=shape_const)

    # Reshape节点
    reshape_node = helper.make_node(
        "Reshape",
        inputs=["input", "reshape_shape"],
        outputs=["reshape_output"],
        name="reshape_0",
    )

    # Add常量
    add_const = helper.make_tensor("add_bias", TensorProto.FLOAT, [8, 8], [0.1] * 64)
    add_const_node = helper.make_node("Constant", [], ["add_bias_value"], value=add_const)

    # Add节点 (element-wise，所有轴都可切分)
    add_node = helper.make_node(
        "Add",
        inputs=["reshape_output", "add_bias_value"],
        outputs=["output"],
        name="add_0",
    )

    # 输出张量
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [18, 8, 8])

    # 构建图
    graph = helper.make_graph(
        [const_node, reshape_node, add_const_node, add_node],
        "reshape_batch_test",
        [input_tensor],
        [output_tensor],
    )

    model = helper.make_model(graph)
    model = onnx.shape_inference.infer_shapes(model)
    return model


def test_reshape_batch_18_split_into_4_parts_should_fail():
    """测试Reshape输出batch=18时，分成4份应该失败

    18 % 4 != 0，所以不应该允许这种切分
    """
    model = create_reshape_model_with_batch_18()
    analyzer = ModelAnalyzer.from_model_proto(model)

    # 验证模型结构正确
    reshape_op = analyzer.get_operator("reshape_0")
    add_op = analyzer.get_operator("add_0")

    assert reshape_op is not None
    assert add_op is not None

    # 验证Add的输入形状是 [18, 8, 8]
    assert add_op.input_tensors[0].shape == (18, 8, 8)

    # 配置Add算子切分成4份，axis=0 (batch)
    config = SplitConfig(
        global_config=GlobalConfig(default_parts=1),
        operators={
            "add_0": OperatorConfig(parts=4, axis=0),
        },
    )

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    # 预期: Add算子不应该被切分 (因为18不能被4整除)
    # 或者parts应该被调整为6, 9, 或18

    add_plans = [p for p in report.plans if p.operator_name == "add_0"]

    if not add_plans:
        # 没有生成切分计划，这是正确的行为
        assert True, "Correctly skipped splitting add_0"
    else:
        # 如果生成了切分计划，parts应该被调整
        plan = add_plans[0]
        # parts应该是18的因数: 1, 2, 3, 6, 9, 18
        # 且由于配置了4，应该向上调整为能整除18的值
        assert 18 % plan.parts == 0, f"parts={plan.parts} should evenly divide 18"


def test_reshape_batch_18_split_with_auto_adjust():
    """测试自动调整功能应该找到合适的parts

    对于batch=18，parts=4应该被调整为6或9
    """
    model = create_reshape_model_with_batch_18()
    analyzer = ModelAnalyzer.from_model_proto(model)

    add_op = analyzer.get_operator("add_0")
    assert add_op.input_tensors[0].shape == (18, 8, 8)

    # 直接测试planner的_find_suitable_parts方法
    config = SplitConfig()
    planner = SplitPlanner(analyzer, config)

    # 尝试为add_0找4份的切分方案
    found, parts, warning = planner._find_suitable_parts(add_op, axis=0, initial_parts=4)

    # 预期: 应该找到一个能整除18的parts值
    # 或者返回False表示找不到合适的
    if found:
        assert 18 % parts == 0, f"parts={parts} should evenly divide 18"
        assert parts >= 4, f"parts={parts} should be >= initial_parts=4"
    else:
        assert warning is not None, "Should have a warning message"


def test_reshape_batch_18_split_should_not_cause_shape_inference_error():
    """测试切分不应该导致形状推断失败

    这是一个端到端测试，确保即使配置了不合理的切分，
    系统也能正确处理，不会产生形状推断错误的模型
    """
    model = create_reshape_model_with_batch_18()
    analyzer = ModelAnalyzer.from_model_proto(model)

    config = SplitConfig(
        global_config=GlobalConfig(default_parts=1),
        operators={
            "add_0": OperatorConfig(parts=4, axis=0),
        },
    )

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    # 如果生成了切分计划，应用它并验证形状推断不会失败
    add_plans = [p for p in report.plans if p.operator_name == "add_0"]

    if add_plans:
        plan = add_plans[0]
        transformer = GraphTransformer(analyzer)

        # 这应该不会抛出异常，或者返回的模型应该有有效的形状
        try:
            result_model = transformer.apply_split_plan(plan)

            # 验证形状推断成功
            # 检查结果模型中所有Split节点的输出是否能被正确分割
            for node in result_model.graph.node:
                if node.op_type == "Split":
                    # 获取split的axis属性
                    axis = 0
                    for attr in node.attribute:
                        if attr.name == "axis":
                            axis = attr.i
                            break

                    # 检查输入的维度
                    input_name = node.input[0]
                    for vi in result_model.graph.input:
                        if vi.name == input_name:
                            shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
                            if shape:
                                dim_size = shape[axis]
                                # 验证parts能整除维度
                                assert dim_size % len(node.output) == 0, \
                                    f"Split with {len(node.output)} parts cannot evenly divide dimension {dim_size}"

        except Exception as e:
            pytest.fail(f"Shape inference failed: {e}")


def test_reshape_batch_18_valid_split_6_parts():
    """测试Reshape输出batch=18时，分成6份应该成功

    18 % 6 == 0，这是合法的切分
    """
    model = create_reshape_model_with_batch_18()
    analyzer = ModelAnalyzer.from_model_proto(model)

    config = SplitConfig(
        global_config=GlobalConfig(default_parts=1),
        operators={
            "add_0": OperatorConfig(parts=6, axis=0),
        },
    )

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    add_plans = [p for p in report.plans if p.operator_name == "add_0"]
    assert len(add_plans) == 1, "Should generate split plan for 6 parts"

    plan = add_plans[0]
    assert plan.parts == 6
    assert plan.axis == 0

    # 应用切分并验证结果
    transformer = GraphTransformer(analyzer)
    result_model = transformer.apply_split_plan(plan)

    # 验证模型结构
    node_types = [n.op_type for n in result_model.graph.node]
    assert "Split" in node_types
    assert node_types.count("Add") == 6
    assert "Concat" in node_types


if __name__ == "__main__":
    # 运行单个测试
    test_reshape_batch_18_split_into_4_parts_should_fail()
    print("test_reshape_batch_18_split_into_4_parts_should_fail: PASSED")

    test_reshape_batch_18_split_with_auto_adjust()
    print("test_reshape_batch_18_split_with_auto_adjust: PASSED")

    test_reshape_batch_18_split_should_not_cause_shape_inference_error()
    print("test_reshape_batch_18_split_should_not_cause_shape_inference_error: PASSED")

    test_reshape_batch_18_valid_split_6_parts()
    print("test_reshape_batch_18_valid_split_6_parts: PASSED")

    print("\nAll tests passed!")
