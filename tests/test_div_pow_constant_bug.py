"""测试Div算子中Pow常数输出被错误split的bug复现

Bug描述:
- Div算子的输入A来自MatMul
- Div算子的输入B来自Pow算子
- Pow的输入都是常数，所以输出也是常数
- 但Pow的输出被错误地split了

根本原因:
- _is_weight只检查Constant节点和initializer
- 没有检查像Pow这样的计算节点，当输入都是常数时，输出也是常数
"""

import onnx
import pytest
from onnx import TensorProto, helper

from onnxsplit.analyzer.model import ModelAnalyzer
from onnxsplit.splitter.plan import SplitPlan
from onnxsplit.transform.executor import GraphTransformer


def create_div_with_pow_constant() -> onnx.ModelProto:
    """创建一个Div模型，其中一个输入是Pow的结果（常数）

    模型结构:
    input: [4, 8] -> MatMul -> [4, 4] -> Div -> [4, 4]
                                    ^
                                    Pow(constant^2) -> [1] (广播到[4,4])

    Pow的两个输入都是常数，所以输出也是常数。
    """
    # 输入张量
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [4, 8])

    # MatMul权重
    weight_data = [0.1] * (8 * 4)
    weight_tensor = helper.make_tensor(
        "matmul_weight",
        TensorProto.FLOAT,
        [8, 4],
        weight_data,
    )

    # MatMul节点
    matmul_node = helper.make_node(
        "MatMul",
        inputs=["input", "matmul_weight"],
        outputs=["matmul_output"],
        name="matmul_0",
    )

    # Pow的底数（常数）
    base_value = 2.0
    base_tensor = helper.make_tensor("pow_base", TensorProto.FLOAT, [], [base_value])
    base_const = helper.make_node("Constant", [], ["pow_base_value"], value=base_tensor)

    # Pow的指数（常数）
    exp_value = 2.0
    exp_tensor = helper.make_tensor("pow_exp", TensorProto.FLOAT, [], [exp_value])
    exp_const = helper.make_node("Constant", [], ["pow_exp_value"], value=exp_tensor)

    # Pow节点 (base^exp = 2^2 = 4, 常数)
    pow_node = helper.make_node(
        "Pow",
        inputs=["pow_base_value", "pow_exp_value"],
        outputs=["pow_output"],
        name="pow_0",
    )

    # Div节点 (matmul_output / pow_output)
    div_node = helper.make_node(
        "Div",
        inputs=["matmul_output", "pow_output"],
        outputs=["output"],
        name="div_0",
    )

    # 输出张量
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [4, 4])

    # 构建图
    graph = helper.make_graph(
        [base_const, exp_const, pow_node, matmul_node, div_node],
        "div_with_pow_constant",
        [input_tensor],
        [output_tensor],
        [weight_tensor],  # initializer
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model = onnx.shape_inference.infer_shapes(model)
    return model


def test_div_with_pow_constant_should_not_split_pow_output():
    """测试Div有Pow常数输入时，Pow输出不应该被split"""
    model = create_div_with_pow_constant()
    analyzer = ModelAnalyzer.from_model_proto(model)
    transformer = GraphTransformer(analyzer)

    # 验证模型结构
    div_op = analyzer.get_operator("div_0")
    assert div_op is not None

    # 验证Div的输入形状
    # input_0 (matmul_output): [4, 4]
    # input_1 (pow_output): [] (标量)
    input_0_shape = div_op.input_tensors[0].shape
    assert input_0_shape == (4, 4), f"Expected [4, 4], got {input_0_shape}"

    # pow_output 是 Pow 的输出，由于 Pow 的输入都是常数，输出也是常数
    # 修复后：_is_weight 应该识别 pow_output 为权重

    # 验证 pow_output 被识别为权重
    assert transformer._is_weight("pow_output") is True, \
        "pow_output (constant computation result) should be recognized as weight"

    # 获取div_0节点
    div_node = None
    for node in model.graph.node:
        if node.name == "div_0":
            div_node = node
            break
    assert div_node is not None

    # 尝试split axis=0, parts=2
    plan = SplitPlan(operator_name="div_0", parts=2, axis=0)

    # 创建输入split
    split_nodes, input_split_map = transformer._create_input_splits(
        model.graph, div_node, plan
    )

    # 修复后：只有 matmul_output 被split，pow_output 不应该被split
    assert len(split_nodes) == 1, \
        f"Should only create 1 split node (for matmul_output), got {len(split_nodes)}"
    assert "matmul_output" in input_split_map, \
        "matmul_output should be in input_split_map"
    assert "pow_output" not in input_split_map, \
        "pow_output (constant computation result) should NOT be in input_split_map"


def test_div_split_creates_shape_mismatch():
    """测试验证split后的行为"""
    model = create_div_with_pow_constant()
    analyzer = ModelAnalyzer.from_model_proto(model)
    transformer = GraphTransformer(analyzer)

    plan = SplitPlan(operator_name="div_0", parts=2, axis=0)

    # 应用split
    result_model = transformer.apply_split_plan(plan)

    # 检查结果模型中的节点
    div_nodes = [n for n in result_model.graph.node if n.op_type == "Div"]
    split_nodes = [n for n in result_model.graph.node if n.op_type == "Split"]

    # 应该有2个Div节点和1个Split节点
    assert len(div_nodes) == 2, f"Should have 2 Div nodes, got {len(div_nodes)}"
    assert len(split_nodes) == 1, f"Should have 1 Split node (for matmul_output), got {len(split_nodes)}"

    # 验证Split只针对matmul_output
    assert split_nodes[0].input[0] == "matmul_output"


def test_check_if_producer_is_constant_computation():
    """测试检查节点的前置节点是否是常量计算"""
    model = create_div_with_pow_constant()
    analyzer = ModelAnalyzer.from_model_proto(model)
    transformer = GraphTransformer(analyzer)

    # 获取pow_output的生产者
    pow_producer = analyzer.get_tensor_producer("pow_output")
    assert pow_producer == "pow_0", f"Expected pow_0, got {pow_producer}"

    # 获取pow节点的输入
    pow_node = None
    for node in model.graph.node:
        if node.name == "pow_0":
            pow_node = node
            break

    assert pow_node is not None
    assert list(pow_node.input) == ["pow_base_value", "pow_exp_value"]

    # 检查Pow的输入是否都是常数
    assert transformer._is_weight("pow_base_value") is True
    assert transformer._is_weight("pow_exp_value") is True

    # 验证pow_output也被识别为权重（常数计算结果）
    assert transformer._is_weight("pow_output") is True

    # 验证_is_constant_computation方法
    assert transformer._is_constant_computation("pow_output") is True


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
