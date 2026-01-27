"""测试Clip算子中Constant 0维常数被错误split的bug复现

Bug描述:
- Clip算子的输入来自Add算子和Constant权重
- Constant是一个0维常数（标量）
- executor._is_weight没有检查Constant节点，导致0维常数被尝试split
- 0维常数无法split，应该被识别为权重并跳过
"""

import onnx
import pytest
from onnx import TensorProto, helper

from onnxsplit.analyzer.model import ModelAnalyzer
from onnxsplit.splitter.plan import SplitPlan
from onnxsplit.transform.executor import GraphTransformer


def create_clip_with_zero_dim_constant() -> onnx.ModelProto:
    """创建一个Clip模型，其中一个输入是0维Constant

    模型结构:
    input: [4, 8] -> Add -> [4, 8] -> Clip (min=0.0, max=6.0来自0维Constant) -> output: [4, 8]
                                                           ^
                                                   0维常量（标量）
    """
    # 输入张量
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [4, 8])

    # Add常量（广播到 [4, 8]）
    add_bias = helper.make_tensor("add_bias", TensorProto.FLOAT, [], [1.0])  # 0维标量
    add_const_node = helper.make_node("Constant", [], ["add_bias_value"], value=add_bias)

    # Add节点
    add_node = helper.make_node(
        "Add",
        inputs=["input", "add_bias_value"],
        outputs=["add_output"],
        name="add_0",
    )

    # Clip的min和max常量（0维标量）
    clip_min = helper.make_tensor("clip_min", TensorProto.FLOAT, [], [0.0])  # 0维
    clip_max = helper.make_tensor("clip_max", TensorProto.FLOAT, [], [6.0])  # 0维
    clip_min_node = helper.make_node("Constant", [], ["clip_min_value"], value=clip_min)
    clip_max_node = helper.make_node("Constant", [], ["clip_max_value"], value=clip_max)

    # Clip节点 (3个输入: data, min, max)
    clip_node = helper.make_node(
        "Clip",
        inputs=["add_output", "clip_min_value", "clip_max_value"],
        outputs=["output"],
        name="clip_0",
    )

    # 输出张量
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [4, 8])

    # 构建图
    graph = helper.make_graph(
        [add_const_node, add_node, clip_min_node, clip_max_node, clip_node],
        "clip_with_zero_dim_constant",
        [input_tensor],
        [output_tensor],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model = onnx.shape_inference.infer_shapes(model)
    return model


def test_executor_is_weight_should_detect_constant_nodes():
    """测试executor._is_weight应该正确识别Constant节点产生的张量"""
    model = create_clip_with_zero_dim_constant()
    analyzer = ModelAnalyzer.from_model_proto(model)
    transformer = GraphTransformer(analyzer)

    # clip_min_value 和 clip_max_value 是由 Constant 节点产生的0维常数
    # 应该被识别为权重，不需要split
    assert transformer._is_weight("clip_min_value") is True, \
        "clip_min_value should be recognized as weight (from Constant node)"
    assert transformer._is_weight("clip_max_value") is True, \
        "clip_max_value should be recognized as weight (from Constant node)"
    assert transformer._is_weight("add_bias_value") is True, \
        "add_bias_value should be recognized as weight (from Constant node)"

    # input 不是权重
    assert transformer._is_weight("input") is False


def test_clip_split_should_not_create_split_for_constant_inputs():
    """测试Clip切分时不应该为Constant输入创建Split节点"""
    model = create_clip_with_zero_dim_constant()
    analyzer = ModelAnalyzer.from_model_proto(model)
    transformer = GraphTransformer(analyzer)

    # 找到clip_0节点
    clip_node = None
    for node in model.graph.node:
        if node.name == "clip_0":
            clip_node = node
            break

    assert clip_node is not None, "clip_0 node should exist"

    plan = SplitPlan(operator_name="clip_0", parts=2, axis=0)

    # 创建输入split
    split_nodes, input_split_map = transformer._create_input_splits(
        model.graph, clip_node, plan
    )

    # Clip有3个输入: add_output, clip_min_value, clip_max_value
    # 其中 clip_min_value 和 clip_max_value 是Constant，不应该被split
    # 只有 add_output 应该被split
    assert len(split_nodes) == 1, \
        f"Should only create 1 split node for non-constant input, got {len(split_nodes)}"
    assert "add_output" in input_split_map, \
        "add_output should be in input_split_map"
    assert "clip_min_value" not in input_split_map, \
        "clip_min_value (Constant) should NOT be in input_split_map"
    assert "clip_max_value" not in input_split_map, \
        "clip_max_value (Constant) should NOT be in input_split_map"


def test_clip_full_split_integration():
    """测试Clip完整切分集成

    端到端测试，确保Clip切分后不会尝试split Constant输入
    """
    model = create_clip_with_zero_dim_constant()
    analyzer = ModelAnalyzer.from_model_proto(model)
    transformer = GraphTransformer(analyzer)

    plan = SplitPlan(operator_name="clip_0", parts=2, axis=0)

    # 应用切分
    result_model = transformer.apply_split_plan(plan)

    # 验证模型结构
    node_types = [n.op_type for n in result_model.graph.node]

    # 应该有: 2个Clip, 1个Split (只为add_output), 1个Concat
    assert node_types.count("Clip") == 2, \
        f"Should have 2 Clip nodes, got {node_types.count('Clip')}"
    assert node_types.count("Split") == 1, \
        f"Should have 1 Split node (for add_output only), got {node_types.count('Split')}"
    assert node_types.count("Concat") == 1, \
        f"Should have 1 Concat node, got {node_types.count('Concat')}"

    # 验证Split节点只split add_output，不是Constant输入
    split_node = None
    for node in result_model.graph.node:
        if node.op_type == "Split":
            split_node = node
            break

    assert split_node is not None, "Should have a Split node"
    assert split_node.input[0] == "add_output", \
        "Split should be applied to add_output, not to Constant inputs"

    # 验证形状推断成功
    assert result_model.graph.value_info is not None, \
        "Shape inference should succeed"


def test_clip_constant_zero_dim_shape():
    """测试验证Constant产生的0维张量的形状"""
    model = create_clip_with_zero_dim_constant()

    # 检查模型中clip_0节点的输入
    clip_node = None
    for node in model.graph.node:
        if node.name == "clip_0":
            clip_node = node
            break

    assert clip_node is not None, "clip_0 node should exist"
    assert len(clip_node.input) == 3, f"Clip should have 3 inputs, got {len(clip_node.input)}"

    # 验证第2和第3个输入是0维（通过检查value_info）
    # Clip的min和max输入是Constant节点的输出，是0维张量
    clip_min_input = clip_node.input[1]
    clip_max_input = clip_node.input[2]

    # 检查value_info中的形状
    for vi in model.graph.value_info:
        if vi.name == clip_min_input:
            shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            assert shape == [], f"min tensor should be 0-dimensional, got {shape}"
        if vi.name == clip_max_input:
            shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            assert shape == [], f"max tensor should be 0-dimensional, got {shape}"


def test_add_with_zero_dim_constant_split():
    """测试Add算子有0维Constant输入时的切分"""
    model = create_clip_with_zero_dim_constant()
    analyzer = ModelAnalyzer.from_model_proto(model)
    transformer = GraphTransformer(analyzer)

    # 找到add_0节点
    add_node = None
    for node in model.graph.node:
        if node.name == "add_0":
            add_node = node
            break

    assert add_node is not None, "add_0 node should exist"

    plan = SplitPlan(operator_name="add_0", parts=2, axis=0)

    # 创建输入split
    split_nodes, input_split_map = transformer._create_input_splits(
        model.graph, add_node, plan
    )

    # Add有2个输入: input 和 add_bias_value
    # add_bias_value 是0维Constant，不应该被split
    # 只有 input 应该被split
    assert len(split_nodes) == 1, \
        f"Should only create 1 split node for non-constant input, got {len(split_nodes)}"
    assert "input" in input_split_map, \
        "input should be in input_split_map"
    assert "add_bias_value" not in input_split_map, \
        "add_bias_value (0-dim Constant) should NOT be in input_split_map"


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
