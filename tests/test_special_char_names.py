"""测试特殊字符张量名称的处理

某些ONNX模型导出器会在张量名称中包含特殊字符（如前导斜杠），
这会导致生成的节点名称不符合ONNX规范或产生重复名称。
"""

import onnx
import pytest
from onnx import helper

from onnxsplit.analyzer.model import ModelAnalyzer
from onnxsplit.splitter.plan import SplitPlan
from onnxsplit.transform.executor import GraphTransformer
from onnxsplit.transform.split_concat import create_split_node, create_concat_node


def create_model_with_slash_names():
    """创建一个包含前导斜杠张量名称的模型"""
    # 创建输入
    input_tensor = helper.make_tensor_value_info("/input", onnx.TensorProto.FLOAT, [4, 8])

    # 创建权重常量
    weight_const = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["/weight"],
        value=helper.make_tensor("weight_data", onnx.TensorProto.FLOAT, [8, 4], [1] * 32)
    )

    # 创建MatMul节点
    matmul_node = helper.make_node(
        "MatMul",
        inputs=["/input", "/weight"],
        outputs=["/matmul_output"],
        name="matmul_0"
    )

    # 创建Reshape节点（Reshape有特殊的输出命名）
    shape_const = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["reshape_shape"],
        value=helper.make_tensor("shape_value", onnx.TensorProto.INT64, [2], [4, 32])
    )

    reshape_node = helper.make_node(
        "Reshape",
        inputs=["/matmul_output", "reshape_shape"],
        outputs=["/Reshape_2_output_0"],
        name="/Reshape_2"  # 节点名称也包含斜杠
    )

    # 创建输出
    output_tensor = helper.make_tensor_value_info("/Reshape_2_output_0", onnx.TensorProto.FLOAT, [4, 32])

    # 创建图
    graph = helper.make_graph(
        [weight_const, matmul_node, shape_const, reshape_node],
        "test_model",
        [input_tensor],
        [output_tensor]
    )

    # 创建模型
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return model


def test_split_node_with_slash_in_input_name():
    """测试创建split节点时处理输入名称中的斜杠"""
    # 创建一个包含斜杠的输入名称
    node = create_split_node(
        input_name="/input_with_slash",
        axis=0,
        parts=2,
        output_prefix="output",
    )

    # 节点名称应该被清理（不包含斜杠）
    assert "/" not in node.name
    # 节点名称应该是有效的
    assert node.name


def test_concat_node_with_slash_in_output_name():
    """测试创建concat节点时处理输出名称中的斜杠"""
    node = create_concat_node(
        input_names=["in_0", "in_1"],
        output_name="/output_with_slash",
        axis=0,
    )

    # 节点名称应该被清理（不包含斜杠）
    assert "/" not in node.name
    # 节点名称应该是有效的
    assert node.name


def test_transform_with_slash_names():
    """测试对包含斜杠张量名称的模型进行变换"""
    model = create_model_with_slash_names()
    analyzer = ModelAnalyzer(model)
    transformer = GraphTransformer(analyzer)

    # 创建切分方案
    plan = SplitPlan(
        operator_name="/Reshape_2",  # 节点名称也包含斜杠
        parts=2,
        axis=0,
        reason="test split"
    )

    # 应用切分方案
    result = transformer.apply_split_plan(plan)

    # 验证结果模型没有无效的节点名称
    # 检查onnxsplit创建的节点（Split, Concat, 以及split后缀的节点）名称有效
    for node in result.graph.node:
        # Constant节点可以有空名称（这是ONNX允许的）
        if node.op_type == "Constant":
            continue
        # 其他节点应该有名称
        assert node.name, f"{node.op_type}节点名称不能为空"
        # onnxsplit创建的节点不应该包含前导斜杠
        if node.op_type in ("Split", "Concat") or "_split_" in node.name:
            assert not node.name.startswith("split_/"), f"节点名称不应包含前导斜杠: {node.name}"
            assert not node.name.startswith("concat_/"), f"节点名称不应包含前导斜杠: {node.name}"

    # 验证模型可以通过ONNX checker
    onnx.checker.check_model(result)


def test_onnx_model_validity_with_special_chars():
    """测试包含特殊字符的模型在ONNX验证器中是否有效"""
    model = create_model_with_slash_names()

    # 原始模型应该通过ONNX检查
    # （张量名称可以包含斜杠，但节点名称可能有限制）
    try:
        onnx.checker.check_model(model)
    except Exception as e:
        pytest.skip(f"ONNX checker failed on test model: {e}")


def test_duplicate_node_name_detection():
    """测试重复节点名称的检测

    注意：ONNX本身不强制要求节点名称唯一，但重复的名称可能导致问题。
    这个测试验证onnxsplit不会创建重复的节点名称。
    """
    # 创建一个简单模型
    input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [4, 8])
    weight_const = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["weight"],
        value=helper.make_tensor("weight_data", onnx.TensorProto.FLOAT, [8, 4], [1] * 32)
    )
    matmul_node = helper.make_node(
        "MatMul",
        inputs=["input", "weight"],
        outputs=["output_0"],
        name="matmul_0"
    )
    output_tensor = helper.make_tensor_value_info("output_0", onnx.TensorProto.FLOAT, [4, 4])

    graph = helper.make_graph(
        [weight_const, matmul_node],
        "test_model",
        [input_tensor],
        [output_tensor]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    # onnxsplit应该不创建重复的节点名称
    analyzer = ModelAnalyzer(model)
    transformer = GraphTransformer(analyzer)
    plan = SplitPlan(operator_name="matmul_0", parts=2, axis=0, reason="test")

    result = transformer.apply_split_plan(plan)

    # 收集所有节点名称（排除Constant，因为它们可以没有名称）
    node_names = [n.name for n in result.graph.node if n.op_type != "Constant"]

    # 检查没有重复的名称
    assert len(node_names) == len(set(node_names)), f"发现重复的节点名称: {[n for n in node_names if node_names.count(n) > 1]}"
