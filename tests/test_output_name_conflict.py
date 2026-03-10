"""测试输出名称冲突问题 - clone 节点时 name_i 格式与已存在节点冲突。

这个测试修复了一个 bug：当 split 节点生成 {output}_{i} 格式的输出名称时，
如果图中已存在同名输出（如 transpose_1_0），会导致 SSA 违规。

修复方案：使用 {output}_split_{i} 格式替代 {output}_{i} 格式，
以避免与图中可能已存在的简单数字后缀名称冲突。
"""

import numpy as np
import onnx
import pytest
from onnx import helper, TensorProto

from onnxsplit.analyzer.model import ModelAnalyzer
from onnxsplit.splitter.plan import SplitPlan
from onnxsplit.transform.executor import GraphTransformer


def check_ssa_violations(model: onnx.ModelProto) -> list[str]:
    """检查模型是否有 SSA 违规。

    Returns:
        重复使用的输出名称列表
    """
    output_names = []
    duplicates = []

    for node in model.graph.node:
        for output in node.output:
            if output in output_names:
                if output not in duplicates:
                    duplicates.append(output)
            else:
                output_names.append(output)

    return duplicates


def create_model_with_conflicting_output_names():
    """创建一个模型，其中存在与 split 生成的输出名称格式相同的节点输出。

    场景：
    1. 第一个 Transpose 输出名为 "transpose_1"
    2. 第二个 Transpose 输出名为 "transpose_1_0" (这个名称会在 split transpose_1 时生成！)
    3. 当 split 第一个 Transpose 时，如果使用 {output}_{i} 格式，
       生成的 "transpose_1_0" 与已存在的输出冲突
    """
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [4, 8])

    # 第一个 Transpose 节点，输出名为 "transpose_1"
    transpose1 = helper.make_node(
        "Transpose",
        inputs=["input"],
        outputs=["transpose_1"],
        name="Transpose_1",
        perm=[1, 0],
    )

    # 第二个 Transpose 节点，输出名为 "transpose_1_0"
    # 这个名称与 split transpose_1 时生成的名称格式相同！
    transpose2 = helper.make_node(
        "Transpose",
        inputs=["transpose_1"],
        outputs=["transpose_1_0"],
        name="Transpose_2",
        perm=[1, 0],
    )

    # MatMul 节点使用 transpose_1_0
    weight = helper.make_tensor(
        "weight", TensorProto.FLOAT, [4, 4], vals=np.random.randn(4, 4).astype(np.float32)
    )
    matmul = helper.make_node(
        "MatMul",
        inputs=["transpose_1_0", "weight"],
        outputs=["output"],
        name="MatMul_1",
    )

    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [8, 4])

    graph = helper.make_graph(
        [transpose1, transpose2, matmul],
        "test_model",
        [input_tensor],
        [output],
        [weight],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    return model


def test_split_does_not_create_ssa_violation_with_conflicting_names():
    """测试 split 不会因为输出名称与已存在名称冲突而导致 SSA 违规。"""
    model = create_model_with_conflicting_output_names()

    # Split Transpose_1
    analyzer = ModelAnalyzer(model)
    transformer = GraphTransformer(analyzer)

    plan = SplitPlan(operator_name="Transpose_1", axis=0, parts=2)
    result = transformer.apply_split_plan(plan)

    # 检查没有 SSA 违规
    duplicates = check_ssa_violations(result)
    assert not duplicates, f"SSA violations found: {duplicates}"

    # 验证模型可以通过 ONNX checker
    onnx.checker.check_model(result)


def test_split_uses_unique_output_names():
    """测试 split 生成的输出名称是唯一的，不会与现有名称冲突。"""
    model = create_model_with_conflicting_output_names()

    # 记录原始输出名称
    original_outputs = set()
    for node in model.graph.node:
        original_outputs.update(node.output)

    # Split Transpose_1
    analyzer = ModelAnalyzer(model)
    transformer = GraphTransformer(analyzer)

    plan = SplitPlan(operator_name="Transpose_1", axis=0, parts=2)
    result = transformer.apply_split_plan(plan)

    # 收集新的输出名称
    new_outputs_from_split = set()
    for node in result.graph.node:
        for output in node.output:
            if output not in original_outputs:
                new_outputs_from_split.add(output)

    # 检查新生成的输出名称应该使用 _split_ 前缀而不是简单的数字后缀
    # 例如应该生成 transpose_1_split_0 而不是 transpose_1_0
    split_outputs = [o for o in new_outputs_from_split if "transpose_1" in o and "_split_" in o]
    assert len(split_outputs) >= 2, f"Expected at least 2 split outputs, got: {split_outputs}"


def test_multiple_splits_with_conflicting_patterns():
    """测试多次 split 时不会产生名称冲突。

    场景：
    1. 模型中有多个节点，其输出名称使用 {name}_{i} 格式
    2. split 这些节点时不应产生冲突
    """
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [8, 8])

    # 创建多个节点，使用 {name}_{i} 格式的输出名称
    transpose1 = helper.make_node(
        "Transpose",
        inputs=["input"],
        outputs=["data_0"],
        name="Transpose_1",
        perm=[1, 0],
    )

    # 这个节点的输出名为 data_0_0，与 split data_0 时生成的名称格式相同
    transpose2 = helper.make_node(
        "Transpose",
        inputs=["data_0"],
        outputs=["data_0_0"],
        name="Transpose_2",
        perm=[1, 0],
    )

    # 这个节点的输出名为 data_0_1，也与 split data_0 时生成的名称格式相同
    add = helper.make_node(
        "Add",
        inputs=["data_0", "data_0"],
        outputs=["data_0_1"],
        name="Add_1",
    )

    output1 = helper.make_tensor_value_info("data_0_0", TensorProto.FLOAT, [8, 8])
    output2 = helper.make_tensor_value_info("data_0_1", TensorProto.FLOAT, [8, 8])

    graph = helper.make_graph(
        [transpose1, transpose2, add],
        "test_model",
        [input_tensor],
        [output1, output2],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])

    # Split Transpose_1
    analyzer = ModelAnalyzer(model)
    transformer = GraphTransformer(analyzer)

    plan = SplitPlan(operator_name="Transpose_1", axis=0, parts=2)
    result = transformer.apply_split_plan(plan)

    # 检查没有 SSA 违规
    duplicates = check_ssa_violations(result)
    assert not duplicates, f"SSA violations found: {duplicates}"

    # 验证模型可以通过 ONNX checker
    onnx.checker.check_model(result)
