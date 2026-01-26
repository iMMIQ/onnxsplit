"""测试内存调整器在无法满足约束时正确放弃切分

Bug场景:
- 设置内存限制 -m 参数
- batch 维度较小（如18）
- 内存约束需要切分成很多份（如24、46等）
- 但 18 不能被 24 或 46 整除
- 预期行为：放弃切分（parts=1），而不是应用无效的切分
"""

import onnx
from onnx import TensorProto, helper
import pytest

from onnxsplit.analyzer.model import ModelAnalyzer
from onnxsplit.memory.estimator import MemoryEstimator
from onnxsplit.memory.auto_adjust import AutoSplitAdjuster
from onnxsplit.splitter.plan import SplitPlan


def create_reshape_model_batch_18() -> onnx.ModelProto:
    """创建Reshape模型，batch=18"""
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

    # Add节点
    add_node = helper.make_node(
        "Add",
        inputs=["reshape_output", "add_bias_value"],
        outputs=["output"],
        name="add_0",
    )

    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [18, 8, 8])

    graph = helper.make_graph(
        [const_node, reshape_node, add_const_node, add_node],
        "reshape_batch_test",
        [input_tensor],
        [output_tensor],
    )

    model = helper.make_model(graph)
    model = onnx.shape_inference.infer_shapes(model)
    return model


def test_memory_constraint_should_give_up_when_dimension_not_divisible():
    """测试当内存约束要求的parts无法整除维度时，应放弃切分

    Bug: batch=18, 内存约束需要46份，但 18 % 46 != 0
    预期: 放弃切分，返回 parts=1
    """
    model = create_reshape_model_batch_18()
    analyzer = ModelAnalyzer.from_model_proto(model)
    estimator = MemoryEstimator(analyzer)
    adjuster = AutoSplitAdjuster(estimator)

    # 创建一个初始计划 parts=1
    plan = SplitPlan(operator_name="add_0", parts=1, axis=0, reason="test")

    # 设置一个很小的内存限制，强制需要很多份
    # 假设总内存约 18*64*4 bytes，限制每份 0.2KB
    max_memory_mb = 0.0002

    adjusted_plan = adjuster.adjust_plan(plan, max_memory_mb=max_memory_mb)

    # 验证：放弃切分，返回 parts=1
    assert adjusted_plan.parts == 1, (
        f"Expected parts=1 (no split), got parts={adjusted_plan.parts}. "
        f"When memory constraint requires parts that cannot divide the dimension, "
        f"should give up splitting."
    )

    # 验证原因描述
    assert "not divisible" in adjusted_plan.reason.lower() or "no split" in adjusted_plan.reason.lower(), (
        f"Expected reason to mention giving up split, got: {adjusted_plan.reason}"
    )

    # 验证最终 parts 值确实能整除维度
    add_op = analyzer.get_operator("add_0")
    input_shape = add_op.input_tensors[0].shape
    dim_size = input_shape[0]  # batch 维度
    assert dim_size % adjusted_plan.parts == 0, (
        f"Final parts={adjusted_plan.parts} must divide dimension={dim_size}"
    )


def test_memory_constraint_should_find_valid_parts_when_possible():
    """测试当内存约束可以满足时，应找到有效的parts

    batch=18, 如果内存约束需要3份，18 % 3 == 0，应该成功
    """
    model = create_reshape_model_batch_18()
    analyzer = ModelAnalyzer.from_model_proto(model)
    estimator = MemoryEstimator(analyzer)
    adjuster = AutoSplitAdjuster(estimator)

    # 创建一个初始计划 parts=1
    plan = SplitPlan(operator_name="add_0", parts=1, axis=0, reason="test")

    # 设置内存限制需要约3份
    # 总内存约 0.009 MB
    # 如果限制 0.003 MB, 需要 0.009 / 0.003 = 3 份
    max_memory_mb = 0.003  # 需要约3份

    adjusted_plan = adjuster.adjust_plan(plan, max_memory_mb=max_memory_mb)

    # 验证：找到有效的切分数
    assert adjusted_plan.parts > 1, f"Should find valid parts > 1, got {adjusted_plan.parts}"

    # 验证最终 parts 值确实能整除维度
    add_op = analyzer.get_operator("add_0")
    input_shape = add_op.input_tensors[0].shape
    dim_size = input_shape[0]  # batch 维度 = 18
    assert dim_size % adjusted_plan.parts == 0, (
        f"Final parts={adjusted_plan.parts} must divide dimension={dim_size}"
    )

    # 18的因数: 1, 2, 3, 6, 9, 18
    valid_parts_for_18 = [1, 2, 3, 6, 9, 18]
    assert adjusted_plan.parts in valid_parts_for_18, (
        f"Expected parts in {valid_parts_for_18}, got {adjusted_plan.parts}"
    )


def test_memory_constraint_batch_18_parts_24():
    """测试 batch=18 时，内存约束需要24份的情况

    18 % 24 != 0，应该放弃切分
    """
    model = create_reshape_model_batch_18()
    analyzer = ModelAnalyzer.from_model_proto(model)
    estimator = MemoryEstimator(analyzer)
    adjuster = AutoSplitAdjuster(estimator)

    plan = SplitPlan(operator_name="add_0", parts=1, axis=0, reason="test")

    # 设置内存限制需要约24份
    # 18*64*4 / (1024*1024) ≈ 0.0044 MB
    # 0.0044 / 24 ≈ 0.00018 MB
    max_memory_mb = 0.0002

    adjusted_plan = adjuster.adjust_plan(plan, max_memory_mb=max_memory_mb)

    # 由于18的因数只有 1,2,3,6,9,18，无法满足24份
    # 应该放弃切分或调整到18份
    # 检查最终 parts 必须是18的因数
    add_op = analyzer.get_operator("add_0")
    input_shape = add_op.input_tensors[0].shape
    dim_size = input_shape[0]  # 18

    valid_parts_for_18 = [1, 2, 3, 6, 9, 18]
    assert adjusted_plan.parts in valid_parts_for_18, (
        f"Expected parts in {valid_parts_for_18}, got {adjusted_plan.parts}"
    )
    assert dim_size % adjusted_plan.parts == 0, (
        f"Final parts={adjusted_plan.parts} must divide dimension={dim_size}"
    )


if __name__ == "__main__":
    test_memory_constraint_should_give_up_when_dimension_not_divisible()
    print("test_memory_constraint_should_give_up_when_dimension_not_divisible: PASSED")

    test_memory_constraint_should_find_valid_parts_when_possible()
    print("test_memory_constraint_should_find_valid_parts_when_possible: PASSED")

    test_memory_constraint_batch_18_parts_24()
    print("test_memory_constraint_batch_18_parts_24: PASSED")

    print("\nAll tests passed!")
