"""测试内存调整器的bug

Bug 1: AutoSplitAdjuster.adjust_plan() 在 parts=1 时提前返回
         - 当 parts=1 时，is_split=False
         - adjust_plan 直接返回，不做内存调整
         - 导致即使设置了 max_memory，也不会触发切分

Bug 2: AutoSplitAdjuster 不验证维度整除性
         - adjuster 计算出 parts 后，不验证是否能整除目标维度
         - 可能创建 18 % 4 != 0 这样的非法切分
"""

import onnx
from onnx import TensorProto, helper

from onnxsplit.analyzer.model import ModelAnalyzer
from onnxsplit.config import GlobalConfig, OperatorConfig, SplitConfig
from onnxsplit.memory.auto_adjust import AutoSplitAdjuster
from onnxsplit.memory.estimator import MemoryEstimator
from onnxsplit.splitter.plan import SplitPlan
from onnxsplit.splitter.planner import SplitPlanner
from onnxsplit.transform.executor import GraphTransformer


def create_large_memory_model_batch_18() -> onnx.ModelProto:
    """创建大内存模型，batch=18"""
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [18, 1024, 1024])
    add_const = helper.make_tensor("add_bias", TensorProto.FLOAT, [1], [0.1])
    add_const_node = helper.make_node("Constant", [], ["add_bias_value"], value=add_const)
    add_node = helper.make_node(
        "Add", inputs=["input", "add_bias_value"], outputs=["output"], name="add_0"
    )
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [18, 1024, 1024])
    graph = helper.make_graph([add_const_node, add_node], "test", [input_tensor], [output_tensor])
    model = helper.make_model(graph)
    model = onnx.shape_inference.infer_shapes(model)
    return model


def test_bug1_adjuster_skips_parts_equal_1():
    """Bug 1: adjuster 在 parts=1 时跳过调整"""
    model = create_large_memory_model_batch_18()
    analyzer = ModelAnalyzer.from_model_proto(model)

    add_op = analyzer.get_operator("add_0")
    estimator = MemoryEstimator(analyzer)
    op_mem = estimator.get_operator_memory(add_op)

    # 创建 parts=1 的计划
    plan_parts_1 = SplitPlan(operator_name="add_0", parts=1, axis=0, reason="test")
    print(f"plan.parts = {plan_parts_1.parts}")
    print(f"plan.is_split = {plan_parts_1.is_split}")

    # 设置很低的内存限制，理论上应该增加 parts
    max_memory_mb = 1.0  # 1MB，而算子占用 144MB

    adjuster = AutoSplitAdjuster(estimator, max_parts=256)
    adjusted = adjuster.adjust_plan(plan_parts_1, max_memory_mb)

    print(f"Original parts: {plan_parts_1.parts}")
    print(f"Adjusted parts: {adjusted.parts}")

    # BUG: adjusted.parts 仍然是 1，因为 is_split=False 导致提前返回
    if adjusted.parts == 1:
        print("❌ BUG 1 CONFIRMED: adjuster did not adjust parts=1 plan")
        print(f"   Expected: parts >= {op_mem.total_memory_mb / max_memory_mb:.0f}")
        print(f"   Got: parts = {adjusted.parts}")
    else:
        print(f"✓ Bug 1 fixed: parts adjusted from 1 to {adjusted.parts}")


def test_bug2_adjuster_does_not_validate_divisibility():
    """Bug 2: adjuster 不验证维度整除性"""
    model = create_large_memory_model_batch_18()
    analyzer = ModelAnalyzer.from_model_proto(model)

    add_op = analyzer.get_operator("add_0")
    batch_dim = add_op.input_tensors[0].shape[0]  # 18

    estimator = MemoryEstimator(analyzer)
    op_mem = estimator.get_operator_memory(add_op)

    # 创建 parts=4 的计划（不是从 planner 来的，可能是手动创建或配置来的）
    # 18 % 4 = 2，不能整除
    plan_parts_4 = SplitPlan(operator_name="add_0", parts=4, axis=0, reason="test")
    print(f"\nBatch dimension: {batch_dim}")
    print(f"Plan parts: {plan_parts_4.parts}")
    print(f"{batch_dim} % {plan_parts_4.parts} = {batch_dim % plan_parts_4.parts}")

    # 假设内存限制导致 parts 需要增加到某个不能整除 18 的值
    # 比如 parts=4, 5, 7, 8 等
    max_memory_mb = op_mem.total_memory_mb / 4  # 约 36MB

    adjuster = AutoSplitAdjuster(estimator, max_parts=256)

    # 先测试 parts=1 -> 可能计算出 parts=5 左右
    # 18 的因数: 1, 2, 3, 6, 9, 18
    # 如果计算出 5, 7, 8, 10, 11, 13, 14, 15, 16, 17 都会有问题

    # 直接模拟：假设我们有一个 parts=5 的计划
    # 144 / 5 = 28.8 MB per part
    plan_parts_5 = SplitPlan(operator_name="add_0", parts=5, axis=0, reason="test")
    max_memory_mb2 = 30.0

    adjusted5 = adjuster.adjust_plan(plan_parts_5, max_memory_mb2)
    print(f"\nWith parts=5, max_memory=30MB:")
    print(f"  Adjusted parts: {adjusted5.parts}")
    print(f"  {batch_dim} % {adjusted5.parts} = {batch_dim % adjusted5.parts}")

    if batch_dim % adjusted5.parts != 0:
        print("❌ BUG 2 CONFIRMED: adjuster created parts that don't divide evenly")
    else:
        print("✓ Bug 2 fixed: parts divides evenly")


def test_valid_parts_for_batch_18():
    """测试对于 batch=18，哪些 parts 值是有效的"""
    batch_dim = 18
    valid_parts = [p for p in range(1, 37) if batch_dim % p == 0]
    print(f"\nValid parts for batch_dim={batch_dim}: {valid_parts}")
    print(f"Invalid parts examples: {[p for p in range(1, 37) if batch_dim % p != 0][:10]}")


if __name__ == "__main__":
    print("=" * 60)
    print("Bug 1: Adjuster skips parts=1")
    print("=" * 60)
    test_bug1_adjuster_skips_parts_equal_1()

    print("\n" + "=" * 60)
    print("Bug 2: Adjuster doesn't validate divisibility")
    print("=" * 60)
    test_bug2_adjuster_does_not_validate_divisibility()

    print("\n" + "=" * 60)
    print("Valid parts for batch=18")
    print("=" * 60)
    test_valid_parts_for_batch_18()
