"""测试Mul算子中权重batch维度与数据不匹配的bug复现

Bug描述:
- Mul节点有两个输入：A是4维常量权重 [N, C, H, W]，B是Cast的结果 [N, C, H, W]
- B被split了，但A（权重）没有
- 导致batch维度不匹配：A=[N, C, H, W] vs B_split=[N/2, C, H, W]
- Element-wise操作无法进行

根本原因:
- _is_weight只检查initializer/Constant，没有考虑形状是否需要同步split
- 当权重的batch维度与数据输入相同时，只split数据输入会导致形状不匹配
"""

import onnx
import pytest
from onnx import TensorProto, helper

from onnxsplit.analyzer.model import ModelAnalyzer
from onnxsplit.splitter.plan import SplitPlan
from onnxsplit.transform.executor import GraphTransformer


def create_mul_with_4d_weight() -> onnx.ModelProto:
    """创建一个Mul模型，其中一个输入是4维权重，另一个是Cast的结果

    模型结构:
    input: [4, 3, 8, 8] -> Cast(float32) -> [4, 3, 8, 8] -> Mul(weight=[4, 3, 8, 8]) -> output

    权重形状: [4, 3, 8, 8] - 注意batch维度是4，与数据相同
    """
    # 输入张量 - float32
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [4, 3, 8, 8])

    # Cast节点 (input已经是float32，但模拟Cast操作)
    cast_node = helper.make_node(
        "Cast",
        inputs=["input"],
        outputs=["cast_output"],
        name="cast_0",
        to=TensorProto.FLOAT,
    )

    # 4维权重 - 注意这是[4, 3, 8, 8]，batch维度是4
    weight_data = [0.5] * (4 * 3 * 8 * 8)  # 全部0.5
    weight_tensor = helper.make_tensor(
        "mul_weight",
        TensorProto.FLOAT,
        [4, 3, 8, 8],  # 4维，batch=4
        weight_data,
    )

    # Mul节点
    mul_node = helper.make_node(
        "Mul",
        inputs=["cast_output", "mul_weight"],
        outputs=["output"],
        name="mul_0",
    )

    # 输出张量
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [4, 3, 8, 8])

    # 构建图
    graph = helper.make_graph(
        [cast_node, mul_node],
        "mul_with_4d_weight",
        [input_tensor],
        [output_tensor],
        [weight_tensor],  # initializer
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model = onnx.shape_inference.infer_shapes(model)
    return model


def test_mul_weight_batch_mismatch_should_not_split():
    """测试Mul有4维权重且batch维度相同时，不应该split

    当权重的batch维度与数据输入相同时，split数据输入会导致形状不匹配。
    这种情况下应该检测到冲突并拒绝split。
    """
    model = create_mul_with_4d_weight()
    analyzer = ModelAnalyzer.from_model_proto(model)

    # 验证模型结构
    mul_op = analyzer.get_operator("mul_0")
    assert mul_op is not None

    # 验证输入形状（analyzer只跟踪非权重输入）
    # input_0 (cast_output): [4, 3, 8, 8]
    input_0_shape = mul_op.input_tensors[0].shape
    assert input_0_shape == (4, 3, 8, 8), f"Expected [4, 3, 8, 8], got {input_0_shape}"

    # 从initializer获取权重形状
    weight_shape = None
    for init in model.graph.initializer:
        if init.name == "mul_weight":
            weight_shape = tuple(init.dims)
            break
    assert weight_shape == (4, 3, 8, 8), f"Expected [4, 3, 8, 8], got {weight_shape}"

    # 验证 mul_weight 是权重
    transformer = GraphTransformer(analyzer)
    assert transformer._is_weight("mul_weight") is True
    assert transformer._is_weight("cast_output") is False

    # 获取mul_0节点
    mul_node = None
    for node in model.graph.node:
        if node.name == "mul_0":
            mul_node = node
            break
    assert mul_node is not None

    # 尝试split axis=0, parts=2
    plan = SplitPlan(operator_name="mul_0", parts=2, axis=0)

    # 创建输入split
    split_nodes, input_split_map = transformer._create_input_splits(
        model.graph, mul_node, plan
    )

    # 修复后：由于权重batch维度与数据相同，split被拒绝
    assert len(split_nodes) == 0, "Should not create split nodes when weight batch dimension matches data"
    assert len(input_split_map) == 0, "Should not have any input splits when weight batch dimension matches data"

    # 验证形状兼容性检查
    assert transformer._check_weight_shape_compatibility(mul_node, axis=0) is False, \
        "Weight shape compatibility check should return False"


def test_mul_split_creates_shape_mismatch():
    """测试验证split后的形状不匹配问题

    修复后：当权重batch维度与数据相同时，split被拒绝，
    返回原始模型（不split）。
    """
    model = create_mul_with_4d_weight()
    analyzer = ModelAnalyzer.from_model_proto(model)
    transformer = GraphTransformer(analyzer)

    plan = SplitPlan(operator_name="mul_0", parts=2, axis=0)

    # 应用split
    # 修复后：由于形状不兼容，不执行split，返回原始模型
    result_model = transformer.apply_split_plan(plan)

    # 检查结果模型中的Mul节点
    mul_nodes = [n for n in result_model.graph.node if n.op_type == "Mul"]

    # 应该只有1个Mul节点（原始的），因为split被拒绝了
    assert len(mul_nodes) == 1, f"Should have 1 Mul node (no split), got {len(mul_nodes)}"

    # 检查Mul节点的输入没有被split
    mul_0 = mul_nodes[0]
    assert mul_0.input[0] == "cast_output"  # 没有split
    assert mul_0.input[1] == "mul_weight"

    # 验证没有创建Split节点
    split_nodes = [n for n in result_model.graph.node if n.op_type == "Split"]
    assert len(split_nodes) == 0, "Should not create Split nodes when weight shape is incompatible"


def test_mul_with_batch1_weight_should_split():
    """测试当权重batch维度为1时，应该可以正常split

    如果权重的形状是 [1, C, H, W]，数据是 [N, C, H, W]，
    那么split数据输入是可以的，因为可以广播。
    """
    # 创建权重batch为1的模型
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [4, 3, 8, 8])

    cast_node = helper.make_node(
        "Cast",
        inputs=["input"],
        outputs=["cast_output"],
        name="cast_0",
        to=TensorProto.FLOAT,
    )

    # 权重形状是 [1, 3, 8, 8] - batch维度为1
    weight_data = [0.5] * (1 * 3 * 8 * 8)
    weight_tensor = helper.make_tensor(
        "mul_weight",
        TensorProto.FLOAT,
        [1, 3, 8, 8],  # batch=1，可以广播
        weight_data,
    )

    mul_node = helper.make_node(
        "Mul",
        inputs=["cast_output", "mul_weight"],
        outputs=["output"],
        name="mul_0",
    )

    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [4, 3, 8, 8])

    graph = helper.make_graph(
        [cast_node, mul_node],
        "mul_with_batch1_weight",
        [input_tensor],
        [output_tensor],
        [weight_tensor],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model = onnx.shape_inference.infer_shapes(model)

    analyzer = ModelAnalyzer.from_model_proto(model)
    transformer = GraphTransformer(analyzer)

    plan = SplitPlan(operator_name="mul_0", parts=2, axis=0)

    # 这种情况下应该可以正常split
    split_nodes, input_split_map = transformer._create_input_splits(
        model.graph, mul_node, plan
    )

    # 只有cast_output被split
    assert "cast_output" in input_split_map
    assert "mul_weight" not in input_split_map

    # 结果模型应该形状有效
    result_model = transformer.apply_split_plan(plan)

    # 验证形状推断成功
    assert result_model.graph.value_info is not None


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
