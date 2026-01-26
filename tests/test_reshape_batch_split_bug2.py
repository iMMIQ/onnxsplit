"""测试直接应用非法切分计划时的形状推断失败

Bug场景:
- 直接创建一个parts=4的SplitPlan (绕过planner的验证)
- 应用到batch=18的Reshape输出
- 验证是否能正确处理
"""

import onnx
from onnx import TensorProto, helper

from onnxsplit.analyzer.model import ModelAnalyzer
from onnxsplit.splitter.plan import SplitPlan
from onnxsplit.transform.executor import GraphTransformer


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

    # Add常量 (广播)
    add_const = helper.make_tensor("add_bias", TensorProto.FLOAT, [1], [0.1])
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


def test_direct_split_plan_with_invalid_parts():
    """测试直接创建非法的SplitPlan并应用

    这模拟了planner验证被绕过的情况
    """
    model = create_reshape_model_batch_18()
    analyzer = ModelAnalyzer.from_model_proto(model)

    # 创建一个parts=4的切分计划 (18不能被4整除)
    invalid_plan = SplitPlan(
        operator_name="add_0",
        parts=4,  # 18 % 4 != 0
        axis=0,
        reason="test",
    )

    transformer = GraphTransformer(analyzer)

    # 应用切分计划
    result_model = transformer.apply_split_plan(invalid_plan)

    # 验证结果
    # 检查Split节点
    split_nodes = [n for n in result_model.graph.node if n.op_type == "Split"]

    assert len(split_nodes) > 0, "Should have created Split nodes"

    for split_node in split_nodes:
        # 获取axis属性
        axis = 0
        for attr in split_node.attribute:
            if attr.name == "axis":
                axis = attr.i
                break

        # 检查输入张量
        input_name = split_node.input[0]

        # 在value_info中查找输入形状
        input_shape = None
        for vi in result_model.graph.value_info:
            if vi.name == input_name:
                input_shape = tuple(d.dim_value for d in vi.type.tensor_type.shape.dim)
                break

        if input_shape and len(input_shape) > axis:
            dim_size = input_shape[axis]
            num_parts = len(split_node.output)

            # 这里是bug所在: 如果创建了不均匀的切分，形状推断会失败
            # 或产生不正确的输出形状
            print(f"Split node: input={input_name}, shape={input_shape}, axis={axis}, parts={num_parts}")
            print(f"  dim_size={dim_size}, dim_size % parts = {dim_size % num_parts}")

            # 如果dim_size不能被parts整除，这就是问题所在
            if dim_size % num_parts != 0:
                # 这是bug情况
                print(f"WARNING: Invalid split detected! {dim_size} cannot be split into {num_parts} parts")

    # 形状推断应该已经运行，检查是否成功
    # 如果形状推断失败，模型可能是无效的
    assert result_model is not None


def test_shape_inference_with_invalid_split():
    """测试形状推断在非法切分时的行为"""
    model = create_reshape_model_batch_18()
    analyzer = ModelAnalyzer.from_model_proto(model)

    # 获取add_0的输入形状
    add_op = analyzer.get_operator("add_0")
    input_shape = add_op.input_tensors[0].shape
    print(f"Add input shape: {input_shape}")  # 应该是 (18, 8, 8)

    # 创建parts=4的计划
    invalid_plan = SplitPlan(
        operator_name="add_0",
        parts=4,
        axis=0,
        reason="test",
    )

    transformer = GraphTransformer(analyzer)

    # 应用切分
    result_model = transformer.apply_split_plan(invalid_plan)

    # 检查形状推断是否成功
    # ONNX shape_inference.infer_shapes会抛出异常如果形状不一致
    # 或者静默失败但产生无效的形状

    # 让我们手动运行形状推断并捕获错误
    try:
        validated_model = onnx.shape_inference.infer_shapes(result_model)
        print("Shape inference succeeded (no exception thrown)")
    except Exception as e:
        print(f"Shape inference failed with error: {e}")
        raise


if __name__ == "__main__":
    print("Testing direct split plan with invalid parts...")
    test_direct_split_plan_with_invalid_parts()
    print("\nTesting shape inference with invalid split...")
    test_shape_inference_with_invalid_split()
    print("\nTests completed!")
