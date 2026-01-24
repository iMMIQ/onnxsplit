"""创建简单测试模型"""

import onnx
from onnx import TensorProto, helper


def create_simple_conv_model() -> onnx.ModelProto:
    """创建一个简单的卷积模型用于测试"""
    # 输入
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])

    # 权重
    weight_const = helper.make_tensor("weight", TensorProto.FLOAT, [2, 3, 3, 3], [0.1] * 54)
    weight_node = helper.make_node("Constant", [], ["weight_value"], value=weight_const)

    # 卷积
    conv_node = helper.make_node(
        "Conv",
        inputs=["input", "weight_value"],
        outputs=["conv_output"],
        name="conv_0",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )

    # ReLU
    relu_node = helper.make_node(
        "Relu",
        inputs=["conv_output"],
        outputs=["relu_output"],
        name="relu_0",
    )

    # 输出
    output_tensor = helper.make_tensor_value_info("relu_output", TensorProto.FLOAT, [1, 2, 8, 8])

    # 图
    graph = helper.make_graph(
        [weight_node, conv_node, relu_node],
        "simple_conv",
        [input_tensor],
        [output_tensor],
    )

    # 模型
    model = helper.make_model(graph)
    model = onnx.shape_inference.infer_shapes(model)
    return model


def create_matmul_model() -> onnx.ModelProto:
    """创建一个MatMul模型"""
    input_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 4])
    input_b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [4, 3])

    matmul_node = helper.make_node("MatMul", inputs=["A", "B"], outputs=["C"], name="matmul_0")

    output_c = helper.make_tensor_value_info("C", TensorProto.FLOAT, [2, 3])

    graph = helper.make_graph([matmul_node], "matmul_model", [input_a, input_b], [output_c])

    model = helper.make_model(graph)
    model = onnx.shape_inference.infer_shapes(model)
    return model


def create_model_with_branches() -> onnx.ModelProto:
    """创建有分支的模型"""
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])

    # 分支1
    conv1 = helper.make_node(
        "Conv",
        inputs=["input", "w1"],
        outputs=["conv1_out"],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )

    # 分支2
    conv2 = helper.make_node(
        "Conv",
        inputs=["input", "w2"],
        outputs=["conv2_out"],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )

    # 合并
    add = helper.make_node("Add", inputs=["conv1_out", "conv2_out"], outputs=["output"])

    w1 = helper.make_tensor("w1", TensorProto.FLOAT, [2, 3, 3, 3], [0.1] * 54)
    w2 = helper.make_tensor("w2", TensorProto.FLOAT, [2, 3, 3, 3], [0.2] * 54)

    const1 = helper.make_node("Constant", [], ["w1_value"], value=w1)
    const2 = helper.make_node("Constant", [], ["w2_value"], value=w2)

    # 修正conv节点的输入
    conv1.input[1] = "w1_value"
    conv2.input[1] = "w2_value"

    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 8, 8])

    graph = helper.make_graph(
        [const1, const2, conv1, conv2, add],
        "branch_model",
        [input_tensor],
        [output_tensor],
    )

    model = helper.make_model(graph)
    model = onnx.shape_inference.infer_shapes(model)
    return model


if __name__ == "__main__":
    import pathlib

    output_dir = pathlib.Path("tests/fixtures/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存模型
    onnx.save(create_simple_conv_model(), output_dir / "simple_conv.onnx")
    onnx.save(create_matmul_model(), output_dir / "simple_matmul.onnx")
    onnx.save(create_model_with_branches(), output_dir / "model_with_branches.onnx")

    print("Test models created successfully!")
