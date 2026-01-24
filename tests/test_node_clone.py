"""测试节点克隆功能"""

import pytest
from onnx import TensorProto, helper

from onnxsplit.transform.node_clone import clone_node, generate_split_name


def test_generate_split_name():
    """测试生成切分后的节点名"""
    assert generate_split_name("conv_0", 0) == "conv_0_split_0"
    assert generate_split_name("conv_0", 1) == "conv_0_split_1"
    assert generate_split_name("matmul", 5) == "matmul_split_5"


def test_generate_split_name_custom_suffix():
    """测试自定义后缀"""
    assert generate_split_name("conv_0", 0, suffix="part") == "conv_0_part_0"


def test_clone_simple_node():
    """测试克隆简单节点"""
    node = helper.make_node(
        "Relu",
        inputs=["input"],
        outputs=["output"],
        name="relu_0",
    )

    cloned = clone_node(node, suffix="_split_0", new_outputs=["output_0"])

    assert cloned.op_type == "Relu"
    assert cloned.name == "relu_0_split_0"
    assert list(cloned.input) == ["input"]
    assert list(cloned.output) == ["output_0"]


def test_clone_node_with_attributes():
    """测试克隆带属性的节点"""
    node = helper.make_node(
        "Conv",
        inputs=["input", "weight"],
        outputs=["output"],
        name="conv_0",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )

    cloned = clone_node(node, suffix="_split_0", new_outputs=["output_0"])

    assert cloned.op_type == "Conv"
    # 检查属性保留
    kernel_shape = None
    pads = None
    for attr in cloned.attribute:
        if attr.name == "kernel_shape":
            kernel_shape = list(attr.ints)
        elif attr.name == "pads":
            pads = list(attr.ints)

    assert kernel_shape == [3, 3]
    assert pads == [1, 1, 1, 1]


def test_clone_node_multiple_outputs():
    """测试克隆多输出节点"""
    node = helper.make_node(
        "Split",
        inputs=["input"],
        outputs=["output_0", "output_1", "output_2"],
        name="split_0",
        axis=0,
    )

    cloned = clone_node(
        node,
        suffix="_split_0",
        new_outputs=["output_0_0", "output_1_0", "output_2_0"],
    )

    assert len(cloned.output) == 3
    assert cloned.output[0] == "output_0_0"


def test_clone_node_preserve_domain():
    """测试保留domain信息"""
    node = helper.make_node(
        "CustomOp",
        inputs=["input"],
        outputs=["output"],
        name="custom_0",
        domain="custom.domain",
    )

    cloned = clone_node(node, suffix="_copy", new_outputs=["output_copy"])

    assert cloned.domain == "custom.domain"


def test_clone_node_without_name():
    """测试克隆无名称节点"""
    node = helper.make_node(
        "Relu",
        inputs=["input"],
        outputs=["output"],
        # 没有name
    )

    cloned = clone_node(node, suffix="_split_0", new_outputs=["output_0"])

    # 应该生成默认名称
    assert "_split_0" in cloned.name


def test_clone_batch_norm():
    """测试克隆BatchNorm（多属性）"""
    node = helper.make_node(
        "BatchNormalization",
        inputs=["input", "scale", "b", "mean", "var"],
        outputs=["output"],
        name="bn_0",
        epsilon=0.001,
        momentum=0.9,
    )

    cloned = clone_node(node, suffix="_split_0", new_outputs=["output_0"])

    assert cloned.op_type == "BatchNormalization"
    epsilon = None
    momentum = None
    for attr in cloned.attribute:
        if attr.name == "epsilon":
            epsilon = attr.f
        elif attr.name == "momentum":
            momentum = attr.f

    assert epsilon == pytest.approx(0.001)
    assert momentum == pytest.approx(0.9)


def test_clone_transpose():
    """测试克隆Transpose（perm属性）"""
    node = helper.make_node(
        "Transpose",
        inputs=["input"],
        outputs=["output"],
        name="transpose_0",
        perm=[0, 2, 3, 1],
    )

    cloned = clone_node(node, suffix="_split_0", new_outputs=["output_0"])

    perm = None
    for attr in cloned.attribute:
        if attr.name == "perm":
            perm = list(attr.ints)

    assert perm == [0, 2, 3, 1]


def test_clone_reshape():
    """测试克隆Reshape"""
    node = helper.make_node(
        "Reshape",
        inputs=["input", "shape"],
        outputs=["output"],
        name="reshape_0",
    )

    cloned = clone_node(node, suffix="_split_0", new_outputs=["output_0"])

    assert cloned.op_type == "Reshape"
    assert len(cloned.input) == 2


def test_clone_multiple():
    """测试克隆多个副本"""
    node = helper.make_node(
        "Relu",
        inputs=["input"],
        outputs=["output"],
        name="relu_0",
    )

    clones = []
    for i in range(3):
        cloned = clone_node(
            node,
            suffix=f"_split_{i}",
            new_outputs=[f"output_{i}"],
        )
        clones.append(cloned)

    assert len(clones) == 3
    assert clones[0].name == "relu_0_split_0"
    assert clones[1].name == "relu_0_split_1"
    assert clones[2].name == "relu_0_split_2"


def test_clone_gather():
    """测试克隆Gather（带int属性）"""
    node = helper.make_node(
        "Gather",
        inputs=["input", "indices"],
        outputs=["output"],
        name="gather_0",
        axis=1,
    )

    cloned = clone_node(node, suffix="_split_0", new_outputs=["output_0"])

    axis = None
    for attr in cloned.attribute:
        if attr.name == "axis":
            axis = attr.i

    assert axis == 1


def test_clone_cast():
    """测试克隆Cast（带to属性）"""
    node = helper.make_node(
        "Cast",
        inputs=["input"],
        outputs=["output"],
        name="cast_0",
        to=TensorProto.INT64,
    )

    cloned = clone_node(node, suffix="_split_0", new_outputs=["output_0"])

    to_attr = None
    for attr in cloned.attribute:
        if attr.name == "to":
            to_attr = attr.i

    assert to_attr == TensorProto.INT64
