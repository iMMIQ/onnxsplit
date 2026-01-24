"""测试Split和Concat节点生成"""

from onnxsplit.transform.split_concat import (
    create_concat_node,
    create_slice_node,
    create_split_node,
)


def test_create_split_node_equal():
    """测试创建均分Split节点"""
    node = create_split_node(
        input_name="input",
        axis=0,
        parts=4,
        output_prefix="split_out",
    )

    assert node.op_type == "Split"
    assert node.input[0] == "input"
    assert len(node.output) == 4
    assert node.output[0] == "split_out_0"
    assert node.output[3] == "split_out_3"


def test_create_split_node_with_split_attr():
    """测试创建带split属性的Split节点"""
    node = create_split_node(
        input_name="input",
        axis=0,
        parts=3,
        output_prefix="out",
        split_sizes=[10, 20, 30],
    )

    assert node.op_type == "Split"
    # 检查split属性
    split_attr = None
    for attr in node.attribute:
        if attr.name == "split":
            split_attr = list(attr.ints)

    assert split_attr == [10, 20, 30]


def test_create_split_node_axis():
    """测试Split节点axis属性"""
    for axis in [0, 1, 2, -1]:
        node = create_split_node(
            input_name="input",
            axis=axis,
            parts=2,
            output_prefix="out",
        )

        axis_attr = None
        for attr in node.attribute:
            if attr.name == "axis":
                if attr.type == 2:  # INT
                    axis_attr = attr.i
                elif attr.type == 7:  # INTS
                    axis_attr = list(attr.ints)

        assert axis_attr == axis or (isinstance(axis_attr, list) and axis_attr[0] == axis)


def test_create_concat_node():
    """测试创建Concat节点"""
    node = create_concat_node(
        input_names=["input_0", "input_1", "input_2"],
        output_name="output",
        axis=0,
    )

    assert node.op_type == "Concat"
    assert len(node.input) == 3
    assert node.input[0] == "input_0"
    assert node.input[2] == "input_2"
    assert node.output[0] == "output"


def test_create_concat_node_axis():
    """测试Concat节点axis属性"""
    for axis in [0, 1, 2]:
        node = create_concat_node(
            input_names=["a", "b"],
            output_name="out",
            axis=axis,
        )

        axis_attr = None
        for attr in node.attribute:
            if attr.name == "axis":
                axis_attr = attr.i

        assert axis_attr == axis


def test_create_slice_node():
    """测试创建Slice节点"""
    node = create_slice_node(
        input_name="input",
        output_name="output",
        starts=[0],
        ends=[10],
        axes=[0],
    )

    assert node.op_type == "Slice"
    assert node.input[0] == "input"
    assert node.output[0] == "output"


def test_create_slice_node_multi_dim():
    """测试创建多维Slice节点"""
    node = create_slice_node(
        input_name="input",
        output_name="output",
        starts=[0, 0],
        ends=[10, 20],
        axes=[0, 1],
    )

    assert node.op_type == "Slice"
    # Slice的输入是: input, starts, ends, axes（不指定steps时不包含）
    assert len(node.input) == 4  # input + starts + ends + axes


def test_create_slice_node_with_steps():
    """测试创建带步长的Slice节点"""
    node = create_slice_node(
        input_name="input",
        output_name="output",
        starts=[0],
        ends=[10],
        axes=[0],
        steps=[2],
    )

    assert node.op_type == "Slice"
    assert len(node.input) == 5


def test_create_split_node_name():
    """测试Split节点命名"""
    node = create_split_node(
        input_name="data",
        axis=0,
        parts=2,
        output_prefix="split",
        node_name="my_split",
    )

    assert node.name == "my_split"


def test_create_concat_node_name():
    """测试Concat节点命名"""
    node = create_concat_node(
        input_names=["a", "b"],
        output_name="out",
        axis=0,
        node_name="my_concat",
    )

    assert node.name == "my_concat"


def test_create_slice_node_name():
    """测试Slice节点命名"""
    node = create_slice_node(
        input_name="input",
        output_name="output",
        starts=[0],
        ends=[10],
        axes=[0],
        node_name="my_slice",
    )

    assert node.name == "my_slice"


def test_create_split_single_part():
    """测试创建单份Split（退化情况）"""
    node = create_split_node(
        input_name="input",
        axis=0,
        parts=1,
        output_prefix="out",
    )

    assert node.op_type == "Split"
    assert len(node.output) == 1


def test_create_concat_single_input():
    """测试创建单输入Concat（退化情况）"""
    node = create_concat_node(
        input_names=["a"],
        output_name="out",
        axis=0,
    )

    assert node.op_type == "Concat"
    assert len(node.input) == 1
