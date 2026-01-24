"""测试算子信息结构"""
import pytest
from onnx import TensorProto, helper
from onnxsplit.analyzer.tensor import TensorMetadata
from onnxsplit.analyzer.operator import OperatorInfo


def test_operator_info_creation():
    """测试创建算子信息"""
    op = OperatorInfo(
        name="/model/Conv_0",
        op_type="Conv",
        attributes={"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]},
        input_tensors=[
            TensorMetadata("input", shape=(1, 3, 224, 224), dtype=TensorProto.FLOAT),
            TensorMetadata("weight", shape=(64, 3, 3, 3), dtype=TensorProto.FLOAT),
        ],
        output_tensors=[
            TensorMetadata("output", shape=(1, 64, 112, 112), dtype=TensorProto.FLOAT),
        ],
    )
    assert op.name == "/model/Conv_0"
    assert op.op_type == "Conv"
    assert op.attributes["kernel_shape"] == [3, 3]
    assert len(op.input_tensors) == 2
    assert len(op.output_tensors) == 1


def test_operator_info_input_memory():
    """测试算子输入内存计算"""
    op = OperatorInfo(
        name="test",
        op_type="Add",
        attributes={},
        input_tensors=[
            TensorMetadata("a", shape=(1000, 1000), dtype=TensorProto.FLOAT),
            TensorMetadata("b", shape=(1000, 1000), dtype=TensorProto.FLOAT),
        ],
        output_tensors=[
            TensorMetadata("c", shape=(1000, 1000), dtype=TensorProto.FLOAT),
        ],
    )
    # 两个输入各约3.81MB (1000*1000*4/1024/1024)
    assert op.input_memory_mb == pytest.approx(7.63, rel=0.01)


def test_operator_info_output_memory():
    """测试算子输出内存计算"""
    op = OperatorInfo(
        name="test",
        op_type="Relu",
        attributes={},
        input_tensors=[
            TensorMetadata("input", shape=(500, 500), dtype=TensorProto.FLOAT),
        ],
        output_tensors=[
            TensorMetadata("output", shape=(500, 500), dtype=TensorProto.FLOAT),
        ],
    )
    # 输出约0.95MB (500*500*4/1024/1024)
    assert op.output_memory_mb == pytest.approx(0.95, rel=0.01)


def test_operator_info_total_memory():
    """测试算子总内存计算"""
    op = OperatorInfo(
        name="test",
        op_type="Add",
        attributes={},
        input_tensors=[
            TensorMetadata("a", shape=(1000, 1000), dtype=TensorProto.FLOAT),
            TensorMetadata("b", shape=(1000, 1000), dtype=TensorProto.FLOAT),
        ],
        output_tensors=[
            TensorMetadata("c", shape=(1000, 1000), dtype=TensorProto.FLOAT),
        ],
    )
    # 输入7.63MB + 输出3.81MB = 11.44MB
    assert op.total_memory_mb == pytest.approx(11.44, rel=0.01)


def test_operator_info_no_inputs():
    """测试无输入的算子"""
    op = OperatorInfo(
        name="test",
        op_type="Constant",
        attributes={},
        input_tensors=[],
        output_tensors=[
            TensorMetadata("output", shape=(10,), dtype=TensorProto.FLOAT),
        ],
    )
    assert op.input_memory_mb == 0
    assert op.output_memory_mb > 0


def test_operator_info_no_outputs():
    """测试无输出的算子"""
    op = OperatorInfo(
        name="test",
        op_type="Dropout",
        attributes={},
        input_tensors=[
            TensorMetadata("input", shape=(100,), dtype=TensorProto.FLOAT),
        ],
        output_tensors=[],
    )
    assert op.output_memory_mb == 0
    assert op.input_memory_mb > 0


def test_operator_info_repr():
    """测试算子信息字符串表示"""
    op = OperatorInfo(
        name="/model/Conv_0",
        op_type="Conv",
        attributes={},
        input_tensors=[],
        output_tensors=[],
    )
    repr_str = repr(op)
    assert "Conv" in repr_str
    assert "/model/Conv_0" in repr_str


def test_operator_info_attribute_get():
    """测试获取算子属性"""
    op = OperatorInfo(
        name="test",
        op_type="Conv",
        attributes={"kernel_shape": [3, 3], "strides": [1, 1]},
        input_tensors=[],
        output_tensors=[],
    )
    assert op.get_attribute("kernel_shape") == [3, 3]
    assert op.get_attribute("strides") == [1, 1]


def test_operator_info_attribute_get_default():
    """测试获取不存在的属性返回默认值"""
    op = OperatorInfo(
        name="test",
        op_type="Conv",
        attributes={},
        input_tensors=[],
        output_tensors=[],
    )
    assert op.get_attribute("nonexistent", default=5) == 5
    assert op.get_attribute("nonexistent") is None


def test_operator_info_get_input_shape():
    """测试获取输入形状"""
    op = OperatorInfo(
        name="test",
        op_type="Add",
        attributes={},
        input_tensors=[
            TensorMetadata("a", shape=(2, 3, 4), dtype=TensorProto.FLOAT),
            TensorMetadata("b", shape=(2, 3, 4), dtype=TensorProto.FLOAT),
        ],
        output_tensors=[],
    )
    assert op.get_input_shape(0) == (2, 3, 4)
    assert op.get_input_shape(1) == (2, 3, 4)


def test_operator_info_get_input_shape_out_of_bounds():
    """测试获取不存在的输入形状"""
    op = OperatorInfo(
        name="test",
        op_type="Add",
        attributes={},
        input_tensors=[
            TensorMetadata("a", shape=(2, 3), dtype=TensorProto.FLOAT),
        ],
        output_tensors=[],
    )
    assert op.get_input_shape(5) is None
    assert op.get_input_shape(1) is None


def test_operator_info_get_output_shape():
    """测试获取输出形状"""
    op = OperatorInfo(
        name="test",
        op_type="Relu",
        attributes={},
        input_tensors=[],
        output_tensors=[
            TensorMetadata("output", shape=(1, 64, 56, 56), dtype=TensorProto.FLOAT),
        ],
    )
    assert op.get_output_shape(0) == (1, 64, 56, 56)


def test_operator_info_from_node_proto():
    """测试从ONNX NodeProto创建算子信息"""
    node = helper.make_node(
        "Conv",
        inputs=["input", "weight", "bias"],
        outputs=["output"],
        name="conv_0",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )

    # 此时没有形状信息，需要后续补充
    op = OperatorInfo.from_node_proto(node)
    assert op.name == "conv_0"
    assert op.op_type == "Conv"
    assert op.attributes["kernel_shape"] == [3, 3]
    assert op.attributes["pads"] == [1, 1, 1, 1]
    assert op.input_names == ["input", "weight", "bias"]
    assert op.output_names == ["output"]


def test_operator_info_with_dynamic_shape():
    """测试处理动态形状（包含-1或None）"""
    op = OperatorInfo(
        name="test",
        op_type="Reshape",
        attributes={},
        input_tensors=[
            # 动态batch
            TensorMetadata("input", shape=(-1, 3, 224, 224), dtype=TensorProto.FLOAT),
        ],
        output_tensors=[
            TensorMetadata("output", shape=(-1, 3, 224, 224), dtype=TensorProto.FLOAT),
        ],
    )
    # 动态维度无法计算内存，应该返回0或特殊处理
    assert op.input_memory_mb == 0
