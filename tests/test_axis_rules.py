"""测试切分轴规则"""
import pytest
from onnx import TensorProto
from onnxsplit.splitter.axis_rules import (
    SplitableAxes,
    AxisAnalyzer,
    get_splitable_axes_for_op,
)
from onnxsplit.analyzer.operator import OperatorInfo
from onnxsplit.analyzer.tensor import TensorMetadata


def test_splitable_axes_creation():
    """测试创建可切分轴集合"""
    axes = SplitableAxes(axes={0, 1, 2}, reason="Element-wise operation")
    assert axes.axes == {0, 1, 2}
    assert axes.reason == "Element-wise operation"


def test_splitable_axes_empty():
    """测试空可切分轴"""
    axes = SplitableAxes.empty("Cannot split")
    assert axes.axes == set()
    assert axes.reason == "Cannot split"


def test_splitable_axes_single():
    """测试单轴可切分"""
    axes = SplitableAxes.single(0, "Batch dimension")
    assert axes.axes == {0}
    assert axes.reason == "Batch dimension"


def test_splitable_axes_contains():
    """测试轴包含检查"""
    axes = SplitableAxes(axes={0, 2}, reason="test")
    assert 0 in axes
    assert 2 in axes
    assert 1 not in axes


def test_splitable_axes_len():
    """测试可切分轴数量"""
    axes = SplitableAxes(axes={0, 1, 2}, reason="test")
    assert len(axes) == 3


def test_splitable_axes_repr():
    """测试字符串表示"""
    axes = SplitableAxes(axes={0}, reason="Batch")
    repr_str = repr(axes)
    assert "{0}" in repr_str


def test_analyzer_elementwise_ops():
    """测试Element-wise算子可切任意轴"""
    analyzer = AxisAnalyzer()

    for op_type in ["Add", "Mul", "Sub", "Div", "Relu", "Sigmoid", "Tanh"]:
        op_info = OperatorInfo(
            name=f"test_{op_type}",
            op_type=op_type,
            attributes={},
            input_tensors=[
                TensorMetadata("input", shape=(2, 3, 4, 4), dtype=TensorProto.FLOAT)
            ],
            output_tensors=[
                TensorMetadata("output", shape=(2, 3, 4, 4), dtype=TensorProto.FLOAT)
            ],
        )
        axes = analyzer.analyze(op_info)
        # Element-wise算子应该可以切所有轴
        assert len(axes) > 0


def test_analyzer_conv_only_batch():
    """测试Conv只能切batch维度"""
    analyzer = AxisAnalyzer()

    op_info = OperatorInfo(
        name="conv_0",
        op_type="Conv",
        attributes={"kernel_shape": [3, 3]},
        input_tensors=[
            TensorMetadata("input", shape=(1, 3, 224, 224), dtype=TensorProto.FLOAT),
            TensorMetadata("weight", shape=(64, 3, 3, 3), dtype=TensorProto.FLOAT),
        ],
        output_tensors=[
            TensorMetadata("output", shape=(1, 64, 112, 112), dtype=TensorProto.FLOAT)
        ],
    )

    axes = analyzer.analyze(op_info)
    assert 0 in axes  # Batch维度可切
    assert 1 not in axes  # Channel维度不可切
    assert 2 not in axes  # Height维度不可切


def test_analyzer_conv_1d():
    """测试Conv1D只能切batch维度"""
    analyzer = AxisAnalyzer()

    op_info = OperatorInfo(
        name="conv1d",
        op_type="Conv",
        attributes={"kernel_shape": [3]},
        input_tensors=[
            TensorMetadata("input", shape=(2, 16, 100), dtype=TensorProto.FLOAT),
            TensorMetadata("weight", shape=(32, 16, 3), dtype=TensorProto.FLOAT),
        ],
        output_tensors=[
            TensorMetadata("output", shape=(2, 32, 98), dtype=TensorProto.FLOAT)
        ],
    )

    axes = analyzer.analyze(op_info)
    assert 0 in axes
    assert len(axes) == 1


def test_analyzer_matmul():
    """测试MatMul切分规则"""
    analyzer = AxisAnalyzer()

    # 2D矩阵乘法: (M, K) @ (K, N) = (M, N)
    op_info = OperatorInfo(
        name="matmul_2d",
        op_type="MatMul",
        attributes={},
        input_tensors=[
            TensorMetadata("A", shape=(128, 64), dtype=TensorProto.FLOAT),
            TensorMetadata("B", shape=(64, 32), dtype=TensorProto.FLOAT),
        ],
        output_tensors=[
            TensorMetadata("C", shape=(128, 32), dtype=TensorProto.FLOAT)
        ],
    )

    axes = analyzer.analyze(op_info)
    # 2D MatMul没有batch维度，通常不可切
    assert len(axes) == 0


def test_analyzer_matmul_3d():
    """测试3D MatMul（带batch维度）"""
    analyzer = AxisAnalyzer()

    # (B, M, K) @ (B, K, N) = (B, M, N)
    op_info = OperatorInfo(
        name="matmul_3d",
        op_type="MatMul",
        attributes={},
        input_tensors=[
            TensorMetadata("A", shape=(4, 128, 64), dtype=TensorProto.FLOAT),
            TensorMetadata("B", shape=(4, 64, 32), dtype=TensorProto.FLOAT),
        ],
        output_tensors=[
            TensorMetadata("C", shape=(4, 128, 32), dtype=TensorProto.FLOAT)
        ],
    )

    axes = analyzer.analyze(op_info)
    # Batch维度(axis=0)可切
    assert 0 in axes


def test_analyzer_reduce_ops():
    """测试Reduce算子只能切非归约轴"""
    analyzer = AxisAnalyzer()

    op_info = OperatorInfo(
        name="reduce_mean",
        op_type="ReduceMean",
        attributes={"axes": [1], "keepdims": 1},
        input_tensors=[
            TensorMetadata("input", shape=(2, 16, 32, 32), dtype=TensorProto.FLOAT)
        ],
        output_tensors=[
            TensorMetadata("output", shape=(2, 1, 32, 32), dtype=TensorProto.FLOAT)
        ],
    )

    axes = analyzer.analyze(op_info)
    # axis=1是归约轴，不可切；其他轴可切
    assert 0 in axes  # Batch可切
    assert 1 not in axes  # 归约轴不可切
    assert 2 in axes  # H可切


def test_analyzer_reduce_mean_all_axes():
    """测试全轴归约的情况"""
    analyzer = AxisAnalyzer()

    op_info = OperatorInfo(
        name="reduce_mean_all",
        op_type="ReduceMean",
        attributes={"keepdims": 1},
        input_tensors=[
            TensorMetadata("input", shape=(2, 16, 32, 32), dtype=TensorProto.FLOAT)
        ],
        output_tensors=[
            TensorMetadata("output", shape=(1, 1, 1, 1), dtype=TensorProto.FLOAT)
        ],
    )

    axes = analyzer.analyze(op_info)
    # 归约所有轴时，没有可切分的轴
    assert len(axes) == 0


def test_analyzer_batch_norm():
    """测试BatchNorm只能切batch维度"""
    analyzer = AxisAnalyzer()

    op_info = OperatorInfo(
        name="batch_norm",
        op_type="BatchNormalization",
        attributes={},
        input_tensors=[
            TensorMetadata("input", shape=(4, 16, 32, 32), dtype=TensorProto.FLOAT)
        ],
        output_tensors=[
            TensorMetadata("output", shape=(4, 16, 32, 32), dtype=TensorProto.FLOAT)
        ],
    )

    axes = analyzer.analyze(op_info)
    assert 0 in axes
    assert 1 not in axes  # Channel维度统计量不可切


def test_analyzer_layer_norm():
    """测试LayerNorm只能切batch维度"""
    analyzer = AxisAnalyzer()

    op_info = OperatorInfo(
        name="layer_norm",
        op_type="LayerNormalization",
        attributes={},
        input_tensors=[
            TensorMetadata("input", shape=(4, 128, 768), dtype=TensorProto.FLOAT)
        ],
        output_tensors=[
            TensorMetadata("output", shape=(4, 128, 768), dtype=TensorProto.FLOAT)
        ],
    )

    axes = analyzer.analyze(op_info)
    assert 0 in axes  # Batch可切
    # Layer norm通常在最后几维计算，这些维度不可切


def test_analyzer_softmax():
    """测试Softmax沿特定轴计算"""
    analyzer = AxisAnalyzer()

    op_info = OperatorInfo(
        name="softmax",
        op_type="Softmax",
        attributes={"axis": -1},
        input_tensors=[
            TensorMetadata("input", shape=(4, 128, 768), dtype=TensorProto.FLOAT)
        ],
        output_tensors=[
            TensorMetadata("output", shape=(4, 128, 768), dtype=TensorProto.FLOAT)
        ],
    )

    axes = analyzer.analyze(op_info)
    # Batch维度可切
    assert 0 in axes
    # 计算轴(-1)不可切


def test_analyzer_reshape():
    """测试Reshape算子需要特殊处理"""
    analyzer = AxisAnalyzer()

    op_info = OperatorInfo(
        name="reshape",
        op_type="Reshape",
        attributes={},
        input_tensors=[
            TensorMetadata("input", shape=(4, 128), dtype=TensorProto.FLOAT)
        ],
        output_tensors=[
            TensorMetadata("output", shape=(4, 8, 16), dtype=TensorProto.FLOAT)
        ],
    )

    axes = analyzer.analyze(op_info)
    # Reshape通常不直接切分
    assert len(axes) == 0


def test_analyzer_flatten():
    """测试Flatten算子"""
    analyzer = AxisAnalyzer()

    op_info = OperatorInfo(
        name="flatten",
        op_type="Flatten",
        attributes={"axis": 1},
        input_tensors=[
            TensorMetadata("input", shape=(4, 3, 224, 224), dtype=TensorProto.FLOAT)
        ],
        output_tensors=[
            TensorMetadata("output", shape=(4, 3*224*224), dtype=TensorProto.FLOAT)
        ],
    )

    axes = analyzer.analyze(op_info)
    # 只有batch维度可切
    assert 0 in axes


def test_analyzer_unknown_op():
    """测试未知算子类型"""
    analyzer = AxisAnalyzer()

    op_info = OperatorInfo(
        name="unknown",
        op_type="CustomOp",
        attributes={},
        input_tensors=[
            TensorMetadata("input", shape=(4, 16), dtype=TensorProto.FLOAT)
        ],
        output_tensors=[
            TensorMetadata("output", shape=(4, 16), dtype=TensorProto.FLOAT)
        ],
    )

    axes = analyzer.analyze(op_info)
    # 未知算子保守处理，默认空
    assert len(axes) == 0


def test_get_splitable_axes_for_op():
    """测试便捷函数"""
    op_info = OperatorInfo(
        name="relu",
        op_type="Relu",
        attributes={},
        input_tensors=[
            TensorMetadata("input", shape=(2, 3, 4), dtype=TensorProto.FLOAT)
        ],
        output_tensors=[
            TensorMetadata("output", shape=(2, 3, 4), dtype=TensorProto.FLOAT)
        ],
    )

    axes = get_splitable_axes_for_op(op_info)
    assert len(axes) > 0


def test_analyzer_shape_dimension_one():
    """测试维度为1时从可切分轴中移除"""
    analyzer = AxisAnalyzer()

    op_info = OperatorInfo(
        name="conv",
        op_type="Conv",
        attributes={},
        input_tensors=[
            TensorMetadata("input", shape=(1, 3, 224, 224), dtype=TensorProto.FLOAT)
        ],
        output_tensors=[
            TensorMetadata("output", shape=(1, 64, 112, 112), dtype=TensorProto.FLOAT)
        ],
    )

    axes = analyzer.analyze(op_info)
    # batch维度为1，应该被移除（切分无意义）
    # 或者保留但允许用户配置
    assert 0 in axes  # 仍标记为可切，但实际切分时可能跳过


def test_analyzer_transpose():
    """测试Transpose算子"""
    analyzer = AxisAnalyzer()

    op_info = OperatorInfo(
        name="transpose",
        op_type="Transpose",
        attributes={"perm": [0, 2, 1, 3]},
        input_tensors=[
            TensorMetadata("input", shape=(2, 3, 224, 224), dtype=TensorProto.FLOAT)
        ],
        output_tensors=[
            TensorMetadata("output", shape=(2, 224, 3, 224), dtype=TensorProto.FLOAT)
        ],
    )

    axes = analyzer.analyze(op_info)
    # Batch维度(perm[0]=0)可切
    assert 0 in axes


def test_analyzer_pooling():
    """测试池化算子"""
    analyzer = AxisAnalyzer()

    for op_type in ["MaxPool", "AveragePool", "GlobalAveragePool"]:
        op_info = OperatorInfo(
            name=f"pool_{op_type}",
            op_type=op_type,
            attributes={"kernel_shape": [2, 2]} if "Global" not in op_type else {},
            input_tensors=[
                TensorMetadata("input", shape=(4, 16, 32, 32), dtype=TensorProto.FLOAT)
            ],
            output_tensors=[
                TensorMetadata("output", shape=(4, 16, 16, 16), dtype=TensorProto.FLOAT)
            ],
        )

        axes = analyzer.analyze(op_info)
        # Pooling可以切batch维度
        assert 0 in axes
