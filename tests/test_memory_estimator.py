"""测试内存估算器"""

from pathlib import Path

import pytest
from onnx import TensorProto

from onnxsplit.analyzer import ModelAnalyzer
from onnxsplit.analyzer.tensor import dtype_to_bytes
from onnxsplit.memory.estimator import (
    MemoryEstimator,
    TensorMemoryInfo,
    estimate_tensor_memory,
)


def test_dtype_to_bytes():
    """测试各数据类型的字节大小"""
    assert dtype_to_bytes(TensorProto.FLOAT) == 4
    assert dtype_to_bytes(TensorProto.FLOAT16) == 2
    assert dtype_to_bytes(TensorProto.BFLOAT16) == 2
    assert dtype_to_bytes(TensorProto.DOUBLE) == 8
    assert dtype_to_bytes(TensorProto.INT8) == 1
    assert dtype_to_bytes(TensorProto.INT16) == 2
    assert dtype_to_bytes(TensorProto.INT32) == 4
    assert dtype_to_bytes(TensorProto.INT64) == 8
    assert dtype_to_bytes(TensorProto.BOOL) == 1
    assert dtype_to_bytes(TensorProto.UINT8) == 1
    assert dtype_to_bytes(TensorProto.COMPLEX64) == 8
    assert dtype_to_bytes(TensorProto.COMPLEX128) == 16
    assert dtype_to_bytes(TensorProto.STRING) == 0


def test_estimate_tensor_memory():
    """测试估算张量内存"""
    # FLOAT32: 100 * 200 * 4 bytes = 80KB
    mem = estimate_tensor_memory((100, 200), TensorProto.FLOAT)
    assert mem == 100 * 200 * 4


def test_estimate_tensor_memory_1d():
    """测试1D张量"""
    mem = estimate_tensor_memory((1024,), TensorProto.FLOAT)
    assert mem == 1024 * 4


def test_estimate_tensor_memory_empty():
    """测试空张量（标量）"""
    mem = estimate_tensor_memory((), TensorProto.FLOAT)
    assert mem == 4  # 标量占用一个元素


def test_estimate_tensor_memory_float16():
    """测试FLOAT16"""
    mem = estimate_tensor_memory((100, 100), TensorProto.FLOAT16)
    assert mem == 100 * 100 * 2


def test_memory_estimator_creation():
    """测试创建内存估算器"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    estimator = MemoryEstimator(analyzer)

    assert estimator is not None


def test_estimator_get_tensor_memory():
    """测试获取张量内存"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    estimator = MemoryEstimator(analyzer)

    inputs = analyzer.get_inputs()
    if inputs:
        info = estimator.get_tensor_memory(inputs[0].name)
        assert info is not None
        assert info.tensor_name == inputs[0].name
        assert info.memory_bytes > 0


def test_estimator_get_operator_memory():
    """测试获取算子内存"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    estimator = MemoryEstimator(analyzer)

    conv_op = analyzer.get_operator("conv_0")
    if conv_op:
        info = estimator.get_operator_memory(conv_op)
        assert info is not None
        assert info.operator_name == "conv_0"
        assert info.total_memory_mb > 0


def test_estimator_get_total_model_memory():
    """测试获取总模型内存"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    estimator = MemoryEstimator(analyzer)

    total = estimator.get_total_model_memory()
    assert total > 0


def test_estimator_get_peak_memory():
    """测试获取峰值内存"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    estimator = MemoryEstimator(analyzer)

    peak = estimator.get_peak_memory()
    assert peak >= 0


def test_estimator_get_memory_breakdown():
    """测试获取内存分解"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    estimator = MemoryEstimator(analyzer)

    breakdown = estimator.get_memory_breakdown()
    assert len(breakdown) > 0
    assert all(info.total_memory_mb >= 0 for info in breakdown)


def test_tensor_memory_info():
    """测试张量内存信息"""
    info = TensorMemoryInfo(
        tensor_name="input",
        shape=(1, 3, 224, 224),
        dtype=TensorProto.FLOAT,
        memory_bytes=1 * 3 * 224 * 224 * 4,
    )

    assert info.tensor_name == "input"
    assert info.size_mb == pytest.approx(0.6, rel=0.1)


def test_operator_memory_info():
    """测试算子内存信息"""
    from onnxsplit.memory.estimator import OperatorMemoryInfo

    info = OperatorMemoryInfo(
        operator_name="conv_0",
        op_type="Conv",
        input_memory_mb=1.0,
        output_memory_mb=0.5,
        weights_memory_mb=0.0,
        total_memory_mb=1.5,
        peak_memory_mb=1.5,
    )

    assert info.operator_name == "conv_0"
    assert info.total_memory_mb == 1.5


def test_estimator_with_dynamic_shape():
    """测试处理动态形状"""
    import onnx
    from onnx import helper

    # 创建带动态形状的模型
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 224, 224])
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, ["batch", 16, 224, 224]
    )
    conv_node = helper.make_node(
        "Conv",
        inputs=["input", "weight"],
        outputs=["output"],
        name="conv_0",
    )
    weight = helper.make_tensor("weight", TensorProto.FLOAT, [16, 3, 3, 3], [0.1] * 432)
    const_node = helper.make_node("Constant", [], ["weight_value"], value=weight)
    conv_node.input[1] = "weight_value"

    graph = helper.make_graph([const_node, conv_node], "test", [input_tensor], [output_tensor])
    model = helper.make_model(graph)
    model = onnx.shape_inference.infer_shapes(model)

    analyzer = ModelAnalyzer.from_model_proto(model)
    estimator = MemoryEstimator(analyzer)

    # 动态形状的内存应该返回0或特殊处理
    breakdown = estimator.get_memory_breakdown()
    # 应该能正常处理，动态维度返回0
    assert len(breakdown) > 0


def test_estimator_weights_memory():
    """测试权重内存计算"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    estimator = MemoryEstimator(analyzer)

    # 获取初始器（权重）
    weights_memory = estimator.get_weights_memory()
    assert weights_memory >= 0


def test_dtype_to_bytes_unknown():
    """测试未知类型"""
    # 假设999是未知类型
    assert dtype_to_bytes(999) == 4  # 默认值


def test_estimate_tensor_memory_large():
    """测试大张量"""
    # 1000x1000x1000 FLOAT32 = 4GB
    mem = estimate_tensor_memory((1000, 1000, 1000), TensorProto.FLOAT)
    assert mem == 4_000_000_000
