"""测试模型分析器"""

from pathlib import Path

import onnx

from onnxsplit.analyzer.model import ModelAnalyzer


def test_analyzer_load_model():
    """测试加载ONNX模型"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    assert analyzer.model is not None
    assert analyzer.graph is not None


def test_analyzer_get_input_info():
    """测试获取模型输入信息"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    inputs = analyzer.get_inputs()
    assert len(inputs) == 1
    assert inputs[0].name == "input"
    assert inputs[0].shape == (1, 3, 8, 8)


def test_analyzer_get_output_info():
    """测试获取模型输出信息"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    outputs = analyzer.get_outputs()
    assert len(outputs) == 1
    assert outputs[0].name == "relu_output"


def test_analyzer_get_operators():
    """测试获取所有算子"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    ops = analyzer.get_operators()
    assert len(ops) == 2  # Conv + Relu (Constant被跳过)

    op_types = [op.op_type for op in ops]
    assert "Conv" in op_types
    assert "Relu" in op_types


def test_analyzer_get_operator_by_name():
    """测试按名称获取算子"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    conv_op = analyzer.get_operator("conv_0")
    assert conv_op is not None
    assert conv_op.op_type == "Conv"
    assert conv_op.name == "conv_0"


def test_analyzer_get_nonexistent_operator():
    """测试获取不存在的算子"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    op = analyzer.get_operator("nonexistent")
    assert op is None


def test_analyzer_tensor_shapes():
    """测试张量形状推断"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    conv_op = analyzer.get_operator("conv_0")
    assert conv_op is not None
    assert len(conv_op.input_tensors) > 0
    assert len(conv_op.output_tensors) > 0


def test_analyzer_matmul_model():
    """测试分析MatMul模型"""
    model_path = Path("tests/fixtures/models/simple_matmul.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    ops = analyzer.get_operators()
    assert len(ops) == 1
    assert ops[0].op_type == "MatMul"


def test_analyzer_branch_model():
    """测试分析有分支的模型"""
    model_path = Path("tests/fixtures/models/model_with_branches.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    ops = analyzer.get_operators()
    assert len(ops) == 3  # 2 Conv + 1 Add

    # 检查输入被两个Conv使用
    conv_ops = [op for op in ops if op.op_type == "Conv"]
    assert len(conv_ops) == 2


def test_analyzer_model_ir_version():
    """测试获取模型IR版本"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    assert analyzer.ir_version > 0
    assert analyzer.opset_version > 0


def test_analyzer_producer_info():
    """测试获取生产者信息"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    # 生产者信息可能为空，但不应该报错
    assert hasattr(analyzer, "producer_name")
    assert hasattr(analyzer, "producer_version")


def test_analyzer_from_model_proto():
    """测试从ModelProto创建分析器"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    model = onnx.load(model_path)

    analyzer = ModelAnalyzer.from_model_proto(model)
    assert analyzer.model is not None

    ops = analyzer.get_operators()
    assert len(ops) >= 1


def test_analyzer_constant_skipped():
    """测试Constant算子被正确处理"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    # Constant算子通常不作为独立算子分析
    ops = analyzer.get_operators()
    for op in ops:
        assert op.op_type != "Constant"


def test_analyzer_get_operator_cached():
    """测试算子查询使用缓存"""
    model_path = Path("tests/fixtures/models/model_with_branches.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    # 多次查询应该返回相同结果
    op1 = analyzer.get_operator("Conv_conv1_out")
    op2 = analyzer.get_operator("Conv_conv1_out")
    op3 = analyzer.get_operator("Conv_conv2_out")

    assert op1 is op2  # 同一个对象引用（缓存）
    assert op1 is not None
    assert op3 is not None
    assert op1.name == "Conv_conv1_out"
    assert op3.name == "Conv_conv2_out"


def test_analyzer_get_operator_nonexistent_returns_none():
    """测试查询不存在的算子返回None（优化后行为不变）"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    op = analyzer.get_operator("definitely_not_exists")
    assert op is None


def test_analyzer_all_operators_accessible():
    """测试所有算子都可通过名称访问"""
    model_path = Path("tests/fixtures/models/model_with_branches.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    all_ops = analyzer.get_operators()
    for op in all_ops:
        by_name = analyzer.get_operator(op.name)
        assert by_name is not None
        assert by_name.name == op.name
