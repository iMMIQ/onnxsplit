"""使用onnxruntime验证切分前后模型等价性的测试"""

from pathlib import Path

import numpy as np
import onnx
import onnx.helper
import pytest

from onnxsplit import ModelAnalyzer, SplitConfig, SplitPlanner, GraphTransformer
from onnxsplit.config.schema import OperatorConfig

# 尝试导入onnxruntime，如果不可用则跳过测试
try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False


@pytest.fixture
def resnet18_model_path() -> Path:
    """ResNet18模型路径"""
    return Path(__file__).parent.parent / "models" / "resnet18.onnx"


@pytest.fixture
def resnet18_model(resnet18_model_path: Path) -> onnx.ModelProto:
    """加载ResNet18模型"""
    if not resnet18_model_path.exists():
        pytest.skip(f"Model file not found: {resnet18_model_path}")
    return onnx.load(str(resnet18_model_path))


@pytest.fixture
def sample_input_for_resnet() -> dict[str, np.ndarray]:
    """为ResNet创建示例输入"""
    # ResNet通常接受 [batch_size, 3, 224, 224] 的输入
    # 注意: models/resnet18.onnx 的batch_size固定为1
    np.random.seed(42)
    return {"input": np.random.randn(1, 3, 224, 224).astype(np.float32)}


@pytest.fixture
def simple_conv_model_with_batch() -> onnx.ModelProto:
    """创建一个具有可切分batch维度的简单卷积模型"""
    # 创建输入 (batch_size=4, channels=3, height=8, width=8)
    input_tensor = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, [4, 3, 8, 8]
    )

    # 创建权重
    weight_init = onnx.helper.make_tensor(
        "conv_weight", onnx.TensorProto.FLOAT, [3, 3, 3, 3], np.random.randn(3, 3, 3, 3).astype(np.float32)
    )

    # 创建Conv节点
    conv_node = onnx.helper.make_node(
        "Conv",
        inputs=["input", "conv_weight"],
        outputs=["conv_output"],
        name="conv_layer",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )

    # 创建输出 (名称与conv输出一致)
    output_tensor = onnx.helper.make_tensor_value_info(
        "conv_output", onnx.TensorProto.FLOAT, [4, 3, 8, 8]
    )

    # 创建图
    graph = onnx.helper.make_graph(
        [conv_node],
        "simple_conv_model",
        [input_tensor],
        [output_tensor],
        [weight_init],
    )

    # 创建模型 (使用opset 11以兼容更多版本的onnxruntime)
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 11)])
    model.ir_version = 11  # 设置IR版本以兼容onnxruntime
    return model


@pytest.fixture
def sample_input_for_simple_conv() -> dict[str, np.ndarray]:
    """为简单卷积模型创建示例输入"""
    np.random.seed(42)
    return {"input": np.random.randn(4, 3, 8, 8).astype(np.float32)}


def run_inference(model: onnx.ModelProto, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """使用onnxruntime运行推理"""
    if not ONNXRUNTIME_AVAILABLE:
        pytest.skip("onnxruntime not available")

    # 创建临时文件保存模型
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        temp_path = f.name
        onnx.save(model, temp_path)

    try:
        # 创建推理session，使用最优的可用EP
        # 按性能优先级: CUDA > TensorRT > ROCm > OpenVINO > DNNL > CoreML > XNNPACK > CPU
        available_providers = ort.get_available_providers()
        providers = [p for p in [
            "CUDAExecutionProvider",
            "TensorrtExecutionProvider",
            "ROCmExecutionProvider",
            "OpenVINOExecutionProvider",
            "DnnlExecutionProvider",
            "CoreMLExecutionProvider",
            "XnnpackExecutionProvider",
            "CPUExecutionProvider",
        ] if p in available_providers]
        sess = ort.InferenceSession(temp_path, providers=providers)

        # 准备输入
        input_dict = {}
        for inp in sess.get_inputs():
            if inp.name in inputs:
                input_dict[inp.name] = inputs[inp.name]

        # 运行推理
        outputs = sess.run(None, input_dict)

        # 收集输出
        output_dict = {}
        for i, out in enumerate(sess.get_outputs()):
            output_dict[out.name] = outputs[i]

        return output_dict
    finally:
        import os
        if os.path.exists(temp_path):
            os.unlink(temp_path)


class TestRuntimeEquivalence:
    """测试切分前后模型的运行时等价性"""

    @pytest.mark.skipif(not ONNXRUNTIME_AVAILABLE, reason="onnxruntime not available")
    def test_simple_conv_split_output_matches_original(
        self, simple_conv_model_with_batch: onnx.ModelProto, sample_input_for_simple_conv: dict[str, np.ndarray]
    ) -> None:
        """测试简单卷积模型切分后输出与原模型一致"""
        # 获取原始模型输出
        original_outputs = run_inference(simple_conv_model_with_batch, sample_input_for_simple_conv)

        # 分析模型
        analyzer = ModelAnalyzer(simple_conv_model_with_batch)

        # 创建切分配置 - 切分卷积层为2路
        config = SplitConfig(
            operators={"conv_layer": OperatorConfig(parts=2, axis=0)}
        )

        # 生成切分计划
        planner = SplitPlanner(analyzer, config)
        report = planner.generate()

        # 应用切分
        transformer = GraphTransformer(analyzer)
        split_model = transformer.apply_split_plan(report.plans[0])

        # 获取切分后模型输出
        split_outputs = run_inference(split_model, sample_input_for_simple_conv)

        # 验证输出数量相同
        assert len(original_outputs) == len(split_outputs)

        # 验证每个输出在数值上等价
        for name in original_outputs:
            assert name in split_outputs, f"Output {name} missing in split model"
            np.testing.assert_allclose(
                original_outputs[name],
                split_outputs[name],
                rtol=1e-4,
                atol=1e-5,
                err_msg=f"Output {name} differs between original and split models"
            )

    @pytest.mark.skipif(not ONNXRUNTIME_AVAILABLE, reason="onnxruntime not available")
    def test_simple_conv_four_way_split(
        self, simple_conv_model_with_batch: onnx.ModelProto, sample_input_for_simple_conv: dict[str, np.ndarray]
    ) -> None:
        """测试4路切分的等价性"""
        # 获取原始模型输出
        original_outputs = run_inference(simple_conv_model_with_batch, sample_input_for_simple_conv)

        # 分析模型
        analyzer = ModelAnalyzer(simple_conv_model_with_batch)

        # 创建4路切分配置
        config = SplitConfig(
            operators={"conv_layer": OperatorConfig(parts=4, axis=0)}
        )

        planner = SplitPlanner(analyzer, config)
        report = planner.generate()

        transformer = GraphTransformer(analyzer)
        split_model = transformer.apply_split_plan(report.plans[0])

        # 获取切分后模型输出
        split_outputs = run_inference(split_model, sample_input_for_simple_conv)

        # 验证输出
        for name in original_outputs:
            assert name in split_outputs
            np.testing.assert_allclose(
                original_outputs[name],
                split_outputs[name],
                rtol=1e-4,
                atol=1e-5,
            )

    @pytest.mark.skipif(not ONNXRUNTIME_AVAILABLE, reason="onnxruntime not available")
    def test_simple_conv_different_inputs(
        self, simple_conv_model_with_batch: onnx.ModelProto
    ) -> None:
        """测试不同输入下切分模型的等价性"""
        analyzer = ModelAnalyzer(simple_conv_model_with_batch)

        # 创建切分配置
        config = SplitConfig(
            operators={"conv_layer": OperatorConfig(parts=2, axis=0)}
        )

        planner = SplitPlanner(analyzer, config)
        report = planner.generate()

        transformer = GraphTransformer(analyzer)
        split_model = transformer.apply_split_plan(report.plans[0])

        # 测试不同的随机输入
        for seed in [42, 123, 456, 789]:
            np.random.seed(seed)
            inputs = {"input": np.random.randn(4, 3, 8, 8).astype(np.float32)}

            original_outputs = run_inference(simple_conv_model_with_batch, inputs)
            split_outputs = run_inference(split_model, inputs)

            for name in original_outputs:
                np.testing.assert_allclose(
                    original_outputs[name],
                    split_outputs[name],
                    rtol=1e-4,
                    atol=1e-5,
                    err_msg=f"Seed {seed} mismatch for output {name}"
                )


class TestModelIntegrity:
    """测试切分后模型的完整性"""

    @pytest.mark.skipif(not ONNXRUNTIME_AVAILABLE, reason="onnxruntime not available")
    def test_split_model_has_valid_metadata(
        self, simple_conv_model_with_batch: onnx.ModelProto
    ) -> None:
        """测试切分后模型元数据有效"""
        analyzer = ModelAnalyzer(simple_conv_model_with_batch)

        config = SplitConfig(
            operators={"conv_layer": OperatorConfig(parts=2, axis=0)}
        )

        planner = SplitPlanner(analyzer, config)
        report = planner.generate()

        transformer = GraphTransformer(analyzer)
        split_model = transformer.apply_split_plan(report.plans[0])

        # 使用onnxruntime验证模型可以被加载
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name
            onnx.save(split_model, temp_path)

        try:
            sess = ort.InferenceSession(temp_path)
            # 验证输入输出存在
            assert len(sess.get_inputs()) > 0
            assert len(sess.get_outputs()) > 0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @pytest.mark.skipif(not ONNXRUNTIME_AVAILABLE, reason="onnxruntime not available")
    def test_split_model_preserves_input_output_names(
        self, simple_conv_model_with_batch: onnx.ModelProto
    ) -> None:
        """测试切分后模型保持输入输出名称"""
        analyzer = ModelAnalyzer(simple_conv_model_with_batch)

        config = SplitConfig(
            operators={"conv_layer": OperatorConfig(parts=2, axis=0)}
        )

        planner = SplitPlanner(analyzer, config)
        report = planner.generate()

        transformer = GraphTransformer(analyzer)
        split_model = transformer.apply_split_plan(report.plans[0])

        # 获取原始模型的输入输出名称
        original_inputs = [inp.name for inp in simple_conv_model_with_batch.graph.input]
        original_outputs = [out.name for out in simple_conv_model_with_batch.graph.output]

        # 获取切分后模型的输入输出名称
        split_inputs = [inp.name for inp in split_model.graph.input]
        split_outputs = [out.name for out in split_model.graph.output]

        # 验证输入输出名称保持一致
        assert set(original_inputs) == set(split_inputs)
        assert set(original_outputs) == set(split_outputs)
