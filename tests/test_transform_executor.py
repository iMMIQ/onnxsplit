"""图变换执行器测试"""

import tempfile
from pathlib import Path

import onnx
import pytest
from onnx import ModelProto

from onnxsplit.analyzer.model import ModelAnalyzer
from onnxsplit.splitter.plan import SplitPlan
from onnxsplit.transform.executor import (
    GraphTransformer,
    TransformContext,
    TransformResult,
)


@pytest.fixture
def simple_conv_model() -> ModelProto:
    """加载简单的卷积模型"""
    model_path = Path(__file__).parent / "fixtures" / "models" / "simple_conv.onnx"
    return onnx.load(str(model_path))


@pytest.fixture
def simple_matmul_model() -> ModelProto:
    """加载简单的矩阵乘法模型"""
    model_path = Path(__file__).parent / "fixtures" / "models" / "simple_matmul.onnx"
    return onnx.load(str(model_path))


@pytest.fixture
def model_with_branches() -> ModelProto:
    """加载带分支的模型"""
    model_path = Path(__file__).parent / "fixtures" / "models" / "model_with_branches.onnx"
    return onnx.load(str(model_path))


@pytest.fixture
def simple_split_plan() -> SplitPlan:
    """创建简单的切分方案"""
    return SplitPlan(
        operator_name="conv_0",  # 匹配simple_conv.onnx中的节点名
        parts=2,
        axis=0,
        reason="batch split",
    )


class TestTransformContext:
    """测试TransformContext类"""

    def test_initialization(self, simple_conv_model: ModelProto) -> None:
        """测试初始化"""
        ctx = TransformContext(simple_conv_model, batch_dim=0)

        assert ctx.model == simple_conv_model
        assert ctx.batch_dim == 0
        assert ctx.original_graph is not None
        assert len(ctx.new_nodes) == 0
        assert len(ctx.new_inputs) == 0
        assert len(ctx.new_outputs) == 0
        assert len(ctx.new_initializers) == 0

    def test_analyzer_property(self, simple_conv_model: ModelProto) -> None:
        """测试analyzer属性"""
        ctx = TransformContext(simple_conv_model, batch_dim=0)

        analyzer = ctx.analyzer
        assert isinstance(analyzer, ModelAnalyzer)
        assert analyzer.graph == simple_conv_model.graph

    def test_get_operator_by_name(self, simple_conv_model: ModelProto) -> None:
        """测试通过名称获取算子"""
        ctx = TransformContext(simple_conv_model, batch_dim=0)

        # 查找第一个算子
        first_node = simple_conv_model.graph.node[0]
        if first_node.name:
            op = ctx.get_operator(first_node.name)
            assert op is not None
            assert op.name == first_node.name


class TestTransformResult:
    """测试TransformResult类"""

    def test_empty_result(self) -> None:
        """测试空结果"""
        result = TransformResult()

        assert result.success is True
        assert result.error_message == ""
        assert len(result.split_operators) == 0
        assert len(result.new_nodes) == 0
        assert result.metrics["split_count"] == 0

    def test_result_with_data(self) -> None:
        """测试带数据的结果"""
        result = TransformResult()
        result.split_operators = ["conv1", "conv2"]
        result.new_nodes = ["conv1_split_0", "conv1_split_1"]

        assert len(result.split_operators) == 2
        assert len(result.new_nodes) == 2
        assert result.metrics["split_count"] == 0

    def test_error_result(self) -> None:
        """测试错误结果"""
        result = TransformResult(success=False, error_message="Test error")

        assert result.success is False
        assert result.error_message == "Test error"


class TestGraphTransformer:
    """测试GraphTransformer类"""

    def test_initialization(self, simple_conv_model: ModelProto) -> None:
        """测试初始化"""
        transformer = GraphTransformer(simple_conv_model, batch_dim=0)

        assert transformer.model == simple_conv_model
        assert transformer.batch_dim == 0
        assert transformer._analyzer is not None

    def test_create_context(self, simple_conv_model: ModelProto) -> None:
        """测试创建上下文"""
        transformer = GraphTransformer(simple_conv_model, batch_dim=0)
        ctx = transformer.create_context()

        assert isinstance(ctx, TransformContext)
        assert ctx.model == simple_conv_model

    def test_apply_single_split(
        self, simple_conv_model: ModelProto, simple_split_plan: SplitPlan
    ) -> None:
        """测试应用单个切分方案"""
        transformer = GraphTransformer(simple_conv_model, batch_dim=0)
        result = transformer.apply_split(simple_split_plan)

        assert result.success is True
        assert len(result.split_operators) == 1
        assert simple_split_plan.operator_name in result.split_operators

    def test_apply_multiple_splits(self, simple_conv_model: ModelProto) -> None:
        """测试应用多个切分方案"""
        plans = [
            SplitPlan(operator_name="conv_0", parts=2, axis=0),
            SplitPlan(operator_name="relu_0", parts=2, axis=0),
        ]

        transformer = GraphTransformer(simple_conv_model, batch_dim=0)
        result = transformer.apply_splits(plans)

        assert result.success is True
        # Note: Only conv_0 will be found and split, relu_0 may not have suitable input
        assert len(result.split_operators) >= 1

    def test_apply_splits_empty_list(self, simple_conv_model: ModelProto) -> None:
        """测试应用空的切分列表"""
        transformer = GraphTransformer(simple_conv_model, batch_dim=0)
        result = transformer.apply_splits([])

        assert result.success is True
        assert len(result.split_operators) == 0

    def test_build_transformed_model(
        self, simple_conv_model: ModelProto, simple_split_plan: SplitPlan
    ) -> None:
        """测试构建变换后的模型"""
        transformer = GraphTransformer(simple_conv_model, batch_dim=0)
        transformer.apply_split(simple_split_plan)

        new_model = transformer.build()

        assert isinstance(new_model, ModelProto)
        assert new_model.ir_version == simple_conv_model.ir_version
        assert len(new_model.graph.node) > 0

    def test_preserve_model_metadata(
        self, simple_conv_model: ModelProto, simple_split_plan: SplitPlan
    ) -> None:
        """测试保留模型元数据"""
        transformer = GraphTransformer(simple_conv_model, batch_dim=0)
        transformer.apply_split(simple_split_plan)

        new_model = transformer.build()

        assert new_model.ir_version == simple_conv_model.ir_version
        assert new_model.producer_name == simple_conv_model.producer_name
        assert new_model.producer_version == simple_conv_model.producer_version

    def test_transform_with_custom_axis(self, simple_matmul_model: ModelProto) -> None:
        """测试使用自定义轴进行变换"""
        plan = SplitPlan(operator_name="matmul_0", parts=2, axis=1)

        transformer = GraphTransformer(simple_matmul_model, batch_dim=1)
        result = transformer.apply_split(plan)

        assert result.success is True
        assert plan.operator_name in result.split_operators

    def test_execute_and_save(
        self, simple_conv_model: ModelProto, simple_split_plan: SplitPlan
    ) -> None:
        """测试执行变换并保存"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "split_model.onnx"

            transformer = GraphTransformer(simple_conv_model, batch_dim=0)
            result = transformer.execute_and_save([simple_split_plan], output_path)

            assert result.success is True
            assert output_path.exists()

            # 加载保存的模型验证
            loaded_model = onnx.load(str(output_path))
            assert isinstance(loaded_model, ModelProto)

    def test_execute_and_save_creates_directory(
        self, simple_conv_model: ModelProto, simple_split_plan: SplitPlan
    ) -> None:
        """测试execute_and_save创建不存在的目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "split_model.onnx"

            transformer = GraphTransformer(simple_conv_model, batch_dim=0)
            result = transformer.execute_and_save([simple_split_plan], output_path)

            assert result.success is True
            assert output_path.exists()

    def test_model_validity_after_transform(
        self, simple_conv_model: ModelProto, simple_split_plan: SplitPlan
    ) -> None:
        """测试变换后的模型有效性"""
        transformer = GraphTransformer(simple_conv_model, batch_dim=0)
        transformer.apply_split(simple_split_plan)

        new_model = transformer.build()

        # 检查模型有基本结构
        assert len(new_model.graph.node) > 0
        assert len(new_model.graph.input) > 0
        assert len(new_model.graph.output) > 0

        # 检查模型可以序列化
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=True) as f:
            onnx.save(new_model, f.name)
            # 可以重新加载
            loaded = onnx.load(f.name)
            assert loaded is not None

    def test_get_transform_metrics(
        self, simple_conv_model: ModelProto, simple_split_plan: SplitPlan
    ) -> None:
        """测试获取变换指标"""
        transformer = GraphTransformer(simple_conv_model, batch_dim=0)
        transformer.apply_split(simple_split_plan)

        metrics = transformer.get_metrics()

        assert "split_count" in metrics
        assert "new_node_count" in metrics
        assert "original_node_count" in metrics
        assert metrics["split_count"] >= 1

    def test_nonexistent_operator_split(self, simple_conv_model: ModelProto) -> None:
        """测试对不存在的算子进行切分"""
        plan = SplitPlan(operator_name="nonexistent_op", parts=2, axis=0)

        transformer = GraphTransformer(simple_conv_model, batch_dim=0)
        result = transformer.apply_split(plan)

        # 应该返回失败或跳过
        assert result.success is True  # 跳过不报错

    def test_batch_dim_parameter(self, simple_conv_model: ModelProto) -> None:
        """测试batch_dim参数传递"""
        transformer = GraphTransformer(simple_conv_model, batch_dim=1)

        assert transformer.batch_dim == 1

        ctx = transformer.create_context()
        assert ctx.batch_dim == 1


@pytest.mark.parametrize(
    ("parts", "axis", "expected_split_count"),
    [
        (2, 0, 1),
        (3, 0, 1),
        (4, 1, 1),
    ],
)
def test_various_split_configs(
    simple_conv_model: ModelProto,
    parts: int,
    axis: int,
    expected_split_count: int,
) -> None:
    """测试各种切分配置"""
    plan = SplitPlan(operator_name="conv_0", parts=parts, axis=axis)

    transformer = GraphTransformer(simple_conv_model, batch_dim=axis)
    result = transformer.apply_split(plan)

    assert result.success is True
    assert result.metrics["split_count"] == expected_split_count
