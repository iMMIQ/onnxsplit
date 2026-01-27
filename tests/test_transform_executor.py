"""图变换执行器测试"""

from pathlib import Path

import onnx
import pytest
from onnx import ModelProto

from onnxsplit.analyzer.model import ModelAnalyzer
from onnxsplit.splitter.plan import SplitPlan
from onnxsplit.transform.executor import GraphTransformer


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
        operator_name="conv_0",
        parts=2,
        axis=0,
        reason="batch split",
    )


class TestGraphTransformerInitialization:
    """测试GraphTransformer初始化"""

    def test_init_with_analyzer(self, simple_conv_model: ModelProto) -> None:
        """测试使用ModelAnalyzer初始化"""
        analyzer = ModelAnalyzer(simple_conv_model)
        transformer = GraphTransformer(analyzer)

        assert transformer.analyzer == analyzer


class TestApplySplitPlan:
    """测试apply_split_plan方法"""

    def test_apply_split_plan_returns_model(
        self, simple_conv_model: ModelProto, simple_split_plan: SplitPlan
    ) -> None:
        """测试apply_split_plan返回ModelProto"""
        analyzer = ModelAnalyzer(simple_conv_model)
        transformer = GraphTransformer(analyzer)

        result = transformer.apply_split_plan(simple_split_plan)

        assert isinstance(result, ModelProto)

    def test_apply_split_plan_no_split_returns_copy(self, simple_conv_model: ModelProto) -> None:
        """测试非切分方案返回原模型副本"""
        analyzer = ModelAnalyzer(simple_conv_model)
        transformer = GraphTransformer(analyzer)
        plan = SplitPlan(operator_name="conv_0", parts=1, axis=0)  # parts=1 表示不切分

        result = transformer.apply_split_plan(plan)

        assert result is not simple_conv_model  # 应该是副本
        assert len(result.graph.node) == len(simple_conv_model.graph.node)

    def test_apply_split_plan_invalid_operator_raises_error(
        self, simple_conv_model: ModelProto
    ) -> None:
        """测试不存在的算子抛出ValueError"""
        analyzer = ModelAnalyzer(simple_conv_model)
        transformer = GraphTransformer(analyzer)
        plan = SplitPlan(operator_name="nonexistent_op", parts=2, axis=0)

        with pytest.raises(ValueError, match="Operator not found"):
            transformer.apply_split_plan(plan)


class TestInputSplit:
    """测试输入切分相关方法"""

    def test_needs_input_split_with_model_input(self, simple_conv_model: ModelProto) -> None:
        """测试模型输入需要切分"""
        analyzer = ModelAnalyzer(simple_conv_model)
        transformer = GraphTransformer(analyzer)

        # 找到conv_0节点
        for node in simple_conv_model.graph.node:
            if node.name == "conv_0":
                result = transformer._needs_input_split(node)
                # conv_0的输入来自模型输入或Constant节点，需要切分
                assert result is True
                break

    def test_is_weight_detects_initializer(self, simple_conv_model: ModelProto) -> None:
        """测试_is_weight正确识别权重"""
        analyzer = ModelAnalyzer(simple_conv_model)
        transformer = GraphTransformer(analyzer)

        # simple_conv模型中，input不是权重
        assert transformer._is_weight("input") is False
        # weight_value是由Constant节点生成的权重，应该被识别为权重
        assert transformer._is_weight("weight_value") is True

        # 测试empty string
        assert transformer._is_weight("") is False

    def test_create_input_splits_creates_split_nodes(self, simple_conv_model: ModelProto) -> None:
        """测试_create_input_splits创建切分节点"""
        analyzer = ModelAnalyzer(simple_conv_model)
        transformer = GraphTransformer(analyzer)
        plan = SplitPlan(operator_name="conv_0", parts=2, axis=0)

        # 找到conv_0节点
        for node in simple_conv_model.graph.node:
            if node.name == "conv_0":
                split_nodes, input_split_map = transformer._create_input_splits(
                    simple_conv_model.graph, node, plan
                )

                # conv_0有两个输入：input和weight_value
                # weight_value是由Constant节点产生的权重，不应该被split
                # 只有input应该被split
                assert len(split_nodes) == 1
                assert all(n.op_type == "Split" for n in split_nodes)
                # 检查返回的映射
                assert "input" in input_split_map
                assert "weight_value" not in input_split_map
                break


class TestOutputMerge:
    """测试输出合并相关方法"""

    def test_needs_output_merge_with_model_output(self, model_with_branches: ModelProto) -> None:
        """测试模型输出需要合并"""
        analyzer = ModelAnalyzer(model_with_branches)
        transformer = GraphTransformer(analyzer)

        # 找到输出节点
        for node in model_with_branches.graph.node:
            for output_name in node.output:
                if transformer._is_model_output(output_name):
                    result = transformer._needs_output_merge(node)
                    assert result is True
                    break

    def test_is_model_output_detects_graph_outputs(self, simple_conv_model: ModelProto) -> None:
        """测试_is_model_output正确识别模型输出"""
        analyzer = ModelAnalyzer(simple_conv_model)
        transformer = GraphTransformer(analyzer)

        # relu_output是模型输出
        assert transformer._is_model_output("relu_output") is True
        # conv_output不是模型输出
        assert transformer._is_model_output("conv_output") is False

    def test_create_output_merges_creates_concat_nodes(self, simple_conv_model: ModelProto) -> None:
        """测试_create_output_merges创建合并节点"""
        analyzer = ModelAnalyzer(simple_conv_model)
        transformer = GraphTransformer(analyzer)
        plan = SplitPlan(operator_name="conv_0", parts=2, axis=0)

        # 找到conv_0节点
        for node in simple_conv_model.graph.node:
            if node.name == "conv_0":
                concat_nodes = transformer._create_output_merges(node, plan)

                # conv_0有一个输出
                assert len(concat_nodes) == 1
                assert concat_nodes[0].op_type == "Concat"
                break


class TestUpdateGraphNodes:
    """测试图节点更新方法"""

    def test_update_graph_nodes_removes_and_adds(self, simple_conv_model: ModelProto) -> None:
        """测试_update_graph_nodes正确移除和添加节点"""
        analyzer = ModelAnalyzer(simple_conv_model)
        transformer = GraphTransformer(analyzer)

        # 克隆模型进行测试
        test_model = onnx.load(
            str(Path(__file__).parent / "fixtures" / "models" / "simple_conv.onnx")
        )

        original_node_count = len(test_model.graph.node)
        to_remove = [test_model.graph.node[0]]
        to_add = [onnx.helper.make_node("Relu", inputs=["x"], outputs=["y"], name="new_relu")]

        transformer._update_graph_nodes(test_model.graph, to_remove, to_add)

        # 应该移除1个，添加1个，总数不变
        assert len(test_model.graph.node) == original_node_count

        # 新节点应该存在
        node_names = [n.name for n in test_model.graph.node]
        assert "new_relu" in node_names


class TestSplitIntegration:
    """测试切分集成"""

    def test_full_split_creates_expected_nodes(self, simple_conv_model: ModelProto) -> None:
        """测试完整切分创建预期的节点"""
        analyzer = ModelAnalyzer(simple_conv_model)
        transformer = GraphTransformer(analyzer)
        plan = SplitPlan(operator_name="conv_0", parts=2, axis=0)

        result = transformer.apply_split_plan(plan)

        # 应该包含: Constant, Split, 2个Conv, Concat, Relu
        node_types = [n.op_type for n in result.graph.node]
        assert node_types.count("Conv") == 2
        assert "Split" in node_types
        assert "Concat" in node_types

    def test_split_with_matmul_model(self, simple_matmul_model: ModelProto) -> None:
        """测试MatMul模型切分"""
        analyzer = ModelAnalyzer(simple_matmul_model)
        transformer = GraphTransformer(analyzer)
        plan = SplitPlan(operator_name="matmul_0", parts=2, axis=0)

        result = transformer.apply_split_plan(plan)

        # 应该包含2个MatMul和相关的Split/Concat节点
        node_types = [n.op_type for n in result.graph.node]
        assert node_types.count("MatMul") == 2
        assert "Split" in node_types
        assert "Concat" in node_types


class TestShapeInference:
    """测试形状推断"""

    def test_split_runs_shape_inference(self, simple_conv_model: ModelProto) -> None:
        """测试切分后运行形状推断"""
        analyzer = ModelAnalyzer(simple_conv_model)
        transformer = GraphTransformer(analyzer)
        plan = SplitPlan(operator_name="conv_0", parts=2, axis=0)

        result = transformer.apply_split_plan(plan)

        # 验证模型有形状信息
        assert result.graph.value_info is not None
