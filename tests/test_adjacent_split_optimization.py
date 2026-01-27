"""Adjacent split nodes optimization tests

Tests for the optimization where adjacent split nodes (e.g., Matmul -> Add -> Relu)
should not have redundant concat/split inserted between them. When both upstream
and downstream nodes are split with the same parts/axis, they should connect
directly (1-to-1).
"""

import copy

import onnx
import pytest
from onnx import TensorProto, helper

from onnxsplit.analyzer.model import ModelAnalyzer
from onnxsplit.splitter.plan import SplitPlan
from onnxsplit.transform.executor import GraphTransformer


@pytest.fixture
def adjacent_ops_model() -> onnx.ModelProto:
    """Create a simple ONNX model with Matmul -> Add -> Relu

    Input: [4, 10] FLOAT tensor
    Weight: [10, 10] initializer
    Bias: [10] initializer
    Nodes: MatMul("matmul1") -> Add("add1") -> Relu("relu1")
    """
    # Input tensor
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [4, 10])

    # Weight initializer [10, 10]
    weight_init = helper.make_tensor("weight", TensorProto.FLOAT, [10, 10], [0.1] * 100)

    # Bias initializer [10]
    bias_init = helper.make_tensor("bias", TensorProto.FLOAT, [10], [0.0] * 10)

    # MatMul node
    matmul_node = helper.make_node(
        "MatMul",
        inputs=["input", "weight"],
        outputs=["matmul_out"],
        name="matmul1",
    )

    # Add node
    add_node = helper.make_node(
        "Add",
        inputs=["matmul_out", "bias"],
        outputs=["add_out"],
        name="add1",
    )

    # Relu node
    relu_node = helper.make_node(
        "Relu",
        inputs=["add_out"],
        outputs=["relu_out"],
        name="relu1",
    )

    # Output tensor
    output_tensor = helper.make_tensor_value_info("relu_out", TensorProto.FLOAT, [4, 10])

    # Graph
    graph = helper.make_graph(
        [matmul_node, add_node, relu_node],
        "adjacent_ops_model",
        [input_tensor],
        [output_tensor],
        [weight_init, bias_init],
    )

    # Model
    model = helper.make_model(graph)
    model = onnx.shape_inference.infer_shapes(model)
    return model


class TestAdjacentSplitOptimization:
    """Test adjacent split nodes optimization"""

    def test_adjacent_same_split_no_redundant_concat_split(
        self, adjacent_ops_model: onnx.ModelProto
    ) -> None:
        """Split Matmul first (parts=2, axis=0), then split Add with same config.

        Verify no redundant Split/Concat nodes are added between matmul and add.

        Expected (Optimized) Behavior:
            Matmul (split=2, axis=0) -> Add (split=2, axis=0)
        After transformation:
            input -> Split -> [Matmul_0, Matmul_1]
            Matmul_0 -> Add_0    <-- Direct connection!
            Matmul_1 -> Add_1    <-- Direct connection!

        Current (Buggy) Behavior:
            input -> Split -> [Matmul_0, Matmul_1]
            [Matmul_0, Matmul_1] -> Concat -> matmul_out
            matmul_out -> Split -> [Add_0, Add_1]    <-- REDUNDANT!
        """
        # First split: Matmul with parts=2, axis=0
        analyzer = ModelAnalyzer(adjacent_ops_model)
        transformer = GraphTransformer(analyzer)

        matmul_plan = SplitPlan(operator_name="matmul1", parts=2, axis=0, reason="test split")
        model_after_matmul_split = transformer.apply_split_plan(matmul_plan)

        # Second split: Add with same config (parts=2, axis=0)
        analyzer2 = ModelAnalyzer(model_after_matmul_split)
        transformer2 = GraphTransformer(analyzer2)

        add_plan = SplitPlan(operator_name="add1", parts=2, axis=0, reason="test split")
        final_model = transformer2.apply_split_plan(add_plan)

        # Verify the graph structure
        node_types = [n.op_type for n in final_model.graph.node]
        node_names = [n.name for n in final_model.graph.node]

        # Count nodes by type
        split_count = node_types.count("Split")
        concat_count = node_types.count("Concat")
        matmul_count = node_types.count("MatMul")
        add_count = node_types.count("Add")

        # Expected: 1 Split (for input), 2 MatMul, 2 Add, 1 Concat (for final output)
        # No redundant Concat between MatMul and Add!
        assert split_count == 1, f"Expected 1 Split node, got {split_count}. Nodes: {node_names}"
        assert concat_count == 1, f"Expected 1 Concat node (for final output), got {concat_count}. Nodes: {node_names}"
        assert matmul_count == 2, f"Expected 2 MatMul nodes, got {matmul_count}"
        assert add_count == 2, f"Expected 2 Add nodes, got {add_count}"

        # Verify direct connection: Matmul_split_0 -> Add_split_0, Matmul_split_1 -> Add_split_1
        matmul_split_0_output = None
        matmul_split_1_output = None
        add_split_0_input = None
        add_split_1_input = None

        for node in final_model.graph.node:
            if node.name == "matmul1_split_0":
                matmul_split_0_output = node.output[0]
            elif node.name == "matmul1_split_1":
                matmul_split_1_output = node.output[0]
            elif node.name == "add1_split_0":
                add_split_0_input = node.input[0]
            elif node.name == "add1_split_1":
                add_split_1_input = node.input[0]

        assert matmul_split_0_output is not None, "MatMul split_0 output not found"
        assert matmul_split_1_output is not None, "MatMul split_1 output not found"
        assert add_split_0_input is not None, "Add split_0 input not found"
        assert add_split_1_input is not None, "Add split_1 input not found"

        # Direct connections: no intermediate concat
        assert matmul_split_0_output == add_split_0_input, \
            f"Expected direct connection MatMul_split_0 -> Add_split_0, got {matmul_split_0_output} -> {add_split_0_input}"
        assert matmul_split_1_output == add_split_1_input, \
            f"Expected direct connection MatMul_split_1 -> Add_split_1, got {matmul_split_1_output} -> {add_split_1_input}"

    def test_adjacent_different_parts_should_have_adaptation(
        self, adjacent_ops_model: onnx.ModelProto
    ) -> None:
        """Split Matmul with 2 parts, then Add with 4 parts.

        Verify adaptation nodes exist to handle the parts mismatch.
        """
        # First split: Matmul with parts=2, axis=0
        analyzer = ModelAnalyzer(adjacent_ops_model)
        transformer = GraphTransformer(analyzer)

        matmul_plan = SplitPlan(operator_name="matmul1", parts=2, axis=0, reason="test split")
        model_after_matmul_split = transformer.apply_split_plan(matmul_plan)

        # Second split: Add with different config (parts=4, axis=0)
        analyzer2 = ModelAnalyzer(model_after_matmul_split)
        transformer2 = GraphTransformer(analyzer2)

        add_plan = SplitPlan(operator_name="add1", parts=4, axis=0, reason="test split")
        final_model = transformer2.apply_split_plan(add_plan)

        # Verify the graph structure
        node_types = [n.op_type for n in final_model.graph.node]

        # With different parts, we need adaptation nodes
        # Expected: Multiple Splits, Concats to handle 2->4 transition
        split_count = node_types.count("Split")
        concat_count = node_types.count("Concat")

        # Should have more than 1 split due to parts mismatch
        assert split_count >= 1, f"Expected at least 1 Split node, got {split_count}"
        # Should have concats for adaptation
        assert concat_count >= 1, f"Expected at least 1 Concat node, got {concat_count}"

        # Verify the Add nodes are split into 4 parts
        add_split_nodes = [n for n in final_model.graph.node if n.name.startswith("add1_split_")]
        assert len(add_split_nodes) == 4, f"Expected 4 Add split nodes, got {len(add_split_nodes)}"

    def test_detect_upstream_split_outputs(
        self, adjacent_ops_model: onnx.ModelProto
    ) -> None:
        """After splitting Matmul, verify the split outputs exist with
        _split_0 and _split_1 suffixes.
        """
        analyzer = ModelAnalyzer(adjacent_ops_model)
        transformer = GraphTransformer(analyzer)

        matmul_plan = SplitPlan(operator_name="matmul1", parts=2, axis=0, reason="test split")
        model_after_split = transformer.apply_split_plan(matmul_plan)

        # Find the MatMul split nodes
        matmul_split_nodes = [
            n for n in model_after_split.graph.node if n.name.startswith("matmul1_split_")
        ]

        assert len(matmul_split_nodes) == 2, \
            f"Expected 2 MatMul split nodes, got {len(matmul_split_nodes)}"

        # Verify the naming pattern
        split_names = [n.name for n in matmul_split_nodes]
        assert "matmul1_split_0" in split_names, "Expected matmul1_split_0 node"
        assert "matmul1_split_1" in split_names, "Expected matmul1_split_1 node"

        # Verify the outputs
        matmul_split_0 = next(n for n in matmul_split_nodes if n.name == "matmul1_split_0")
        matmul_split_1 = next(n for n in matmul_split_nodes if n.name == "matmul1_split_1")

        # Outputs should have _0 and _1 suffixes
        assert matmul_split_0.output[0].endswith("_0"), \
            f"Expected MatMul_split_0 output to end with _0, got {matmul_split_0.output[0]}"
        assert matmul_split_1.output[0].endswith("_1"), \
            f"Expected MatMul_split_1 output to end with _1, got {matmul_split_1.output[0]}"

        # Verify shape inference was run
        for value_info in model_after_split.graph.value_info:
            if value_info.name == matmul_split_0.output[0]:
                # Shape should be [2, 10] (half of [4, 10])
                shape = [d.dim_value for d in value_info.type.tensor_type.shape.dim]
                assert shape == [2, 10], f"Expected shape [2, 10], got {shape}"
            if value_info.name == matmul_split_1.output[0]:
                # Shape should be [2, 10] (half of [4, 10])
                shape = [d.dim_value for d in value_info.type.tensor_type.shape.dim]
                assert shape == [2, 10], f"Expected shape [2, 10], got {shape}"
