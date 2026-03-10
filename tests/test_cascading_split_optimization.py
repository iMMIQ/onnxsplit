"""Cascading split optimization tests

Tests for the optimization where upstream split (e.g., 3 parts) and downstream
split (e.g., 6 parts) should use SPLIT_SOURCE strategy (split each upstream
output further) instead of CONCAT_SOURCE strategy (concat then split).

Example scenario:
    Add1 (split=3, axis=0) -> Add2 (split=6, axis=0)

Optimal behavior:
    Add1_0 -> Split -> [Add2_0, Add2_1]
    Add1_1 -> Split -> [Add2_2, Add2_3]
    Add1_2 -> Split -> [Add2_4, Add2_5]

Suboptimal (current buggy) behavior:
    [Add1_0, Add1_1, Add1_2] -> Concat -> Split -> [Add2_0, Add2_1, ..., Add2_5]
"""

import copy

import onnx
import pytest
from onnx import TensorProto, helper

from onnxsplit.analyzer.model import ModelAnalyzer
from onnxsplit.splitter.plan import SplitPlan
from onnxsplit.transform.executor import GraphTransformer


@pytest.fixture
def cascading_adds_model() -> onnx.ModelProto:
    """Create a simple ONNX model with two Add operations.

    Input: [6, 10] FLOAT tensor
    Bias1: [10] initializer
    Bias2: [10] initializer
    Nodes: Add("add1") -> Add("add2")

    This model is designed to test:
    - add1 split into 3 parts (each [2, 10])
    - add2 split into 6 parts (each [1, 10])
    """
    # Input tensor
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [6, 10])

    # Bias1 initializer [10]
    bias1_init = helper.make_tensor("bias1", TensorProto.FLOAT, [10], [0.1] * 10)

    # Bias2 initializer [10]
    bias2_init = helper.make_tensor("bias2", TensorProto.FLOAT, [10], [0.2] * 10)

    # Add1 node
    add1_node = helper.make_node(
        "Add",
        inputs=["input", "bias1"],
        outputs=["add1_out"],
        name="add1",
    )

    # Add2 node
    add2_node = helper.make_node(
        "Add",
        inputs=["add1_out", "bias2"],
        outputs=["add2_out"],
        name="add2",
    )

    # Output tensor
    output_tensor = helper.make_tensor_value_info("add2_out", TensorProto.FLOAT, [6, 10])

    # Graph
    graph = helper.make_graph(
        [add1_node, add2_node],
        "cascading_adds_model",
        [input_tensor],
        [output_tensor],
        [bias1_init, bias2_init],
    )

    # Model
    model = helper.make_model(graph)
    model = onnx.shape_inference.infer_shapes(model)
    return model


class TestCascadingSplitOptimization:
    """Test cascading split optimization (SPLIT_SOURCE strategy)"""

    def test_cascading_split_3_to_6_should_split_source_not_concat(
        self, cascading_adds_model: onnx.ModelProto
    ) -> None:
        """Split add1 into 3 parts, then add2 into 6 parts.

        Expected (Optimized) Behavior:
            Since 6 % 3 == 0, each add1 output should be further split into 2 parts.

            add1 (split=3, axis=0) -> add2 (split=6, axis=0)

            add1_out_0 -> Split(2) -> [add2_0_input, add2_1_input]
            add1_out_1 -> Split(2) -> [add2_2_input, add2_3_input]
            add1_out_2 -> Split(2) -> [add2_4_input, add2_5_input]

        Suboptimal (Current Buggy) Behavior:
            [add1_out_0, add1_out_1, add1_out_2] -> Concat -> add1_out
            add1_out -> Split(6) -> [add2_0_input, ..., add2_5_input]
        """
        # Create both plans upfront
        add1_plan = SplitPlan(operator_name="add1", parts=3, axis=0, reason="test split")
        add2_plan = SplitPlan(operator_name="add2", parts=6, axis=0, reason="test split")

        # First split: add1 with parts=3, axis=0
        # Pass the planned splits so add1 knows add2 will be split with parts=6
        planned_splits = {"add2": add2_plan}
        analyzer = ModelAnalyzer(cascading_adds_model)
        transformer = GraphTransformer(analyzer, planned_splits=planned_splits)

        model_after_add1_split = transformer.apply_split_plan(add1_plan)

        # Second split: add2 with parts=6, axis=0
        analyzer2 = ModelAnalyzer(model_after_add1_split)
        transformer2 = GraphTransformer(analyzer2)

        final_model = transformer2.apply_split_plan(add2_plan)

        # Verify the graph structure
        node_types = [n.op_type for n in final_model.graph.node]
        node_names = [n.name for n in final_model.graph.node]

        print("Node types:", node_types)
        print("Node names:", node_names)

        # Count nodes by type
        split_count = node_types.count("Split")
        concat_count = node_types.count("Concat")
        add_count = node_types.count("Add")

        # Expected:
        # - 1 Split for input (splitting input into 3 for add1)
        # - 3 Splits for add1 outputs (splitting each into 2 for add2)
        # - 1 Concat for final output
        #
        # Total: 4 Splits, 1 Concat
        # NOT: 1 Split for input + 1 Concat (for add1) + 1 Split (for add2) = 1 Split, 2 Concat

        # With optimization:
        # - 1 Split for input
        # - 3 Splits for cascading (each add1 output split into 2)
        # - 1 Concat for final output
        # Total: 4 Splits, 1 Concat

        # Current buggy behavior would have:
        # - 1 Split for input
        # - 1 Concat for add1 output (because parts don't match)
        # - 1 Split for add2 input
        # - 1 Concat for final output
        # Total: 2 Splits, 2 Concats

        # Check for the key optimization: should have SPLIT_SOURCE pattern
        # Each add1 output should be split (not concatenated first)
        add1_split_nodes = [n for n in final_model.graph.node if n.name.startswith("add1_split_")]
        assert len(add1_split_nodes) == 3, f"Expected 3 add1 split nodes, got {len(add1_split_nodes)}"

        add2_split_nodes = [n for n in final_model.graph.node if n.name.startswith("add2_split_")]
        assert len(add2_split_nodes) == 6, f"Expected 6 add2 split nodes, got {len(add2_split_nodes)}"

        # The key assertion: should NOT have a concat that merges add1 outputs before splitting
        # If optimization is working, there should be NO concat that takes add1 outputs as inputs

        for node in final_model.graph.node:
            if node.op_type == "Concat":
                # Check if this concat is merging add1 outputs
                add1_outputs_in_concat = sum(
                    1 for inp in node.input if inp.startswith("add1_out_")
                )
                # If concat is for final output, it should have 6 inputs (all add2 outputs)
                # If concat is for intermediate (buggy), it would have 3 inputs (add1 outputs)
                if add1_outputs_in_concat > 0:
                    # This is a concat of add1 outputs - should NOT exist!
                    pytest.fail(
                        f"Found intermediate Concat merging {add1_outputs_in_concat} add1 outputs. "
                        f"Expected SPLIT_SOURCE strategy (split each add1 output further), "
                        f"not CONCAT_SOURCE (concat then split)."
                    )

        # Alternative: check that we have the expected number of splits
        # With SPLIT_SOURCE: 4 splits (1 for input + 3 cascading)
        # With CONCAT_SOURCE: 2 splits (1 for input + 1 for add2)
        assert split_count == 4, (
            f"Expected 4 Split nodes (1 input + 3 cascading), got {split_count}. "
            f"Nodes: {node_names}"
        )

        # With optimization: only 1 concat for final output
        assert concat_count == 1, (
            f"Expected 1 Concat node (for final output), got {concat_count}. "
            f"Nodes: {node_names}"
        )

    def test_cascading_split_2_to_4_should_split_source_not_concat(
        self, cascading_adds_model: onnx.ModelProto
    ) -> None:
        """Split add1 into 2 parts, then add2 into 4 parts.

        Similar test with different ratio (2 -> 4).
        """
        # Create both plans upfront
        add1_plan = SplitPlan(operator_name="add1", parts=2, axis=0, reason="test split")
        add2_plan = SplitPlan(operator_name="add2", parts=4, axis=0, reason="test split")

        # First split: add1 with parts=2, axis=0
        planned_splits = {"add2": add2_plan}
        analyzer = ModelAnalyzer(cascading_adds_model)
        transformer = GraphTransformer(analyzer, planned_splits=planned_splits)

        model_after_add1_split = transformer.apply_split_plan(add1_plan)

        # Second split: add2 with parts=4, axis=0
        analyzer2 = ModelAnalyzer(model_after_add1_split)
        transformer2 = GraphTransformer(analyzer2)

        final_model = transformer2.apply_split_plan(add2_plan)

        # Verify the graph structure
        node_types = [n.op_type for n in final_model.graph.node]

        split_count = node_types.count("Split")
        concat_count = node_types.count("Concat")

        # With SPLIT_SOURCE optimization:
        # - 1 Split for input (2 parts)
        # - 2 Splits for cascading (each add1 output split into 2)
        # - 1 Concat for final output
        # Total: 3 Splits, 1 Concat
        assert split_count == 3, f"Expected 3 Split nodes, got {split_count}"
        assert concat_count == 1, f"Expected 1 Concat node, got {concat_count}"

        # Check for intermediate concat (should not exist)
        for node in final_model.graph.node:
            if node.op_type == "Concat":
                add1_outputs_in_concat = sum(
                    1 for inp in node.input if inp.startswith("add1_out_")
                )
                if add1_outputs_in_concat > 0:
                    pytest.fail(
                        f"Found intermediate Concat merging add1 outputs. "
                        f"Expected SPLIT_SOURCE strategy."
                    )
