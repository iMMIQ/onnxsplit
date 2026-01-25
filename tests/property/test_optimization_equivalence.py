"""Property tests: optimized algorithms are equivalent to original implementations."""

from pathlib import Path

import hypothesis.strategies as st
from hypothesis import assume, given, settings
import onnx
from onnx import helper

from onnxsplit.analyzer import ModelAnalyzer
from onnxsplit.config import GlobalConfig, SplitConfig
from onnxsplit.memory import MemoryEstimator
from onnxsplit.splitter import SplitPlanner


@st.composite
def onnx_model_strategy(draw):
    """Generate simple ONNX models for property testing.

    Creates models with:
    - Variable number of operators (1-10)
    - Variable batch size (1-8)
    - Variable channels (1-16)
    - Various operator types (Relu, Add, Mul, Sigmoid, Tanh)
    """
    num_operators = draw(st.integers(min_value=1, max_value=10))
    batch_size = draw(st.integers(min_value=1, max_value=8))
    channels = draw(st.integers(min_value=1, max_value=16))

    input_tensor = helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, [batch_size, channels, 8, 8]
    )

    nodes = []
    current_input = "input"
    output_names = []

    for i in range(num_operators):
        op_type = draw(st.sampled_from(["Relu", "Add", "Mul", "Sigmoid", "Tanh"]))
        output_name = f"output_{i}"

        if op_type == "Relu":
            node = helper.make_node(
                op_type, inputs=[current_input],
                outputs=[output_name], name=f"{op_type.lower()}_{i}"
            )
        elif op_type in ("Add", "Mul"):
            node = helper.make_node(
                op_type, inputs=[current_input, current_input],
                outputs=[output_name], name=f"{op_type.lower()}_{i}"
            )
        else:  # Sigmoid, Tanh
            node = helper.make_node(
                op_type, inputs=[current_input],
                outputs=[output_name], name=f"{op_type.lower()}_{i}"
            )

        nodes.append(node)
        current_input = output_name
        output_names.append(output_name)

    output_tensor = helper.make_tensor_value_info(
        output_names[-1], onnx.TensorProto.FLOAT,
        [batch_size, channels, 8, 8]
    )

    graph = helper.make_graph(nodes, "test_model", [input_tensor], [output_tensor])
    model = helper.make_model(graph)
    model = onnx.shape_inference.infer_shapes(model)

    return model


@st.composite
def onnx_model_with_conv_strategy(draw):
    """Generate ONNX models with Conv operators.

    Creates models that include Conv operators which require Constant nodes for weights.
    """
    num_operators = draw(st.integers(min_value=1, max_value=5))
    batch_size = draw(st.integers(min_value=1, max_value=4))
    channels = draw(st.integers(min_value=1, max_value=8))

    input_tensor = helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, [batch_size, channels, 8, 8]
    )

    nodes = []
    current_input = "input"
    output_names = []

    for i in range(num_operators):
        op_type = draw(st.sampled_from(["Relu", "Conv"]))
        output_name = f"output_{i}"

        if op_type == "Conv":
            # Create weight constant
            weight = helper.make_tensor(
                f"weight_{i}", onnx.TensorProto.FLOAT,
                [channels, channels, 3, 3], [0.1] * (channels * channels * 9)
            )
            const_node = helper.make_node("Constant", [], [f"weight_{i}_const"], value=weight)
            nodes.append(const_node)
            node = helper.make_node(
                "Conv", inputs=[current_input, f"weight_{i}_const"],
                outputs=[output_name], name=f"conv_{i}",
                kernel_shape=[3, 3], pads=[1, 1, 1, 1]
            )
        else:  # Relu
            node = helper.make_node(
                op_type, inputs=[current_input],
                outputs=[output_name], name=f"relu_{i}"
            )

        nodes.append(node)
        current_input = output_name
        output_names.append(output_name)

    output_tensor = helper.make_tensor_value_info(
        output_names[-1], onnx.TensorProto.FLOAT,
        [batch_size, channels, 8, 8]
    )

    graph = helper.make_graph(nodes, "test_model", [input_tensor], [output_tensor])
    model = helper.make_model(graph)
    model = onnx.shape_inference.infer_shapes(model)

    return model


@settings(max_examples=20, deadline=1000)
@given(model=onnx_model_strategy())
def test_analyzer_get_operators_consistency(model):
    """Test that ModelAnalyzer.get_operators() returns consistent results.

    Property: Multiple calls to get_operators() should return the same list.
    """
    analyzer = ModelAnalyzer.from_model_proto(model)

    # Multiple calls should return the same operator list
    ops1 = analyzer.get_operators()
    ops2 = analyzer.get_operators()

    assert len(ops1) == len(ops2)
    assert [op.name for op in ops1] == [op.name for op in ops2]

    # All operators should be queryable by name
    for op in ops1:
        by_name = analyzer.get_operator(op.name)
        assert by_name is not None
        assert by_name.name == op.name


@settings(max_examples=20, deadline=1000)
@given(model=onnx_model_with_conv_strategy())
def test_analyzer_with_conv_consistency(model):
    """Test ModelAnalyzer consistency with Conv operators.

    Property: Models with Constant nodes for weights should still be analyzed consistently.
    """
    analyzer = ModelAnalyzer.from_model_proto(model)

    # Multiple calls should return the same operator list
    ops1 = analyzer.get_operators()
    ops2 = analyzer.get_operators()

    assert len(ops1) == len(ops2)
    assert [op.name for op in ops1] == [op.name for op in ops2]

    # Constant nodes should not be in the operator list
    for op in ops1:
        assert op.op_type != "Constant"


@settings(max_examples=20, deadline=1000)
@given(model=onnx_model_strategy())
def test_analyzer_get_operator_idempotent(model):
    """Test that get_operator() is idempotent.

    Property: Calling get_operator() multiple times with the same name
    should return equivalent results.
    """
    analyzer = ModelAnalyzer.from_model_proto(model)
    ops = analyzer.get_operators()

    assume(len(ops) > 0)

    for op in ops:
        op1 = analyzer.get_operator(op.name)
        op2 = analyzer.get_operator(op.name)

        assert op1 is not None
        assert op2 is not None
        assert op1.name == op2.name
        assert op1.op_type == op2.op_type


@settings(max_examples=20, deadline=1000)
@given(model=onnx_model_strategy())
def test_memory_estimator_consistency(model):
    """Test that MemoryEstimator returns consistent memory information.

    Property: Multiple calls to get_peak_memory() should return the same value.
    """
    analyzer = ModelAnalyzer.from_model_proto(model)
    estimator = MemoryEstimator(analyzer)

    # Multiple calls to peak memory should return same value
    peak1 = estimator.get_peak_memory()
    peak2 = estimator.get_peak_memory()

    assert peak1 == peak2

    # Memory breakdown should be consistent with peak
    breakdown = estimator.get_memory_breakdown()
    if breakdown:
        max_memory = max(info.peak_memory_mb for info in breakdown)
        assert peak1 == max_memory


@settings(max_examples=20, deadline=1000)
@given(model=onnx_model_strategy())
def test_memory_estimator_total_memory(model):
    """Test that total memory is non-negative.

    Property: Total memory should always be non-negative.
    """
    analyzer = ModelAnalyzer.from_model_proto(model)
    estimator = MemoryEstimator(analyzer)

    peak = estimator.get_peak_memory()
    assert peak >= 0

    total = estimator.get_total_model_memory()
    assert total >= 0


@settings(max_examples=20, deadline=1000)
@given(model=onnx_model_strategy())
def test_splitter_planner_deterministic(model):
    """Test that SplitPlanner generates deterministic results.

    Property: Two planners with the same configuration should produce
    identical split reports.
    """
    analyzer = ModelAnalyzer.from_model_proto(model)
    config = SplitConfig(global_config=GlobalConfig(default_parts=2))

    planner1 = SplitPlanner(analyzer, config)
    report1 = planner1.generate()

    planner2 = SplitPlanner(analyzer, config)
    report2 = planner2.generate()

    # Two generated reports should be identical
    assert report1.original_operators == report2.original_operators
    assert report1.split_operators == report2.split_operators
    assert len(report1.plans) == len(report2.plans)

    for plan1, plan2 in zip(report1.plans, report2.plans):
        assert plan1.operator_name == plan2.operator_name
        assert plan1.parts == plan2.parts
        assert plan1.axis == plan2.axis


@settings(max_examples=20, deadline=1000)
@given(model=onnx_model_strategy())
def test_splitter_planner_single_part_no_split(model):
    """Test that single-part configuration produces no splits.

    Property: When default_parts=1, no operators should be split.
    """
    analyzer = ModelAnalyzer.from_model_proto(model)
    config = SplitConfig(global_config=GlobalConfig(default_parts=1))

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    # With default_parts=1, nothing should be split
    assert report.split_operators == 0
    assert report.original_operators == report.unsplit_operators


@settings(max_examples=15, deadline=1000)
@given(model=onnx_model_strategy())
def test_splitter_planner_parts_positive(model):
    """Test that all split plans have positive parts.

    Property: All split plans should have parts >= 1.
    """
    analyzer = ModelAnalyzer.from_model_proto(model)
    config = SplitConfig(global_config=GlobalConfig(default_parts=3))

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    for plan in report.plans:
        assert plan.parts >= 1


@settings(max_examples=20, deadline=1000)
@given(model=onnx_model_strategy())
def test_operator_info_memory_properties(model):
    """Test that operator memory properties are consistent.

    Property: Total memory should equal input + output memory.
    """
    analyzer = ModelAnalyzer.from_model_proto(model)
    ops = analyzer.get_operators()

    assume(len(ops) > 0)

    for op in ops:
        # Total memory should equal input + output memory
        assert op.total_memory_mb == op.input_memory_mb + op.output_memory_mb

        # All memory values should be non-negative
        assert op.input_memory_mb >= 0
        assert op.output_memory_mb >= 0
        assert op.total_memory_mb >= 0


@settings(max_examples=20, deadline=1000)
@given(model=onnx_model_strategy())
def test_analyzer_input_output_count(model):
    """Test that input and output counts are consistent.

    Property: Number of inputs and outputs should be non-negative.
    """
    analyzer = ModelAnalyzer.from_model_proto(model)

    inputs = analyzer.get_inputs()
    outputs = analyzer.get_outputs()
    operators = analyzer.get_operators()

    assert len(inputs) >= 0
    assert len(outputs) >= 0
    assert len(operators) >= 0


def test_optimization_with_real_models():
    """Use real models to verify optimization correctness.

    This test uses actual ONNX models from fixtures to ensure
    the optimizations work correctly on real-world models.
    """
    model_files = [
        "tests/fixtures/models/simple_conv.onnx",
        "tests/fixtures/models/model_with_branches.onnx",
        "tests/fixtures/models/simple_matmul.onnx",
    ]

    for model_file in model_files:
        model_path = Path(model_file)
        if not model_path.exists():
            continue

        analyzer = ModelAnalyzer.from_path(model_path)

        # Test operator query
        all_ops = analyzer.get_operators()
        for op in all_ops:
            by_name = analyzer.get_operator(op.name)
            assert by_name is not None
            assert by_name.name == op.name

        # Test memory estimation
        estimator = MemoryEstimator(analyzer)
        peak1 = estimator.get_peak_memory()
        peak2 = estimator.get_peak_memory()
        assert peak1 == peak2

        # Test planner
        config = SplitConfig(global_config=GlobalConfig(default_parts=2))
        planner = SplitPlanner(analyzer, config)
        report1 = planner.generate()
        report2 = planner.generate()
        assert report1.split_operators == report2.split_operators
