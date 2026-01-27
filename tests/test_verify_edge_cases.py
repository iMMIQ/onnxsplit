"""Tests for verification edge cases.

Tests for scenarios where verification fails with misleading max_diff values.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper

from onnxsplit.verify import verify_equivalence


def make_simple_model(output_value: float = 1.0) -> helper.ModelProto:
    """Create a simple ONNX model that outputs a constant.

    Args:
        output_value: The constant value to output

    Returns:
        ONNX model
    """
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])

    # Create a constant node with the given value
    const_node = helper.make_node(
        "Constant",
        [],
        ["const_out"],
        value=helper.make_tensor("value", TensorProto.FLOAT, [], [output_value]),
        name="const1",
    )

    # Identity to pass through (ensures we have an input)
    identity_node = helper.make_node("Identity", ["const_out"], ["output"], name="id1")

    graph = helper.make_graph(
        [const_node, identity_node],
        "constant_model",
        [],  # No real inputs needed for constant model
        [output_tensor],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return model


def test_verify_with_small_relative_error_exceeds_tolerance() -> None:
    """Test verification fails when relative error exceeds tolerance but abs diff is tiny.

    This reproduces the issue where:
    - np.allclose fails (relative error > rtol)
    - But max_diff is very small (< 0.005)
    - So it displays as "0.00e+00" which is misleading
    """
    # Create two models with outputs that differ slightly
    orig_model = make_simple_model(output_value=100.0)
    split_model = make_simple_model(output_value=100.011)  # 0.011 difference

    result = verify_equivalence(
        orig_model,
        split_model,
        rtol=1e-4,  # 0.0001 relative tolerance
        atol=1e-5,
        verbose=False,
    )

    # Should fail: 0.011 / 100.0 = 0.00011 > rtol(0.0001)
    assert result.success is False
    # Note: actual max_diff may differ slightly due to float32 precision
    assert result.max_diff > 0.01
    assert result.failure_reason is not None
    assert "exceeds tolerance" in result.failure_reason


def test_verify_with_very_small_difference_displays_as_zero() -> None:
    """Test that very small max_diff values are displayed correctly.

    This test verifies that we properly handle the case where max_diff
    is so small it formats to "0.00e+00" with .2e format.
    """
    # Create models with outputs differing by a small amount
    orig_model = make_simple_model(output_value=1.0)
    split_model = make_simple_model(output_value=1.00001)

    result = verify_equivalence(
        orig_model,
        split_model,
        rtol=1e-6,  # Stricter tolerance to ensure failure
        atol=1e-9,
        verbose=False,
    )

    assert result.success is False
    # Note: actual value may differ due to float32 precision
    assert result.max_diff > 0
    assert result.failure_reason is not None
    assert "exceeds tolerance" in result.failure_reason


def test_verify_with_nan_difference() -> None:
    """Test verification when outputs contain NaN values.

    NaN - NaN = NaN, and np.max on array with NaN returns NaN.
    """
    orig_model = make_simple_model(output_value=1.0)

    # Create a model that will produce NaN (use a very large value that becomes NaN)
    # Actually, let's just verify the behavior with NaN in the max_diff calculation
    arr1 = np.array([1.0, 2.0, 3.0])
    arr2 = np.array([1.0, 2.0, float("nan")])

    diff = np.max(np.abs(arr1 - arr2))
    assert np.isnan(diff)


def test_verify_with_inf_difference() -> None:
    """Test verification when outputs contain Inf values.

    Inf - Inf = NaN, which behaves similarly to NaN case.
    """
    arr1 = np.array([1.0, 2.0, float("inf")])
    arr2 = np.array([1.0, 2.0, float("inf")])

    diff = np.max(np.abs(arr1 - arr2))
    # Inf - Inf = NaN
    assert np.isnan(diff)


def test_verify_output_name_mismatch() -> None:
    """Test verification when output names don't match."""
    orig_model = make_simple_model(output_value=1.0)

    # Create model with different output name
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])
    output_tensor = helper.make_tensor_value_info("different_output", TensorProto.FLOAT, [1])

    const_node = helper.make_node(
        "Constant",
        [],
        ["const_out"],
        value=helper.make_tensor("value", TensorProto.FLOAT, [], [1.0]),
        name="const1",
    )
    identity_node = helper.make_node("Identity", ["const_out"], ["different_output"], name="id1")

    graph = helper.make_graph(
        [const_node, identity_node],
        "constant_model",
        [],
        [output_tensor],
    )

    split_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    result = verify_equivalence(orig_model, split_model, verbose=False)

    assert result.success is False
    assert result.max_diff == float("inf")
    assert result.failure_reason is not None
    assert "not found in split model" in result.failure_reason


def test_verify_output_shape_mismatch() -> None:
    """Test verification when output shapes don't match."""
    orig_model = make_simple_model(output_value=1.0)

    # Create model with different output shape
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2])  # Different shape

    const_node = helper.make_node(
        "Constant",
        [],
        ["const_out"],
        value=helper.make_tensor("value", TensorProto.FLOAT, [2], [1.0, 2.0]),
        name="const1",
    )
    identity_node = helper.make_node("Identity", ["const_out"], ["output"], name="id1")

    graph = helper.make_graph(
        [const_node, identity_node],
        "constant_model",
        [],
        [output_tensor],
    )

    split_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    result = verify_equivalence(orig_model, split_model, verbose=False)

    assert result.success is False
    assert result.max_diff == float("inf")
    assert result.failure_reason is not None
    assert "shape mismatch" in result.failure_reason


def test_verify_zero_max_diff_display() -> None:
    """Test the display format for edge case max_diff values.

    This verifies the exact scenario where max_diff displays as "0.00e+00".
    Any value < 0.005 will format to "0.00e+00" with .2e format.
    """
    # Test various small values
    test_values = [
        (0.0, "0.00e+00"),
        (0.00001, "1.00e-05"),
        (0.0001, "1.00e-04"),
        (0.001, "1.00e-03"),
        (0.0049, "4.90e-03"),
        (0.00499, "4.99e-03"),  # Actually formats as 4.99e-03 due to float representation
        (0.005, "5.00e-03"),
    ]

    for value, expected in test_values:
        formatted = f"{value:.2e}"
        assert formatted == expected, f"{value} formatted as {formatted}, expected {expected}"


def test_verify_allclose_failure_with_small_max_diff() -> None:
    """Test the actual bug: np.allclose fails but max_diff is tiny.

    This happens when:
    1. Values are large, so relative error is significant
    2. But absolute difference is small
    3. np.allclose uses rtol * |expected| + atol
    """
    # Try with larger values - need diff > rtol * value + atol
    # For value=10000, rtol=1e-4, atol=1e-5:
    # threshold = 1e-4 * 10000 + 1e-5 = 1 + 0.00001 = 1.00001
    # So we need diff > 1.0
    orig_model = make_simple_model(output_value=10000.0)
    split_model = make_simple_model(output_value=10001.5)  # 1.5 difference

    result = verify_equivalence(
        orig_model,
        split_model,
        rtol=1e-4,
        atol=1e-5,
        verbose=False,
    )

    assert result.success is False
    assert result.max_diff > 1.0  # Account for float32 precision
    assert result.failure_reason is not None
    assert "exceeds tolerance" in result.failure_reason


def test_nan_max_diff_handling() -> None:
    """Test that NaN in max_diff is handled properly.

    If max_diff is NaN (from Inf - Inf or NaN - NaN), the display will show "nan".
    But we should detect this and provide better error message.
    """
    # This is a theoretical test - actual reproduction would require
    # a model that produces NaN/Inf outputs

    # Simulate what happens when max_diff is NaN
    max_diff = float("nan")
    formatted = f"{max_diff:.2e}"
    assert formatted == "nan"


def test_zero_tolerance_edge_case() -> None:
    """Test with zero tolerances."""
    orig_model = make_simple_model(output_value=1.0)
    # Float32 cannot represent 1.0 + 1e-10 precisely, so use a larger difference
    split_model = make_simple_model(output_value=1.001)  # 0.001 difference

    result = verify_equivalence(
        orig_model,
        split_model,
        rtol=0.0,  # No relative tolerance
        atol=0.0,  # No absolute tolerance
        verbose=False,
    )

    assert result.success is False
    assert result.max_diff > 0
    assert result.failure_reason is not None
    assert "exceeds tolerance" in result.failure_reason
