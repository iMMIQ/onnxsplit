"""Model equivalence verification using onnxruntime."""

import numpy as np
from onnx import ModelProto

from onnxsplit.verify.result import VerifyResult
from onnxsplit.verify.runtime import RuntimeChecker, ONNXRUNTIME_AVAILABLE


def verify_equivalence(
    original_model: ModelProto,
    split_model: ModelProto,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    seed: int = 42,
    verbose: bool = False,
) -> VerifyResult:
    """Verify that split model produces same outputs as original.

    Args:
        original_model: Original ONNX model before splitting
        split_model: ONNX model after splitting transformation
        rtol: Relative tolerance for numerical comparison
        atol: Absolute tolerance for numerical comparison
        seed: Random seed for input generation
        verbose: Whether to print detailed information

    Returns:
        VerifyResult with verification outcome
    """
    # Check if onnxruntime is available
    if not RuntimeChecker.is_available():
        return VerifyResult.skipped_result(
            "onnxruntime not installed. Install with: pip install onnxruntime"
        )

    checker = RuntimeChecker()

    try:
        # Generate random inputs based on original model
        inputs = checker.generate_random_inputs(original_model, seed=seed)

        if verbose:
            print(f"  Generated {len(inputs)} input(s)")

        # Run inference on both models
        original_outputs = checker.run_inference(original_model, inputs)
        split_outputs = checker.run_inference(split_model, inputs)

        # Verify output counts match
        if len(original_outputs) != len(split_outputs):
            return VerifyResult.failed_result(
                outputs_compared=min(len(original_outputs), len(split_outputs)),
                max_diff=float("inf"),
            )

        # Compare outputs
        max_diff = 0.0
        outputs_compared = len(original_outputs)

        for name in original_outputs:
            if name not in split_outputs:
                return VerifyResult.failed_result(
                    outputs_compared=outputs_compared,
                    max_diff=float("inf"),
                )

            orig = original_outputs[name]
            split = split_outputs[name]

            # Check shapes match
            if orig.shape != split.shape:
                return VerifyResult.failed_result(
                    outputs_compared=outputs_compared,
                    max_diff=float("inf"),
                )

            # Calculate max difference
            diff = np.max(np.abs(orig - split))
            max_diff = max(max_diff, diff)

            # Check if within tolerance
            if not np.allclose(orig, split, rtol=rtol, atol=atol):
                return VerifyResult.failed_result(
                    outputs_compared=outputs_compared,
                    max_diff=max_diff,
                )

        return VerifyResult.passed_result(
            outputs_compared=outputs_compared,
            max_diff=max_diff,
        )

    except Exception as e:
        return VerifyResult.skipped_result(f"Verification failed with error: {e}")


__all__ = [
    "verify_equivalence",
    "VerifyResult",
    "RuntimeChecker",
    "ONNXRUNTIME_AVAILABLE",
]
