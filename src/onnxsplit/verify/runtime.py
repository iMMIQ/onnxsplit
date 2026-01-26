"""Runtime verification using onnxruntime."""

from typing import Optional

import numpy as np
import onnx
from onnx import ModelProto

# Try to import onnxruntime, but make it optional
try:
    import onnxruntime as ort

    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    ort = None


class RuntimeChecker:
    """Checker for onnxruntime availability and inference execution."""

    # EP priority order (fastest first)
    EP_PRIORITY = [
        "CUDAExecutionProvider",
        "TensorrtExecutionProvider",
        "ROCmExecutionProvider",
        "OpenVINOExecutionProvider",
        "DnnlExecutionProvider",
        "CoreMLExecutionProvider",
        "XnnpackExecutionProvider",
        "CPUExecutionProvider",
    ]

    @staticmethod
    def get_available_providers() -> list[str]:
        """Get available execution providers in priority order.

        Returns:
            List of available provider names sorted by performance priority.
        """
        if not ONNXRUNTIME_AVAILABLE:
            return []

        available = ort.get_available_providers()
        # Return providers in priority order, but only those that are available
        return [p for p in RuntimeChecker.EP_PRIORITY if p in available]

    @staticmethod
    def is_available() -> bool:
        """Check if onnxruntime is available."""
        return ONNXRUNTIME_AVAILABLE

    @staticmethod
    def run_inference(
        model: ModelProto,
        inputs: dict[str, np.ndarray],
        providers: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Run inference using onnxruntime.

        Args:
            model: ONNX model to run inference on
            inputs: Dictionary mapping input names to numpy arrays
            providers: Optional list of execution providers. If None, uses
                       available providers in priority order (fastest first).

        Returns:
            Dictionary mapping output names to numpy arrays

        Raises:
            RuntimeError: If onnxruntime is not available
            ValueError: If inference fails
        """
        if not ONNXRUNTIME_AVAILABLE:
            raise RuntimeError("onnxruntime is not available")

        import tempfile
        import os

        # Use optimal providers if not specified
        if providers is None:
            providers = RuntimeChecker.get_available_providers()

        # Create temporary file for the model
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name
            onnx.save(model, temp_path)

        try:
            # Create inference session with optimal providers
            sess = ort.InferenceSession(temp_path, providers=providers)

            # Prepare input dict matching session's expected inputs
            input_dict = {}
            for inp in sess.get_inputs():
                if inp.name in inputs:
                    input_dict[inp.name] = inputs[inp.name]

            # Run inference
            outputs = sess.run(None, input_dict)

            # Collect outputs
            output_dict = {}
            for i, out in enumerate(sess.get_outputs()):
                output_dict[out.name] = outputs[i]

            return output_dict
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @staticmethod
    def generate_random_inputs(model: ModelProto, seed: int = 42) -> dict[str, np.ndarray]:
        """Generate random input data based on model's input shapes.

        Args:
            model: ONNX model to generate inputs for
            seed: Random seed for reproducibility

        Returns:
            Dictionary mapping input names to numpy arrays
        """
        rng = np.random.default_rng(seed)

        inputs = {}
        for inp in model.graph.input:
            shape = []
            for dim in inp.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else:
                    # Use a reasonable default for dynamic dimensions
                    shape.append(1)

            # Get dtype
            dtype = np.float32  # default
            onnx_dtype = inp.type.tensor_type.elem_type
            if onnx_dtype == onnx.TensorProto.FLOAT:
                dtype = np.float32
            elif onnx_dtype == onnx.TensorProto.DOUBLE:
                dtype = np.float64
            elif onnx_dtype == onnx.TensorProto.INT32:
                dtype = np.int32
            elif onnx_dtype == onnx.TensorProto.INT64:
                dtype = np.int64
            elif onnx_dtype == onnx.TensorProto.BOOL:
                dtype = np.bool_

            # Generate random data with appropriate distribution
            if dtype == np.bool_:
                inputs[inp.name] = rng.random(shape).astype(np.float32) > 0.5
            elif np.issubdtype(dtype, np.integer):
                inputs[inp.name] = rng.integers(-10, 10, size=shape).astype(dtype)
            else:
                # For float types, use standard normal distribution
                inputs[inp.name] = rng.standard_normal(size=shape).astype(dtype)

        return inputs
