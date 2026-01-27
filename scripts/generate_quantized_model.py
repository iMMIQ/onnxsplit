"""Generate quantized ONNX model with QDQ format using onnxruntime."""

import onnx
import numpy as np
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat
from onnxruntime.quantization import CalibrationDataReader
from pathlib import Path

MODEL_FP32 = "/home/ayd/code/onnxsplit/models/resnet18.onnx"
MODEL_QUANT = "/home/ayd/code/onnxsplit/models/resnet18_quan.onnx"


class DummyDataReader(CalibrationDataReader):
    """Minimal calibration data reader for static quantization."""

    def __init__(self, input_name: str, batch_size: int = 1, num_samples: int = 10):
        self.input_name = input_name
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.current = 0

    def get_next(self):
        if self.current < self.num_samples:
            self.current += 1
            # Generate random data matching ResNet18 input shape (batch_size, 3, 224, 224)
            return {self.input_name: np.random.rand(self.batch_size, 3, 224, 224).astype(np.float32)}
        return None


# Get the input name from the model
model = onnx.load(MODEL_FP32)
input_name = model.graph.input[0].name

# Create calibration data reader
dr = DummyDataReader(input_name, batch_size=1, num_samples=5)

# Quantize the model with QDQ format (QuantizeLinear/DequantizeLinear nodes)
quantize_static(
    MODEL_FP32,
    MODEL_QUANT,
    calibration_data_reader=dr,
    quant_format=QuantFormat.QDQ,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8,
)

print(f"Quantized model saved to {MODEL_QUANT}")

# Verify the model
from onnx import checker
model = onnx.load(MODEL_QUANT)
checker.check_model(model)
print("Model is valid!")

# Print some stats
op_counts = {}
for node in model.graph.node:
    op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

print("\nOperator counts:")
for op_type, count in sorted(op_counts.items()):
    if op_type in ['QuantizeLinear', 'DequantizeLinear']:
        print(f"  {op_type}: {count} <- QDQ nodes!")
    else:
        print(f"  {op_type}: {count}")
