# onnxsplit

ONNX model operator splitting tool for memory optimization.

Split large ONNX models by replicating operators and partitioning their inputs/outputs, reducing peak memory usage during inference.

## Installation

```bash
pip install onnxsplit
```

This installs the published runtime dependencies declared by the package, including
`onnx`, `onnxruntime`, `onnxsim`, `pyyaml`, and `typer`.

Or with uv:

```bash
uv pip install onnxsplit
```

## CLI Usage

### Validate a Model

Check if an ONNX model is valid:

```bash
onnxsplit validate model.onnx
```

### Analyze a Model

Get detailed information about model structure:

```bash
onnxsplit analyze model.onnx
```

Output:
```
Model Analysis:
  IR Version: 7
  Opset Version: 13
  Producer: pytorch
  Graph: main_graph
  Inputs: 1
    - input: (1, 3, 224, 224)
  Outputs: 1
    - output: (1, 1000)
  Operators: 65
  Report saved to: output/analysis_report.json
```

### Split a Model

Generate a transformed model plus a split report:

```bash
# Split into 4 parts using a config file
onnxsplit split model.onnx --config onnxsplit.yaml --parts 4 --output output_dir

# Limit memory per split (in MB) while loading config rules
onnxsplit split model.onnx --config onnxsplit.yaml --max-memory 1024 --output output_dir

# Verify output equivalence using onnxruntime (optional)
onnxsplit split model.onnx --config onnxsplit.yaml --parts 2 --verify

# Verbose output
onnxsplit -v split model.onnx --config onnxsplit.yaml --parts 2
```

Output:
```
Model split successfully!
  Output: output_dir/split_model.onnx
  Report: output_dir/split_report.json
  Summary: SplitReport: 1/65 operators split (1.5%), total 2 parts
  ✓ Verification passed: 1 outputs match
```

> **Note**: The `--verify` option uses the package's `onnxruntime` dependency at runtime.
> If `onnxruntime` is unavailable in the current environment, verification is skipped with a warning.

## Python API

```python
import onnx

from onnxsplit import GraphTransformer, ModelAnalyzer, SplitPlanner, verify_equivalence
from onnxsplit.config import OperatorConfig, SplitConfig

# Load the original model
original_model = onnx.load("model.onnx")
current_model = original_model

# Analyze the current model and build a split plan
analyzer = ModelAnalyzer.from_model_proto(current_model)
print(f"Operators: {len(analyzer.get_operators())}")

# Create split plan configuration
config = SplitConfig(
    operators={
        "conv_layer": OperatorConfig(parts=2, axis=0),
    }
)

planner = SplitPlanner(analyzer, config)
report = planner.generate()

# Apply each split plan to the progressively transformed model
for plan in report.plans:
    if not plan.is_split:
        continue

    current_analyzer = ModelAnalyzer.from_model_proto(current_model)
    transformer = GraphTransformer(current_analyzer)
    current_model = transformer.apply_split_plan(plan)

split_model = current_model

# Save result
onnx.save(split_model, "split_model.onnx")

# Verify equivalence (requires onnxruntime)
result = verify_equivalence(original_model, split_model)
if result.success:
    print(f"Verification passed: {result.outputs_compared} outputs match")
elif result.skipped:
    print(f"Verification skipped: {result.skip_reason}")
else:
    print(f"Verification failed: {result.failure_reason}")
```

## Features

- **Operator splitting**: Split large operators by partitioning inputs/outputs
- **Memory estimation**: Calculate memory usage for operators and tensors
- **Auto adjustment**: Automatically determine optimal split count based on memory limits
- **Output verification**: Verify split model produces same outputs as original using onnxruntime
- **Configurable rules**: YAML-based configuration for split strategies per operator type
- **CLI & API**: Use as command-line tool or Python library

## How It Works

1. **Analyze** the model graph to identify operators that can be split
2. **Plan** which operators to split based on configuration and memory rules
3. **Transform** the graph by:
   - Inserting Split nodes to partition inputs
   - Cloning the target operator for each partition
   - Inserting Concat nodes to merge outputs

## Configuration

Create a `onnxsplit.yaml` file for custom split rules:

```yaml
global:
  default_parts: 2
  max_memory_mb: 1024

operators:
  "/encoder/Conv_0":
    parts: 4
    axis: 0

  "/decoder/MatMul_*":
    parts: 2

axis_rules:
  - op_type: Conv
    prefer_axis: 0

  - op_type: MatMul
    prefer_axis: batch

  - op_type: LayerNormalization
    prefer_axis: null

memory_rules:
  auto_adjust: true
  overflow_strategy: binary_split
```

`memory_rules.overflow_strategy` accepts `binary_split` or `linear_split` when auto-adjusting
parts to satisfy a memory limit.

## Requirements

Package/runtime requirements from `pyproject.toml`:

- Python 3.13+
- onnx >= 1.20.1
- onnxruntime >= 1.23.2
- onnxsim >= 0.4.34
- pyyaml >= 6.0.3
- typer >= 0.21.1

## License

MIT
