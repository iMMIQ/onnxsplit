# onnxsplit

ONNX model operator splitting tool for memory optimization.

Split large ONNX models by replicating operators and partitioning their inputs/outputs, reducing peak memory usage during inference.

## Installation

```bash
pip install onnxsplit
```

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

Split a model into multiple parts:

```bash
# Split into 4 parts
onnxsplit split model.onnx --parts 4 --output output_dir

# Limit memory per split (in MB)
onnxsplit split model.onnx --max-memory 1024 --output output_dir

# Verbose output
onnxsplit split -v model.onnx --parts 2
```

Output:
```
Model split successfully!
  Output: output_dir/split_model.onnx
  Report: output_dir/split_report.json
  Summary: SplitReport: 5/65 operators split (7.7%), total 2 parts
```

## Python API

```python
from onnxsplit import ModelAnalyzer, SplitPlanner, GraphTransformer

# Load and analyze model
analyzer = ModelAnalyzer.from_path("model.onnx")
print(f"Operators: {len(analyzer.get_operators())}")

# Create split plan
from onnxsplit.config import SplitConfig
config = SplitConfig()
planner = SplitPlanner(analyzer, config)
report = planner.generate()

# Apply transformation
transformer = GraphTransformer(analyzer)
for plan in report.plans:
    if plan.is_split:
        model = transformer.apply_split_plan(model)

# Save result
import onnx
onnx.save(model, "split_model.onnx")
```

## Features

- **Operator splitting**: Split large operators by partitioning inputs/outputs
- **Memory estimation**: Calculate memory usage for operators and tensors
- **Auto adjustment**: Automatically determine optimal split count based on memory limits
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
global_config:
  default_parts: 2
  max_memory_mb: 1024

operator_rules:
  - op_types: [Conv, MatMul, Gemm]
    axis: 0
    default_parts: 4

  - op_types: [Add, Mul]
    split: false

memory_rules:
  auto_adjust: true
```

## Requirements

- Python 3.9+
- onnx >= 1.14.0

## License

MIT
