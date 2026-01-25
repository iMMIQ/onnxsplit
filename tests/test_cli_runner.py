"""Tests for CLI runner module."""

import json
from pathlib import Path

import pytest
from onnx import TensorProto, helper

from onnxsplit.cli.runner import RunContext, _generate_report, run_split


@pytest.fixture
def simple_onnx_model(tmp_path: Path) -> Path:
    """Create a simple ONNX model for testing."""
    # Create a simple linear model: Input -> MatMul -> Add -> Output
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])

    # Create weight initializer
    weight_data = [0.1] * 100  # 10x10 weight matrix
    weight = helper.make_tensor("weight", TensorProto.FLOAT, [10, 10], weight_data)
    bias_data = [0.0] * 10
    bias = helper.make_tensor("bias", TensorProto.FLOAT, [10], bias_data)

    # Create nodes
    matmul_node = helper.make_node("MatMul", ["input", "weight"], ["matmul_out"], name="matmul1")
    add_node = helper.make_node("Add", ["matmul_out", "bias"], ["output"], name="add1")

    # Create graph
    graph = helper.make_graph(
        [matmul_node, add_node],
        "simple_model",
        [input_tensor],
        [output_tensor],
        [weight, bias],
    )

    # Create model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model_path = tmp_path / "simple_model.onnx"
    model_path.write_bytes(model.SerializeToString())
    return model_path


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """Create a test config file."""
    config_content = """
global:
  default_parts: 2
  max_memory_mb: 1000

operators:
  matmul1:
    parts: 4
    axis: 0

axis_rules:
  - op_type: MatMul
    prefer_axis: 0

memory_rules:
  auto_adjust: true
  overflow_strategy: binary_split
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)
    return config_path


class TestRunContext:
    """Tests for RunContext dataclass."""

    def test_run_context_creation(self) -> None:
        """Test RunContext can be created with all fields."""
        ctx = RunContext(
            model_path="/path/to/model.onnx",
            output_dir="/path/to/output",
            config_path="/path/to/config.yaml",
            cli_parts=4,
            cli_max_memory=2000,
            verbose=True,
            quiet=False,
        )
        assert ctx.model_path == "/path/to/model.onnx"
        assert ctx.output_dir == "/path/to/output"
        assert ctx.config_path == "/path/to/config.yaml"
        assert ctx.cli_parts == 4
        assert ctx.cli_max_memory == 2000
        assert ctx.verbose is True
        assert ctx.quiet is False

    def test_run_context_defaults(self) -> None:
        """Test RunContext default values."""
        ctx = RunContext(model_path="/path/to/model.onnx")
        assert ctx.model_path == "/path/to/model.onnx"
        assert ctx.output_dir == "output"
        assert ctx.config_path is None
        assert ctx.cli_parts is None
        assert ctx.cli_max_memory is None
        assert ctx.verbose is False
        assert ctx.quiet is False
        assert ctx.verify is False


class TestRunSplit:
    """Tests for run_split function."""

    def test_run_split_basic(self, simple_onnx_model: Path, tmp_path: Path) -> None:
        """Test basic run_split execution."""
        output_dir = tmp_path / "output"
        ctx = RunContext(
            model_path=str(simple_onnx_model),
            output_dir=str(output_dir),
            verbose=False,
            quiet=False,
            verify=True,
        )

        # Should not raise
        run_split(ctx)

        # Check output directory was created
        assert output_dir.exists()
        # Check model was saved
        assert (output_dir / "split_model.onnx").exists()

    def test_run_split_with_config(
        self, simple_onnx_model: Path, config_file: Path, tmp_path: Path
    ) -> None:
        """Test run_split with config file."""
        output_dir = tmp_path / "output2"
        ctx = RunContext(
            model_path=str(simple_onnx_model),
            output_dir=str(output_dir),
            config_path=str(config_file),
            verbose=True,
            quiet=False,
        )

        # Should not raise
        run_split(ctx)

        # Check output directory was created
        assert output_dir.exists()
        assert (output_dir / "split_model.onnx").exists()

    def test_run_split_with_cli_args(self, simple_onnx_model: Path, tmp_path: Path) -> None:
        """Test run_split with CLI arguments overriding config."""
        output_dir = tmp_path / "output3"
        ctx = RunContext(
            model_path=str(simple_onnx_model),
            output_dir=str(output_dir),
            cli_parts=3,
            cli_max_memory=1500,
            verbose=False,
            quiet=False,
            verify=True,
        )

        # Should not raise
        run_split(ctx)

        assert output_dir.exists()
        assert (output_dir / "split_model.onnx").exists()

    def test_run_split_invalid_model_path(self, tmp_path: Path) -> None:
        """Test run_split with invalid model path."""
        output_dir = tmp_path / "output4"
        ctx = RunContext(
            model_path="/nonexistent/model.onnx",
            output_dir=str(output_dir),
        )

        result = run_split(ctx)
        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_run_split_invalid_config(self, simple_onnx_model: Path, tmp_path: Path) -> None:
        """Test run_split with invalid config file."""
        output_dir = tmp_path / "output5"
        config_path = tmp_path / "invalid.yaml"
        config_path.write_text("invalid: yaml: content:")

        ctx = RunContext(
            model_path=str(simple_onnx_model),
            output_dir=str(output_dir),
            config_path=str(config_path),
        )

        result = run_split(ctx)
        assert result.success is False
        assert result.error is not None
        assert "configuration" in result.error.lower() or "yaml" in result.error.lower()

    def test_run_split_creates_output_dir(self, simple_onnx_model: Path, tmp_path: Path) -> None:
        """Test that run_split creates output directory if it doesn't exist."""
        output_dir = tmp_path / "deep" / "nested" / "output"
        ctx = RunContext(
            model_path=str(simple_onnx_model),
            output_dir=str(output_dir),
        )

        run_split(ctx)

        assert output_dir.exists()
        assert (output_dir / "split_model.onnx").exists()

    def test_run_split_verbose_output(
        self, simple_onnx_model: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test verbose output from run_split."""
        output_dir = tmp_path / "output6"
        ctx = RunContext(
            model_path=str(simple_onnx_model),
            output_dir=str(output_dir),
            verbose=True,
            quiet=False,
            verify=True,
        )

        run_split(ctx)

        captured = capsys.readouterr()
        # Check for verbose output
        assert "model" in captured.out.lower() or "split" in captured.out.lower()
        # Check for verification output
        assert "verification" in captured.out.lower() or "verif" in captured.out.lower()

    def test_run_split_quiet_mode(
        self, simple_onnx_model: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test quiet mode suppresses output."""
        output_dir = tmp_path / "output7"
        ctx = RunContext(
            model_path=str(simple_onnx_model),
            output_dir=str(output_dir),
            verbose=False,
            quiet=True,
        )

        run_split(ctx)

        captured = capsys.readouterr()
        # Quiet mode should have minimal output
        assert len(captured.out) == 0 or "Saved" not in captured.out

    def test_run_split_saves_report(self, simple_onnx_model: Path, tmp_path: Path) -> None:
        """Test that run_split saves a JSON report."""
        output_dir = tmp_path / "output8"
        ctx = RunContext(
            model_path=str(simple_onnx_model),
            output_dir=str(output_dir),
            verbose=False,
            quiet=False,
            verify=True,
        )

        run_split(ctx)

        # Check for report file
        report_path = output_dir / "split_report.json"
        assert report_path.exists()

        # Verify it's valid JSON
        data = json.loads(report_path.read_text())
        assert "original_operators" in data
        assert "split_operators" in data


class TestGenerateReport:
    """Tests for _generate_report function."""

    def test_generate_report_basic(self, tmp_path: Path) -> None:
        """Test basic report generation."""
        from onnxsplit.splitter.plan import SplitReport

        report = SplitReport(
            original_operators=10,
            split_operators=3,
            unsplit_operators=7,
            plans=[],
        )

        output_path = tmp_path / "report.json"
        _generate_report(report, str(output_path))

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["original_operators"] == 10
        assert data["split_operators"] == 3
        assert data["unsplit_operators"] == 7

    def test_generate_report_with_plans(self, tmp_path: Path) -> None:
        """Test report generation with split plans."""
        from onnxsplit.splitter.plan import SplitPlan, SplitReport

        plans = [
            SplitPlan(operator_name="op1", parts=4, axis=0, reason="Large tensor"),
            SplitPlan(operator_name="op2", parts=2, axis=1, reason="Memory limit"),
        ]
        report = SplitReport(
            original_operators=5,
            split_operators=2,
            unsplit_operators=3,
            plans=plans,
        )

        output_path = tmp_path / "report2.json"
        _generate_report(report, str(output_path))

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert len(data["plans"]) == 2
        assert data["plans"][0]["operator_name"] == "op1"
        assert data["plans"][0]["parts"] == 4
        assert data["plans"][1]["operator_name"] == "op2"
        assert data["plans"][1]["parts"] == 2

    def test_generate_report_creates_directory(self, tmp_path: Path) -> None:
        """Test that report generation creates parent directories."""
        from onnxsplit.splitter.plan import SplitReport

        report = SplitReport(
            original_operators=1,
            split_operators=0,
            unsplit_operators=1,
            plans=[],
        )

        output_path = tmp_path / "deep" / "nested" / "report.json"
        _generate_report(report, str(output_path))

        assert output_path.exists()
        assert output_path.parent.is_dir()

    def test_generate_report_overwrites_existing(self, tmp_path: Path) -> None:
        """Test that report generation overwrites existing file."""
        from onnxsplit.splitter.plan import SplitReport

        output_path = tmp_path / "report.json"
        output_path.write_text('{"old": "data"}')

        report = SplitReport(
            original_operators=5,
            split_operators=2,
            unsplit_operators=3,
            plans=[],
        )

        _generate_report(report, str(output_path))

        data = json.loads(output_path.read_text())
        assert "old" not in data
        assert data["original_operators"] == 5
