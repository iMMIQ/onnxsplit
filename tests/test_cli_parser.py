"""CLI parser tests."""

import json
from pathlib import Path

import onnx
from onnx import helper
from typer.testing import CliRunner

from onnxsplit.cli.parser import CliOptions, app

runner = CliRunner()


def _create_minimal_onnx_model(path: Path) -> None:
    """Create a minimal valid ONNX model for testing.

    Args:
        path: Path where to save the model
    """
    # Create a simple graph with one input, one output, and one identity node
    input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3, 224, 224])
    output_tensor = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 3, 224, 224])

    # Create an identity node
    node = helper.make_node("Identity", inputs=["input"], outputs=["output"])

    # Create the graph
    graph = helper.make_graph(
        [node],
        "test_model",
        [input_tensor],
        [output_tensor],
    )

    # Create the model
    model = helper.make_model(graph)
    model.opset_import[0].version = 18

    # Save the model
    onnx.save(model, str(path))


def _create_matmul_onnx_model(path: Path) -> None:
    """Create a small MatMul model whose split plan can be config-driven."""
    input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [4, 8, 10])
    output_tensor = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [4, 8, 6])

    weight_data = [0.1] * (4 * 10 * 6)
    weight = helper.make_tensor("weight", onnx.TensorProto.FLOAT, [4, 10, 6], weight_data)

    node = helper.make_node("MatMul", inputs=["input", "weight"], outputs=["output"], name="matmul1")
    graph = helper.make_graph(
        [node],
        "config_driven_test_model",
        [input_tensor],
        [output_tensor],
        [weight],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, str(path))


def test_cli_options_defaults():
    """Test CliOptions default values."""
    options = CliOptions()
    assert options.verbose is False
    assert options.quiet is False
    assert options.output is None


def test_cli_options_with_values():
    """Test CliOptions with explicit values."""
    options = CliOptions(verbose=True, quiet=True, output="output_dir")
    assert options.verbose is True
    assert options.quiet is True
    assert options.output == "output_dir"


def test_split_command_basic():
    """Test split command with basic arguments."""
    with runner.isolated_filesystem():
        model_path = Path("model.onnx")
        _create_minimal_onnx_model(model_path)

        result = runner.invoke(app, ["split", "model.onnx", "--no-simplify"])
        assert result.exit_code == 0
        # Should create output directory
        assert Path("output").exists()


def test_split_command_with_options():
    """Test split command with all options."""
    with runner.isolated_filesystem():
        model_path = Path("model.onnx")
        _create_minimal_onnx_model(model_path)

        result = runner.invoke(
            app,
            [
                "split",
                "model.onnx",
                "--output",
                "custom_output",
                "--parts",
                "4",
                "--no-simplify",
            ],
        )
        assert result.exit_code == 0
        # Should create custom output directory
        assert Path("custom_output").exists()


def test_analyze_command_basic():
    """Test analyze command with basic arguments."""
    with runner.isolated_filesystem():
        model_path = Path("model.onnx")
        _create_minimal_onnx_model(model_path)

        result = runner.invoke(app, ["analyze", "model.onnx"])
        assert result.exit_code == 0
        assert "Model Analysis:" in result.stdout
        # Should create output directory
        assert Path("output").exists()


def test_analyze_command_with_options():
    """Test analyze command with output option."""
    with runner.isolated_filesystem():
        model_path = Path("model.onnx")
        _create_minimal_onnx_model(model_path)

        result = runner.invoke(
            app,
            [
                "analyze",
                "model.onnx",
                "--output",
                "custom_output",
            ],
        )
        assert result.exit_code == 0
        # Should create custom output directory
        assert Path("custom_output").exists()


def test_validate_command_basic():
    """Test validate command with basic arguments."""
    with runner.isolated_filesystem():
        model_path = Path("model.onnx")
        _create_minimal_onnx_model(model_path)

        result = runner.invoke(app, ["validate", "model.onnx"])
        assert result.exit_code == 0
        assert "validation passed" in result.stdout.lower()


def test_validate_command_with_options():
    """Test validate command is available."""
    result = runner.invoke(app, ["validate", "--help"])
    assert result.exit_code == 0
    assert "validate" in result.stdout.lower()


def test_help_displays_all_commands():
    """Test help displays all commands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "split" in result.stdout.lower()
    assert "analyze" in result.stdout.lower()
    assert "validate" in result.stdout.lower()


def test_split_command_help():
    """Test split command help."""
    result = runner.invoke(app, ["split", "--help"])
    assert result.exit_code == 0
    assert "parts" in result.stdout.lower()
    assert "max-memory" in result.stdout.lower()


def test_split_command_help_shows_config():
    """Test split help lists the config option."""
    result = runner.invoke(app, ["split", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.stdout


def test_split_command_with_config_affects_generated_report():
    """Test split command uses --config to change the actual split plan."""
    with runner.isolated_filesystem():
        model_path = Path("model.onnx")
        _create_matmul_onnx_model(model_path)

        config_path = Path("config.yaml")
        config_path.write_text(
            "global:\n"
            "  default_parts: 1\n"
            "\n"
            "operators:\n"
            "  matmul1:\n"
            "    parts: 2\n"
            "    axis: 0\n"
        )

        baseline_output_dir = Path("baseline_output")
        baseline_result = runner.invoke(
            app,
            [
                "split",
                "model.onnx",
                "--output",
                "baseline_output",
                "--no-simplify",
            ],
        )

        assert baseline_result.exit_code == 0
        baseline_report = json.loads((baseline_output_dir / "split_report.json").read_text())
        assert baseline_report["split_operators"] == 0
        assert baseline_report["total_parts"] == 0

        output_dir = Path("config_output")
        config_result = runner.invoke(
            app,
            [
                "split",
                "model.onnx",
                "--config",
                "config.yaml",
                "--output",
                "config_output",
                "--no-simplify",
            ],
        )

        assert config_result.exit_code == 0
        assert (output_dir / "split_model.onnx").exists()
        report_path = output_dir / "split_report.json"
        assert report_path.exists()

        report = json.loads(report_path.read_text())
        assert report["split_operators"] == 1
        assert report["total_parts"] == 2

        plans_by_name = {plan["operator_name"]: plan for plan in report["plans"]}
        assert plans_by_name["matmul1"]["parts"] == 2
        assert plans_by_name["matmul1"]["axis"] == 0


def test_analyze_command_help():
    """Test analyze command help."""
    result = runner.invoke(app, ["analyze", "--help"])
    assert result.exit_code == 0
    assert "output" in result.stdout.lower()


def test_validate_command_help():
    """Test validate command help."""
    result = runner.invoke(app, ["validate", "--help"])
    assert result.exit_code == 0
    assert "validate" in result.stdout.lower()


def test_no_command_provided():
    """Test no command provided shows help."""
    result = runner.invoke(app, [])
    # Should show help when no command provided
    assert result.exit_code == 0 or "split" in result.stdout.lower()


def test_invalid_command():
    """Test invalid command returns error."""
    result = runner.invoke(app, ["invalid_command"])
    assert result.exit_code != 0


def test_split_command_nonexistent_file():
    """Test split command with non-existent file returns error."""
    result = runner.invoke(app, ["split", "nonexistent.onnx"])
    assert result.exit_code != 0


def test_validate_invalid_model():
    """Test validate command with invalid model."""
    with runner.isolated_filesystem():
        # Create a file with invalid ONNX content
        Path("invalid.onnx").write_text("not a valid onnx file")

        result = runner.invoke(app, ["validate", "invalid.onnx"])
        assert result.exit_code != 0


def test_split_command_help_shows_verify():
    """Test split command help shows verify option."""
    result = runner.invoke(app, ["split", "--help"])
    assert result.exit_code == 0
    assert "verify" in result.stdout.lower()
    assert "no-simplify" in result.stdout.lower() or "simplify" in result.stdout.lower()


def test_split_command_with_verify_option():
    """Test split command with verify option."""
    with runner.isolated_filesystem():
        model_path = Path("model.onnx")
        _create_minimal_onnx_model(model_path)

        result = runner.invoke(app, ["split", "model.onnx", "--verify", "--no-simplify"])
        # Command should succeed even if onnxruntime is not available
        assert result.exit_code == 0
        assert Path("output").exists()
        # Should show verification output (either passed or skipped)
        assert "verification" in result.stdout.lower() or "verify" in result.stdout.lower()
