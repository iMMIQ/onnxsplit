"""CLI parser tests."""

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

        result = runner.invoke(app, ["split", "model.onnx"])
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


def test_split_command_with_verify_option():
    """Test split command with verify option."""
    with runner.isolated_filesystem():
        model_path = Path("model.onnx")
        _create_minimal_onnx_model(model_path)

        result = runner.invoke(app, ["split", "model.onnx", "--verify"])
        # Command should succeed even if onnxruntime is not available
        assert result.exit_code == 0
        assert Path("output").exists()
        # Should show verification output (either passed or skipped)
        assert "verification" in result.stdout.lower() or "verify" in result.stdout.lower()
