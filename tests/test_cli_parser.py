"""CLI parser tests."""

from typer.testing import CliRunner

from onnxsplit.cli.parser import CliOptions, app

runner = CliRunner()


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
    result = runner.invoke(app, ["split", "model.onnx"])
    assert result.exit_code == 0
    assert "split" in result.stdout.lower()
    assert "model.onnx" in result.stdout


def test_split_command_with_options():
    """Test split command with all options."""
    result = runner.invoke(
        app,
        [
            "split",
            "model.onnx",
            "--output",
            "output_dir",
            "--strategy",
            "size-based",
            "--max-size",
            "512",
            "--target-ops",
            "100",
        ],
    )
    assert result.exit_code == 0
    assert "output_dir" in result.stdout
    assert "size-based" in result.stdout


def test_analyze_command_basic():
    """Test analyze command with basic arguments."""
    result = runner.invoke(app, ["analyze", "model.onnx"])
    assert result.exit_code == 0
    assert "analyze" in result.stdout.lower()
    assert "model.onnx" in result.stdout


def test_analyze_command_with_options():
    """Test analyze command with all options."""
    result = runner.invoke(
        app,
        [
            "analyze",
            "model.onnx",
            "--output",
            "analysis.json",
            "--format",
            "json",
            "--include-graph",
        ],
    )
    assert result.exit_code == 0
    assert "analysis.json" in result.stdout


def test_validate_command_basic():
    """Test validate command with basic arguments."""
    result = runner.invoke(app, ["validate", "model.onnx"])
    assert result.exit_code == 0
    assert "validate" in result.stdout.lower()
    assert "model.onnx" in result.stdout


def test_validate_command_with_options():
    """Test validate command with all options."""
    result = runner.invoke(
        app,
        [
            "validate",
            "model.onnx",
            "--check",
            "graph",
            "--check",
            "metadata",
        ],
    )
    assert result.exit_code == 0
    assert "graph" in result.stdout
    assert "metadata" in result.stdout


def test_global_verbose_flag():
    """Test global verbose flag."""
    result = runner.invoke(app, ["--verbose", "split", "model.onnx"])
    assert result.exit_code == 0
    # Verbose mode adds additional output lines
    assert "splitting model:" in result.stdout.lower()


def test_global_quiet_flag():
    """Test global quiet flag."""
    result = runner.invoke(app, ["--quiet", "split", "model.onnx"])
    assert result.exit_code == 0
    # Quiet mode suppresses all normal output
    assert result.stdout == ""


def test_global_output_option():
    """Test global output option."""
    result = runner.invoke(app, ["--output", "custom_output", "split", "model.onnx"])
    assert result.exit_code == 0
    assert "custom_output" in result.stdout


def test_help_displays_all_commands():
    """Test that help displays all available commands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "split" in result.stdout
    assert "analyze" in result.stdout
    assert "validate" in result.stdout


def test_split_command_help():
    """Test split command help."""
    result = runner.invoke(app, ["split", "--help"])
    assert result.exit_code == 0
    assert "strategy" in result.stdout
    assert "max-size" in result.stdout
    assert "target-ops" in result.stdout


def test_analyze_command_help():
    """Test analyze command help."""
    result = runner.invoke(app, ["analyze", "--help"])
    assert result.exit_code == 0
    assert "format" in result.stdout
    assert "include-graph" in result.stdout


def test_validate_command_help():
    """Test validate command help."""
    result = runner.invoke(app, ["validate", "--help"])
    assert result.exit_code == 0
    assert "check" in result.stdout


def test_no_command_provided():
    """Test behavior when no command is provided."""
    result = runner.invoke(app, [])
    # CliRunner returns exit code 2 even with no_args_is_help=True
    # but still displays the help message
    assert result.exit_code != 0
    # Should display help when no command is provided
    assert "Usage" in result.stdout or "usage" in result.stdout


def test_invalid_command():
    """Test behavior with invalid command."""
    result = runner.invoke(app, ["invalid_command", "model.onnx"])
    assert result.exit_code != 0
    # Error message is in stderr, not stdout
    assert (
        "No such command" in result.stderr
        or "not found" in result.stderr
        or "invalid" in result.stderr.lower()
    )
