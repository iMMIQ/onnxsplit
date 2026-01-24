"""CLI parser for onnxsplit."""

from dataclasses import dataclass, field
from typing import Optional

import typer

app = typer.Typer(
    help="ONNX model splitting tool for partitioning large models into smaller components.",
    no_args_is_help=True,
    add_completion=False,
)


@dataclass
class CliOptions:
    """Global CLI options passed to commands."""

    verbose: bool = field(default=False)
    quiet: bool = field(default=False)
    output: Optional[str] = field(default=None)


@app.callback()
def cli_options(
    ctx: typer.Context,
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress non-error output.",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for generated files.",
    ),
) -> None:
    """Global CLI options."""
    ctx.ensure_object(dict)
    ctx.obj = CliOptions(verbose=verbose, quiet=quiet, output=output)


@app.command()
def split(
    ctx: typer.Context,
    model_path: str = typer.Argument(
        ...,
        help="Path to the ONNX model file to split.",
        exists=True,
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for split model files.",
    ),
    strategy: str = typer.Option(
        "size-based",
        "--strategy",
        "-s",
        help="Splitting strategy: 'size-based', 'op-based', or 'custom'.",
    ),
    max_size: int = typer.Option(
        1024,
        "--max-size",
        "-m",
        help="Maximum size (MB) per split file.",
    ),
    target_ops: int = typer.Option(
        50,
        "--target-ops",
        "-t",
        help="Target number of operations per split for op-based strategy.",
    ),
) -> None:
    """Split an ONNX model into smaller components.

    This command partitions a large ONNX model into multiple smaller models
    based on the specified splitting strategy.
    """
    options: CliOptions = ctx.obj
    output_path = output or options.output or "output"

    if options.verbose:
        typer.echo(f"Splitting model: {model_path}")
        typer.echo(f"Strategy: {strategy}")
        typer.echo(f"Output directory: {output_path}")

    if not options.quiet:
        typer.echo(f"Split {model_path} with {strategy} strategy to {output_path}")
        if strategy == "size-based":
            typer.echo(f"Max size per split: {max_size} MB")
        elif strategy == "op-based":
            typer.echo(f"Target ops per split: {target_ops}")


@app.command()
def analyze(
    ctx: typer.Context,
    model_path: str = typer.Argument(
        ...,
        help="Path to the ONNX model file to analyze.",
        exists=True,
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for analysis results.",
    ),
    format: str = typer.Option(
        "text",
        "--format",
        "-f",
        help="Output format: 'text', 'json', or 'yaml'.",
    ),
    include_graph: bool = typer.Option(
        False,
        "--include-graph",
        help="Include graph structure in analysis.",
    ),
) -> None:
    """Analyze an ONNX model and report structure information.

    This command examines the model's computational graph and provides
    detailed information about its structure, operations, and size.
    """
    options: CliOptions = ctx.obj
    output_file = output or f"analysis.{format}"

    if options.verbose:
        typer.echo(f"Analyzing model: {model_path}")
        typer.echo(f"Output format: {format}")
        typer.echo(f"Output file: {output_file}")

    if not options.quiet:
        typer.echo(f"Analyze {model_path} and save to {output_file} ({format} format)")
        if include_graph:
            typer.echo("Including graph structure in analysis")


@app.command()
def validate(
    ctx: typer.Context,
    model_path: str = typer.Argument(
        ...,
        help="Path to the ONNX model file to validate.",
        exists=True,
    ),
    check: list[str] = typer.Option(
        ["all"],
        "--check",
        "-c",
        help="Validation checks to run: 'graph', 'metadata', or 'all'.",
    ),
) -> None:
    """Validate an ONNX model for correctness and compatibility.

    This command performs various validation checks on the model to ensure
    it meets ONNX specification and can be properly loaded and executed.
    """
    options: CliOptions = ctx.obj

    if options.verbose:
        typer.echo(f"Validating model: {model_path}")
        typer.echo(f"Checks: {', '.join(check)}")

    if not options.quiet:
        typer.echo(f"Validate {model_path} with checks: {', '.join(check)}")


if __name__ == "__main__":
    app()
