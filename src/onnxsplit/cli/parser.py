"""CLI parser for onnxsplit."""

from dataclasses import dataclass, field
from typing import Optional

import typer

from onnxsplit.utils.constants import DEFAULT_VERIFY_ATOL, DEFAULT_VERIFY_RTOL

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
    parts: int = typer.Option(
        1,
        "--parts",
        "-p",
        help="Number of parts to split the model into.",
    ),
    max_memory: Optional[float] = typer.Option(
        None,
        "--max-memory",
        "-m",
        help="Max memory per split (MB). Supports decimal values for finer control.",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for generated files.",
    ),
    verify: bool = typer.Option(
        False,
        "--verify",
        help="Verify split model produces same outputs as original using onnxruntime.",
    ),
    verify_rtol: float = typer.Option(
        DEFAULT_VERIFY_RTOL,
        "--verify-rtol",
        help=f"Relative tolerance for verification (default: {DEFAULT_VERIFY_RTOL}).",
    ),
    verify_atol: float = typer.Option(
        DEFAULT_VERIFY_ATOL,
        "--verify-atol",
        help=f"Absolute tolerance for verification (default: {DEFAULT_VERIFY_ATOL}).",
    ),
    no_simplify: bool = typer.Option(
        False,
        "--no-simplify",
        help="Skip model simplification with onnxsim after splitting.",
    ),
    skip_validation: bool = typer.Option(
        False,
        "--skip-validation",
        help="Skip ONNX model validation before processing.",
    ),
) -> None:
    """Split an ONNX model into smaller components.

    This command partitions a large ONNX model into multiple smaller models
    based on the specified splitting strategy.
    """
    from onnxsplit.cli.parser import CliOptions
    from onnxsplit.cli.runner import RunContext, run_split

    options: CliOptions = ctx.obj

    run_ctx = RunContext(
        model_path=model_path,
        output_dir=output or options.output or "output",
        config_path=None,
        cli_parts=parts,
        cli_max_memory=max_memory,
        verbose=options.verbose,
        quiet=options.quiet,
        verify=verify,
        simplify=not no_simplify,
        skip_validation=skip_validation,
        verify_rtol=verify_rtol,
        verify_atol=verify_atol,
    )

    result = run_split(run_ctx)
    if not result.success:
        raise typer.Exit(1)


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
        help="Output directory for analysis report.",
    ),
) -> None:
    """Analyze an ONNX model and report structure information.

    This command examines the model's computational graph and provides
    detailed information about its structure, operations, and size.
    """
    from onnxsplit.cli.runner import RunContext, run_analyze

    options: CliOptions = ctx.obj

    run_ctx = RunContext(
        model_path=model_path,
        output_dir=output or options.output or "output",
        verbose=options.verbose,
        quiet=options.quiet,
    )

    result = run_analyze(run_ctx)
    if not result.success:
        raise typer.Exit(1)


@app.command()
def validate(
    ctx: typer.Context,
    model_path: str = typer.Argument(
        ...,
        help="Path to the ONNX model file to validate.",
        exists=True,
    ),
) -> None:
    """Validate an ONNX model for correctness and compatibility.

    This command performs various validation checks on the model to ensure
    it meets ONNX specification and can be properly loaded and executed.
    """
    from onnxsplit.cli.runner import RunContext, run_validate

    options: CliOptions = ctx.obj

    run_ctx = RunContext(
        model_path=model_path,
        verbose=options.verbose,
        quiet=options.quiet,
    )

    result = run_validate(run_ctx)
    if not result.success:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
