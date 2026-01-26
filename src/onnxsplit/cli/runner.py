"""CLI runner for onnxsplit.

This module provides the main execution logic for the onnxsplit CLI,
including model loading, validation, configuration merging, and transformation.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import onnx
import typer
from onnx import ModelProto

from onnxsplit.analyzer.model import ModelAnalyzer
from onnxsplit.config import ConfigError, SplitConfig, load_config, merge_cli_args
from onnxsplit.config.merger import ConfigMergeError
from onnxsplit.memory.auto_adjust import AutoSplitAdjuster
from onnxsplit.memory.estimator import MemoryEstimator
from onnxsplit.splitter.plan import SplitReport
from onnxsplit.splitter.planner import SplitPlanner
from onnxsplit.transform.executor import GraphTransformer


@dataclass
class RunContext:
    """Context for running the split operation.

    Attributes:
        model_path: Path to the ONNX model file
        output_dir: Directory for output files
        config_path: Path to configuration file (optional)
        cli_parts: Number of parts from CLI argument (optional)
        cli_max_memory: Max memory from CLI argument in MB (optional)
        verbose: Enable verbose output
        quiet: Suppress non-error output
        verify: Verify split model equivalence using onnxruntime
        simplify: Simplify model with onnxsim after splitting
    """

    model_path: str
    output_dir: str = "output"
    config_path: Optional[str] = None
    cli_parts: Optional[int] = None
    cli_max_memory: Optional[int] = None
    verbose: bool = False
    quiet: bool = False
    verify: bool = False
    simplify: bool = True


@dataclass
class RunResult:
    """Result of running the split operation.

    Attributes:
        success: Whether the operation completed successfully
        output_path: Path to the saved model
        report_path: Path to the JSON report
        error: Error message if failed
    """

    success: bool
    output_path: Optional[str] = None
    report_path: Optional[str] = None
    error: Optional[str] = None

    @classmethod
    def ok(cls, output_path: str, report_path: str) -> "RunResult":
        """Create a successful result."""
        return cls(success=True, output_path=output_path, report_path=report_path)

    @classmethod
    def fail(cls, message: str) -> "RunResult":
        """Create a failed result."""
        return cls(success=False, error=message)


def _load_model(model_path: str) -> ModelProto:
    """Load ONNX model from file.

    Args:
        model_path: Path to the ONNX model file

    Returns:
        Loaded ONNX model

    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If model file is invalid
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        model = onnx.load(str(path))
        return model
    except Exception as e:
        raise ValueError(f"Failed to load ONNX model: {e}") from e


def _validate_model(model: ModelProto) -> bool:
    """Validate ONNX model.

    Args:
        model: ONNX model to validate

    Returns:
        True if valid

    Raises:
        ValueError: If model is invalid
    """
    try:
        onnx.checker.check_model(model)
        return True
    except Exception as e:
        raise ValueError(f"Model validation failed: {e}") from e


def _load_config(config_path: Optional[str]) -> Optional[SplitConfig]:
    """Load configuration from file.

    Args:
        config_path: Path to configuration file, or None

    Returns:
        Configuration object, or None if no config file

    Raises:
        FileNotFoundError: If config file doesn't exist
        ConfigError: If config is invalid
    """
    if config_path is None:
        return None

    return load_config(config_path)


def _prepare_config(
    file_config: Optional[SplitConfig],
    cli_parts: Optional[int],
    cli_max_memory: Optional[int],
) -> SplitConfig:
    """Prepare final configuration by merging file config with CLI args.

    Args:
        file_config: Configuration from file, or None
        cli_parts: CLI parts argument
        cli_max_memory: CLI max memory argument

    Returns:
        Final configuration
    """
    base_config = file_config if file_config is not None else SplitConfig()

    try:
        return merge_cli_args(base_config, cli_parts, cli_max_memory)
    except ConfigMergeError as e:
        raise ValueError(f"Configuration merge failed: {e}") from e


def _generate_report(report: SplitReport, output_path: str) -> None:
    """Generate and save JSON report.

    Args:
        report: Split report to save
        output_path: Path to save the report
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert report to dict
    report_dict = {
        "original_operators": report.original_operators,
        "split_operators": report.split_operators,
        "unsplit_operators": report.unsplit_operators,
        "total_parts": report.total_parts,
        "split_ratio": report.split_ratio,
        "max_parts": report.max_parts,
        "plans": [
            {
                "operator_name": plan.operator_name,
                "parts": plan.parts,
                "axis": plan.axis,
                "reason": plan.reason,
            }
            for plan in report.plans
        ],
    }

    with open(path, "w") as f:
        json.dump(report_dict, f, indent=2)


def run_split(ctx: RunContext) -> RunResult:
    """Run the split operation.

    This is the main entry point for the split functionality. It:
    1. Loads the ONNX model
    2. Validates the model
    3. Loads and merges configuration
    4. Generates split plan
    5. Applies transformations
    6. Saves the result

    Args:
        ctx: Run context with all parameters

    Returns:
        RunResult indicating success or failure
    """
    try:
        # Step 1: Load model
        if ctx.verbose:
            typer.echo(f"Loading model: {ctx.model_path}")

        model = _load_model(ctx.model_path)

        if ctx.verbose:
            typer.echo("Model loaded successfully")

        # Step 2: Validate model
        if ctx.verbose:
            typer.echo("Validating model...")

        _validate_model(model)

        if ctx.verbose:
            typer.echo("Model validation passed")

        # Step 3: Load and merge configuration
        if ctx.config_path and ctx.verbose:
            typer.echo(f"Loading configuration: {ctx.config_path}")

        file_config = _load_config(ctx.config_path)
        config = _prepare_config(file_config, ctx.cli_parts, ctx.cli_max_memory)

        if ctx.verbose:
            typer.echo(
                f"Configuration: default_parts={config.global_config.default_parts}, "
                f"max_memory_mb={config.global_config.max_memory_mb}"
            )

        # Step 4: Analyze model and generate split plan
        if ctx.verbose:
            typer.echo("Analyzing model...")

        analyzer = ModelAnalyzer.from_model_proto(model)
        planner = SplitPlanner(analyzer, config)
        report = planner.generate()

        # 输出规划器收集的警告
        planner_warnings = planner.get_warnings()
        if planner_warnings and not ctx.quiet:
            for warning in planner_warnings:
                typer.echo(
                    typer.style(f"  ⚠ {warning}", fg=typer.colors.YELLOW),
                    err=True,
                )

        if ctx.verbose:
            typer.echo(f"Analysis complete: {report.summary()}")

        # Step 5: Apply memory adjustments if configured or if max_memory specified via CLI
        # If max_memory_mb is set (from config or CLI), apply memory adjustment automatically
        should_adjust = (
            config.memory_rules and config.memory_rules.auto_adjust
        ) or config.global_config.max_memory_mb is not None

        if should_adjust:
            if ctx.verbose:
                typer.echo("Applying memory adjustments...")

            estimator = MemoryEstimator(analyzer)
            adjuster = AutoSplitAdjuster(
                estimator,
                max_parts=256,
                warn_threshold=64,
            )

            max_memory = config.global_config.max_memory_mb
            adjusted_plans = adjuster.adjust_report(report.plans, max_memory)

            # Update report with adjusted plans
            report = SplitReport(
                original_operators=report.original_operators,
                split_operators=report.split_operators,
                unsplit_operators=report.unsplit_operators,
                plans=adjusted_plans,
            )

            if ctx.verbose:
                typer.echo("Memory adjustments applied")

        # Step 6: Apply transformations
        if ctx.verbose:
            typer.echo("Applying transformations...")

        # Apply split plans sequentially
        transformed_model = model
        for plan in report.plans:
            if plan.is_split:
                if ctx.verbose:
                    typer.echo(
                        f"  Splitting {plan.operator_name}: {plan.parts} parts on axis {plan.axis}"
                    )
                # Re-create analyzer with transformed model for sequential splits
                current_analyzer = ModelAnalyzer.from_model_proto(transformed_model)
                current_transformer = GraphTransformer(current_analyzer)
                transformed_model = current_transformer.apply_split_plan(plan)

        # Step 7: Save output
        output_dir = Path(ctx.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "split_model.onnx"
        report_path = output_dir / "split_report.json"

        if ctx.verbose:
            typer.echo(f"Saving model to: {output_path}")

        onnx.save(transformed_model, str(output_path))

        # Simplify model with onnxsim if enabled
        simplified_model = None  # Track if simplification succeeded
        if ctx.simplify:
            if ctx.verbose:
                typer.echo("Simplifying model with onnxsim...")

            try:
                import onnxsim

                simplified_model, check_ok = onnxsim.simplify(
                    transformed_model,
                    perform_optimization=True,
                    check_n=False,
                )

                if not check_ok:
                    raise ValueError("onnxsim validation failed: model has correctness issues")

                # Save simplified model
                onnx.save(simplified_model, str(output_path))

                if ctx.verbose:
                    typer.echo("Model simplified successfully")

            except ImportError:
                typer.echo(
                    typer.style(
                        "  ⚠ onnxsim not available, skipping simplification",
                        fg=typer.colors.YELLOW,
                    ),
                    err=True,
                )
            except Exception as e:
                error_msg = f"Model simplification failed: {e}"
                typer.echo(error_msg, err=True)
                return RunResult.fail(error_msg)

        # Generate and save report
        if ctx.verbose:
            typer.echo(f"Saving report to: {report_path}")

        _generate_report(report, str(report_path))

        # Verify model equivalence if requested
        verify_result = None
        if ctx.verify:
            if ctx.verbose:
                typer.echo("Verifying model equivalence...")

            from onnxsplit.verify import verify_equivalence

            # Use the actual output model (simplified if successful, otherwise transformed)
            model_to_verify = simplified_model if simplified_model is not None else transformed_model

            verify_result = verify_equivalence(
                original_model=model,
                split_model=model_to_verify,
                rtol=1e-4,
                atol=1e-5,
                verbose=ctx.verbose,
            )

            if verify_result.skipped:
                typer.echo(
                    typer.style(f"  ⚠ Verification skipped: {verify_result.skip_reason}", fg=typer.colors.YELLOW)
                )
            elif verify_result.success:
                typer.echo(
                    typer.style(f"  ✓ Verification passed: {verify_result.outputs_compared} outputs match", fg=typer.colors.GREEN)
                )
                if ctx.verbose and verify_result.max_diff > 0:
                    typer.echo(f"    Max difference: {verify_result.max_diff:.2e}")
            else:
                typer.echo(
                    typer.style(f"  ✗ Verification failed: outputs differ", fg=typer.colors.RED)
                )
                if ctx.verbose:
                    typer.echo(f"    Max difference: {verify_result.max_diff:.2e}")

        if not ctx.quiet:
            typer.echo("Model split successfully!")
            typer.echo(f"  Output: {output_path}")
            typer.echo(f"  Report: {report_path}")
            typer.echo(f"  Summary: {report.summary()}")

        return RunResult.ok(str(output_path), str(report_path))

    except FileNotFoundError as e:
        error_msg = f"File not found: {e}"
        typer.echo(error_msg, err=True)
        return RunResult.fail(error_msg)

    except (ValueError, ConfigError, ConfigMergeError) as e:
        error_msg = f"Configuration error: {e}"
        typer.echo(error_msg, err=True)
        return RunResult.fail(error_msg)

    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        typer.echo(error_msg, err=True)
        return RunResult.fail(error_msg)


def run_analyze(ctx: RunContext) -> RunResult:
    """Run model analysis.

    Args:
        ctx: Run context with parameters

    Returns:
        RunResult indicating success or failure
    """
    try:
        if ctx.verbose:
            typer.echo(f"Loading model: {ctx.model_path}")

        model = _load_model(ctx.model_path)
        analyzer = ModelAnalyzer.from_model_proto(model)

        if ctx.verbose:
            typer.echo("Analyzing model...")

        inputs = analyzer.get_inputs()
        outputs = analyzer.get_outputs()
        operators = analyzer.get_operators()

        if not ctx.quiet:
            typer.echo("Model Analysis:")
            typer.echo(f"  IR Version: {analyzer.ir_version}")
            typer.echo(f"  Opset Version: {analyzer.opset_version}")
            typer.echo(f"  Producer: {analyzer.producer_name}")
            typer.echo(f"  Graph: {analyzer.graph_name}")
            typer.echo(f"  Inputs: {len(inputs)}")
            for inp in inputs:
                typer.echo(f"    - {inp.name}: {inp.shape}")
            typer.echo(f"  Outputs: {len(outputs)}")
            for out in outputs:
                typer.echo(f"    - {out.name}: {out.shape}")
            typer.echo(f"  Operators: {len(operators)}")
            for op in operators:
                typer.echo(f"    - {op.name} ({op.op_type})")

        # Save analysis report
        output_dir = Path(ctx.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "analysis_report.json"

        analysis = {
            "ir_version": analyzer.ir_version,
            "opset_version": analyzer.opset_version,
            "producer_name": analyzer.producer_name,
            "producer_version": analyzer.producer_version,
            "graph_name": analyzer.graph_name,
            "inputs": [
                {"name": inp.name, "shape": list(inp.shape), "dtype": inp.dtype} for inp in inputs
            ],
            "outputs": [
                {"name": out.name, "shape": list(out.shape), "dtype": out.dtype} for out in outputs
            ],
            "operators": [
                {
                    "name": op.name,
                    "op_type": op.op_type,
                    "inputs": list(op.input_names),
                    "outputs": list(op.output_names),
                }
                for op in operators
            ],
        }

        with open(report_path, "w") as f:
            json.dump(analysis, f, indent=2)

        if not ctx.quiet:
            typer.echo(f"  Report saved to: {report_path}")

        return RunResult.ok(str(report_path), str(report_path))

    except Exception as e:
        error_msg = f"Analysis failed: {e}"
        typer.echo(error_msg, err=True)
        return RunResult.fail(error_msg)


def run_validate(ctx: RunContext) -> RunResult:
    """Run model validation.

    Args:
        ctx: Run context with parameters

    Returns:
        RunResult indicating success or failure
    """
    try:
        if ctx.verbose:
            typer.echo(f"Loading model: {ctx.model_path}")

        model = _load_model(ctx.model_path)

        if ctx.verbose:
            typer.echo("Validating model...")

        _validate_model(model)

        if not ctx.quiet:
            typer.echo("Model validation passed!")

        return RunResult.ok("", "")

    except Exception as e:
        error_msg = f"Validation failed: {e}"
        typer.echo(error_msg, err=True)
        return RunResult.fail(error_msg)
