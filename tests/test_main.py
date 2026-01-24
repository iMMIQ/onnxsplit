"""Tests for main entry point and module exports."""

from pathlib import Path


class TestModuleExports:
    """Tests for public API exports from __init__.py."""

    def test_main_package_exports(self):
        """Test that main package exports are available."""
        # Check version is available
        from onnxsplit import __version__

        assert isinstance(__version__, str)
        assert __version__ == "0.1.0"

    def test_analyzer_exports(self):
        """Test that analyzer module exports are available."""
        from onnxsplit.analyzer import (  # noqa: F401
            DependencyEdge,
            DependencyGraph,
            DependencyNode,
            ModelAnalyzer,
            OperatorInfo,
            TensorMetadata,
            dtype_to_bytes,
        )

    def test_cli_exports(self):
        """Test that CLI module exports are available."""
        from onnxsplit.cli import CliOptions, app  # noqa: F401

    def test_config_exports(self):
        """Test that config module exports are available."""
        from onnxsplit.config import (  # noqa: F401
            AxisRule,
            ConfigError,
            ConfigMergeError,
            GlobalConfig,
            MemoryRule,
            OperatorConfig,
            SplitConfig,
            load_config,
            merge_cli_args,
        )

    def test_memory_exports(self):
        """Test that memory module exports are available."""
        from onnxsplit.memory import (  # noqa: F401
            AutoSplitAdjuster,
            MemoryEstimator,
            OperatorMemoryInfo,
            TensorMemoryInfo,
            dtype_to_bytes,
            estimate_tensor_memory,
        )

    def test_splitter_exports(self):
        """Test that splitter module exports are available."""
        from onnxsplit.splitter import (  # noqa: F401
            AxisAnalyzer,
            SplitableAxes,
            SplitPlan,
            SplitPlanner,
            SplitReport,
            get_splitable_axes_for_op,
        )

    def test_transform_exports(self):
        """Test that transform module exports are available."""
        from onnxsplit.transform import (  # noqa: F401
            GraphTransformer,
            ReconnectConnection,
            ReconnectPlan,
            ReconnectStrategy,
            clone_node,
            create_concat_node,
            create_slice_node,
            create_split_node,
            generate_reconnect_plan,
            generate_split_name,
            get_slice_initializers,
        )


class TestMainEntry:
    """Tests for __main__.py entry point."""

    def test_main_module_exists(self):
        """Test that __main__ module can be imported."""
        import onnxsplit.__main__  # noqa: F401

    def test_main_has_app(self):
        """Test that main module exposes the typer app."""
        from onnxsplit.__main__ import app  # noqa: F401

        assert app is not None

    def test_main_callable(self):
        """Test that main module can be called."""
        from onnxsplit import __main__ as main  # noqa: F401

        assert callable(getattr(main, "app", None))


class TestCliIntegration:
    """Tests for CLI integration."""

    def test_cli_app_exists(self):
        """Test that CLI app is available."""
        from onnxsplit.cli import app  # noqa: F401

        assert app is not None

    def test_cli_options_exists(self):
        """Test that CliOptions is available."""
        from onnxsplit.cli import CliOptions  # noqa: F401

        assert CliOptions is not None

    def test_cli_runner_functions_exist(self):
        """Test that CLI runner functions are available."""
        from onnxsplit.cli import run_analyze, run_split  # noqa: F401

        assert callable(run_split)
        assert callable(run_analyze)


class TestPackageStructure:
    """Tests for package structure."""

    def test_package_dir_exists(self):
        """Test that package directory exists."""
        import onnxsplit  # noqa: F401

        package_path = Path(onnxsplit.__file__).parent
        assert package_path.exists()
        assert package_path.is_dir()

    def test_all_expected_modules_exist(self):
        """Test that all expected modules exist."""
        import onnxsplit  # noqa: F401

        package_path = Path(onnxsplit.__file__).parent

        expected_modules = [
            "__main__.py",
            "analyzer",
            "cli",
            "config",
            "memory",
            "splitter",
            "transform",
            "utils",
        ]

        for module in expected_modules:
            if module.endswith(".py"):
                module_path = package_path / module
            else:
                module_path = package_path / module / "__init__.py"

            assert module_path.exists(), f"Module {module} does not exist"

    def test_version_is_string(self):
        """Test that version is a string."""
        from onnxsplit import __version__  # noqa: F401

        assert isinstance(__version__, str)
        assert len(__version__) > 0


class TestReexports:
    """Tests for re-exports in main package."""

    def test_dtype_to_bytes_reexport(self):
        """Test that dtype_to_bytes is re-exported from main package."""
        from onnxsplit import dtype_to_bytes  # noqa: F401

        assert callable(dtype_to_bytes)

    def test_estimate_tensor_memory_reexport(self):
        """Test that estimate_tensor_memory is re-exported from main package."""
        from onnxsplit import estimate_tensor_memory  # noqa: F401

        assert callable(estimate_tensor_memory)

    def test_autosplit_adjuster_reexport(self):
        """Test that AutoSplitAdjuster is re-exported from main package."""
        from onnxsplit import AutoSplitAdjuster  # noqa: F401

        assert AutoSplitAdjuster is not None
