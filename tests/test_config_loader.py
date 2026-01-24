"""Tests for config loader module."""

import tempfile
from pathlib import Path

import pytest

from onnxsplit.config.loader import ConfigError, load_config
from onnxsplit.config.schema import (
    AxisRule,
    GlobalConfig,
    MemoryRule,
    OperatorConfig,
    SplitConfig,
)


class TestLoadValidConfig:
    """Test loading a valid config file."""

    def test_load_valid_config(self, fixtures_dir: Path) -> None:
        """Test loading valid config with all fields."""
        config = load_config(fixtures_dir / "configs" / "valid_config.yaml")

        assert isinstance(config, SplitConfig)
        assert config.global_config.default_parts == 2
        assert config.global_config.max_memory_mb == 512

        # Check operator configs
        assert "/model/Conv_0" in config.operators
        assert config.operators["/model/Conv_0"].parts == 4
        assert config.operators["/model/Conv_0"].axis == 0

        assert "/model/MatMul_*" in config.operators
        assert config.operators["/model/MatMul_*"].parts == 2

        assert "/model/LayerNorm_*" in config.operators
        assert config.operators["/model/LayerNorm_*"].parts == 1

        # Check axis rules
        assert len(config.axis_rules) == 3
        conv_rule = [r for r in config.axis_rules if r.op_type == "Conv"][0]
        assert conv_rule.prefer_axis == 0

        matmul_rule = [r for r in config.axis_rules if r.op_type == "MatMul"][0]
        assert matmul_rule.prefer_axis == "batch"

        layernorm_rule = [r for r in config.axis_rules if r.op_type == "LayerNorm"][0]
        assert layernorm_rule.prefer_axis is None

        # Check memory rules
        assert config.memory_rules.auto_adjust is True
        assert config.memory_rules.overflow_strategy == "binary_split"


class TestLoadMinimalConfig:
    """Test loading a minimal config file."""

    def test_load_minimal_config(self, fixtures_dir: Path) -> None:
        """Test loading minimal config with only required fields."""
        config = load_config(fixtures_dir / "configs" / "minimal_config.yaml")

        assert isinstance(config, SplitConfig)
        assert config.global_config.default_parts == 1
        assert config.operators == {}
        assert config.axis_rules == []
        # When memory_rules is not specified, it should have default MemoryRule
        assert config.memory_rules is not None
        assert config.memory_rules.auto_adjust is False
        assert config.memory_rules.overflow_strategy is None


class TestLoadErrors:
    """Test error cases for config loading."""

    def test_load_nonexistent_file(self) -> None:
        """Test loading a file that doesn't exist."""
        with pytest.raises(ConfigError, match="Config file not found"):
            load_config(Path("/nonexistent/config.yaml"))

    def test_load_invalid_yaml_syntax(self) -> None:
        """Test loading a file with invalid YAML syntax."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content:\n  - broken\n")
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(ConfigError, match="Failed to parse YAML"):
                load_config(path)
        finally:
            path.unlink()

    def test_load_invalid_config_values(self, fixtures_dir: Path) -> None:
        """Test loading a config with invalid values."""
        with pytest.raises(ConfigError, match="default_parts.*must be an integer"):
            load_config(fixtures_dir / "configs" / "invalid_config.yaml")


class TestOperatorConfigWildcard:
    """Test operator config with wildcard patterns."""

    def test_operator_config_wildcard(self, fixtures_dir: Path) -> None:
        """Test that wildcard patterns are preserved in operator configs."""
        config = load_config(fixtures_dir / "configs" / "valid_config.yaml")

        # Wildcard patterns should be stored as-is
        assert "/model/MatMul_*" in config.operators
        assert "/model/LayerNorm_*" in config.operators


class TestAxisRulesNullAxis:
    """Test axis rules with null (None) axis preference."""

    def test_axis_rules_null_axis(self, fixtures_dir: Path) -> None:
        """Test that null axis is properly loaded as None."""
        config = load_config(fixtures_dir / "configs" / "valid_config.yaml")

        layernorm_rule = [r for r in config.axis_rules if r.op_type == "LayerNorm"][0]
        assert layernorm_rule.prefer_axis is None


class TestMemoryRulesOptional:
    """Test memory rules are optional with defaults."""

    def test_memory_rules_optional(self, fixtures_dir: Path) -> None:
        """Test that memory rules have default values when not specified."""
        config = load_config(fixtures_dir / "configs" / "minimal_config.yaml")

        # memory_rules should have a default MemoryRule instance
        assert config.memory_rules is not None
        assert config.memory_rules.auto_adjust is False
        assert config.memory_rules.overflow_strategy is None


class TestLoadConfigWithComments:
    """Test loading config with comments."""

    def test_load_config_with_comments(self) -> None:
        """Test that comments are ignored during loading."""
        yaml_content = """
        # This is a comment
        global:
          default_parts: 3  # inline comment
          # max_memory_mb: 1024

        operators:
          # Comment before operator
          "/model/Conv":  # inline after
            parts: 2
            # axis: 1
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        try:
            config = load_config(path)

            assert config.global_config.default_parts == 3
            assert config.global_config.max_memory_mb is None  # default value
            assert config.operators["/model/Conv"].parts == 2
            assert config.operators["/model/Conv"].axis is None  # not specified
        finally:
            path.unlink()


class TestLoadConfigFromString:
    """Test loading config from string content."""

    def test_load_config_from_string(self) -> None:
        """Test loading config directly from YAML string."""
        yaml_content = """
        global:
          default_parts: 4
          max_memory_mb: 2048

        operators:
          "/model/Op":
            parts: 2
            axis: 1
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        try:
            config = load_config(path)

            assert config.global_config.default_parts == 4
            assert config.global_config.max_memory_mb == 2048
            assert config.operators["/model/Op"].parts == 2
            assert config.operators["/model/Op"].axis == 1
        finally:
            path.unlink()


# Fixtures
@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to fixtures directory."""
    return Path(__file__).parent / "fixtures"
