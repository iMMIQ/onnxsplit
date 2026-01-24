"""Coverage tests for config module - covering edge cases and error paths."""

import tempfile
from pathlib import Path

import pytest

from onnxsplit.config.loader import ConfigError, load_config
from onnxsplit.config.merger import merge_cli_args, ConfigMergeError
from onnxsplit.config.schema import (
    GlobalConfig,
    OperatorConfig,
    AxisRule,
    MemoryRule,
    SplitConfig,
)


class TestConfigErrorIsException:
    """Test ConfigError is an Exception subclass."""

    def test_config_error_is_exception(self):
        """Test that ConfigError is a subclass of Exception."""
        assert issubclass(ConfigError, Exception)
        error = ConfigError("test error")
        assert isinstance(error, Exception)
        assert str(error) == "test error"


class TestConfigMergeErrorIsException:
    """Test ConfigMergeError is an Exception subclass."""

    def test_config_merge_error_is_exception(self):
        """Test that ConfigMergeError is a subclass of Exception."""
        assert issubclass(ConfigMergeError, Exception)
        error = ConfigMergeError("merge error")
        assert isinstance(error, Exception)
        assert str(error) == "merge error"


class TestLoadEmptyYaml:
    """Test loading empty or minimal YAML configs."""

    def test_load_empty_yaml(self):
        """Test that empty YAML file raises ConfigError (not a dict)."""
        yaml_content = ""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(ConfigError, match="root must be a dict"):
                load_config(path)
        finally:
            path.unlink()

    def test_load_yaml_only_global(self):
        """Test loading YAML with only global section."""
        yaml_content = """
        global:
          default_parts: 5
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        try:
            config = load_config(path)
            assert config.global_config.default_parts == 5
        finally:
            path.unlink()


class TestOperatorConfigWithoutParts:
    """Test operator config without required parts field."""

    def test_operator_config_missing_parts(self):
        """Test that missing parts field raises ConfigError."""
        yaml_content = """
        operators:
          "/model/Conv":
            axis: 0
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(ConfigError, match="missing required field 'parts'"):
                load_config(path)
        finally:
            path.unlink()

    def test_operator_config_not_dict(self):
        """Test that non-dict operator config raises ConfigError."""
        yaml_content = """
        operators:
          "/model/Conv": "invalid"
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(ConfigError, match="must be a dict"):
                load_config(path)
        finally:
            path.unlink()


class TestMemoryRulesWithLinearStrategy:
    """Test memory rules with linear_split overflow strategy."""

    def test_memory_rules_with_linear_strategy(self):
        """Test that linear_split overflow strategy is valid."""
        yaml_content = """
        memory_rules:
          auto_adjust: true
          overflow_strategy: linear_split
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        try:
            config = load_config(path)
            assert config.memory_rules.overflow_strategy == "linear_split"
        finally:
            path.unlink()


class TestInvalidOverflowStrategy:
    """Test invalid overflow strategy values."""

    def test_invalid_overflow_strategy(self):
        """Test that invalid overflow_strategy raises ConfigError."""
        yaml_content = """
        memory_rules:
          overflow_strategy: invalid_strategy
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(ConfigError, match="overflow_strategy.*must be"):
                load_config(path)
        finally:
            path.unlink()

    def test_overflow_strategy_not_string(self):
        """Test that non-string overflow_strategy raises ConfigError."""
        yaml_content = """
        memory_rules:
          overflow_strategy: 123
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(ConfigError, match="overflow_strategy.*must be a string"):
                load_config(path)
        finally:
            path.unlink()


class TestAxisRuleWithoutPreferAxis:
    """Test axis rule without prefer_axis field."""

    def test_axis_rule_without_prefer_axis(self):
        """Test axis_rule without prefer_axis defaults to None."""
        yaml_content = """
        axis_rules:
          - op_type: "Conv"
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        try:
            config = load_config(path)
            assert len(config.axis_rules) == 1
            assert config.axis_rules[0].op_type == "Conv"
            assert config.axis_rules[0].prefer_axis is None
        finally:
            path.unlink()

    def test_axis_rule_missing_op_type(self):
        """Test that missing op_type raises ConfigError."""
        yaml_content = """
        axis_rules:
          - prefer_axis: 0
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(ConfigError, match="missing required field 'op_type'"):
                load_config(path)
        finally:
            path.unlink()

    def test_axis_rule_not_dict(self):
        """Test that non-dict axis rule raises ConfigError."""
        yaml_content = """
        axis_rules:
          - "invalid"
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(ConfigError, match="must be a dict"):
                load_config(path)
        finally:
            path.unlink()

    def test_axis_rule_op_type_not_string(self):
        """Test that non-string op_type raises ConfigError."""
        yaml_content = """
        axis_rules:
          - op_type: 123
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(ConfigError, match="op_type.*must be a string"):
                load_config(path)
        finally:
            path.unlink()

    def test_axis_rule_prefer_axis_invalid_type(self):
        """Test that invalid prefer_axis type raises ConfigError."""
        yaml_content = """
        axis_rules:
          - op_type: "Conv"
            prefer_axis: [1, 2]
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(ConfigError, match="prefer_axis.*must be int, str, or None"):
                load_config(path)
        finally:
            path.unlink()


class TestMergeWithZeroCliParts:
    """Test merge with zero CLI parts."""

    def test_merge_with_zero_cli_parts(self):
        """Test that cli_parts=0 raises ConfigMergeError."""
        config = SplitConfig(global_config=GlobalConfig(default_parts=2))
        with pytest.raises(ConfigMergeError, match="cli_parts.*必须大于0"):
            merge_cli_args(config, cli_parts=0, cli_max_memory=None)


class TestMergeWithNegativeMaxMemory:
    """Test merge with negative max_memory."""

    def test_merge_with_negative_max_memory(self):
        """Test that negative cli_max_memory raises ConfigMergeError."""
        config = SplitConfig(global_config=GlobalConfig(default_parts=2))
        with pytest.raises(ConfigMergeError, match="cli_max_memory.*必须大于0"):
            merge_cli_args(config, cli_parts=None, cli_max_memory=-100)


class TestInvalidDefaultParts:
    """Test invalid default_parts values."""

    def test_default_parts_less_than_one(self):
        """Test that default_parts < 1 raises ConfigError."""
        yaml_content = """
        global:
          default_parts: 0
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(ConfigError, match="default_parts.*must be >= 1"):
                load_config(path)
        finally:
            path.unlink()

    def test_default_parts_negative(self):
        """Test that negative default_parts raises ConfigError."""
        yaml_content = """
        global:
          default_parts: -5
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(ConfigError, match="default_parts.*must be >= 1"):
                load_config(path)
        finally:
            path.unlink()


class TestInvalidMaxMemory:
    """Test invalid max_memory_mb values."""

    def test_max_memory_less_than_one(self):
        """Test that max_memory_mb < 1 raises ConfigError."""
        yaml_content = """
        global:
          max_memory_mb: 0
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(ConfigError, match="max_memory_mb.*must be >= 1"):
                load_config(path)
        finally:
            path.unlink()


class TestInvalidOperatorParts:
    """Test invalid operator parts values."""

    def test_operator_parts_less_than_one(self):
        """Test that parts < 1 raises ConfigError."""
        yaml_content = """
        operators:
          "/model/Conv":
            parts: 0
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(ConfigError, match="parts.*must be >= 1"):
                load_config(path)
        finally:
            path.unlink()


class TestInvalidAutoAdjust:
    """Test invalid auto_adjust values."""

    def test_auto_adjust_not_bool(self):
        """Test that non-bool auto_adjust raises ConfigError."""
        yaml_content = """
        memory_rules:
          auto_adjust: "true"
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(ConfigError, match="auto_adjust.*must be a boolean"):
                load_config(path)
        finally:
            path.unlink()


class TestLoadConfigNotDictRoot:
    """Test loading config with non-dict root."""

    def test_load_config_list_root(self):
        """Test that list root raises ConfigError."""
        yaml_content = """
        - item1
        - item2
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(ConfigError, match="root must be a dict"):
                load_config(path)
        finally:
            path.unlink()

    def test_load_config_string_root(self):
        """Test that string root raises ConfigError."""
        yaml_content = "just a string"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(ConfigError, match="root must be a dict"):
                load_config(path)
        finally:
            path.unlink()


class TestLoadConfigWithStringPath:
    """Test load_config accepts string path."""

    def test_load_config_with_string_path(self):
        """Test that load_config works with string path."""
        yaml_content = """
        global:
          default_parts: 3
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            string_path = f.name

        try:
            config = load_config(string_path)
            assert config.global_config.default_parts == 3
        finally:
            Path(string_path).unlink()


class TestLoadConfigPathExistsCheck:
    """Test path exists check coverage."""

    def test_load_config_nonexistent(self):
        """Test loading non-existent file raises ConfigError."""
        with pytest.raises(ConfigError, match="Config file not found"):
            load_config("/nonexistent/path/config.yaml")

    def test_load_config_directory_raises_oserror(self):
        """Test that loading a directory raises ConfigError with OSError message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Try to load a directory as a config file - should fail with OSError
            with pytest.raises(ConfigError, match="Failed to read file"):
                load_config(tmpdir)
