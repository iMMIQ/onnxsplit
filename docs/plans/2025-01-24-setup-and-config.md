# Plan 1: 项目基础设置和配置管理

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 建立项目基础结构，实现YAML配置文件的加载、验证和合并逻辑。

**Architecture:**
- 使用 `pyproject.toml` 管理依赖和项目元数据
- 配置模块使用 `dataclasses` 定义配置结构，使用 `PyYAML` 加载
- 配置优先级：算子实例 > 算子类型 > 全局配置 > 命令行参数

**Tech Stack:** Python 3.13+, PyYAML, dataclasses, pytest

---

## Task 1: 项目目录结构创建

**Files:**
- Create: `src/onnxsplit/__init__.py`
- Create: `src/onnxsplit/cli/__init__.py`
- Create: `src/onnxsplit/config/__init__.py`
- Create: `src/onnxsplit/analyzer/__init__.py`
- Create: `src/onnxsplit/splitter/__init__.py`
- Create: `src/onnxsplit/memory/__init__.py`
- Create: `src/onnxsplit/transform/__init__.py`
- Create: `src/onnxsplit/utils/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/fixtures/__init__.py`
- Create: `tests/fixtures/models/.gitkeep`

**Step 1: 创建所有必要的目录和 `__init__.py` 文件**

```bash
mkdir -p src/onnxsplit/{cli,config,analyzer,splitter,memory,transform,utils}
mkdir -p tests/fixtures/models
touch -p src/onnxsplit/__init__.py
touch -p src/onnxsplit/cli/__init__.py
touch -p src/onnxsplit/config/__init__.py
touch -p src/onnxsplit/analyzer/__init__.py
touch -p src/onnxsplit/splitter/__init__.py
touch -p src/onnxsplit/memory/__init__.py
touch -p src/onnxsplit/transform/__init__.py
touch -p src/onnxsplit/utils/__init__.py
touch -p tests/__init__.py
touch -p tests/fixtures/__init__.py
touch -p tests/fixtures/models/.gitkeep
```

**Step 2: 验证目录结构创建成功**

Run: `tree src/onnxsplit tests -I __pycache__`
Expected: 显示完整的目录树结构

**Step 3: 提交**

```bash
git add src/onnxsplit tests
git commit -m "chore: create project directory structure"
```

---

## Task 2: 更新 pyproject.toml 依赖

**Files:**
- Modify: `pyproject.toml`

**Step 1: 添加完整的依赖配置**

编辑 `pyproject.toml`:

```toml
[project]
name = "onnxsplit"
version = "0.1.0"
description = "ONNX model operator splitting tool for memory optimization"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "onnx>=1.20.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pyyaml>=6.0.0",
    "black>=24.0.0",
    "ruff>=0.1.0",
]

[project.scripts]
onnxsplit = "onnxsplit.__main__:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py313"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W"]
ignore = ["E501"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

**Step 2: 安装依赖**

Run: `uv sync --all-extras`
Expected: 依赖安装成功，无错误

**Step 3: 验证 pytest 可用**

Run: `uv run pytest --version`
Expected: 显示 pytest 版本号

**Step 4: 提交**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: update project dependencies"
```

---

## Task 3: 配置数据结构定义

**Files:**
- Create: `src/onnxsplit/config/schema.py`
- Test: `tests/test_config_schema.py`

**Step 1: 编写配置数据结构的测试**

创建 `tests/test_config_schema.py`:

```python
"""测试配置数据结构"""
from dataclasses import dataclass
from onnxsplit.config.schema import (
    GlobalConfig,
    OperatorConfig,
    AxisRule,
    MemoryRule,
    SplitConfig,
)


def test_global_config_default_values():
    """测试全局配置的默认值"""
    config = GlobalConfig()
    assert config.default_parts == 1
    assert config.max_memory_mb is None


def test_global_config_with_values():
    """测试创建带值的全局配置"""
    config = GlobalConfig(default_parts=4, max_memory_mb=512)
    assert config.default_parts == 4
    assert config.max_memory_mb == 512


def test_operator_config_creation():
    """测试算子配置创建"""
    config = OperatorConfig(parts=2, axis=0)
    assert config.parts == 2
    assert config.axis == 0


def test_operator_config_without_axis():
    """测试不带axis的算子配置"""
    config = OperatorConfig(parts=2)
    assert config.parts == 2
    assert config.axis is None


def test_axis_rule_creation():
    """测试切分轴规则创建"""
    rule = AxisRule(op_type="Conv", prefer_axis=0)
    assert rule.op_type == "Conv"
    assert rule.prefer_axis == 0


def test_axis_rule_with_null_axis():
    """测试不可切分轴规则"""
    rule = AxisRule(op_type="LayerNorm", prefer_axis=None)
    assert rule.op_type == "LayerNorm"
    assert rule.prefer_axis is None


def test_axis_rule_with_string_axis():
    """测试字符串形式的轴规则"""
    rule = AxisRule(op_type="MatMul", prefer_axis="batch")
    assert rule.op_type == "MatMul"
    assert rule.prefer_axis == "batch"


def test_memory_rule_creation():
    """测试内存规则创建"""
    rule = MemoryRule(auto_adjust=True, overflow_strategy="binary_split")
    assert rule.auto_adjust is True
    assert rule.overflow_strategy == "binary_split"


def test_memory_rule_default_values():
    """测试内存规则默认值"""
    rule = MemoryRule()
    assert rule.auto_adjust is False
    assert rule.overflow_strategy is None


def test_split_config_creation():
    """测试完整配置创建"""
    config = SplitConfig(
        global_config=GlobalConfig(default_parts=2),
        operators={"/model/Conv": OperatorConfig(parts=4)},
        axis_rules=[AxisRule(op_type="Conv", prefer_axis=0)],
        memory_rules=MemoryRule(auto_adjust=True)
    )
    assert config.global_config.default_parts == 2
    assert config.operators["/model/Conv"].parts == 4
    assert len(config.axis_rules) == 1
    assert config.memory_rules.auto_adjust is True


def test_split_config_empty_operators():
    """测试空算子配置"""
    config = SplitConfig(
        global_config=GlobalConfig(),
        operators={},
        axis_rules=[],
        memory_rules=None
    )
    assert config.operators == {}
    assert config.axis_rules == []
    assert config.memory_rules is None
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_config_schema.py -v`
Expected: FAIL - `ModuleNotFoundError: onnxsplit.config.schema`

**Step 3: 实现配置数据结构**

创建 `src/onnxsplit/config/schema.py`:

```python
"""配置数据结构定义"""
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class GlobalConfig:
    """全局配置"""
    default_parts: int = 1
    max_memory_mb: Optional[int] = None


@dataclass
class OperatorConfig:
    """算子级别的配置"""
    parts: int
    axis: Optional[int] = None


@dataclass
class AxisRule:
    """切分轴规则"""
    op_type: str
    prefer_axis: Optional[int | str] = None


@dataclass
class MemoryRule:
    """内存限制规则"""
    auto_adjust: bool = False
    overflow_strategy: Optional[Literal["binary_split", "linear_split"]] = None


@dataclass
class SplitConfig:
    """完整的切分配置"""
    global_config: GlobalConfig = field(default_factory=GlobalConfig)
    operators: dict[str, OperatorConfig] = field(default_factory=dict)
    axis_rules: list[AxisRule] = field(default_factory=list)
    memory_rules: Optional[MemoryRule] = None
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_config_schema.py -v`
Expected: PASS - 所有10个测试通过

**Step 5: 提交**

```bash
git add src/onnxsplit/config/schema.py tests/test_config_schema.py
git commit -m "feat: add config schema data structures"
```

---

## Task 4: YAML配置加载器

**Files:**
- Create: `src/onnxsplit/config/loader.py`
- Test: `tests/test_config_loader.py`
- Create: `tests/fixtures/configs/valid_config.yaml`
- Create: `tests/fixtures/configs/minimal_config.yaml`
- Create: `tests/fixtures/configs/invalid_config.yaml`

**Step 1: 创建测试配置文件**

创建 `tests/fixtures/configs/valid_config.yaml`:

```yaml
# 完整的测试配置
global:
  default_parts: 2
  max_memory_mb: 512

operators:
  "/model/Conv_0":
    parts: 4
    axis: 0

  "/model/MatMul_*":
    parts: 2

  "/model/LayerNorm_*":
    parts: 1

axis_rules:
  - op_type: "Conv"
    prefer_axis: 0

  - op_type: "MatMul"
    prefer_axis: "batch"

  - op_type: "LayerNorm"
    prefer_axis: null

memory_rules:
  auto_adjust: true
  overflow_strategy: "binary_split"
```

创建 `tests/fixtures/configs/minimal_config.yaml`:

```yaml
# 最小配置
global:
  default_parts: 1

operators: {}
```

创建 `tests/fixtures/configs/invalid_config.yaml`:

```yaml
# 无效配置 - parts 应该是整数
global:
  default_parts: "invalid"

operators:
  "/model/Conv_0":
    parts: "not_a_number"
```

创建测试目录:

```bash
mkdir -p tests/fixtures/configs
```

**Step 2: 编写配置加载器测试**

创建 `tests/test_config_loader.py`:

```python
"""测试配置加载器"""
import pytest
from pathlib import Path
from onnxsplit.config.loader import load_config, ConfigError
from onnxsplit.config.schema import (
    GlobalConfig,
    OperatorConfig,
    AxisRule,
    MemoryRule,
    SplitConfig,
)


def test_load_valid_config():
    """测试加载有效配置"""
    config_path = Path("tests/fixtures/configs/valid_config.yaml")
    config = load_config(config_path)

    assert isinstance(config, SplitConfig)
    assert config.global_config.default_parts == 2
    assert config.global_config.max_memory_mb == 512
    assert "/model/Conv_0" in config.operators
    assert config.operators["/model/Conv_0"].parts == 4
    assert config.operators["/model/Conv_0"].axis == 0
    assert "/model/MatMul_*" in config.operators
    assert len(config.axis_rules) == 3
    assert config.axis_rules[0].op_type == "Conv"
    assert config.memory_rules.auto_adjust is True
    assert config.memory_rules.overflow_strategy == "binary_split"


def test_load_minimal_config():
    """测试加载最小配置"""
    config_path = Path("tests/fixtures/configs/minimal_config.yaml")
    config = load_config(config_path)

    assert config.global_config.default_parts == 1
    assert config.global_config.max_memory_mb is None
    assert config.operators == {}
    assert config.axis_rules == []
    assert config.memory_rules is None


def test_load_nonexistent_file():
    """测试加载不存在的文件"""
    with pytest.raises(ConfigError, match="Config file not found"):
        load_config(Path("nonexistent.yaml"))


def test_load_invalid_yaml_syntax():
    """测试加载语法错误的YAML"""
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("invalid: yaml: content:\n  - broken\n")
        f.flush()
        temp_path = Path(f.name)

    try:
        with pytest.raises(ConfigError, match="YAML syntax error"):
            load_config(temp_path)
    finally:
        temp_path.unlink()


def test_load_invalid_config_values():
    """测试加载值类型错误的配置"""
    config_path = Path("tests/fixtures/configs/invalid_config.yaml")

    with pytest.raises(ConfigError, match="Invalid config value"):
        load_config(config_path)


def test_operator_config_wildcard():
    """测试通配符算子配置"""
    config_path = Path("tests/fixtures/configs/valid_config.yaml")
    config = load_config(config_path)

    assert "/model/MatMul_*" in config.operators
    assert "/model/LayerNorm_*" in config.operators


def test_axis_rules_null_axis():
    """测试null轴规则"""
    config_path = Path("tests/fixtures/configs/valid_config.yaml")
    config = load_config(config_path)

    layer_norm_rule = [r for r in config.axis_rules if r.op_type == "LayerNorm"][0]
    assert layer_norm_rule.prefer_axis is None


def test_memory_rules_optional():
    """测试内存规则可选"""
    config_path = Path("tests/fixtures/configs/minimal_config.yaml")
    config = load_config(config_path)

    assert config.memory_rules is None


def test_load_config_from_string():
    """测试从字符串加载配置"""
    yaml_content = """
global:
  default_parts: 3

operators:
  "/test/Op":
    parts: 5
    axis: 1
"""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        temp_path = Path(f.name)

    try:
        config = load_config(temp_path)
        assert config.global_config.default_parts == 3
        assert config.operators["/test/Op"].parts == 5
        assert config.operators["/test/Op"].axis == 1
    finally:
        temp_path.unlink()
```

**Step 3: 运行测试验证失败**

Run: `uv run pytest tests/test_config_loader.py -v`
Expected: FAIL - `ModuleNotFoundError: onnxsplit.config.loader`

**Step 4: 实现配置加载器**

创建 `src/onnxsplit/config/loader.py`:

```python
"""YAML配置文件加载器"""
from pathlib import Path
from typing import Any
import yaml

from onnxsplit.config.schema import (
    GlobalConfig,
    OperatorConfig,
    AxisRule,
    MemoryRule,
    SplitConfig,
)


class ConfigError(Exception):
    """配置错误"""


def _validate_int(value: Any, field_name: str) -> int:
    """验证值为整数"""
    if isinstance(value, int):
        return value
    raise ConfigError(f"Invalid config value: {field_name} must be an integer, got {type(value).__name__}")


def _load_global_config(data: dict[str, Any]) -> GlobalConfig:
    """加载全局配置"""
    if "global" not in data:
        return GlobalConfig()

    global_data = data["global"]
    default_parts = 1
    max_memory_mb = None

    if "default_parts" in global_data:
        default_parts = _validate_int(global_data["default_parts"], "global.default_parts")

    if "max_memory_mb" in global_data:
        max_memory_mb = _validate_int(global_data["max_memory_mb"], "global.max_memory_mb")

    return GlobalConfig(default_parts=default_parts, max_memory_mb=max_memory_mb)


def _load_operator_configs(data: dict[str, Any]) -> dict[str, OperatorConfig]:
    """加载算子配置"""
    operators = {}

    if "operators" not in data:
        return operators

    for name, op_data in data["operators"].items():
        if not isinstance(op_data, dict):
            raise ConfigError(f"Invalid operator config for {name}: must be a dict")

        parts = _validate_int(op_data.get("parts", 1), f"operators.{name}.parts")
        axis = op_data.get("axis")

        if axis is not None:
            axis = _validate_int(axis, f"operators.{name}.axis")

        operators[name] = OperatorConfig(parts=parts, axis=axis)

    return operators


def _load_axis_rules(data: dict[str, Any]) -> list[AxisRule]:
    """加载切分轴规则"""
    rules = []

    if "axis_rules" not in data:
        return rules

    for rule_data in data["axis_rules"]:
        if not isinstance(rule_data, dict):
            raise ConfigError("Invalid axis_rule: must be a dict")

        op_type = rule_data.get("op_type")
        if not isinstance(op_type, str):
            raise ConfigError("axis_rule.op_type must be a string")

        prefer_axis = rule_data.get("prefer_axis")
        if prefer_axis is not None and not isinstance(prefer_axis, (int, str)):
            raise ConfigError("axis_rule.prefer_axis must be int, str, or null")

        rules.append(AxisRule(op_type=op_type, prefer_axis=prefer_axis))

    return rules


def _load_memory_rules(data: dict[str, Any]) -> MemoryRule | None:
    """加载内存规则"""
    if "memory_rules" not in data:
        return None

    rule_data = data["memory_rules"]
    if not isinstance(rule_data, dict):
        raise ConfigError("Invalid memory_rules: must be a dict")

    auto_adjust = rule_data.get("auto_adjust", False)
    if not isinstance(auto_adjust, bool):
        raise ConfigError("memory_rules.auto_adjust must be a boolean")

    overflow_strategy = rule_data.get("overflow_strategy")
    if overflow_strategy is not None and overflow_strategy not in ("binary_split", "linear_split"):
        raise ConfigError(f"Invalid overflow_strategy: {overflow_strategy}")

    return MemoryRule(auto_adjust=auto_adjust, overflow_strategy=overflow_strategy)


def load_config(config_path: Path) -> SplitConfig:
    """从YAML文件加载配置

    Args:
        config_path: 配置文件路径

    Returns:
        SplitConfig: 解析后的配置对象

    Raises:
        ConfigError: 配置文件不存在或格式错误
    """
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML syntax error: {e}") from e
    except Exception as e:
        raise ConfigError(f"Error reading config file: {e}") from e

    if not isinstance(data, dict):
        raise ConfigError("Config file must contain a YAML dict")

    global_config = _load_global_config(data)
    operators = _load_operator_configs(data)
    axis_rules = _load_axis_rules(data)
    memory_rules = _load_memory_rules(data)

    return SplitConfig(
        global_config=global_config,
        operators=operators,
        axis_rules=axis_rules,
        memory_rules=memory_rules,
    )
```

**Step 5: 运行测试验证通过**

Run: `uv run pytest tests/test_config_loader.py -v`
Expected: PASS - 所有10个测试通过

**Step 6: 提交**

```bash
git add src/onnxsplit/config/loader.py tests/test_config_loader.py tests/fixtures/configs
git commit -m "feat: add YAML config loader"
```

---

## Task 5: 配置合并逻辑（命令行参数与配置文件）

**Files:**
- Create: `src/onnxsplit/config/merger.py`
- Test: `tests/test_config_merger.py`

**Step 1: 编写配置合并测试**

创建 `tests/test_config_merger.py`:

```python
"""测试配置合并逻辑"""
from onnxsplit.config.schema import (
    GlobalConfig,
    OperatorConfig,
    SplitConfig,
)
from onnxsplit.config.merger import merge_cli_args, ConfigMergeError


def test_merge_cli_parts_only():
    """测试只合并命令行parts参数"""
    config = SplitConfig(global_config=GlobalConfig(default_parts=1))
    merged = merge_cli_args(config, cli_parts=4, cli_max_memory=None)

    assert merged.global_config.default_parts == 4


def test_merge_cli_max_memory_only():
    """测试只合并命令行max_memory参数"""
    config = SplitConfig(global_config=GlobalConfig())
    merged = merge_cli_args(config, cli_parts=None, cli_max_memory=512)

    assert merged.global_config.max_memory_mb == 512


def test_merge_both_cli_args():
    """测试合并所有命令行参数"""
    config = SplitConfig(global_config=GlobalConfig(default_parts=1))
    merged = merge_cli_args(config, cli_parts=8, cli_max_memory=256)

    assert merged.global_config.default_parts == 8
    assert merged.global_config.max_memory_mb == 256


def test_cli_parts_lower_than_config():
    """测试命令行parts小于配置文件时，配置文件优先"""
    config = SplitConfig(
        global_config=GlobalConfig(default_parts=4),
        operators={"/model/Conv": OperatorConfig(parts=6)}
    )
    merged = merge_cli_args(config, cli_parts=2, cli_max_memory=None)

    # 命令行作为默认值，不影响已配置的算子
    assert merged.global_config.default_parts == 2
    # 已配置的算子保持不变
    assert merged.operators["/model/Conv"].parts == 6


def test_merge_creates_new_config():
    """测试合并返回新配置对象，不修改原配置"""
    config = SplitConfig(global_config=GlobalConfig(default_parts=1))
    merged = merge_cli_args(config, cli_parts=4, cli_max_memory=None)

    # 原配置不变
    assert config.global_config.default_parts == 1
    # 新配置有新值
    assert merged.global_config.default_parts == 4


def test_merge_with_no_cli_args():
    """测试无命令行参数时返回原配置"""
    config = SplitConfig(global_config=GlobalConfig(default_parts=3))
    merged = merge_cli_args(config, cli_parts=None, cli_max_memory=None)

    assert merged is config
    assert merged.global_config.default_parts == 3


def test_invalid_cli_parts():
    """测试无效的cli_parts值"""
    config = SplitConfig(global_config=GlobalConfig())

    with pytest.raises(ConfigMergeError, match="cli_parts must be positive"):
        merge_cli_args(config, cli_parts=0, cli_max_memory=None)


def test_invalid_cli_max_memory():
    """测试无效的cli_max_memory值"""
    config = SplitConfig(global_config=GlobalConfig())

    with pytest.raises(ConfigMergeError, match="cli_max_memory must be positive"):
        merge_cli_args(config, cli_parts=None, cli_max_memory=-100)


def test_merge_preserves_operators():
    """测试合并保留所有算子配置"""
    config = SplitConfig(
        global_config=GlobalConfig(default_parts=1),
        operators={
            "/model/A": OperatorConfig(parts=2),
            "/model/B": OperatorConfig(parts=3),
        }
    )
    merged = merge_cli_args(config, cli_parts=5, cli_max_memory=100)

    assert len(merged.operators) == 2
    assert merged.operators["/model/A"].parts == 2
    assert merged.operators["/model/B"].parts == 3
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_config_merger.py -v`
Expected: FAIL - `ModuleNotFoundError: onnxsplit.config.merger`

**Step 3: 实现配置合并逻辑**

创建 `src/onnxsplit/config/merger.py`:

```python
"""配置合并逻辑 - 合并命令行参数和配置文件"""
from onnxsplit.config.schema import SplitConfig, GlobalConfig


class ConfigMergeError(Exception):
    """配置合并错误"""


def merge_cli_args(
    config: SplitConfig,
    cli_parts: int | None,
    cli_max_memory: int | None
) -> SplitConfig:
    """将命令行参数合并到配置中

    命令行参数作为默认值，优先级低于配置文件中的具体配置。
    例如：配置文件中指定某算子parts=6，则不受cli_parts影响。

    Args:
        config: 从配置文件加载的配置
        cli_parts: 命令行指定的切分数
        cli_max_memory: 命令行指定的内存限制

    Returns:
        SplitConfig: 合并后的新配置对象

    Raises:
        ConfigMergeError: 参数值无效
    """
    if cli_parts is None and cli_max_memory is None:
        return config

    # 验证参数
    if cli_parts is not None and cli_parts <= 0:
        raise ConfigMergeError(f"cli_parts must be positive integer, got {cli_parts}")

    if cli_max_memory is not None and cli_max_memory <= 0:
        raise ConfigMergeError(f"cli_max_memory must be positive integer, got {cli_max_memory}")

    # 创建新的全局配置
    new_global = GlobalConfig(
        default_parts=cli_parts if cli_parts is not None else config.global_config.default_parts,
        max_memory_mb=cli_max_memory if cli_max_memory is not None else config.global_config.max_memory_mb
    )

    # 返回新配置对象（保留原配置的算子配置等）
    return SplitConfig(
        global_config=new_global,
        operators=dict(config.operators),
        axis_rules=list(config.axis_rules),
        memory_rules=config.memory_rules,
    )
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_config_loader.py tests/test_config_merger.py tests/test_config_schema.py -v`
Expected: PASS - 所有配置相关测试通过

**Step 5: 提交**

```bash
git add src/onnxsplit/config/merger.py tests/test_config_merger.py
git commit -m "feat: add config merge logic for CLI args"
```

---

## Task 6: 配置模块导出

**Files:**
- Modify: `src/onnxsplit/config/__init__.py`

**Step 1: 导出配置模块的公共接口**

编辑 `src/onnxsplit/config/__init__.py`:

```python
"""配置管理模块

提供YAML配置文件的加载、验证和合并功能。
"""

from onnxsplit.config.schema import (
    GlobalConfig,
    OperatorConfig,
    AxisRule,
    MemoryRule,
    SplitConfig,
)
from onnxsplit.config.loader import load_config, ConfigError
from onnxsplit.config.merger import merge_cli_args, ConfigMergeError


__all__ = [
    # Schema
    "GlobalConfig",
    "OperatorConfig",
    "AxisRule",
    "MemoryRule",
    "SplitConfig",
    # Loader
    "load_config",
    "ConfigError",
    # Merger
    "merge_cli_args",
    "ConfigMergeError",
]
```

**Step 2: 验证模块导入**

Run: `uv run python -c "from onnxsplit.config import load_config, SplitConfig; print('Import successful')"`
Expected: 打印 "Import successful"

**Step 3: 运行所有配置模块测试**

Run: `uv run pytest tests/test_config_*.py -v`
Expected: PASS - 所有测试通过

**Step 4: 提交**

```bash
git add src/onnxsplit/config/__init__.py
git commit -m "chore: export config module public API"
```

---

## Task 7: 配置模块覆盖率验证

**Files:**
- Test: `tests/test_config_coverage.py`

**Step 1: 编写覆盖率补充测试**

创建 `tests/test_config_coverage.py`:

```python
"""配置模块覆盖率补充测试"""
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile
from onnxsplit.config import load_config, ConfigError, merge_cli_args, ConfigMergeError


def test_config_error_is_exception():
    """测试ConfigError是Exception子类"""
    assert issubclass(ConfigError, Exception)


def test_config_merge_error_is_exception():
    """测试ConfigMergeError是Exception子类"""
    assert issubclass(ConfigMergeError, Exception)


def test_load_empty_yaml():
    """测试加载空YAML文件"""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("{}")
        f.flush()
        temp_path = Path(f.name)

    try:
        config = load_config(temp_path)
        assert config.global_config.default_parts == 1  # 默认值
        assert config.operators == {}
    finally:
        temp_path.unlink()


def test_load_config_with_comments():
    """测试加载带注释的YAML"""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
# This is a comment
global:
  default_parts: 5  # inline comment
""")
        f.flush()
        temp_path = Path(f.name)

    try:
        config = load_config(temp_path)
        assert config.global_config.default_parts == 5
    finally:
        temp_path.unlink()


def test_operator_config_without_parts():
    """测试算子配置中缺少parts字段"""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
operators:
  "/test/Op":
    axis: 1
""")
        f.flush()
        temp_path = Path(f.name)

    try:
        # parts默认为1
        config = load_config(temp_path)
        assert config.operators["/test/Op"].parts == 1
        assert config.operators["/test/Op"].axis == 1
    finally:
        temp_path.unlink()


def test_memory_rules_with_linear_strategy():
    """测试linear_split策略"""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
memory_rules:
  auto_adjust: true
  overflow_strategy: "linear_split"
""")
        f.flush()
        temp_path = Path(f.name)

    try:
        config = load_config(temp_path)
        assert config.memory_rules.overflow_strategy == "linear_split"
    finally:
        temp_path.unlink()


def test_invalid_overflow_strategy():
    """测试无效的overflow_strategy"""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
memory_rules:
  overflow_strategy: "invalid_strategy"
""")
        f.flush()
        temp_path = Path(f.name)

    try:
        with pytest.raises(ConfigError, match="Invalid overflow_strategy"):
            load_config(temp_path)
    finally:
        temp_path.unlink()


def test_axis_rule_without_prefer_axis():
    """测试不带prefer_axis的轴规则"""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
axis_rules:
  - op_type: "Conv"
""")
        f.flush()
        temp_path = Path(f.name)

    try:
        config = load_config(temp_path)
        assert config.axis_rules[0].op_type == "Conv"
        assert config.axis_rules[0].prefer_axis is None
    finally:
        temp_path.unlink()


def test_merge_with_zero_cli_parts():
    """测试cli_parts=0时报错"""
    from onnxsplit.config import SplitConfig, GlobalConfig
    config = SplitConfig(global_config=GlobalConfig())

    with pytest.raises(ConfigMergeError):
        merge_cli_args(config, cli_parts=0, cli_max_memory=None)


def test_merge_with_negative_max_memory():
    """测试cli_max_memory为负数时报错"""
    from onnxsplit.config import SplitConfig, GlobalConfig
    config = SplitConfig(global_config=GlobalConfig())

    with pytest.raises(ConfigMergeError):
        merge_cli_args(config, cli_parts=None, cli_max_memory=-1)
```

**Step 2: 运行所有配置测试并检查覆盖率**

Run: `uv run pytest tests/test_config_*.py -v --cov=onnxsplit/config --cov-report=term-missing`
Expected: PASS - 覆盖率 >= 90%

**Step 3: 提交**

```bash
git add tests/test_config_coverage.py
git commit -m "test: add coverage tests for config module"
```

---

## 完成检查

**Step 1: 运行所有测试**

Run: `uv run pytest tests/ -v`
Expected: PASS - 所有测试通过

**Step 2: 检查代码风格**

Run: `uv run ruff check src/onnxsplit/config tests/`
Expected: 无错误

**Step 3: 验证模块导入**

Run: `uv run python -c "from onnxsplit.config import *; print('All exports available')"`
Expected: 打印 "All exports available"

**Step 4: 最终提交**

```bash
git add .
git commit -m "chore: finalize plan 1 - project setup and config module"
```

---

**Plan 1 完成！** 配置管理模块已实现，包括：
- 项目目录结构
- 依赖配置
- 配置数据结构定义
- YAML配置加载
- 命令行参数合并
- 完整的测试覆盖
