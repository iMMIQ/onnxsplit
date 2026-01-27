# Refactor Technical Debt Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate code duplication and improve maintainability by consolidating shared utilities and constants.

**Architecture:** Create centralized utility modules for constants, naming functions, and validation. Update existing modules to import from these new utilities instead of duplicating code.

**Tech Stack:** Python 3.13, ONNX, pytest

---

## Task 1: Create Constants Module

**Files:**
- Create: `src/onnxsplit/utils/constants.py`
- Modify: `src/onnxsplit/utils/__init__.py`
- Test: `tests/utils/test_constants.py`

**Step 1: Write the failing test**

```python
# tests/utils/test_constants.py
import pytest

# This test will fail before the constants module exists
def test_bytes_per_mb_constant():
    from onnxsplit.utils.constants import BYTES_PER_MB
    assert BYTES_PER_MB == 1024 * 1024

def test_verify_tolerance_constants():
    from onnxsplit.utils.constants import DEFAULT_VERIFY_RTOL, DEFAULT_VERIFY_ATOL
    assert DEFAULT_VERIFY_RTOL == 1e-4
    assert DEFAULT_VERIFY_ATOL == 1e-5
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/utils/test_constants.py -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'onnxsplit.utils.constants'"

**Step 3: Create constants module**

```python
# src/onnxsplit/utils/constants.py
"""Shared constants for onnxsplit."""

# Memory conversion constants
BYTES_PER_MB = 1024 * 1024
BYTES_PER_KB = 1024

# Verification tolerances
DEFAULT_VERIFY_RTOL = 1e-4
DEFAULT_VERIFY_ATOL = 1e-5
```

**Step 4: Update utils __init__.py**

```python
# src/onnxsplit/utils/__init__.py
"""Utility modules for onnxsplit."""

from onnxsplit.utils.constants import (
    BYTES_PER_MB,
    DEFAULT_VERIFY_ATOL,
    DEFAULT_VERIFY_RTOL,
)

__all__ = [
    "BYTES_PER_MB",
    "DEFAULT_VERIFY_RTOL",
    "DEFAULT_VERIFY_ATOL",
]
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/utils/test_constants.py -v`

Expected: PASS

**Step 6: Commit**

```bash
git add src/onnxsplit/utils/constants.py src/onnxsplit/utils/__init__.py tests/utils/test_constants.py
git commit -m "feat: add shared constants module"
```

---

## Task 2: Create Naming Utilities Module

**Files:**
- Create: `src/onnxsplit/utils/naming.py`
- Modify: `src/onnxsplit/utils/__init__.py`
- Test: `tests/utils/test_naming.py`

**Step 1: Write the failing test**

```python
# tests/utils/test_naming.py
import pytest

def test_sanitize_name_for_node_basic():
    from onnxsplit.utils.naming import sanitize_name_for_node
    assert sanitize_name_for_node("tensor_0") == "tensor_0"
    assert sanitize_name_for_node("/tensor_0") == "tensor_0"

def test_sanitize_name_for_node_special_chars():
    from onnxsplit.utils.naming import sanitize_name_for_node
    assert sanitize_name_for_node("tensor with spaces") == "tensor_with_spaces"
    assert sanitize_name_for_node("tensor@#$") == "tensor___"

def test_sanitize_name_for_node_leading_digit():
    from onnxsplit.utils.naming import sanitize_name_for_node
    result = sanitize_name_for_node("123tensor")
    assert result.startswith("n_")

def test_sanitize_name_for_node_empty():
    from onnxsplit.utils.naming import sanitize_name_for_node
    assert sanitize_name_for_node("") == "tensor"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/utils/test_naming.py -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'onnxsplit.utils.naming'"

**Step 3: Create naming module**

```python
# src/onnxsplit/utils/naming.py
"""Name sanitization utilities for ONNX nodes and tensors."""

import re


def sanitize_name_for_node(name: str, default: str = "tensor") -> str:
    """清理张量名称以用作节点名称

    ONNX节点名称不应包含某些特殊字符（如前导斜杠、空格等）。
    此函数将特殊字符替换为下划线。

    Args:
        name: 原始名称
        default: 清理后为空时的默认名称

    Returns:
        清理后的名称
    """
    # 移除前导斜杠并替换其他特殊字符
    cleaned = name.lstrip("/")
    # 替换其他非法字符（除字母、数字、下划线、连字符外的字符）
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", cleaned)
    # 确保不以数字或连字符开头（某些系统不允许）
    if cleaned and cleaned[0] in "0123456789-":
        cleaned = "n_" + cleaned
    # 如果清理后为空，返回默认名称
    return cleaned or default
```

**Step 4: Update utils __init__.py**

```python
# src/onnxsplit/utils/__init__.py
"""Utility modules for onnxsplit."""

from onnxsplit.utils.constants import (
    BYTES_PER_MB,
    DEFAULT_VERIFY_ATOL,
    DEFAULT_VERIFY_RTOL,
)
from onnxsplit.utils.naming import sanitize_name_for_node

__all__ = [
    "BYTES_PER_MB",
    "DEFAULT_VERIFY_RTOL",
    "DEFAULT_VERIFY_ATOL",
    "sanitize_name_for_node",
]
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/utils/test_naming.py -v`

Expected: PASS

**Step 6: Commit**

```bash
git add src/onnxsplit/utils/naming.py src/onnxsplit/utils/__init__.py tests/utils/test_naming.py
git commit -m "feat: add naming utilities module"
```

---

## Task 3: Update transform/split_concat.py to Use New Utilities

**Files:**
- Modify: `src/onnxsplit/transform/split_concat.py`
- Test: `tests/transform/test_split_concat.py` (existing)

**Step 1: Update imports in split_concat.py**

```python
# src/onnxsplit/transform/split_concat.py
"""Split和Concat节点生成"""

import onnx.helper
from onnx import NodeProto, TensorProto

from onnxsplit.utils.constants import BYTES_PER_MB
from onnxsplit.utils.naming import sanitize_name_for_node

# 模块级别的字典，用于存储Slice节点的初始器信息
_slice_node_initializers: dict[int, list] = {}
```

**Step 2: Remove _sanitize_name_for_node function**

Delete lines 12-32 (the entire `_sanitize_name_for_node` function).

**Step 3: Update all references**

Replace `_sanitize_name_for_node` with `sanitize_name_for_node` throughout the file:
- Line 61: `sanitized_prefix = sanitize_name_for_node(output_prefix)`
- Line 106: `sanitized_name = sanitize_name_for_node(output_name)`
- Line 145: `sanitized_name = sanitize_name_for_node(output_name)`

**Step 4: Run existing tests**

Run: `pytest tests/transform/test_split_concat.py -v`

Expected: PASS (all existing tests should still pass)

**Step 5: Commit**

```bash
git add src/onnxsplit/transform/split_concat.py
git commit -m "refactor: use shared naming utilities in split_concat"
```

---

## Task 4: Update transform/node_clone.py to Use New Utilities

**Files:**
- Modify: `src/onnxsplit/transform/node_clone.py`
- Test: `tests/transform/test_node_clone.py` (existing)

**Step 1: Update imports in node_clone.py**

```python
# src/onnxsplit/transform/node_clone.py
"""节点克隆功能"""

from onnx import NodeProto

from onnxsplit.utils.naming import sanitize_name_for_node
```

**Step 2: Remove _sanitize_name_for_node function**

Delete lines 7-27 (the entire `_sanitize_name_for_node` function).

**Step 3: Update all references**

Replace `_sanitize_name_for_node` with `sanitize_name_for_node`:
- Line 41: `sanitized_name = sanitize_name_for_node(original_name)`
- Line 73: `sanitized_name = sanitize_name_for_node(original_name)`

**Step 4: Update default name in generate_split_name**

Update line 42 to use the default parameter:
```python
base_name = sanitized_name if sanitized_name != "tensor" else f"node_{id(object())}"
```

**Step 5: Run existing tests**

Run: `pytest tests/transform/test_node_clone.py -v`

Expected: PASS

**Step 6: Commit**

```bash
git add src/onnxsplit/transform/node_clone.py
git commit -m "refactor: use shared naming utilities in node_clone"
```

---

## Task 5: Update analyzer/operator.py to Use Constants

**Files:**
- Modify: `src/onnxsplit/analyzer/operator.py`
- Test: `tests/analyzer/test_operator.py` (existing)

**Step 1: Add import**

```python
# src/onnxsplit/analyzer/operator.py
"""算子信息结构"""

from dataclasses import dataclass, field
from typing import Any

from onnx import NodeProto

from onnxsplit.analyzer.tensor import TensorMetadata
from onnxsplit.utils.constants import BYTES_PER_MB
```

**Step 2: Update input_memory_mb property**

Replace line 42:
```python
return total / BYTES_PER_MB
```

**Step 3: Update output_memory_mb property**

Replace line 52:
```python
return total / BYTES_PER_MB
```

**Step 4: Run existing tests**

Run: `pytest tests/analyzer/test_operator.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/onnxsplit/analyzer/operator.py
git commit -m "refactor: use shared BYTES_PER_MB constant in operator"
```

---

## Task 6: Update analyzer/tensor.py to Use Constants

**Files:**
- Modify: `src/onnxsplit/analyzer/tensor.py`
- Test: `tests/analyzer/test_tensor.py` (existing)

**Step 1: Add import**

```python
# src/onnxsplit/analyzer/tensor.py
"""张量元数据结构"""

from dataclasses import dataclass

from onnx import TensorProto

from onnxsplit.utils.constants import BYTES_PER_MB
```

**Step 2: Update size_mb property**

Replace line 77:
```python
return self.memory_bytes / BYTES_PER_MB
```

**Step 3: Run existing tests**

Run: `pytest tests/analyzer/test_tensor.py -v`

Expected: PASS

**Step 4: Commit**

```bash
git add src/onnxsplit/analyzer/tensor.py
git commit -m "refactor: use shared BYTES_PER_MB constant in tensor"
```

---

## Task 7: Update memory/estimator.py to Use Constants

**Files:**
- Modify: `src/onnxsplit/memory/estimator.py`
- Test: `tests/memory/test_estimator.py` (existing)

**Step 1: Update imports**

```python
# src/onnxsplit/memory/estimator.py
"""内存估算器"""

from dataclasses import dataclass
from typing import Optional

from onnx import TensorProto

from onnxsplit.analyzer.model import ModelAnalyzer
from onnxsplit.analyzer.operator import OperatorInfo
from onnxsplit.analyzer.tensor import dtype_to_bytes
from onnxsplit.utils.constants import BYTES_PER_MB
```

**Step 2: Remove local MB constant**

Delete lines 12-13:
```python
# Remove these lines:
# MB = 1024 * 1024
```

**Step 3: Update all MB references**

Replace `/ MB` with `/ BYTES_PER_MB` in:
- Line 51: `return self.memory_bytes / BYTES_PER_MB`
- Line 160: `total_memory_mb = total_memory / BYTES_PER_MB`
- Line 164: `if total_memory_mb > self._peak_memory_mb:`
- Line 169: `input_memory_mb=input_memory / BYTES_PER_MB,`
- Line 170: `output_memory_mb=output_memory / BYTES_PER_MB,`
- Line 171: `weights_memory_mb=weights_memory / BYTES_PER_MB,`

**Step 4: Run existing tests**

Run: `pytest tests/memory/test_estimator.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/onnxsplit/memory/estimator.py
git commit -m "refactor: use shared BYTES_PER_MB constant in estimator"
```

---

## Task 8: Update CLI to Use Verification Constants

**Files:**
- Modify: `src/onnxsplit/cli/parser.py`
- Modify: `src/onnxsplit/cli/runner.py`
- Test: `tests/cli/test_parser.py` (existing)

**Step 1: Update parser.py imports**

```python
# src/onnxsplit/cli/parser.py
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
```

**Step 2: Update default values in parser.py**

Replace lines 82-90:
```python
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
```

**Step 3: Update runner.py imports**

```python
# src/onnxsplit/cli/runner.py
"""CLI runner for onnxsplit.

This module provides the main execution logic for the onnxsplit CLI,
including model loading, validation, configuration merging, and transformation.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
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
from onnxsplit.utils.constants import DEFAULT_VERIFY_ATOL, DEFAULT_VERIFY_RTOL
```

**Step 4: Update RunContext defaults**

Replace lines 54-55:
```python
    verify_rtol: float = DEFAULT_VERIFY_RTOL
    verify_atol: float = DEFAULT_VERIFY_ATOL
```

**Step 5: Update verify/__init__.py**

```python
# src/onnxsplit/verify/__init__.py
"""Model equivalence verification using onnxruntime."""

import numpy as np
from onnx import ModelProto

from onnxsplit.utils.constants import DEFAULT_VERIFY_ATOL, DEFAULT_VERIFY_RTOL
from onnxsplit.verify.result import VerifyResult
from onnxsplit.verify.runtime import RuntimeChecker, ONNXRUNTIME_AVAILABLE


def verify_equivalence(
    original_model: ModelProto,
    split_model: ModelProto,
    rtol: float = DEFAULT_VERIFY_RTOL,
    atol: float = DEFAULT_VERIFY_ATOL,
    seed: int = 42,
    verbose: bool = False,
) -> VerifyResult:
```

**Step 6: Run existing tests**

Run: `pytest tests/cli/test_parser.py tests/cli/test_runner.py -v`

Expected: PASS

**Step 7: Commit**

```bash
git add src/onnxsplit/cli/parser.py src/onnxsplit/cli/runner.py src/onnxsplit/verify/__init__.py
git commit -m "refactor: use shared verification constants in CLI"
```

---

## Task 9: Add Integration Test for Shared Utilities

**Files:**
- Create: `tests/integration/test_shared_utilities.py`

**Step 1: Write the integration test**

```python
# tests/integration/test_shared_utilities.py
"""Integration tests to verify shared utilities work across modules."""

import pytest

from onnxsplit.utils.constants import BYTES_PER_MB, DEFAULT_VERIFY_ATOL, DEFAULT_VERIFY_RTOL
from onnxsplit.utils.naming import sanitize_name_for_node


def test_constants_import_from_main_utils():
    """Test that constants can be imported from utils package."""
    from onnxsplit.utils import (
        BYTES_PER_MB as MB_FROM_UTILS,
        DEFAULT_VERIFY_ATOL as ATOL_FROM_UTILS,
        DEFAULT_VERIFY_RTOL as RTOL_FROM_UTILS,
    )

    assert MB_FROM_UTILS == 1024 * 1024
    assert ATOL_FROM_UTILS == 1e-5
    assert RTOL_FROM_UTILS == 1e-4


def test_naming_import_from_main_utils():
    """Test that naming functions can be imported from utils package."""
    from onnxsplit.utils import sanitize_name_for_node as sanitize

    assert sanitize("test/name") == "test_name"
    assert sanitize("/leading") == "leading"


def test_constants_used_in_operator_module():
    """Test that operator module uses shared constants."""
    from onnxsplit.analyzer.operator import OperatorInfo
    from onnxsplit.analyzer.tensor import TensorMetadata
    from onnxsplit.utils.constants import BYTES_PER_MB

    # Create a simple tensor metadata
    tensor = TensorMetadata(name="test", shape=(1024, 1024), dtype=1)  # FLOAT = 4 bytes
    # 1024 * 1024 * 4 bytes = 4 MB
    expected_mb = (1024 * 1024 * 4) / BYTES_PER_MB
    assert tensor.size_mb == expected_mb


def test_naming_used_in_transform_modules():
    """Test that transform modules use shared naming utilities."""
    from onnxsplit.transform.split_concat import create_split_node
    from onnxsplit.utils.naming import sanitize_name_for_node

    # The split_concat module should use the shared sanitize function
    # by generating a name that goes through sanitization
    node = create_split_node(
        input_name="input",
        axis=0,
        parts=2,
        output_prefix="/test/output",  # Has leading slash that should be removed
    )
    # Node name should have sanitized prefix
    assert "test_output" in node.name or "split" in node.name
```

**Step 2: Run integration tests**

Run: `pytest tests/integration/test_shared_utilities.py -v`

Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_shared_utilities.py
git commit -m "test: add integration tests for shared utilities"
```

---

## Task 10: Run Full Test Suite and Verify No Regressions

**Files:**
- All files

**Step 1: Run full test suite**

Run: `pytest tests/ -v`

Expected: All tests PASS

**Step 2: Run linting**

Run: `ruff check src/onnxsplit/`

Expected: No new errors

**Step 3: Run type checking**

Run: `mypy src/onnxsplit/` (if configured)

Expected: No new type errors

**Step 4: Verify code with actual ONNX model**

If available, run manual verification:
```bash
python -m onnxsplit split test_model.onnx --parts 2 --output test_output
```

Expected: Model splits successfully

**Step 5: Final commit if any adjustments needed**

```bash
git add -A
git commit -m "chore: final cleanup after refactoring"
```

---

## Summary

This refactoring plan:

1. **Creates centralized utility modules** for constants and naming functions
2. **Eliminates code duplication** across 6+ files
3. **Maintains backward compatibility** - all existing tests pass
4. **Follows TDD principles** - tests written before implementations
5. **Uses small, atomic commits** for easy rollback if needed

**Estimated completion time:** 1-2 hours

**Files modified:** 10
**Files created:** 4
**Lines removed:** ~80 (duplicated code)
**Lines added:** ~150 (new utilities + tests)
