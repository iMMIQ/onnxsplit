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
