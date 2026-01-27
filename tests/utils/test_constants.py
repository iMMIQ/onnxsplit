"""Tests for onnxsplit.utils.constants module."""

import pytest

# This test will fail before the constants module exists
def test_bytes_per_mb_constant():
    from onnxsplit.utils.constants import BYTES_PER_MB
    assert BYTES_PER_MB == 1024 * 1024

def test_verify_tolerance_constants():
    from onnxsplit.utils.constants import DEFAULT_VERIFY_RTOL, DEFAULT_VERIFY_ATOL
    assert DEFAULT_VERIFY_RTOL == 1e-4
    assert DEFAULT_VERIFY_ATOL == 1e-5
