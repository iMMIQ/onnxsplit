"""Tests for onnxsplit.utils.naming module."""

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
