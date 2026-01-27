"""Verification result dataclass."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class VerifyResult:
    """Result of model equivalence verification.

    Attributes:
        success: Whether verification passed (outputs match within tolerance)
        skipped: Whether verification was skipped (e.g., onnxruntime not available)
        skip_reason: Reason for skipping, if skipped
        outputs_compared: Number of outputs compared
        max_diff: Maximum absolute difference found across all outputs
        failure_reason: Specific reason for failure (e.g., "shape mismatch", "output name mismatch", "nan detected")
        details: Additional details about the verification
    """

    success: bool = False
    skipped: bool = False
    skip_reason: Optional[str] = None
    outputs_compared: int = 0
    max_diff: float = 0.0
    failure_reason: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def skipped_result(cls, reason: str) -> "VerifyResult":
        """Create a skipped result."""
        return cls(skipped=True, skip_reason=reason)

    @classmethod
    def passed_result(cls, outputs_compared: int, max_diff: float = 0.0) -> "VerifyResult":
        """Create a passed result."""
        return cls(success=True, outputs_compared=outputs_compared, max_diff=max_diff)

    @classmethod
    def failed_result(
        cls,
        outputs_compared: int,
        max_diff: float,
        failure_reason: Optional[str] = None,
    ) -> "VerifyResult":
        """Create a failed result."""
        return cls(
            success=False,
            outputs_compared=outputs_compared,
            max_diff=max_diff,
            failure_reason=failure_reason,
        )
