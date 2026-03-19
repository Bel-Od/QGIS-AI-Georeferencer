from __future__ import annotations

from .models import ReviewDecision, ValidationResult


def review_status_from_validation(validation: ValidationResult | None) -> ReviewDecision:
    if validation is None:
        return ReviewDecision(status="review_required", reason="no_validation_result")
    if validation.accepted and not validation.provisional:
        return ReviewDecision(status="accepted", reason=validation.acceptance_reason)
    if validation.accepted and validation.provisional:
        return ReviewDecision(status="review_required", reason="provisional_result")
    return ReviewDecision(status="review_required", reason=validation.acceptance_reason or "rejected")
