from __future__ import annotations

from .decision_engine import AcceptanceDecision, evaluate_quality
from .models import GeoCandidate, GeorefResult, ValidationResult, VisionResult
from .runtime import load_auto_georeference


def validate_candidate(
    candidate: GeoCandidate,
    *,
    src_path,
    vision: VisionResult,
    epsg: int,
) -> tuple[ValidationResult, GeorefResult]:
    """
    Run full georeferencing for *candidate* and return a scored ValidationResult.

    The accepted / provisional flags come directly from the georeferencer's
    LAST_GEOREF_QUALITY dict (hard geometric gates).  The confidence score is
    computed by :func:`decision_engine.evaluate_quality`, which considers NCC
    metrics, coordinate anchors, candidate evidence, and conflicts.
    """
    ag = load_auto_georeference()

    output_path = ag.georeference(
        src_path,
        vision.overview,
        epsg,
        auto_seed=candidate.to_seed_dict(scale_denominator=vision.scale),
    )
    quality = dict(ag.LAST_GEOREF_QUALITY or {})

    decision: AcceptanceDecision = evaluate_quality(
        quality,
        evidence=list(candidate.evidence),
        conflicts=list(candidate.conflicts),
    )

    notes = list(decision.notes)
    # Surface conflict labels as validation notes so they appear in case bundles.
    for conflict in candidate.conflicts:
        note = f"conflict:{conflict}"
        if note not in notes:
            notes.append(note)
    # Record evidence sources for traceability.
    if candidate.evidence:
        sources = ", ".join({e.source for e in candidate.evidence})
        notes.append(f"evidence_sources:{sources}")

    validation = ValidationResult(
        candidate_id=candidate.candidate_id,
        accepted=decision.accepted,
        provisional=decision.provisional,
        confidence=decision.confidence,
        acceptance_reason=decision.reason,
        metrics=quality,
        notes=notes,
    )
    result = GeorefResult(
        output_path=output_path,
        epsg=epsg,
        selected_candidate_id=candidate.candidate_id,
        quality=quality,
        validation=validation,
    )
    return validation, result


def choose_best_result(validations: list[ValidationResult]) -> ValidationResult | None:
    """
    Select the best ValidationResult from a list.

    Ranking priority (highest wins):
      1. Hard accepted and NOT provisional > accepted + provisional > rejected
      2. Within the same acceptance tier, higher confidence wins.
      3. Tiebreak: lower search index (first validated) wins.
    """
    if not validations:
        return None
    return max(
        validations,
        key=lambda v: (_rank_acceptance(v), v.confidence or 0.0),
    )


def _rank_acceptance(item: ValidationResult) -> tuple[int, int]:
    """Return a sortable tuple: (accepted_int, not_provisional_int)."""
    return (
        1 if item.accepted else 0,
        0 if item.provisional else 1,
    )


# ---------------------------------------------------------------------------
# Legacy shim — kept for any callers that still import this directly.
# Prefer using evaluate_quality() from decision_engine instead.
# ---------------------------------------------------------------------------
def _derive_confidence(quality: dict) -> float | None:
    """
    Backwards-compatible wrapper around :func:`evaluate_quality`.

    Returns a float in [0.10, 0.99] or None when quality is empty.
    """
    if not quality:
        return None
    return evaluate_quality(quality).confidence
