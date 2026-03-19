"""
decision_engine.py
------------------
Centralised acceptance / confidence logic for the georeferencing pipeline.

This module computes a richer confidence score from the raw quality dict
returned by auto_georeference.georeference() (via LAST_GEOREF_QUALITY),
combined with candidate evidence and conflicts.

It NEVER overrides the accepted / provisional flags set by the georeferencer
— those come from hard geometric checks.  What it does add is a fine-grained
float confidence in [0.10, 0.99] that downstream ranking can use to
distinguish a borderline acceptance from a rock-solid one.

Nothing here mutates state — every function is a pure transformation over
its inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import CandidateEvidence


# ---------------------------------------------------------------------------
# Base confidence by acceptance reason
# Ordered from most to least reliable.
# ---------------------------------------------------------------------------
_BASE_BY_REASON: dict[str, float] = {
    "feature_refinement":            0.97,  # ORB features + phase: highest geometric certainty
    "ocr_grid_regression":           0.95,  # Direct OCR grid regression: seed-independent
    "ncc_thresholds":                0.88,  # NCC passed all score/gap/confidence gates
    "patch_consensus":               0.80,  # Patch template agreement, no feature lock
    "seed_geotransform":             0.70,  # Seed placed + WMS confirmed shift
    "seed_geotransform_provisional": 0.42,  # Seed placed, WMS did NOT confirm
}

# Reasons we consider unconditionally "hard accepted" (not provisional)
_STRONG_REASONS: frozenset[str] = frozenset({
    "feature_refinement",
    "ocr_grid_regression",
    "ncc_thresholds",
})

# Cap applied when provisional=True regardless of other bonuses
_PROVISIONAL_CAP = 0.60


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class AcceptanceDecision:
    """
    Enriched decision derived from the raw quality dict and candidate context.

    accepted / provisional  – mirrored from the quality dict (not overridden)
    confidence              – float in [0.10, 0.99]; richer than the old 0.9/0.6/0.2
    reason                  – acceptance_reason string from the georeferencer
    notes                   – human-readable list of factors that shifted confidence
    """
    accepted: bool
    provisional: bool
    confidence: float
    reason: str | None
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def evaluate_quality(
    quality: dict[str, Any],
    evidence: list[CandidateEvidence] | None = None,
    conflicts: list[str] | None = None,
) -> AcceptanceDecision:
    """
    Derive a rich AcceptanceDecision from the quality dict and candidate context.

    Parameters
    ----------
    quality:
        The dict from ``auto_georeference.LAST_GEOREF_QUALITY``.
    evidence:
        The ``CandidateEvidence`` list from the validated ``GeoCandidate``.
    conflicts:
        The ``conflicts`` list from the validated ``GeoCandidate``.
    """
    if not quality:
        return AcceptanceDecision(
            accepted=False,
            provisional=False,
            confidence=0.10,
            reason=None,
            notes=["no_quality_data"],
        )

    accepted    = bool(quality.get("accepted"))
    provisional = bool(quality.get("provisional"))
    reason      = str(quality.get("acceptance_reason") or "")
    notes: list[str] = []

    # -----------------------------------------------------------------------
    # 1. Base confidence from acceptance path
    # -----------------------------------------------------------------------
    if accepted:
        base = _BASE_BY_REASON.get(reason, 0.35)
    else:
        base = 0.15
        notes.append("not_accepted")

    # -----------------------------------------------------------------------
    # 2. NCC quality modifiers  (only populated for "ncc_thresholds" path,
    #    but safe to read from other paths — they'll just be 0.0)
    # -----------------------------------------------------------------------
    ncc_confidence = float(quality.get("ncc_confidence") or 0.0)
    ncc_score_gap  = float(quality.get("ncc_score_gap")  or 0.0)

    if ncc_confidence > 0.0:
        if ncc_confidence >= 1.50:
            base += 0.05
            notes.append("ncc_high_confidence")
        elif ncc_confidence >= 1.25:
            base += 0.02
        elif ncc_confidence < 1.08:
            base -= 0.06
            notes.append("ncc_low_confidence")

    if ncc_score_gap > 0.0:
        if ncc_score_gap >= 0.05:
            base += 0.03
        elif ncc_score_gap < 0.01:
            base -= 0.04
            notes.append("ncc_gap_very_small")

    if quality.get("cluster_consensus"):
        base += 0.03
        notes.append("cluster_consensus")

    # -----------------------------------------------------------------------
    # 3. Sub-pixel / feature refinement bonuses
    # -----------------------------------------------------------------------
    if quality.get("feature_refinement_succeeded"):
        base += 0.04
        notes.append("feature_refined")

    if quality.get("patch_consensus_succeeded"):
        base += 0.02
        notes.append("patch_consensus")

    # -----------------------------------------------------------------------
    # 4. Coordinate anchor bonus
    #    Grid anchors mean the pixel↔world mapping is overdetermined;
    #    placement uncertainty is much lower even before WMS.
    # -----------------------------------------------------------------------
    if quality.get("has_coord_anchors"):
        base += 0.04
        notes.append("has_coord_anchors")

    # -----------------------------------------------------------------------
    # 5. Evidence quality modifier
    # -----------------------------------------------------------------------
    ev_list = evidence or []
    ev_count = len(ev_list)
    total_ev_weight = sum(max(float(e.weight), 0.0) for e in ev_list)

    if ev_count >= 3 or total_ev_weight >= 2.5:
        base += 0.05
        notes.append("strong_evidence")
    elif ev_count >= 1:
        base += min(0.03, total_ev_weight * 0.02)
    else:
        base -= 0.04
        notes.append("no_candidate_evidence")

    # -----------------------------------------------------------------------
    # 6. Conflict penalties
    # -----------------------------------------------------------------------
    conflict_list = conflicts or []
    if conflict_list:
        n_city  = sum(1 for c in conflict_list if c.startswith("city_conflict"))
        n_other = len(conflict_list) - n_city
        if n_city:
            base -= 0.08 * n_city
            notes.append(f"city_conflict_x{n_city}")
        if n_other:
            base -= 0.05 * n_other
            notes.append(f"other_conflict_x{n_other}")

    # -----------------------------------------------------------------------
    # 7. Provisional cap — WMS did not confirm the seed, so we hard-cap even
    #    if other signals look good.
    # -----------------------------------------------------------------------
    if provisional:
        base = min(base, _PROVISIONAL_CAP)
        notes.append("provisional_cap_applied")

    # -----------------------------------------------------------------------
    # 8. Clip to valid range
    # -----------------------------------------------------------------------
    confidence = round(max(0.10, min(0.99, base)), 4)
    return AcceptanceDecision(
        accepted=accepted,
        provisional=provisional,
        confidence=confidence,
        reason=reason or None,
        notes=notes,
    )


def confidence_label(confidence: float) -> str:
    """Human-readable tier label for a confidence score."""
    if confidence >= 0.90:
        return "high"
    if confidence >= 0.70:
        return "medium"
    if confidence >= 0.50:
        return "low"
    return "very_low"
