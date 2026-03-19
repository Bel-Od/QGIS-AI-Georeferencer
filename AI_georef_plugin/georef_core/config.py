"""
config.py
---------
Central home for pipeline constants that were previously scattered as magic
numbers across auto_georeference.py and the georef_core modules.

Import from here rather than hard-coding values.  None of these are mutable
at runtime — change them here to affect the whole pipeline.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# WMS refinement acceptance gates
# (mirrors the constants in auto_georeference.py for reference)
# ---------------------------------------------------------------------------

#: Minimum NCC score required to accept a WMS match.
WMS_MIN_SCORE: float = 0.18

#: Minimum score gap between best and second-best WMS tile.
WMS_MIN_SCORE_GAP: float = 0.015

#: Minimum peak-confidence ratio (best / second, both offset by +1).
WMS_MIN_CONFIDENCE: float = 1.08

#: Maximum seed-to-result shift allowed in metres before a result is flagged.
WMS_MAX_SHIFT_M: float = 5_000.0

#: WMS tile size in pixels used for template matching.
WMS_TILE_PX: int = 256

# ---------------------------------------------------------------------------
# Candidate search radii by confidence tier (metres)
# ---------------------------------------------------------------------------
SEARCH_RADIUS_BY_TIER: dict[str, float] = {
    "coordinates": 200.0,
    "parcel":       500.0,
    "address":      250.0,
    "street":       800.0,
    "feature":    2_000.0,
    "city":      10_000.0,
    "fallback":  12_000.0,
}

# ---------------------------------------------------------------------------
# Candidate base rank scores by source
# ---------------------------------------------------------------------------
BASE_RANK_BY_SOURCE: dict[str, float] = {
    "ocr_coordinates": 1.00,
    "parcel_ref":       0.95,
    "site_address":     0.93,
    "project_address":  0.90,
    "street_city":      0.82,
    "road_city":        0.78,
    "landmark_city":    0.68,
    "city_name":        0.45,
}

# ---------------------------------------------------------------------------
# Pipeline validation limits
# ---------------------------------------------------------------------------

#: Maximum candidates to validate via full WMS refinement per run.
MAX_VALIDATION_CANDIDATES: int = 3

#: Minimum confidence required to auto-accept without human review.
MIN_AUTO_ACCEPT_CONFIDENCE: float = 0.70

#: Confidence below which the result is always flagged for human review.
MIN_REVIEW_CONFIDENCE: float = 0.50

# ---------------------------------------------------------------------------
# OCR / thumbnail parameters
# ---------------------------------------------------------------------------

#: Maximum thumbnail dimension (pixels) sent to Vision AI.
THUMB_MAX_PX: int = 2_000

#: Height of the margin strip (pixels) fed to Tesseract for grid labels.
STRIP_HEIGHT: int = 200

#: Width of the margin strip (pixels) for the left/right sides.
STRIP_WIDTH: int = 2_000
