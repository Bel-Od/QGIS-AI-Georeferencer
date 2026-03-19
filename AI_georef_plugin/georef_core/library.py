"""
library.py
----------
Persistent per-plan ground-truth library.

When a georeferenced result is accepted or corrected by the user, the
verified placement is written here so future runs of the SAME plan bypass
the AI/OCR candidate-generation stage and go straight to the verified
position as a high-confidence seed.

The library also records how far the uncorrected pipeline result was from
the verified placement, giving a running accuracy metric per plan.

Format: output/georef_library.json
  {
    "<plan_stem>": { ...GeorefLibraryEntry fields... },
    ...
  }

Usage (typical)
---------------
  library = load_library(library_path)
  entry   = find_library_match("Anhang_4_Olpe", library, plan_path=path)
  if entry:
      # create a GeoCandidate from library_candidate(entry) and prepend to list
      ...

  # After user accepts / corrects:
  update_library(library_path=..., plan_stem=..., ...)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .crs_config import get_active_epsg as _get_active_epsg


#: Default filename written inside the output directory.
LIBRARY_FILENAME = "georef_library.json"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class GeorefLibraryEntry:
    """
    One entry in the ground-truth library, keyed by plan stem.

    Fields
    ------
    plan_stem          : filename without extension (the lookup key)
    plan_hash          : first 16 hex chars of SHA-256(file), for rename detection
    center_easting     : verified map centre in the stored EPSG
    center_northing    : verified map centre in the stored EPSG
    geotransform       : GDAL 6-element affine tuple [origin_e, pixel_w, rot, origin_n, rot, pixel_h]
    epsg               : EPSG code of the geotransform
    scale_denominator  : e.g. 1000 for 1:1000 (None if unknown)
    source             : who/what set this entry ("accepted", "corrected", "interactive_adjustment")
    confidence         : float in [0,1] — pipeline confidence at time of acceptance
    reviewed_at        : ISO-8601 UTC timestamp
    reviewer           : optional reviewer name
    ocr_hints          : structured OCR context (city, road codes, etc.)
    quality            : raw pipeline quality dict at acceptance time
    delta_from_pipeline_m : metres between uncorrected result and this verified position
                           (0.0 for "accepted without changes", None if unknown)
    """
    plan_stem:                str
    plan_hash:                str
    center_easting:           float
    center_northing:          float
    geotransform:             list[float]
    epsg:                     int
    scale_denominator:        int | None
    source:                   str
    confidence:               float
    reviewed_at:              str
    reviewer:                 str | None
    ocr_hints:                dict[str, Any] = field(default_factory=dict)
    quality:                  dict[str, Any] = field(default_factory=dict)
    delta_from_pipeline_m:    float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeorefLibraryEntry":
        return cls(
            plan_stem=str(data["plan_stem"]),
            plan_hash=str(data.get("plan_hash") or ""),
            center_easting=float(data["center_easting"]),
            center_northing=float(data["center_northing"]),
            geotransform=list(data.get("geotransform") or []),
            epsg=int(data.get("epsg") or 0) or _get_active_epsg(),
            scale_denominator=(int(data["scale_denominator"]) if data.get("scale_denominator") else None),
            source=str(data.get("source") or ""),
            confidence=float(data.get("confidence") or 0.0),
            reviewed_at=str(data.get("reviewed_at") or ""),
            reviewer=data.get("reviewer"),
            ocr_hints=dict(data.get("ocr_hints") or {}),
            quality=dict(data.get("quality") or {}),
            delta_from_pipeline_m=(
                float(data["delta_from_pipeline_m"])
                if data.get("delta_from_pipeline_m") is not None
                else None
            ),
        )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_library(library_path: Path) -> dict[str, GeorefLibraryEntry]:
    """Load the library from *library_path*. Returns ``{}`` if missing."""
    if not library_path.exists():
        return {}
    try:
        raw = json.loads(library_path.read_text(encoding="utf-8"))
        return {k: GeorefLibraryEntry.from_dict(v) for k, v in raw.items()}
    except Exception:
        return {}


def save_library(library: dict[str, GeorefLibraryEntry], library_path: Path) -> None:
    """Serialise *library* to disk, creating parent directories if needed."""
    library_path.parent.mkdir(parents=True, exist_ok=True)
    raw = {k: v.to_dict() for k, v in library.items()}
    library_path.write_text(
        json.dumps(raw, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------
def _hash_file(path: Path) -> str:
    """Return first 16 hex chars of SHA-256(file content).  Returns '' on error."""
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()[:16]
    except Exception:
        return ""


def find_library_match(
    plan_stem: str,
    library: dict[str, GeorefLibraryEntry],
    *,
    plan_path: Path | None = None,
) -> GeorefLibraryEntry | None:
    """
    Return the library entry for *plan_stem* (case-insensitive match).

    If *plan_path* is provided AND the stored entry has a hash, the file hash
    is verified.  A hash mismatch means two different files share the same stem
    name — we skip the entry rather than use stale ground truth.
    """
    target = plan_stem.strip().lower()
    for key, entry in library.items():
        if key.strip().lower() == target:
            if plan_path and entry.plan_hash:
                file_hash = _hash_file(plan_path)
                if file_hash and file_hash != entry.plan_hash:
                    # Different file content — treat as a different plan
                    return None
            return entry
    return None


# ---------------------------------------------------------------------------
# Write / update
# ---------------------------------------------------------------------------
def update_library(
    *,
    library_path: Path,
    plan_stem: str,
    plan_path: Path | None,
    center_easting: float,
    center_northing: float,
    geotransform: list[float],
    epsg: int,
    scale_denominator: int | None,
    source: str,
    confidence: float,
    reviewer: str | None,
    ocr_hints: dict[str, Any],
    quality: dict[str, Any],
    pipeline_center_easting: float | None = None,
    pipeline_center_northing: float | None = None,
) -> GeorefLibraryEntry:
    """
    Create or replace the library entry for *plan_stem* and save to disk.

    Parameters
    ----------
    pipeline_center_easting/northing:
        The uncorrected pipeline result centre.  When provided, the entry
        records ``delta_from_pipeline_m`` — how far off the AI was from the
        verified position.  Pass ``None`` (default) when the user accepted
        the result without any manual correction (delta is then 0.0).
    """
    library = load_library(library_path)

    if pipeline_center_easting is not None and pipeline_center_northing is not None:
        delta_m: float | None = round(
            ((center_easting - pipeline_center_easting) ** 2
             + (center_northing - pipeline_center_northing) ** 2) ** 0.5,
            2,
        )
    else:
        delta_m = 0.0  # accepted without changes

    entry = GeorefLibraryEntry(
        plan_stem=plan_stem,
        plan_hash=_hash_file(plan_path) if plan_path else "",
        center_easting=round(center_easting, 2),
        center_northing=round(center_northing, 2),
        geotransform=[round(v, 6) for v in geotransform],
        epsg=epsg,
        scale_denominator=scale_denominator,
        source=source,
        confidence=round(float(confidence), 4),
        reviewed_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        reviewer=reviewer or None,
        ocr_hints={k: v for k, v in ocr_hints.items() if v},
        quality=dict(quality),
        delta_from_pipeline_m=delta_m,
    )
    library[plan_stem] = entry
    save_library(library, library_path)
    return entry


# ---------------------------------------------------------------------------
# Candidate factory
# ---------------------------------------------------------------------------
def library_candidate_kwargs(entry: GeorefLibraryEntry) -> dict[str, Any]:
    """
    Return keyword-argument dict suitable for ``add_candidate()`` in
    ``candidate_generation.build_candidates()``.

    The library source receives a very high base rank (above even
    ``ocr_coordinates``) and a tight 150 m search radius, ensuring it is
    always the first candidate validated by WMS refinement.
    """
    label = (
        f"Library verified: {entry.plan_stem}"
        f" ({entry.source}, {entry.reviewed_at[:10]})"
    )
    return {
        "candidate_id": "library-verified",
        "source": "library",
        "confidence_tier": "coordinates",
        "easting": entry.center_easting,
        "northing": entry.center_northing,
        "label": label,
        "evidence_text": f"Verified by {entry.reviewer or 'operator'} on {entry.reviewed_at[:10]}",
        "evidence_weight": 2.0,
        "metadata": {
            "library_source":      entry.source,
            "library_confidence":  entry.confidence,
            "library_reviewed_at": entry.reviewed_at,
            "library_delta_m":     entry.delta_from_pipeline_m,
        },
    }
