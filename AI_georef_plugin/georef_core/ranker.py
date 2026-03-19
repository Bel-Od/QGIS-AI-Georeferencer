from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .models import GeoCandidate, StructuredHints, VisionResult
from .crs_config import get_active_bounds


def default_ranker_model_path() -> Path:
    return Path(__file__).resolve().parent.parent / "output" / "candidate_ranker_model.json"


@dataclass(slots=True)
class LinearRanker:
    version: str = "1.0"
    bias: float = 0.0
    weights: dict[str, float] = field(default_factory=dict)

    def score_features(self, features: dict[str, float]) -> float:
        total = float(self.bias)
        for key, value in features.items():
            total += float(self.weights.get(key, 0.0)) * float(value)
        return total

    def score_candidate(
        self,
        candidate: GeoCandidate,
        *,
        hints: StructuredHints | None = None,
        vision: VisionResult | None = None,
    ) -> float:
        features = extract_candidate_features(candidate, hints=hints, vision=vision)
        return self.score_features(features)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "bias": self.bias,
            "weights": dict(self.weights),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LinearRanker":
        return cls(
            version=str(data.get("version") or "1.0"),
            bias=float(data.get("bias") or 0.0),
            weights={str(k): float(v) for k, v in dict(data.get("weights") or {}).items()},
        )


def extract_candidate_features(
    candidate: GeoCandidate,
    *,
    hints: StructuredHints | None = None,
    vision: VisionResult | None = None,
) -> dict[str, float]:
    hints = hints or StructuredHints()
    vision = vision or VisionResult()
    source_keys = [
        "library",
        "ocr_coordinates",
        "parcel_ref",
        "site_address",
        "project_address",
        "street_city",
        "road_city",
        "landmark_city",
        "city_name",
    ]
    tier_keys = ["coordinates", "parcel", "address", "street", "feature", "city", "fallback"]
    # Evidence quality: weight-sum is more informative than raw count.
    evidence_weight_total = sum(float(getattr(e, "weight", 0) or 0) for e in candidate.evidence)

    # Conflict severity: count all conflicts and flag city conflicts specifically,
    # since a city-name mismatch is the most common cause of a bad pick.
    conflict_count = float(len(candidate.conflicts))
    has_city_conflict = 1.0 if any("city" in str(c).lower() for c in candidate.conflicts) else 0.0

    # Geographic plausibility: is the candidate within the active CRS bounds?
    # Bounds are updated by set_active_crs_preset() in auto_georeference.py.
    _e_min, _e_max, _n_min, _n_max = get_active_bounds()
    _e = candidate.center_easting
    _n = candidate.center_northing
    in_region = (
        1.0
        if (_e is not None and _n is not None
            and _e_min <= float(_e) <= _e_max
            and _n_min <= float(_n) <= _n_max)
        else 0.0
    )

    features: dict[str, float] = {
        # Raw geometric quality
        "search_radius_km":        float(candidate.search_radius_m) / 1000.0,
        "evidence_count":          float(len(candidate.evidence)),
        "evidence_weight_total":   evidence_weight_total,
        "conflict_count":          conflict_count,
        "has_city_conflict":       has_city_conflict,
        "candidate_in_germany":    in_region,  # legacy key (backwards compat with trained models)
        "candidate_in_region":     in_region,  # new key for retrained models
        # Context presence flags (independent of rank_score)
        "has_road_code":   1.0 if candidate.metadata.get("road_code") or hints.road_codes else 0.0,
        "has_parcel_ref":  1.0 if candidate.source == "parcel_ref" or hints.parcel_refs else 0.0,
        "has_landmark":    1.0 if candidate.metadata.get("landmark") or hints.landmarks else 0.0,
        "has_site_street": 1.0 if hints.site_street else 0.0,
        "has_site_city":   1.0 if hints.site_city else 0.0,
        "has_postcode":    1.0 if hints.site_postcode else 0.0,
        "vision_scale_known": 1.0 if vision.scale else 0.0,
        "vision_epsg_known":  1.0 if vision.epsg else 0.0,
        # NOTE: heuristic_rank (rank_score) is intentionally excluded — it is a
        # non-linear aggregate of source+tier+evidence+conflicts that is already
        # encoded by the one-hot features below.  Including it creates a circular
        # dependency where the model learns to reweight its own input signal.
    }
    for key in source_keys:
        features[f"source:{key}"] = 1.0 if candidate.source == key else 0.0
    for key in tier_keys:
        features[f"tier:{key}"] = 1.0 if candidate.confidence_tier == key else 0.0
    return features


def save_ranker_model(ranker: LinearRanker, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ranker.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_ranker_model(path: Path | None = None) -> LinearRanker | None:
    target = path or default_ranker_model_path()
    if not target.exists():
        return None
    data = json.loads(target.read_text(encoding="utf-8"))
    return LinearRanker.from_dict(data)


def rerank_candidates(
    candidates: list[GeoCandidate],
    *,
    hints: StructuredHints | None = None,
    vision: VisionResult | None = None,
    model_path: Path | None = None,
) -> list[GeoCandidate]:
    ranker = load_ranker_model(model_path)
    if ranker is None:
        return sorted(candidates, key=lambda item: (item.rank_score, -item.search_radius_m), reverse=True)
    ranked: list[GeoCandidate] = []
    for candidate in candidates:
        model_score = ranker.score_candidate(candidate, hints=hints, vision=vision)
        candidate.metadata = dict(candidate.metadata or {})
        candidate.metadata["model_score"] = model_score
        ranked.append(candidate)
    return sorted(
        ranked,
        key=lambda item: (float(item.metadata.get("model_score", -math.inf)), item.rank_score, -item.search_radius_m),
        reverse=True,
    )
