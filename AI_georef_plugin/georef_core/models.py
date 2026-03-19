from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PipelineContext:
    input_path: Path
    is_pdf: bool
    output_dir: Path
    artifact_dir: Path


@dataclass(slots=True)
class IngestResult:
    source_path: Path
    working_path: Path
    is_pdf: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    pdf_text: str = ""
    rendered_from_pdf: bool = False


@dataclass(slots=True)
class OCRResult:
    text: str = ""
    parsed: dict[str, Any] = field(default_factory=dict)
    text_source: str = "ocr"


@dataclass(slots=True)
class StructuredHints:
    site_street: str | None = None
    site_house_number: str | None = None
    site_city: str | None = None
    site_postcode: str | None = None
    road_codes: list[str] = field(default_factory=list)
    station_text: str | None = None
    parcel_refs: list[str] = field(default_factory=list)
    landmarks: list[str] = field(default_factory=list)
    site_addresses: list[dict[str, Any]] = field(default_factory=list)
    office_addresses: list[dict[str, Any]] = field(default_factory=list)
    client_address: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StructuredHints":
        return cls(
            site_street=data.get("site_street"),
            site_house_number=data.get("site_house_number"),
            site_city=data.get("site_city"),
            site_postcode=data.get("site_postcode"),
            road_codes=list(data.get("road_codes") or []),
            station_text=data.get("station_text"),
            parcel_refs=list(data.get("parcel_refs") or []),
            landmarks=list(data.get("landmarks") or []),
            site_addresses=list(data.get("site_addresses") or []),
            office_addresses=list(data.get("office_addresses") or []),
            client_address=data.get("client_address"),
            raw=dict(data),
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("raw", None)
        return d


@dataclass(slots=True)
class VisionResult:
    overview: dict[str, Any] = field(default_factory=dict)
    title_block: dict[str, Any] = field(default_factory=dict)

    @property
    def scale(self) -> int | None:
        value = self.overview.get("scale")
        return int(value) if isinstance(value, (int, float)) else None

    @property
    def epsg(self) -> int | None:
        value = self.overview.get("epsg")
        return int(value) if isinstance(value, (int, float)) else None


@dataclass(slots=True)
class CandidateEvidence:
    source: str
    text: str | None = None
    weight: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GeoCandidate:
    candidate_id: str
    source: str
    confidence_tier: str
    center_easting: float | None
    center_northing: float | None
    search_radius_m: float
    label: str
    rank_score: float = 0.0
    evidence: list[CandidateEvidence] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    conflicts: list[str] = field(default_factory=list)

    def to_seed_dict(self, scale_denominator: int | None = None) -> dict[str, Any]:
        seed = {
            "enabled": True,
            "_source": self.source,
            "_seed_confidence": self.confidence_tier,
            "_candidate_id": self.candidate_id,
            "_label": self.label,
            "center_easting": self.center_easting,
            "center_northing": self.center_northing,
            "apply_after_crop": True,
        }
        if scale_denominator:
            seed["scale_denominator"] = int(scale_denominator)
        return seed


@dataclass(slots=True)
class ValidationResult:
    candidate_id: str
    accepted: bool
    provisional: bool = False
    confidence: float | None = None
    acceptance_reason: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ReviewDecision:
    status: str
    reason: str | None = None
    reviewer: str | None = None
    corrected_center_easting: float | None = None
    corrected_center_northing: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GeorefResult:
    output_path: Path | None = None
    epsg: int | None = None
    selected_candidate_id: str | None = None
    quality: dict[str, Any] = field(default_factory=dict)
    validation: ValidationResult | None = None


@dataclass(slots=True)
class PipelineArtifacts:
    ingest: IngestResult | None = None
    ocr: OCRResult | None = None
    structured_hints: StructuredHints | None = None
    vision: VisionResult | None = None
    candidates: list[GeoCandidate] = field(default_factory=list)
    validations: list[ValidationResult] = field(default_factory=list)
    georef_result: GeorefResult | None = None


@dataclass(slots=True)
class GeorefCaseBundle:
    context: PipelineContext
    artifacts: PipelineArtifacts
    review: ReviewDecision | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": {
                "input_path": str(self.context.input_path),
                "is_pdf": self.context.is_pdf,
                "output_dir": str(self.context.output_dir),
                "artifact_dir": str(self.context.artifact_dir),
            },
            "artifacts": {
                "ingest": _serialize_dataclass(self.artifacts.ingest),
                "ocr": _serialize_dataclass(self.artifacts.ocr),
                "structured_hints": _serialize_dataclass(self.artifacts.structured_hints),
                "vision": _serialize_dataclass(self.artifacts.vision),
                "candidates": [_serialize_dataclass(item) for item in self.artifacts.candidates],
                "validations": [_serialize_dataclass(item) for item in self.artifacts.validations],
                "georef_result": _serialize_dataclass(self.artifacts.georef_result),
            },
            "review": _serialize_dataclass(self.review),
        }


def _serialize_dataclass(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_serialize_dataclass(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_dataclass(item) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        data = asdict(value)
        return _serialize_dataclass(data)
    return value
