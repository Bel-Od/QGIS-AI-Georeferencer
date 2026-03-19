from __future__ import annotations

from .models import IngestResult, OCRResult, StructuredHints
from .runtime import load_auto_georeference


def extract_text(ingest: IngestResult) -> OCRResult:
    ag = load_auto_georeference()

    if ingest.is_pdf:
        image_ocr_text = ag.ocr_extract_text(ingest.working_path)
        merged = ag._merge_text_sources(ingest.pdf_text, image_ocr_text) if ingest.pdf_text.strip() else image_ocr_text
        parsed = ag.parse_coordinates(merged) if merged else {
            "eastings": [],
            "northings": [],
            "pairs": [],
            "scale": None,
            "crs_hints": [],
        }
        if not parsed.get("scale") and ingest.metadata.get("scale_hint"):
            parsed["scale"] = ingest.metadata["scale_hint"]
        return OCRResult(text=merged, parsed=parsed, text_source="pdf+ocr" if ingest.pdf_text.strip() else "ocr")

    text = ag.ocr_extract_text(ingest.working_path)
    parsed = ag.parse_coordinates(text) if text else {
        "eastings": [],
        "northings": [],
        "pairs": [],
        "scale": None,
        "crs_hints": [],
    }
    return OCRResult(text=text, parsed=parsed, text_source="ocr")


def extract_structured_hints(text: str, vision_overview: dict | None = None, title_block: dict | None = None) -> StructuredHints:
    ag = load_auto_georeference()

    hints = ag._extract_structured_location_hints(text or "", vision_overview or {}, title_block or {})
    return StructuredHints.from_dict(hints)
