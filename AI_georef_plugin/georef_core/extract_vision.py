from __future__ import annotations

from PIL import Image

from .models import IngestResult, VisionResult
from .runtime import load_auto_georeference


def extract_vision(ingest: IngestResult, ocr_text: str = "", canvas_img: Image.Image | None = None) -> VisionResult:
    ag = load_auto_georeference()

    overview = ag.openai_vision_analysis(
        ingest.working_path,
        ingest.metadata,
        canvas_img=canvas_img if ag.USE_QGIS_CANVAS_VISION_CONTEXT else None,
        ocr_text=ocr_text or "",
    )
    title_block = overview.get("title_block") if isinstance(overview.get("title_block"), dict) else {}
    return VisionResult(overview=overview, title_block=title_block)
