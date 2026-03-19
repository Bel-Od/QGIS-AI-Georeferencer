from __future__ import annotations

from .models import IngestResult
from .runtime import load_auto_georeference


def detect_crop_bbox(ingest: IngestResult) -> tuple[int, int, int, int] | None:
    ag = load_auto_georeference()

    return ag.detect_main_map_bbox(ingest.working_path)
