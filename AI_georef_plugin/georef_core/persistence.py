from __future__ import annotations

import json
from pathlib import Path

from .models import GeorefCaseBundle


def save_case_bundle(bundle: GeorefCaseBundle, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(bundle.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return target_path


def load_case_bundle(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def update_case_bundle_review(path: Path, review: dict) -> Path:
    payload = load_case_bundle(path)
    history = list(payload.get("review_history") or [])
    if payload.get("review"):
        history.append(payload["review"])
    payload["review"] = dict(review)
    payload["review_history"] = history
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path
