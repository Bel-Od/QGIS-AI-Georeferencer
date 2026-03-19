from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path


_REVIEWED_STATUSES = {"accepted", "corrected", "rejected", "review_required"}
_TRAINABLE_STATUSES = {"accepted", "corrected", "rejected"}


def export_training_dataset(
    *,
    case_dir: Path,
    output_path: Path,
    include_unreviewed: bool = False,
) -> Path:
    records: list[dict] = []
    for case_path, payload in _iter_case_payloads(case_dir):
        review = payload.get("review") or {}
        status = str(review.get("status") or "")
        if not include_unreviewed and status not in _REVIEWED_STATUSES:
            continue
        artifacts = payload.get("artifacts") or {}
        georef_result = artifacts.get("georef_result") or {}
        record = {
            "case_path": str(case_path),
            "input_path": (payload.get("context") or {}).get("input_path"),
            "working_output_path": georef_result.get("output_path"),
            "reviewed_output_path": (review.get("metadata") or {}).get("reviewed_output_path"),
            "epsg": georef_result.get("epsg"),
            "ocr_text": (artifacts.get("ocr") or {}).get("text"),
            "parsed": (artifacts.get("ocr") or {}).get("parsed"),
            "structured_hints": artifacts.get("structured_hints"),
            "vision": artifacts.get("vision"),
            "candidates": artifacts.get("candidates"),
            "selected_candidate_id": georef_result.get("selected_candidate_id"),
            "quality": georef_result.get("quality"),
            "review": review,
        }
        if review.get("corrected_center_easting") is not None and review.get("corrected_center_northing") is not None:
            record["target_center"] = {
                "easting": review.get("corrected_center_easting"),
                "northing": review.get("corrected_center_northing"),
            }
        if (review.get("metadata") or {}).get("corrected_geotransform") is not None:
            record["target_geotransform"] = (review.get("metadata") or {}).get("corrected_geotransform")
        records.append(record)

    return _write_jsonl(output_path, records)


def export_review_label_datasets(
    *,
    case_dir: Path,
    candidate_output_path: Path,
    transform_output_path: Path,
    include_unreviewed: bool = False,
) -> dict[str, Path]:
    candidate_records: list[dict] = []
    transform_records: list[dict] = []

    for case_path, payload in _iter_case_payloads(case_dir):
        review = payload.get("review") or {}
        status = str(review.get("status") or "")
        if not include_unreviewed and status not in _REVIEWED_STATUSES:
            continue
        candidate_record = _build_candidate_label_record(case_path, payload)
        if candidate_record is not None:
            candidate_records.append(candidate_record)
        transform_record = _build_transform_label_record(case_path, payload)
        if transform_record is not None:
            transform_records.append(transform_record)

    return {
        "candidate_labels": _write_jsonl(candidate_output_path, candidate_records),
        "transform_labels": _write_jsonl(transform_output_path, transform_records),
    }


def _iter_case_payloads(case_dir: Path):
    for case_path in sorted(case_dir.glob("*_case_bundle.json")):
        yield case_path, json.loads(case_path.read_text(encoding="utf-8"))


def _write_jsonl(output_path: Path, records: list[dict]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


def _build_candidate_label_record(case_path: Path, payload: dict) -> dict | None:
    review = payload.get("review") or {}
    status = str(review.get("status") or "")
    if status not in _TRAINABLE_STATUSES:
        return None

    artifacts = payload.get("artifacts") or {}
    georef_result = artifacts.get("georef_result") or {}
    candidates = list(artifacts.get("candidates") or [])
    if not candidates:
        return None

    selected_candidate_id = georef_result.get("selected_candidate_id")
    target_candidate_id = _target_candidate_id(candidates, review, selected_candidate_id)
    candidate_ids = [
        str(candidate.get("candidate_id") or "")
        for candidate in candidates
        if candidate.get("candidate_id")
    ]
    if not candidate_ids:
        return None

    if status == "rejected":
        candidate_labels = {candidate_id: 0 for candidate_id in candidate_ids}
    else:
        candidate_labels = {
            candidate_id: 1 if candidate_id == target_candidate_id else 0
            for candidate_id in candidate_ids
        }

    return {
        "case_id": _case_id(case_path),
        "review_id": _review_id(case_path),
        "review_status": status,
        "review_reason": review.get("reason"),
        "reviewer": review.get("reviewer"),
        "selected_candidate_id": selected_candidate_id,
        "target_candidate_id": target_candidate_id,
        "candidate_labels": candidate_labels,
        "context": _compact_context(payload),
    }


def _build_transform_label_record(case_path: Path, payload: dict) -> dict | None:
    review = payload.get("review") or {}
    status = str(review.get("status") or "")
    metadata = review.get("metadata") or {}

    corrected_gt = metadata.get("corrected_geotransform")
    target_center = _target_center(review, corrected_gt)
    if target_center is None:
        return None

    target_gt = corrected_gt if isinstance(corrected_gt, list) and len(corrected_gt) == 6 else None
    transform_target = _transform_target(target_gt)
    original_gt = metadata.get("original_geotransform")

    return {
        "case_id": _case_id(case_path),
        "review_id": _review_id(case_path),
        "review_status": status,
        "review_reason": review.get("reason"),
        "reviewer": review.get("reviewer"),
        "target_center": target_center,
        "target_transform": transform_target,
        "error_from_initial": {
            "delta_easting_m": metadata.get("delta_easting_m"),
            "delta_northing_m": metadata.get("delta_northing_m"),
            "delta_rotation_deg": metadata.get("delta_rotation_deg"),
        },
        "initial_transform": _transform_target(original_gt),
        "context": _compact_context(payload),
    }


def _compact_context(payload: dict) -> dict:
    artifacts = payload.get("artifacts") or {}
    georef_result = artifacts.get("georef_result") or {}
    quality = georef_result.get("quality") or {}
    vision = artifacts.get("vision") or {}
    overview = vision.get("overview") if isinstance(vision.get("overview"), dict) else {}
    ocr = artifacts.get("ocr") or {}
    parsed = ocr.get("parsed") or {}

    return {
        "epsg": georef_result.get("epsg") or overview.get("epsg"),
        "scale_hint": parsed.get("scale") or overview.get("scale"),
        "seed_source": quality.get("seed_source"),
        "seed_confidence": quality.get("seed_confidence"),
        "acceptance_reason": quality.get("acceptance_reason"),
        "provisional": bool(quality.get("provisional")),
        "has_coord_anchors": bool(quality.get("has_coord_anchors")),
    }


def _target_candidate_id(candidates: list[dict], review: dict, selected_candidate_id: str | None) -> str | None:
    corrected_center = _target_center(review, (review.get("metadata") or {}).get("corrected_geotransform"))
    if corrected_center is None:
        return selected_candidate_id

    best_candidate_id: str | None = None
    best_dist2: float | None = None
    for candidate in candidates:
        candidate_id = candidate.get("candidate_id")
        easting = candidate.get("center_easting")
        northing = candidate.get("center_northing")
        if candidate_id is None or easting is None or northing is None:
            continue
        dist2 = (float(easting) - corrected_center["easting"]) ** 2 + (float(northing) - corrected_center["northing"]) ** 2
        if best_dist2 is None or dist2 < best_dist2:
            best_dist2 = dist2
            best_candidate_id = str(candidate_id)
    return best_candidate_id or selected_candidate_id


def _target_center(review: dict, corrected_gt: list | None) -> dict[str, float] | None:
    center_e = review.get("corrected_center_easting")
    center_n = review.get("corrected_center_northing")
    if center_e is not None and center_n is not None:
        return {"easting": float(center_e), "northing": float(center_n)}
    return None


def _transform_target(gt: list | None) -> dict | None:
    if not isinstance(gt, list) or len(gt) != 6:
        return None
    return {
        "geotransform": gt,
        "rotation_deg": _gt_rotation_deg(gt),
        "scale_x": _gt_scale_x(gt),
        "scale_y": _gt_scale_y(gt),
    }


def _case_id(case_path: Path) -> str:
    suffix = "_case_bundle"
    stem = case_path.stem
    return stem[:-len(suffix)] if stem.endswith(suffix) else stem


def _review_id(case_path: Path) -> str:
    return datetime.fromtimestamp(case_path.stat().st_mtime, UTC).isoformat()


def _gt_rotation_deg(gt: list[float]) -> float:
    return math.degrees(math.atan2(-float(gt[2]), float(gt[1])))


def _gt_scale_x(gt: list[float]) -> float:
    return math.sqrt(float(gt[1]) ** 2 + float(gt[4]) ** 2)


def _gt_scale_y(gt: list[float]) -> float:
    return math.sqrt(float(gt[2]) ** 2 + float(gt[5]) ** 2)
