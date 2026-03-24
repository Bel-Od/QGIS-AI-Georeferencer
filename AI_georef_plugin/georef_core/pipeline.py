from __future__ import annotations

from pathlib import Path

from .candidate_generation import build_candidates, select_top_candidates
from .extract_text import extract_structured_hints, extract_text
from .extract_vision import extract_vision
from .ingest import ingest_plan
from .models import GeorefCaseBundle, PipelineArtifacts, PipelineContext
from .persistence import save_case_bundle
from .review import review_status_from_validation
from .validation import choose_best_result, validate_candidate


def run_pipeline(
    *,
    input_path: Path,
    is_pdf: bool,
    output_dir: Path,
    artifact_dir: Path | None = None,
    persist_case: bool = False,
    max_validation_candidates: int = 3,
    library_path: Path | None = None,
) -> GeorefCaseBundle:
    """
    Run the full georeferencing pipeline.

    Parameters
    ----------
    library_path:
        Path to ``georef_library.json``.  When provided, any operator-verified
        entry for the current plan is injected as the top-ranked candidate,
        bypassing the AI/OCR stage for plans that have been reviewed before.
        Pass ``None`` (default) to disable library lookup.
    """
    artifact_dir = artifact_dir or (output_dir / "_cases")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    context = PipelineContext(
        input_path=input_path,
        is_pdf=is_pdf,
        output_dir=output_dir,
        artifact_dir=artifact_dir,
    )
    artifacts = PipelineArtifacts()

    ingest = ingest_plan(context)
    artifacts.ingest = ingest

    ocr = extract_text(ingest)
    artifacts.ocr = ocr

    vision = extract_vision(ingest, ocr.text)
    artifacts.vision = vision

    structured = extract_structured_hints(ocr.text, vision.overview, vision.title_block)
    artifacts.structured_hints = structured

    candidates = build_candidates(
        src_path=ingest.source_path,
        ocr_text=ocr.text,
        hints=structured,
        vision=vision,
        parsed=ocr.parsed,
        library_path=library_path,
    )
    artifacts.candidates = candidates

    from .crs_config import get_active_epsg as _get_epsg
    epsg = vision.epsg or ingest.metadata.get("epsg_hint") or _get_epsg()
    shortlisted = select_top_candidates(
        candidates,
        limit=max_validation_candidates,
        hints=structured,
        vision=vision,
    )
    dynamic_limit = len(shortlisted)
    if shortlisted:
        top = shortlisted[0]
        second = shortlisted[1] if len(shortlisted) > 1 else None
        top_model = float((top.metadata or {}).get("model_score", top.rank_score) or top.rank_score)
        second_model = float((second.metadata or {}).get("model_score", second.rank_score) or second.rank_score) if second else float("-inf")
        margin = top_model - second_model if second is not None else float("inf")
        if top.source == "library":
            dynamic_limit = 1
        elif top.confidence_tier in ("address", "parcel", "coordinates") and margin >= 0.15:
            dynamic_limit = 1
        elif top.confidence_tier in ("street", "feature") and margin >= 0.20:
            dynamic_limit = min(2, len(shortlisted))

    results_by_candidate: dict[str, object] = {}
    for candidate in shortlisted[:dynamic_limit]:
        validation, georef_result = validate_candidate(
            candidate,
            src_path=ingest.working_path,
            vision=vision,
            epsg=int(epsg),
        )
        artifacts.validations.append(validation)
        results_by_candidate[candidate.candidate_id] = georef_result
        if validation.accepted and not validation.provisional:
            break

    best_validation = choose_best_result(artifacts.validations)
    if best_validation is not None:
        artifacts.georef_result = results_by_candidate.get(best_validation.candidate_id)

    bundle = GeorefCaseBundle(
        context=context,
        artifacts=artifacts,
        review=review_status_from_validation(best_validation),
    )

    if persist_case:
        case_name = f"{input_path.stem}_case_bundle.json"
        save_case_bundle(bundle, artifact_dir / case_name)

    return bundle
