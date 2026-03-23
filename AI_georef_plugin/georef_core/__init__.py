"""Core pipeline structures for the georeferencing system."""

from .models import (
    CandidateEvidence,
    GeoCandidate,
    GeorefCaseBundle,
    GeorefResult,
    IngestResult,
    OCRResult,
    PipelineArtifacts,
    PipelineContext,
    ReviewDecision,
    StructuredHints,
    ValidationResult,
    VisionResult,
)
from .training_export import export_review_label_datasets, export_training_dataset
from .ranker import LinearRanker, load_ranker_model, save_ranker_model
from .ranker_training import load_ranking_examples, train_linear_ranker, evaluate_ranker
from .decision_engine import AcceptanceDecision, evaluate_quality, confidence_label
from .library import (
    GeorefLibraryEntry,
    find_library_match,
    load_library,
    save_library,
    update_library,
)
from .map_sources import (
    MapSource,
    load_map_sources,
    save_map_sources,
    add_map_source,
    remove_map_source,
    get_available_presets,
    preset_source,
    as_wms_configs,
    new_source_id,
    MAP_SOURCES_FILENAME,
)

__all__ = [
    # models
    "CandidateEvidence",
    "GeoCandidate",
    "GeorefCaseBundle",
    "GeorefResult",
    "IngestResult",
    "OCRResult",
    "PipelineArtifacts",
    "PipelineContext",
    "ReviewDecision",
    "StructuredHints",
    "ValidationResult",
    "VisionResult",
    # training
    "export_review_label_datasets",
    "export_training_dataset",
    "load_ranking_examples",
    "train_linear_ranker",
    "evaluate_ranker",
    # ranker
    "LinearRanker",
    "load_ranker_model",
    "save_ranker_model",
    # decision engine
    "AcceptanceDecision",
    "evaluate_quality",
    "confidence_label",
    # pipeline
    "run_pipeline",
    # ground-truth library
    "GeorefLibraryEntry",
    "find_library_match",
    "load_library",
    "save_library",
    "update_library",
    # map source registry
    "MapSource",
    "load_map_sources",
    "save_map_sources",
    "add_map_source",
    "remove_map_source",
    "get_available_presets",
    "preset_source",
    "as_wms_configs",
    "new_source_id",
    "MAP_SOURCES_FILENAME",
]


def run_pipeline(*args, **kwargs):
    from .pipeline import run_pipeline as _run_pipeline

    return _run_pipeline(*args, **kwargs)
