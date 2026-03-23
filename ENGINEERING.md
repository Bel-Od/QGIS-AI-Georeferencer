# Engineering Overview

This document describes the current runtime structure of AI Georeferencer and the main tradeoffs behind it. It complements the user-facing setup and usage guidance in `README.md`.

## System objective

The plugin is designed to automate as much of the plan georeferencing workflow as is practical inside QGIS without removing operator control. The target input is engineering plans where coordinate evidence may be partial, noisy, rotated, or embedded in mixed sources such as scanned rasters, vector PDFs, title blocks, and reference-map context.

## Processing model

The current system is a staged pipeline.

1. Ingest reads TIFF metadata or renders PDF input into a raster working file.
2. Text extraction combines PDF text layers, when available, with OCR output.
3. Vision analysis extracts higher-level context such as scale, plan type, title-block fields, and location clues.
4. Structured hint extraction converts OCR and title-block output into normalized location signals such as addresses, road references, parcels, stations, and landmarks.
5. Candidate generation assembles potential geographic seeds from coordinates, geocoded addresses, parcel references, prior reviewed cases, and heuristic fallbacks.
6. Validation and refinement evaluate top candidates against configured map sources and produce the final georeferenced raster.
7. Review persists accept, reject, and adjustment outcomes so later runs can reuse verified information.

This model is reflected in the modular pipeline implementation in `AI_georef_plugin/georef_core/pipeline.py`, while the QGIS plugin still invokes the legacy orchestration surface in `AI_georef_plugin/auto_georeference.py`.

## Architectural layout

The codebase currently has two operational layers.

- `AI_georef_plugin/auto_georeference.py`
  The plugin-facing execution layer used by QGIS. It owns runtime integration, long-lived compatibility surfaces, and the end-to-end georeferencing flow exposed to the UI.
- `AI_georef_plugin/georef_core/`
  Reusable pipeline modules that isolate data structures, extraction helpers, candidate generation, ranking, persistence, review state, and map-source management.

Key modules in `georef_core`:

- `pipeline.py`
  End-to-end modular pipeline assembly.
- `ingest.py`, `extract_text.py`, `extract_vision.py`
  Early-stage processing for source preparation and signal extraction.
- `text_parsing.py`
  OCR and PDF text normalization, merged text handling, scale extraction, and coordinate parsing.
- `location_hints.py`
  Structured extraction of site and office addresses, road codes, parcel references, and related location hints.
- `candidate_generation.py`
  Candidate seed creation and heuristic ranking inputs.
- `ranker.py`, `ranker_training.py`
  Learned reranking and training support.
- `library.py`, `review.py`, `persistence.py`
  Review lifecycle and persistence of accepted results.

## Main design tradeoffs

### Hybrid AI and deterministic GIS

The system uses AI for tasks that are difficult to solve reliably with fixed parsing alone, especially title-block interpretation and ambiguous visual context. Geospatial transformations, reprojection, candidate evaluation, and final raster output remain deterministic.

Tradeoff:
This improves coverage on messy plans, but introduces an external dependency and some non-determinism in interpretation. The deterministic stages are kept explicit so the final output remains inspectable and debuggable.

### Assistive automation over full automation

The plugin is designed to help operators reach a correct result faster, not to hide the decision path completely. Accept, reject, and manual adjustment are normal parts of the workflow.

Tradeoff:
This keeps the tool usable in production settings where plans vary widely in quality, but it means the system optimizes for reviewable automation rather than a one-click guarantee.

### Multiple weak signals instead of one dominant signal

Plans do not always contain clean coordinate grids. Some runs depend more heavily on addresses, parcel references, title-block text, prior reviewed cases, or map matching.

Tradeoff:
The pipeline is more resilient across input types, but candidate generation and ranking become more involved because several imperfect signals must be reconciled rather than a single source of truth being trusted blindly.

### Regional configuration through geopacks

CRS bounds, WMS sources, and related lookup data are externalized through geopacks rather than duplicated in each feature path.

Tradeoff:
This keeps regional behavior configurable and easier to extend, but it also means the shipped experience is strongest in the configured regions, especially NRW.

## Evidence handling and error tolerance

The pipeline is designed around imperfect evidence.

- PDF text and OCR are merged because either source can be incomplete on its own.
- Address extraction distinguishes office and site addresses because title blocks often include both.
- Candidate generation keeps several hypotheses alive long enough for validation rather than committing too early.
- Manual review remains part of the intended operating model because not every plan can be resolved safely from automation alone.

## Verification approach

The repository includes lightweight unit coverage for extracted pure parsing logic in `tests/test_text_parsing.py`. The current tests cover:

- scale extraction from noisy text
- OCR and PDF text merging
- coordinate parsing
- separation of site and office address hints
- road-code extraction and normalization

These tests do not replace end-to-end validation inside QGIS, but they do provide regression coverage for the parsing and hint-extraction logic that now lives in modular form.
