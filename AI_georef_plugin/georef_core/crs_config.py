"""
crs_config.py — shared mutable CRS state for the auto-georeferencer.

This module holds the active EPSG code and coordinate bounds, updated at
runtime by auto_georeference.set_active_crs_preset().  Both the main pipeline
and the georef_core submodules (ranker, candidate_generation) read from here
so that a single call to set_active_crs_preset() propagates everywhere.
"""
from __future__ import annotations

# Active projected CRS code — updated by set_active_crs_preset()
ACTIVE_EPSG: int = 25832

# Valid coordinate bounds for the active CRS (used by ranker + OCR label expansion)
# (e_min, e_max, n_min, n_max)
BOUNDS: tuple[float, float, float, float] = (
    280_000.0, 920_000.0, 5_200_000.0, 6_100_000.0
)


def get_active_epsg() -> int:
    return ACTIVE_EPSG


def get_active_bounds() -> tuple[float, float, float, float]:
    """Return (e_min, e_max, n_min, n_max) for the active projected CRS."""
    return BOUNDS
