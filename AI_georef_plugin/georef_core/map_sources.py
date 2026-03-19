"""
map_sources.py
--------------
Persistent registry of WMS reference-map sources.

Sources are stored in map_sources.json inside the output directory.
On first run (no file present) the registry is pre-populated with all
built-in presets.  Any source — including presets — can be removed; removed
presets can be re-added at any time via get_available_presets().

OSM (terrestris) is always available as a preset and acts as the universal
fallback when the registry is empty.

Security
--------
  • Only http:// / https:// URLs are accepted; file://, ftp://, etc. are
    silently dropped on load.
  • bgcolor is validated against ^(0x[0-9A-Fa-f]{6})?$ to prevent
    parameter injection into WMS tile request URLs.
  • wms_version is allowlisted to "1.3.0" | "1.1.1".
  • format is allowlisted to "image/jpeg" | "image/png".

Usage
-----
  from georef_core.map_sources import (
      load_map_sources, save_map_sources,
      add_map_source, remove_map_source,
      get_available_presets, preset_source,
      as_wms_configs, new_source_id, MapSource,
  )

  sources = load_map_sources(path)              # dict[id -> MapSource]
  sources = add_map_source(sources, entry)
  save_map_sources(sources, path)
  missing_presets = get_available_presets(sources)   # presets not yet active
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Validates bgcolor field — prevents parameter injection in WMS URLs
_BGCOLOR_RE = re.compile(r"^(0x[0-9A-Fa-f]{6})?$")

MAP_SOURCES_FILENAME = "map_sources.json"


# ---------------------------------------------------------------------------
# Preset catalog  (mirrors WMS_CONFIGS in auto_georeference.py)
# These are the factory defaults shown on first launch.
# ---------------------------------------------------------------------------
_PRESET_DEFS: list[dict[str, Any]] = [
    {
        "id":          "aerial",
        "name":        "NRW DOP – aerial orthophoto",
        "url":         "https://www.wms.nrw.de/geobasis/wms_nw_dop",
        "layer":       "nw_dop_rgb",
        "format":      "image/jpeg",
        "wms_version": "1.3.0",
        "bgcolor":     "",
        "description": "NRW true orthophoto (Digital Orthophoto).  Best for aerial/site plans.",
        "builtin":     True,
    },
    {
        "id":          "topo",
        "name":        "NRW DTK – topographic map",
        "url":         "https://www.wms.nrw.de/geobasis/wms_nw_dtk",
        "layer":       "nw_dtk_col",
        "format":      "image/jpeg",
        "wms_version": "1.3.0",
        "bgcolor":     "",
        "description": "NRW digital topographic map.  Best for topo/hiking-style plans.",
        "builtin":     True,
    },
    {
        "id":          "vector",
        "name":        "NRW ALKIS – cadastral/vector",
        "url":         "https://www.wms.nrw.de/geobasis/wms_nw_alkis",
        "layer":       "adv_alkis_flurstuecke,adv_alkis_verkehr,adv_alkis_gebaeude,adv_alkis_gewaesser",
        "format":      "image/png",
        "wms_version": "1.1.1",
        "bgcolor":     "0xFFFFFF",
        "description": "NRW ALKIS cadastral – parcels, roads, buildings, water.  Best for engineering/CAD plans.",
        "builtin":     True,
    },
    {
        "id":          "osm",
        "name":        "OSM – OpenStreetMap (universal fallback)",
        "url":         "https://ows.terrestris.de/osm/service",
        "layer":       "OSM-WMS",
        "format":      "image/jpeg",
        "wms_version": "1.3.0",
        "bgcolor":     "",
        "description": "OpenStreetMap via terrestris.  Works globally without regional restrictions.",
        "builtin":     True,
    },
]

# Legacy alias: select_wms_config() previously used "vector_osm" for OSM
_LEGACY_KEY_MAP = {"vector_osm": "osm"}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class MapSource:
    id:          str
    name:        str
    url:         str
    layer:       str
    format:      str  = "image/jpeg"
    wms_version: str  = "1.3.0"
    bgcolor:     str  = ""
    description: str  = ""
    builtin:     bool = False   # True = originally a preset (informational only)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MapSource":
        raw_url    = str(d.get("url") or "")
        raw_bgcolor = str(d.get("bgcolor") or "")
        # Security: reject non-http/https URLs and malformed bgcolor values
        if raw_url and not raw_url.lower().startswith(("http://", "https://")):
            raw_url = ""
        if not _BGCOLOR_RE.match(raw_bgcolor):
            raw_bgcolor = ""
        # Allowlist wms_version
        raw_version = str(d.get("wms_version") or "1.3.0")
        if raw_version not in ("1.3.0", "1.1.1"):
            raw_version = "1.3.0"
        # Allowlist format
        raw_format = str(d.get("format") or "image/jpeg")
        if raw_format not in ("image/jpeg", "image/png"):
            raw_format = "image/jpeg"
        return cls(
            id=          str(d["id"]),
            name=        str(d.get("name") or d["id"]),
            url=         raw_url,
            layer=       str(d.get("layer") or ""),
            format=      raw_format,
            wms_version= raw_version,
            bgcolor=     raw_bgcolor,
            description= str(d.get("description") or ""),
            builtin=     bool(d.get("builtin", False)),
        )


# ---------------------------------------------------------------------------
# Preset catalog helpers
# ---------------------------------------------------------------------------
def _all_presets() -> dict[str, MapSource]:
    """Return all factory presets as a dict (ordered)."""
    return {d["id"]: MapSource.from_dict(d) for d in _PRESET_DEFS}


def get_available_presets(
    current_sources: dict[str, MapSource],
) -> list[MapSource]:
    """Return presets that are NOT currently in *current_sources* (can be re-added)."""
    return [
        MapSource.from_dict(d)
        for d in _PRESET_DEFS
        if d["id"] not in current_sources
    ]


def preset_source(preset_id: str) -> MapSource | None:
    """Return a fresh copy of the preset with the given id, or None."""
    for d in _PRESET_DEFS:
        if d["id"] == preset_id:
            return MapSource.from_dict(d)
    return None


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_map_sources(path: Path) -> dict[str, MapSource]:
    """
    Load the map-source registry from *path*.

    If the file exists, its contents are used as-is (sources the user has
    explicitly kept).  If the file is missing (first run), the full preset
    catalog is returned so the caller can present the defaults and then save.
    """
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            sources: dict[str, MapSource] = {}
            for entry in (raw if isinstance(raw, list) else []):
                ms = MapSource.from_dict(entry)
                if ms.url:   # skip entries whose URL was stripped by security checks
                    sources[ms.id] = ms
            return sources
        except Exception:
            pass
    # First run — seed with all presets
    return _all_presets()


def save_map_sources(sources: dict[str, MapSource], path: Path) -> None:
    """Persist the full current registry (all sources, including presets) to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([v.to_dict() for v in sources.values()], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Mutation helpers (return new dicts — sources are treated as immutable)
# ---------------------------------------------------------------------------
def add_map_source(
    sources: dict[str, MapSource], entry: MapSource
) -> dict[str, MapSource]:
    """Return updated dict with *entry* added (or replaced if same id)."""
    return {**sources, entry.id: entry}


def remove_map_source(
    sources: dict[str, MapSource], source_id: str
) -> dict[str, MapSource]:
    """Return updated dict without *source_id*.  Any source may be removed."""
    return {k: v for k, v in sources.items() if k != source_id}


# ---------------------------------------------------------------------------
# Conversion to WMS_CONFIGS format
# ---------------------------------------------------------------------------
def as_wms_configs(sources: dict[str, MapSource]) -> dict[str, dict]:
    """
    Convert to the dict format consumed by auto_georeference.WMS_CONFIGS /
    WMS_CONFIGS_EXTRA.
    """
    result: dict[str, dict] = {}
    for sid, ms in sources.items():
        result[sid] = {
            "url":         ms.url,
            "layer":       ms.layer,
            "format":      ms.format,
            "label":       ms.name,
            "wms_version": ms.wms_version,
            "bgcolor":     ms.bgcolor,
        }
        # Register legacy alias so existing config keys still resolve
        for legacy, current in _LEGACY_KEY_MAP.items():
            if sid == current:
                result[legacy] = result[sid]
    return result


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def new_source_id() -> str:
    """Generate a unique ID for a new custom source."""
    return f"custom_{uuid.uuid4().hex[:8]}"
