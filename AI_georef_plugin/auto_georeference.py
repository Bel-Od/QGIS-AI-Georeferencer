"""
auto_georeference.py
--------------------
Automatically georeferences a plan TIFF by:
  1. OCR  – extracts all text (coordinate grid labels, title block, etc.)
  2. OpenAI Vision AI – analyzes a thumbnail to locate grid lines, corner
     coordinates, CRS, scale, and north arrow
  3. GCP builder – pairs pixel positions with real-world coordinates
  4. GDAL georeferencing – writes a georeferenced GeoTIFF

Designed for QGIS Python console OR standalone with OSGeo4W / conda-qgis env.

Dependencies (all present in a standard QGIS/OSGeo4W install + extras):
    osgeo.gdal        – bundled with QGIS
    Pillow            – pip install Pillow
    pytesseract       – pip install pytesseract  (+ install Tesseract binary)
    openai            – pip install openai        (for Vision AI step)
    pyproj            – bundled with QGIS
"""

import os
import sys
import re
import json
import base64
import hashlib
import tempfile
import ssl
import math
import time
import threading
import unicodedata
from pathlib import Path

from georef_core.location_hints import (
    _address_is_specific as _location_address_is_specific,
    _classify_address_confidence as _location_classify_address_confidence,
    _extract_project_city as _location_extract_project_city,
    _extract_road_codes as _location_extract_road_codes,
    extract_structured_location_hints as _build_structured_location_hints,
)
from georef_core.text_parsing import (
    _clean_text_for_match as _text_clean_text_for_match,
    _extract_best_scale as _text_extract_best_scale,
    _extract_scale_candidates as _text_extract_scale_candidates,
    _merge_text_sources as _text_merge_text_sources,
    _parse_scale_value as _text_parse_scale_value,
    parse_coordinates as _text_parse_coordinates,
)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
# __file__ is not defined when exec()'d from the QGIS Python console.
# When installed as a QGIS plugin, __file__ resolves to the plugin folder.
# We fall back to a writable user directory so output files are never written
# into a read-only system path.
try:
    _ROOT = Path(__file__).parent
except NameError:
    _ROOT = Path.home() / "AutoGeoref"

# Default output and plan dirs — the plugin dialog overrides these at runtime.
# Using a user-writable location ensures the plugin works from any install path.
_DEFAULT_WORK_ROOT = Path.home() / "AutoGeoref"

# ---------------------------------------------------------------------------
# Load .env file (zero-dependency — no python-dotenv needed).
# Values are only applied when the env var is not already set, so a real
# system env var always wins over the .env file.
# ---------------------------------------------------------------------------
def _load_dotenv(path: Path) -> None:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except FileNotFoundError:
        pass

# Load .env from multiple locations: plugin dir, project root, user home
for _env_candidate in [_ROOT / ".env", _DEFAULT_WORK_ROOT / ".env", Path.home() / ".auto_georef.env"]:
    _load_dotenv(_env_candidate)

# Plan and output dirs: prefer a user-writable location so the plugin works
# when installed into a read-only QGIS plugin directory.
# The dialog overrides these at runtime with the user's selected paths.
_WORK_ROOT  = _ROOT if (_ROOT / "plan").exists() else _DEFAULT_WORK_ROOT
PLAN_FOLDER = _WORK_ROOT / "plan"
PDF_FOLDER  = _WORK_ROOT / "pdfs"
PLAN_FOLDER.mkdir(parents=True, exist_ok=True)

# Input priority:
#   1. PDF in plan/    (put a PDF here to process it)
#   2. TIFF in plan/   (put a TIFF here to process it)
#   3. PDF in pdfs/    (fallback batch folder – first alphabetically)
_plan_pdf  = next(PLAN_FOLDER.glob("*.pdf"), None)
_plan_tif  = next(PLAN_FOLDER.glob("*.tif"), None) or next(PLAN_FOLDER.glob("*.tiff"), None)
_pdfs_pdf  = next(iter(sorted(PDF_FOLDER.glob("*.pdf"))) if PDF_FOLDER.exists() else iter([]), None)

INPUT_PATH    = _plan_pdf or _plan_tif or _pdfs_pdf
INPUT_IS_PDF  = INPUT_PATH is not None and INPUT_PATH.suffix.lower() == ".pdf"
TIFF_PATH     = _plan_tif   # kept for back-compat with legacy callers

OUTPUT_DIR  = _WORK_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Log file – written from thread so output is never lost
LOG_FILE = OUTPUT_DIR / "run.log"

# Cancellation flag – set by the plugin's Stop button to interrupt a running job.
# The worker checks this between tile downloads and aborts gracefully.
_CANCEL_FLAG: threading.Event = threading.Event()

import builtins as _builtins
_orig_print = _builtins.print
_LOG_MAX_BYTES = 5 * 1024 * 1024  # rotate at 5 MB to prevent unbounded growth

def print(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    _orig_print(msg, **kwargs)
    try:
        # Rotate: rename run.log → run.log.1 when it exceeds 5 MB
        if LOG_FILE.exists() and LOG_FILE.stat().st_size > _LOG_MAX_BYTES:
            _backup = LOG_FILE.with_suffix(".log.1")
            try:
                _backup.unlink(missing_ok=True)
                LOG_FILE.rename(_backup)
            except Exception:
                pass
        with open(LOG_FILE, "a", encoding="utf-8") as _lf:
            _lf.write(msg + "\n")
    except Exception:
        pass  # never let a log-write failure crash the pipeline

# Active target CRS — mutated by set_active_crs_preset() when the user
# switches projection in the dialog.  Default: EPSG:25832 (Germany UTM32N).
TARGET_EPSG: int = 25832

# Set your OpenAI API key via the environment variable (recommended — keeps the
# key out of source files and git history):
#   Windows cmd:      set OPENAI_API_KEY=sk-...
#   OSGeo4W shell:    export OPENAI_API_KEY=sk-...
#   PowerShell:       $env:OPENAI_API_KEY="sk-..."
# The key is read at import time.  If it is missing, Vision AI calls will fail
# with a clear error rather than silently using a stale or shared key.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    import warnings as _warnings
    _warnings.warn(
        "OPENAI_API_KEY environment variable is not set. "
        "Vision AI (GPT-4o) calls will fail. "
        "Set it before running: set OPENAI_API_KEY=sk-...",
        stacklevel=1,
    )

# OpenAI model to use for vision analysis
OPENAI_MODEL = "gpt-4o"

# Common Tesseract install locations (searched if TESSERACT_CMD env var is not set)
_TESSERACT_COMMON = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"D:\Program Files\Tesseract-OCR\tesseract.exe",
    r"D:\Program Files\Tesseract\tesseract.exe",
    r"C:\Program Files\Tesseract\tesseract.exe",
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
]
MANUAL_SEED_FILE       = OUTPUT_DIR / "manual_seed.json"      # user-written overrides only
LAST_RESULT_FILE       = OUTPUT_DIR / "last_result.json"      # auto-written only for trusted results
PROJECT_ADDRESS_FILE   = OUTPUT_DIR / "project_address.json"  # optional address hint
GEOREF_LIBRARY_FILE    = OUTPUT_DIR / "georef_library.json"   # operator-verified per-plan ground truth

# Thumbnail size sent to the model (keeps tokens/cost low)
THUMB_MAX_PX = 2000

# ---------------------------------------------------------------------------
# WMS REFINEMENT (optional)
# After initial coordinate-label georeferencing, download a reference WMS
# orthophoto and use phase correlation to find any remaining translation offset.
# Requires: numpy (bundled with QGIS), internet access.
# Set to False to skip this step.
# ---------------------------------------------------------------------------
ENABLE_WMS_REFINEMENT = True    # required for plans without a coordinate grid

# ---------------------------------------------------------------------------
# CRS PRESETS — each entry defines a supported output projection and its
# associated WMS reference services.  The active preset is switched at
# runtime by set_active_crs_preset() (called from the dialog CRS selector).
#
# Keys per preset:
#   label     – human-readable name shown in the dialog dropdown
#   epsg      – EPSG code of the projected CRS
#   e_min/e_max/n_min/n_max – valid coordinate bounds (used for OCR label
#               expansion and plausibility checks)
#   wms       – dict of plan-type-keyed WMS configs (aerial/topo/vector/…)
#               Leave empty ({}) to disable WMS refinement for this preset.
# ---------------------------------------------------------------------------
# CRS_PRESETS is populated at startup by _load_geopacks() which reads all
# installed .geopack.json files.  Only the "custom" entry is built-in;
# every real regional preset (EPSG codes, WMS, bounds) comes from a pack.
CRS_PRESETS: dict = {
    "custom": {
        "label": "Custom EPSG…",
        "epsg":  25832,          # placeholder — replaced by dialog input
        "e_min": -10_000_000.0, "e_max": 10_000_000.0,
        "n_min": -10_000_000.0, "n_max": 10_000_000.0,
        "wms": {},
    },
}

# Active preset key — mutated by set_active_crs_preset()
# Initialized to "custom"; _load_geopacks() will switch to the default
# geopack preset (e.g. "epsg_25832") once packs are discovered.
ACTIVE_CRS_PRESET_KEY: str = "custom"

# ---------------------------------------------------------------------------
# WMS catalogue – derived from the active CRS preset.
# select_wms_config() updates WMS_URL / WMS_LAYER from this dict.
# set_active_crs_preset() replaces this dict when the preset changes.
# ---------------------------------------------------------------------------
WMS_CONFIGS: dict = {}

# Active WMS endpoints – updated by select_wms_config() and set_active_crs_preset()
ACTIVE_WMS_CONFIG_KEY = "aerial"
WMS_URL     = ""
WMS_LAYER   = ""
WMS_FORMAT  = "image/jpeg"
WMS_VERSION = "1.3.0"
WMS_BGCOLOR = ""
# Maximum plausible refinement shift in metres.  Plans without a coordinate
# grid rely entirely on WMS to locate them, so allow up to 10 km shift from seed.
WMS_MAX_SHIFT_M = 10_000
WMS_SEARCH_EXPAND = 1.8
WMS_COARSE_RANGE = 2          # 5→2: 25 tiles instead of 121
WMS_FINE_RANGE = 1            # 2→1: 9 fine tiles per coarse seed
WMS_FINE_STEP = 0.35
WMS_REF_PX = 512              # higher resolution → better positional accuracy (~2× improvement)
WMS_MIN_CONFIDENCE = 1.08
WMS_MIN_SCORE = 0.18
WMS_MIN_SCORE_GAP = 0.015
WMS_MAX_ROTATION_DEG = 12.0
WMS_LARGE_ROTATION_DEG = 95.0
WMS_COARSE_TOP_K = 3          # use more coarse seeds so fine search sees more nearby evidence
WMS_TILE_TIMEOUT = 12         # seconds per WMS tile request
WMS_PARALLEL_WORKERS = 8      # concurrent tile downloads
LOCAL_REFERENCE_RASTER = OUTPUT_DIR / "reference_raster.tif"
WMS_LOCAL_REFINEMENT_MAX_SHIFT_M = 900.0

# Pre-captured QGIS canvas info (set by the plugin dialog on the main thread
# before the worker thread starts -- canvas.grab() cannot be called off-thread).
# None = not running from the plugin / not yet captured.
CANVAS_INFO_OVERRIDE = None
USE_QGIS_CANVAS_VISION_CONTEXT = False
QGIS_CANVAS_SEED_PRIORITY = "primary"

# WMS reference map override – set by the plugin dialog when the user picks
# a specific map type.  One of "aerial" / "topo" / "vector" / "osm" /
# "custom_XXXX" / None (auto).
WMS_CONFIG_OVERRIDE = None

# Extra WMS configs injected by the plugin dialog from the user's map-source
# registry (map_sources.json).  Merged with WMS_CONFIGS at runtime so
# select_wms_config() can resolve user-added or renamed built-in keys.
WMS_CONFIGS_EXTRA: dict = {}


def set_active_crs_preset(preset_key: str, custom_epsg: int | None = None) -> None:
    """
    Switch the active CRS preset at runtime.

    Updates the module-level globals TARGET_EPSG, _E_MIN/_E_MAX/_N_MIN/_N_MAX,
    WMS_CONFIGS, and all WMS endpoint globals.  Also clears the Nominatim
    geocoding cache so stale projected coordinates are not reused.

    Called by the dialog when the user changes the Projection dropdown.

    Parameters
    ----------
    preset_key   : key in CRS_PRESETS, e.g. "epsg_25832", "epsg_28992",
                   or "custom"
    custom_epsg  : required (and used) only when preset_key == "custom"
    """
    global TARGET_EPSG, _E_MIN, _E_MAX, _N_MIN, _N_MAX, ACTIVE_CRS_PRESET_KEY
    global WMS_CONFIGS, ACTIVE_WMS_CONFIG_KEY
    global WMS_URL, WMS_LAYER, WMS_FORMAT, WMS_VERSION, WMS_BGCOLOR

    if preset_key == "custom":
        if custom_epsg is None:
            raise ValueError("custom_epsg must be provided when preset_key='custom'")
        preset = dict(CRS_PRESETS.get("custom", {}))
        preset["epsg"] = int(custom_epsg)
    elif preset_key in CRS_PRESETS:
        preset = CRS_PRESETS[preset_key]
    else:
        raise ValueError(f"Unknown CRS preset: {preset_key!r}. "
                         f"Valid keys: {list(CRS_PRESETS)}")

    ACTIVE_CRS_PRESET_KEY = preset_key
    TARGET_EPSG = int(preset["epsg"])
    _E_MIN = float(preset.get("e_min", -10_000_000.0))
    _E_MAX = float(preset.get("e_max",  10_000_000.0))
    _N_MIN = float(preset.get("n_min", -10_000_000.0))
    _N_MAX = float(preset.get("n_max",  10_000_000.0))

    # Replace WMS catalogue with the preset's services
    WMS_CONFIGS = dict(preset.get("wms") or {})
    if WMS_CONFIGS:
        # Try to keep the previously-selected plan type (aerial/topo/vector)
        _key = ACTIVE_WMS_CONFIG_KEY if ACTIVE_WMS_CONFIG_KEY in WMS_CONFIGS else next(iter(WMS_CONFIGS))
        ACTIVE_WMS_CONFIG_KEY = _key
        _wms = WMS_CONFIGS[_key]
        WMS_URL     = _wms.get("url", "")
        WMS_LAYER   = _wms.get("layer", "")
        WMS_FORMAT  = _wms.get("format", "image/jpeg")
        WMS_VERSION = _wms.get("wms_version", "1.3.0")
        WMS_BGCOLOR = _wms.get("bgcolor", "")
    else:
        # No WMS for this region — WMS refinement will be skipped
        WMS_URL = WMS_LAYER = WMS_FORMAT = WMS_VERSION = WMS_BGCOLOR = ""

    # Invalidate geocoding cache — coordinates must be re-projected
    _GEOCODE_CACHE.clear()

    # Propagate to the shared crs_config module used by ranker + candidate gen
    try:
        from georef_core import crs_config as _crs_cfg
        _crs_cfg.ACTIVE_EPSG = TARGET_EPSG
        _crs_cfg.BOUNDS = (_E_MIN, _E_MAX, _N_MIN, _N_MAX)
    except ImportError:
        pass

    print(f"[i] Active CRS: {preset.get('label', preset_key)}  (EPSG:{TARGET_EPSG})")
    if WMS_URL:
        print(f"[i] WMS: {WMS_URL}  layer={WMS_LAYER}")
    else:
        print(f"[i] WMS refinement disabled — no WMS configured for this CRS preset")


# When True the plugin dialog handles layer loading on the main thread.
# Set to True by the dialog before starting the worker so load_in_qgis()
# (which is called from inside the worker thread) is skipped -- calling Qt UI
# operations from a background thread causes a hard freeze / crash.
SKIP_AUTO_LAYER_LOAD = False
STRICT_WMS_VALIDATION = True
ENABLE_OSM_VECTOR_SNAPPING = True
GEOCODE_DEBUG = False

# User-supplied scale denominator override (set by the plugin dialog).
# None = auto-detect from Vision AI / OCR / title block scan.
# Example: 1000 → forces 1:1000 scale for the geotransform calculation.
SCALE_OVERRIDE = None
LAST_VISION_RESULT = {}
# Centre of the OCR place-name cluster from the most recent run.
# Set when _geocode_ocr_place_names succeeds; used as tiebreaker for
# ambiguous NCC candidates.
LAST_OCR_PLACES_CENTROID: tuple[float, float] | None = None
# 'city'    → centroid from Standort geocode (city-level, ±5 km)
# 'feature' → centroid aggregated from multiple place-name geocodes (±1–2 km)
LAST_OCR_PLACES_CENTROID_PRECISION: str = "feature"
LAST_GEOREF_QUALITY: dict = {}
LAST_STRUCTURED_LOCATION_HINTS: dict = {}
LAST_OSM_SNAP_INFO: dict = {}
CURRENT_SEED_SOURCE: str = ""
CURRENT_SEED_CONFIDENCE: str = ""


def _artifact_dir() -> Path:
    """
    Keep intermediates inside `_work/` even for standalone runs.
    When OUTPUT_DIR already points to `_work`, do not create nested folders.
    """
    out = Path(OUTPUT_DIR)
    if out.name.lower() == "_work":
        out.mkdir(parents=True, exist_ok=True)
        return out
    work = out / "_work"
    work.mkdir(parents=True, exist_ok=True)
    return work


def _artifact_path(name: str) -> Path:
    return _artifact_dir() / name


def _cache_dir(namespace: str | None = None) -> Path:
    base = _artifact_dir() / "cache"
    base.mkdir(parents=True, exist_ok=True)
    if namespace:
        base = base / namespace
        base.mkdir(parents=True, exist_ok=True)
    return base


def _cache_key(*parts) -> str:
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _path_fingerprint(path: Path) -> dict:
    try:
        stat = Path(path).stat()
        return {
            "path": str(Path(path).resolve()),
            "size": int(stat.st_size),
            "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
        }
    except Exception:
        return {"path": str(path), "size": -1, "mtime_ns": -1}


def _cache_text_path(namespace: str, key: str, suffix: str = ".txt") -> Path:
    return _cache_dir(namespace) / f"{key}{suffix}"


def _cache_read_text(namespace: str, key: str, suffix: str = ".txt") -> str | None:
    target = _cache_text_path(namespace, key, suffix)
    try:
        if target.exists():
            return target.read_text(encoding="utf-8")
    except Exception:
        return None
    return None


def _cache_write_text(namespace: str, key: str, text: str, suffix: str = ".txt") -> Path | None:
    target = _cache_text_path(namespace, key, suffix)
    try:
        target.write_text(text, encoding="utf-8")
        return target
    except Exception:
        return None


def _cache_read_json(namespace: str, key: str) -> object | None:
    raw = _cache_read_text(namespace, key, suffix=".json")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _cache_write_json(namespace: str, key: str, payload: object) -> Path | None:
    return _cache_write_text(
        namespace,
        key,
        json.dumps(payload, indent=2, ensure_ascii=False),
        suffix=".json",
    )


def _cache_binary_path(namespace: str, key: str, suffix: str) -> Path:
    return _cache_dir(namespace) / f"{key}{suffix}"


def _fetch_url_bytes_cached(
    url: str,
    *,
    namespace: str,
    timeout: int,
    headers: dict | None = None,
    context=None,
    suffix: str = ".bin",
) -> tuple[bytes, bool]:
    cache_key = _cache_key("url-bytes", namespace, url)
    cache_path = _cache_binary_path(namespace, cache_key, suffix)
    try:
        if cache_path.exists():
            return cache_path.read_bytes(), True
    except Exception:
        pass
    import urllib.request

    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
        data = resp.read()
    try:
        cache_path.write_bytes(data)
    except Exception:
        pass
    return data, False


def _clean_text_for_match(text: str) -> str:
    return (text or "").replace("ä", "a").replace("ö", "o").replace("ü", "u").replace("ß", "ss")


def _parse_scale_value(raw: str) -> int | None:
    cleaned = re.sub(r"[^\d]", "", raw or "")
    if not cleaned:
        return None
    try:
        scale_val = int(cleaned)
    except ValueError:
        return None
    if 100 <= scale_val <= 100_000:
        return scale_val
    return None


def _extract_scale_candidates(text: str) -> list[tuple[int, float, str]]:
    candidates: list[tuple[int, float, str]] = []
    if not text:
        return candidates
    lines = text.splitlines()
    pat = re.compile(r"(?<!\d)1\s*:\s*([0-9][0-9.\s]{1,12})(?!\d)")
    for idx, line in enumerate(lines):
        for match in pat.finditer(line):
            scale_val = _parse_scale_value(match.group(1))
            if scale_val is None:
                continue
            score = 1.0
            line_norm = _clean_text_for_match(line.lower())
            prev_norm = _clean_text_for_match(lines[idx - 1].lower()) if idx > 0 else ""
            next_norm = _clean_text_for_match(lines[idx + 1].lower()) if idx + 1 < len(lines) else ""
            context = "inline"
            if any(tok in line_norm for tok in ("massstab", "m 1:", "m=", "scale")):
                score += 2.5
                context = "titleblock"
            elif any(tok in prev_norm for tok in ("massstab", "m 1:", "m=", "scale")):
                score += 2.0
                context = "titleblock-prev"
            elif any(tok in next_norm for tok in ("massstab", "m 1:", "m=", "scale")):
                score += 1.5
                context = "titleblock-next"
            if any(tok in line_norm for tok in ("schnitt", "detail", "schema", "schematisch", "querschnitt", "längsschnitt", "langsschnitt")):
                score -= 1.5
                context = f"{context}+detail"
            if scale_val < 200:
                score -= 0.5
            candidates.append((scale_val, score, context))
    candidates.sort(key=lambda item: (item[1], item[0]), reverse=True)
    return candidates


def _extract_best_scale(text: str) -> int | None:
    candidates = _extract_scale_candidates(text)
    if candidates:
        return candidates[0][0]
    return None


def _extract_project_city(project_address: str) -> str | None:
    if not project_address:
        return None
    _pc_m = re.search(r"\b\d{5}\s+([A-ZÄÖÜ][A-Za-zäöüßÄÖÜ\-/ ]+)", project_address or "")
    if _pc_m:
        return _pc_m.group(1).split(",")[0].strip()
    return None


def _address_is_specific(address: str) -> bool:
    if not address:
        return False
    has_number = bool(re.search(r"\b\d+[A-Za-z]?\b", address))
    has_street_token = bool(re.search(
        r"(straße|strasse|str\.|weg|allee|platz|gasse|ufer|chaussee|ring|damm|pfad|kamp|wall|markt)",
        address,
        re.IGNORECASE,
    ))
    return has_number and has_street_token


def _classify_address_confidence(address: str) -> str:
    if not address:
        return "city"
    if _address_is_specific(address):
        return "address"
    has_postcode = bool(re.search(r"\b\d{5}\b", address))
    has_street_token = bool(re.search(
        r"(straße|strasse|str\.|weg|allee|platz|gasse|ufer|chaussee|ring|damm|pfad|kamp|wall|markt)",
        address,
        re.IGNORECASE,
    ))
    if has_street_token:
        return "street"
    if has_postcode:
        return "postcode"
    return "city"


def _extract_road_codes(text: str) -> list[str]:
    """Extract German road identifiers from text.

    Matches Autobahnen (A), Bundesstraßen (B), Landesstraßen (L),
    Kreisstraßen (K), and Staatsstraßen (S).  Handles both compact
    ("L663") and spaced ("L 663") forms.

    Filters out:
    - DIN/paper format codes: "DIN A1", "(A1)", "Blatt A3", etc.
    - Leading-zero variants: A01, B001 (not valid German road codes)
    """
    if not text:
        return []
    codes: list[str] = []
    # Match A/B/L/K/S followed by optional whitespace and 1-4 digits (+ optional letter suffix)
    for match in re.finditer(r"\b([ABLKS])\s*(\d{1,4}[a-z]?)\b", text, re.IGNORECASE):
        letter = match.group(1).upper()
        digits = match.group(2)
        code = f"{letter}{digits}"
        # Skip leading-zero variants (A01, B001) — not valid road codes
        if len(digits) >= 2 and digits[0] == "0":
            continue
        start = max(0, match.start() - 20)
        end   = min(len(text), match.end() + 20)
        context = text[start:end].lower()
        # Skip DIN/paper format references: "DIN A1", "Blatt A3", "(A1)", "Format A4"
        if any(tok in context for tok in ("din a", "blatt a", "format a", "paper a", "size a")):
            continue
        # Skip parenthesized format codes: "(A1)", "(A3)" — paper sizes
        if re.search(r'\(\s*' + re.escape(letter) + r'\s*' + re.escape(digits) + r'\s*\)',
                     text[max(0, match.start() - 3):match.end() + 3]):
            continue
        codes.append(code)
    return list(dict.fromkeys(codes))


def _last_result_is_trusted(data: dict) -> bool:
    quality = data.get("quality")
    if not isinstance(quality, dict):
        return False
    if quality.get("accepted") is not True:
        return False
    if quality.get("provisional") is True:
        return False
    reason = quality.get("acceptance_reason")
    if reason in {
        "ocr_grid_regression",
        "patch_consensus",
        "feature_refinement",
    }:
        return True
    return reason == "ncc_thresholds" and quality.get("has_coord_anchors") is True

# ---------------------------------------------------------------------------
# IMPORTS – graceful degradation
# ---------------------------------------------------------------------------
try:
    from osgeo import gdal, osr
    gdal.UseExceptions()
    HAS_GDAL = True
    print("[✓] GDAL available")
except ImportError:
    HAS_GDAL = False
    print("[✗] GDAL not found – georeferencing output disabled")

try:
    from PIL import Image
    HAS_PIL = True
    print("[✓] Pillow available")
except ImportError:
    HAS_PIL = False
    print("[✗] Pillow not found  →  pip install Pillow")

try:
    import pytesseract
    _tess_candidates = [
        os.environ.get("TESSERACT_CMD", "").strip(),
        r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(
            os.environ.get("USERNAME", "")
        ),
    ] + _TESSERACT_COMMON
    try:
        import subprocess as _sp
        _where = _sp.run(["where", "tesseract"], capture_output=True, text=True, check=False)
        _tess_candidates.extend(l.strip() for l in _where.stdout.splitlines() if l.strip())
    except Exception:
        pass
    for _candidate in _tess_candidates:
        if _candidate and os.path.exists(_candidate):
            pytesseract.pytesseract.tesseract_cmd = _candidate
            break
    pytesseract.get_tesseract_version()
    HAS_TESSERACT = True
    print(f"[✓] Tesseract available: {pytesseract.pytesseract.tesseract_cmd}")
except Exception:
    HAS_TESSERACT = False
    print("[✗] Tesseract binary not found.")
    print("    Download: https://github.com/UB-Mannheim/tesseract/wiki")
    print("    Then set the path in the plugin Setup tab.")

try:
    from openai import OpenAI
    HAS_OPENAI = bool(OPENAI_API_KEY)
    print(f"[{'✓' if HAS_OPENAI else '✗'}] OpenAI SDK {'ready' if HAS_OPENAI else 'found but no API key set'}")
except ImportError:
    HAS_OPENAI = False
    print("[✗] openai not found  →  pip install openai")

def _disable_cv2(reason, exc=None):
    global cv2, HAS_CV2
    cv2 = None
    HAS_CV2 = False
    if exc is not None:
        print(f"[i] OpenCV disabled - {reason}: {exc}")
    else:
        print(f"[i] OpenCV disabled - {reason}")


try:
    import cv2 as _cv2
    try:
        import numpy as _np
        # Smoke-test the extension against the active NumPy ABI before enabling
        # any optional cv2-based refinement paths.
        _ = _cv2.__version__
        _ = _cv2.cvtColor(_np.zeros((1, 1, 3), dtype=_np.uint8), _cv2.COLOR_RGB2GRAY)
        cv2 = _cv2
        HAS_CV2 = True
        print("[✓] OpenCV available")
    except Exception as _cv2_probe_exc:
        _disable_cv2("incompatible with the current QGIS/NumPy runtime; advanced affine refinement disabled", _cv2_probe_exc)
except Exception as _cv2_import_exc:
    _disable_cv2("not available; advanced affine refinement disabled", _cv2_import_exc)

try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

try:
    import fitz as pymupdf  # pymupdf – PDF rendering
    HAS_FITZ = True
except ImportError:
    pymupdf = None
    HAS_FITZ = False


# ---------------------------------------------------------------------------
# STEP 1 – Read basic TIFF metadata with GDAL
# ---------------------------------------------------------------------------
def read_tiff_metadata(path: Path) -> dict:
    """Return existing geotransform / CRS if the TIFF is already geo-tagged."""
    info = {"path": str(path), "already_georef": False}
    if not HAS_GDAL:
        return info
    ds = gdal.Open(str(path))
    if ds is None:
        raise FileNotFoundError(f"GDAL cannot open {path}")
    info["width"]  = ds.RasterXSize
    info["height"] = ds.RasterYSize
    info["bands"]  = ds.RasterCount
    gt = ds.GetGeoTransform()
    info["geotransform"] = gt
    info["projection"]   = ds.GetProjection()
    if gt != (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
        info["already_georef"] = True
        print("[i] TIFF already has a geotransform – will refine with found GCPs")
    print(f"[i] Image size: {info['width']} × {info['height']} px, {info['bands']} band(s)")
    if HAS_PIL:
        try:
            with Image.open(str(path)) as img:
                dpi = img.info.get("dpi")
                if dpi:
                    info["dpi_x"], info["dpi_y"] = float(dpi[0]), float(dpi[1])
                    print(f"[i] TIFF DPI: {info['dpi_x']:.1f} × {info['dpi_y']:.1f}")
        except Exception:
            pass
    ds = None
    return info


# ---------------------------------------------------------------------------
# PDF SUPPORT – render PDF page to PIL image + extract text layer
# ---------------------------------------------------------------------------
PDF_RENDER_DPI = 300  # DPI for rasterising vector PDFs


def render_pdf_to_image(pdf_path: Path, page_index: int = 0) -> "Image.Image":
    """
    Render a PDF page to a PIL Image using pymupdf.
    Returns a high-resolution RGB image suitable for the georeferencing pipeline.
    """
    if not HAS_FITZ:
        raise RuntimeError("pymupdf not installed – run: pip install pymupdf")
    doc = pymupdf.open(str(pdf_path))
    page = doc[page_index]
    mat = pymupdf.Matrix(PDF_RENDER_DPI / 72, PDF_RENDER_DPI / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    doc.close()
    # Convert fitz Pixmap → PIL Image without writing to disk
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return img


def render_pdf_to_tiff_cached(pdf_path: Path, page_index: int = 0, dest_path: Path | None = None) -> Path:
    cache_key = _cache_key("pdf-render", _path_fingerprint(pdf_path), page_index, PDF_RENDER_DPI)
    cache_path = _cache_binary_path("pdf_render", cache_key, ".tif")
    if not cache_path.exists():
        rendered = render_pdf_to_image(pdf_path, page_index=page_index)
        try:
            rendered.save(
                str(cache_path),
                format="TIFF",
                compression="lzw",
                dpi=(PDF_RENDER_DPI, PDF_RENDER_DPI),
            )
        finally:
            rendered.close()
    if dest_path is not None:
        try:
            if dest_path.resolve() != cache_path.resolve():
                dest_path.write_bytes(cache_path.read_bytes())
                return dest_path
        except Exception:
            pass
    return cache_path


def extract_pdf_text(pdf_path: Path, page_index: int = 0) -> str:
    """
    Extract the machine-readable text layer from a PDF page.
    Returns plain text (may be empty if the PDF only contains vector paths).
    """
    if not HAS_FITZ:
        return ""
    cache_key = _cache_key("pdf-text", _path_fingerprint(pdf_path), page_index)
    cached = _cache_read_text("pdf_text", cache_key)
    if cached is not None:
        return cached
    try:
        doc = pymupdf.open(str(pdf_path))
        try:
            if page_index >= len(doc):
                return ""
            page = doc.load_page(page_index)
            text = page.get_text("text") or ""
            _cache_write_text("pdf_text", cache_key, text)
            return text
        finally:
            doc.close()
    except Exception as exc:
        print(f"[~] PDF text extraction failed: {exc}")
        return ""


def _merge_text_sources(primary: str, secondary: str) -> str:
    """
    Merge PDF text-layer content with image OCR content.
    Keep all primary lines, then append secondary lines that add new tokens.
    This preserves clean title-block text while still bringing in raster-only
    map labels that often disambiguate the correct road segment.
    """
    primary = (primary or "").strip()
    secondary = (secondary or "").strip()
    if not primary:
        return secondary
    if not secondary:
        return primary
    merged: list[str] = []
    seen_norm: set[str] = set()

    def _add_lines(text: str, *, fuzzy_against: set[str] | None = None):
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            norm = _normalize_text_token(line)
            if len(norm) < 3 or norm in seen_norm:
                continue
            if fuzzy_against:
                tokens = set(norm.split())
                if tokens and any(len(tokens & set(existing.split())) / max(len(tokens), 1) >= 0.8 for existing in fuzzy_against):
                    continue
            merged.append(line)
            seen_norm.add(norm)

    _add_lines(primary)
    _add_lines(secondary, fuzzy_against=set(seen_norm))
    return "\n".join(merged)
    try:
        doc = pymupdf.open(str(pdf_path))
        page = doc[page_index]
        text = page.get_text("text")
        doc.close()
        return text or ""
    except Exception as exc:
        print(f"[!] PDF text extraction failed: {exc}")
        return ""


def pdf_metadata(pdf_path: Path, page_index: int = 0) -> dict:
    """
    Parse CRS, scale and location hints from a PDF's text layer and title block.
    Returns a partial meta dict (same keys used by read_tiff_metadata).
    """
    info: dict = {"path": str(pdf_path), "already_georef": False}
    if not HAS_FITZ:
        return info
    try:
        doc = pymupdf.open(str(pdf_path))
        try:
            page = doc[page_index]
            rect = page.rect
            info["width"]  = int(rect.width  * PDF_RENDER_DPI / 72)
            info["height"] = int(rect.height * PDF_RENDER_DPI / 72)
            info["bands"]  = 3
            info["dpi_x"]  = float(PDF_RENDER_DPI)
            info["dpi_y"]  = float(PDF_RENDER_DPI)
            text = page.get_text("text") or ""
        finally:
            doc.close()
    except Exception as exc:
        print(f"[!] pdf_metadata error: {exc}")
        return info

    # CRS hint
    if re.search(r'UTM\s*32|ETRS\s*89|25832', text, re.IGNORECASE):
        info["crs_hint"] = "ETRS89/UTM32"
        info["epsg_hint"] = 25832
    elif re.search(r'UTM\s*33|25833', text, re.IGNORECASE):
        info["crs_hint"] = "ETRS89/UTM33"
        info["epsg_hint"] = 25833
    elif re.search(r'Gauss.Kr.ger|31467|31466', text, re.IGNORECASE):
        info["crs_hint"] = "GaussKrueger"
        info["epsg_hint"] = 31467

    # Scale hint from line-local parsing so nearby numbers do not get merged.
    scale_val = _extract_best_scale(text)
    if scale_val:
        info["scale_hint"] = scale_val

    print(f"[i] PDF size: {info['width']} × {info['height']} px  "
          f"(rendered at {PDF_RENDER_DPI} DPI)")
    if info.get("epsg_hint"):
        print(f"[i] PDF CRS hint: {info.get('crs_hint')}  (EPSG:{info['epsg_hint']})")
    if info.get("scale_hint"):
        print(f"[i] PDF scale hint: 1:{info['scale_hint']:,}")
    return info


# ---------------------------------------------------------------------------
# STEP 2 – OCR: extract all text from the image
# ---------------------------------------------------------------------------
def _ocr_text_is_sufficient(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return False
    parsed = parse_coordinates(text)
    if parsed.get("pairs") or parsed.get("eastings") or parsed.get("northings"):
        return True
    hints = _extract_structured_location_hints(text)
    evidence_count = 0
    for key in ("site_city", "site_street", "road_code", "station_text", "landmark_name"):
        if hints.get(key):
            evidence_count += 1
    if hints.get("parcel_refs"):
        evidence_count += 1
    if parsed.get("scale"):
        evidence_count += 1
    return evidence_count >= 2 or len(text) >= 700


def _ocr_image_text(img: "Image.Image", config: str) -> str:
    try:
        return pytesseract.image_to_string(img, config=config) or ""
    except Exception:
        return ""


def _ocr_extract_targeted_text(path: Path) -> str:
    cache_key = _cache_key("ocr-targeted", _path_fingerprint(path), getattr(pytesseract.pytesseract, "tesseract_cmd", ""))
    cached = _cache_read_text("ocr_targeted", cache_key)
    if cached is not None:
        return cached

    _img_raw = Image.open(str(path))
    texts: list[str] = []
    try:
        img = _img_raw.convert("RGB") if _img_raw.mode not in ("RGB", "L") else _img_raw
        W, H = img.width, img.height
        boxes = [
            ("top", (0, 0, W, max(1, int(H * 0.18))), False),
            ("bottom", (0, max(0, int(H * 0.76)), W, H), False),
            ("left", (0, 0, max(1, int(W * 0.18)), H), True),
            ("right", (max(0, int(W * 0.82)), 0, W, H), True),
            ("title", (max(0, int(W * 0.52)), max(0, int(H * 0.58)), W, H), False),
        ]
        for name, box, rotate in boxes:
            crop = img.crop(box)
            try:
                if rotate:
                    crop = crop.rotate(90, expand=True)
                if max(crop.width, crop.height) > 3200:
                    scale = 3200 / max(crop.width, crop.height)
                    crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))), Image.LANCZOS)
                txt = _ocr_image_text(crop, r"--oem 3 --psm 6 -l deu+eng").strip()
                if txt:
                    texts.append(txt)
            finally:
                crop.close()
    finally:
        img.close()
    merged = _merge_text_sources("\n".join(texts), "")
    _cache_write_text("ocr_targeted", cache_key, merged)
    return merged


def _ocr_extract_full_text(path: Path) -> str:
    cache_key = _cache_key("ocr-full", _path_fingerprint(path), getattr(pytesseract.pytesseract, "tesseract_cmd", ""))
    cached = _cache_read_text("ocr_full", cache_key)
    if cached is not None:
        return cached
    print("[~] Running OCR on full image (may take a minute for 132 MB) …")
    _img_raw = Image.open(str(path))
    try:
        if _img_raw.mode not in ("RGB", "L"):
            img = _img_raw.convert("RGB")
            _img_raw.close()
        else:
            img = _img_raw
        max_side = 8000
        if max(img.width, img.height) > max_side:
            scale = max_side / max(img.width, img.height)
            _resized = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
            if img is not _img_raw:
                img.close()
            img = _resized
            print(f"[i] Downsampled to {img.width}×{img.height} for OCR")
        text = _ocr_image_text(img, r"--oem 3 --psm 11 -l deu+eng")
    finally:
        img.close()
    _cache_write_text("ocr_full", cache_key, text)
    return text


def ocr_extract_text(path: Path, dpi: int = 200) -> str:
    """Return OCR text from the TIFF, preferring targeted regions before full-image OCR."""
    if not (HAS_TESSERACT and HAS_PIL):
        print("[!] Skipping OCR (missing dependencies)")
        return ""
    cache_key = _cache_key("ocr-final", _path_fingerprint(path), dpi, getattr(pytesseract.pytesseract, "tesseract_cmd", ""))
    cached = _cache_read_text("ocr_final", cache_key)
    if cached is not None:
        print(f"[i] OCR cache hit → {len(cached)} characters")
        return cached

    print("[~] Running targeted OCR on margins/title block …")
    targeted = _ocr_extract_targeted_text(path)
    if _ocr_text_is_sufficient(targeted):
        print(f"[✓] Targeted OCR extracted {len(targeted)} characters")
        _cache_write_text("ocr_final", cache_key, targeted)
        return targeted

    full_text = _ocr_extract_full_text(path)
    merged = _merge_text_sources(targeted, full_text)
    print(f"[✓] OCR extracted {len(merged)} characters")
    _cache_write_text("ocr_final", cache_key, merged)
    return merged


# ---------------------------------------------------------------------------
# STEP 3 – Parse coordinate patterns from OCR text
# ---------------------------------------------------------------------------
# German engineering plans typically use:
#   UTM easting  ~  300 000 – 900 000   (6-digit, or with zone prefix 32...)
#   UTM northing ~ 5 000 000 – 6 500 000 (7-digit for Central Europe)
#   Gauss-Krüger easting ~  3 400 000 – 3 600 000
#   Gauss-Krüger northing ~ 5 000 000 – 6 500 000

UTM_EASTING_RE  = re.compile(r'\b(3\d{5}|4\d{5}|5\d{5}|6\d{5}|7\d{5}|8\d{5})\b')
UTM_NORTHING_RE = re.compile(r'\b(5[0-9]\d{5}|60\d{5}|61\d{5}|62\d{5}|63\d{5}|64\d{5})\b')
GK_EASTING_RE   = re.compile(r'\b(3[3-5]\d{5})\b')   # Zone 3 Gauss-Krüger
COORD_PAIR_RE   = re.compile(
    r'(?:\b(?:E|R|x|Ost(?:wert)?)\b)[:\s]{0,6}(\d[\d\s.]{4,10})'
    r'[^\n]{0,40}?'
    r'(?:\b(?:N|H|y|Nord(?:wert)?)\b)[:\s]{0,6}(\d[\d\s.]{5,11})',
    re.IGNORECASE
)
SCALE_RE = re.compile(r'(?<!\d)1\s*:\s*([\d\s.]{2,12})(?!\d)')


def parse_coordinates(text: str) -> dict:
    """Extract candidate coordinates and scale from OCR text."""
    result = {"eastings": [], "northings": [], "pairs": [], "scale": None, "crs_hints": []}

    # Named pairs (e.g. "E 395000  N 5723000"), restricted to line-local matches
    # so unrelated title-block numbers do not merge into nonsense coordinates.
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    for idx, line in enumerate(lines):
        search_lines = [line]
        if idx + 1 < len(lines) and len(line) < 80:
            search_lines.append(lines[idx + 1])
        for candidate_line in search_lines:
            for m in COORD_PAIR_RE.finditer(candidate_line):
                try:
                    e = float(re.sub(r'[\s.]', '', m.group(1)))
                    n = float(re.sub(r'[\s.]', '', m.group(2)))
                except ValueError:
                    continue
                if 280_000 <= e <= 850_000 and 5_200_000 <= n <= 6_100_000:
                    result["pairs"].append((e, n))

    # Standalone values
    result["eastings"]  = [int(v) for v in UTM_EASTING_RE.findall(text)]
    result["northings"] = [int(v) for v in UTM_NORTHING_RE.findall(text)]

    # Scale
    scale_candidates = _extract_scale_candidates(text)
    if scale_candidates:
        result["scale"] = scale_candidates[0][0]

    # CRS hints
    for crs_kw in ["UTM", "ETRS", "Gauss", "Krüger", "DHDN", "WGS", "25832", "31467"]:
        if crs_kw.lower() in text.lower():
            result["crs_hints"].append(crs_kw)

    print(f"[✓] Parsed coords – eastings: {result['eastings'][:5]}, "
          f"northings: {result['northings'][:5]}, pairs: {result['pairs'][:3]}, "
          f"scale: {result['scale']}, CRS hints: {result['crs_hints']}")
    return result


# ---------------------------------------------------------------------------
# Delegated pure helpers
# Keep the plugin API stable while routing reusable text parsing through
# dedicated georef_core modules.
def _clean_text_for_match(text: str) -> str:
    return _text_clean_text_for_match(text)


def _parse_scale_value(raw: str) -> int | None:
    return _text_parse_scale_value(raw)


def _extract_scale_candidates(text: str) -> list[tuple[int, float, str]]:
    return _text_extract_scale_candidates(text)


def _extract_best_scale(text: str) -> int | None:
    return _text_extract_best_scale(text)


def _merge_text_sources(primary: str, secondary: str) -> str:
    return _text_merge_text_sources(primary, secondary)


def parse_coordinates(text: str) -> dict:
    return _text_parse_coordinates(text, logger=print)


# STEP 4 – OpenAI Vision analysis
# ---------------------------------------------------------------------------
OVERVIEW_PROMPT = """
You are a GIS specialist analysing a scanned engineering / surveying plan.
If more than one image is provided, the FIRST image(s) are screenshots of the
current QGIS map canvas (showing where the user is currently looking in Germany)
-- use them as geographic context clues. The LAST image is the engineering plan
to be georeferenced.

Identify what you can from the plan image and return a JSON object with EXACTLY
these keys -- use null where unknown:
{
  "coordinate_system": "<e.g. ETRS89/UTM Zone 32N, Gauss-Krüger Zone 3, WGS84, or null>",
  "epsg": <integer EPSG code or null>,
  "scale": <map scale denominator as integer, e.g. 25000, or null>,
  "location_name": "<PROJECT SITE city/municipality -- see rules below>",
  "north_arrow_present": true|false,
  "north_arrow_direction_deg": <clockwise degrees from image-up to the north arrow tip, 0=pointing straight up, 90=pointing right, -90=pointing left, or null>,
  "map_is_north_up": true|false|null,
  "notes": "<any coordinate system clues, plan type (aerial/topo/CAD), or observations>"
}

CRITICAL RULES for location_name:
- Set location_name to the CONSTRUCTION SITE city, i.e. the city or municipality
  where the engineering work physically takes place.
- Look for patterns like "... Straßenname in Kamen", "Radwegbau bei Hamm",
  "Neubau der Umgehung B 55 in Selm", or a "Standort:" label in the title block.
- DO NOT use the contracting authority's office address (e.g. "Harpener Hellweg 1,
  44791 Bochum" is an office -- Bochum is NOT the site city).
- DO NOT use company names or regional office names (e.g. "Regionalniederlassung
  Ruhr" does not mean the site is in the Ruhr area).
- If you can read a street name AND a city name together (e.g. "Dortmunder Allee
  in Kamen"), put the street name too: "Dortmunder Allee, Kamen".
- If only the site city is clear, set location_name to just that city name.
- If the site location cannot be determined with confidence, set location_name=null.

CRITICAL RULES for north_arrow_direction_deg and map_is_north_up:
- Look ONLY at the north arrow symbol (the small compass/arrow icon, typically with an 'N' label or a pointed tip).
- DO NOT infer rotation from the direction of roads, rivers, railways, or other map features. A map can be perfectly north-up while showing diagonal roads or rivers.
- If the north arrow tip points straight up (within ±10°), set north_arrow_direction_deg=0 and map_is_north_up=true.
- map_is_north_up=true whenever |north_arrow_direction_deg| <= 15.
- If you cannot find a north arrow symbol, set north_arrow_present=false and map_is_north_up=null.

Return ONLY the JSON, no prose.
"""

TITLEBLOCK_SCALE_PROMPT = """
You are reading the title block / legend area of a German engineering plan.
Extract ONLY the printed map scale if visible.
Return a JSON object with EXACTLY this shape:
{"scale": <integer denominator like 5000 or 25000, or null>}
Rules:
- Read the scale digit by digit.
- If the image shows "1 : 5000", return {"scale": 5000}
- If no scale is clearly visible, return {"scale": null}
- Return ONLY JSON, no prose.
"""

TITLEBLOCK_SITE_PROMPT = """
You are reading the title block / project header area of a German engineering or roadworks plan.
Extract structured LOCATION metadata only.
Return a JSON object with EXACTLY this shape:
{
  "project_site_street": "<street/road name where the work happens, or null>",
  "project_site_house_number": "<house number only, or null>",
  "project_site_city": "<city/municipality where the work happens, or null>",
  "project_site_postcode": "<5-digit postcode for the work site, or null>",
  "project_road_code": "<road id like A42, B55, L663, K14, or null>",
  "station_text": "<station/chainage text like 'Stat. 0+000 bis 0+175', or null>",
  "landmark_name": "<named nearby landmark or institution physically on site, or null>",
  "client_address": "<office/client address if shown, or null>",
  "has_client_office_address": true|false,
  "notes": "<brief note, or null>"
}

Rules:
- Prefer the physical work site, not the contracting authority office.
- If you see both a client office address and a project site description, keep them separate.
- A line like "Dortmunder Allee in Kamen" means:
  - project_site_street = "Dortmunder Allee"
  - project_site_city = "Kamen"
- A line like "L 663 Westabschnitt" means project_road_code = "L663".
- If a field is not clearly present, return null for that field.
- Return ONLY JSON, no prose.
"""

def _make_margin_prompt(side: str, strip_w: int, strip_h: int,
                        canvas_center: dict = None) -> str:
    """
    Build the margin-analysis prompt for a given side.
    Embeds the EXACT pixel dimensions of the strip thumbnail so the model
    can report pixel_x / pixel_y as concrete integers rather than abstract
    fractions -- this is more accurate and reproducible.
    canvas_center: optional dict with center_easting/center_northing (EPSG:25832)
    used to build a dynamic coordinate range hint instead of a hardcoded location.
    """
    # Build dynamic coordinate range hints from canvas centre if available
    if canvas_center and canvas_center.get("center_easting") and canvas_center.get("center_northing"):
        ce = float(canvas_center["center_easting"])
        cn = float(canvas_center["center_northing"])
        e_lo = int(max(_E_MIN, ce - 50_000))
        e_hi = int(min(_E_MAX, ce + 50_000))
        n_lo = int(max(_N_MIN, cn - 50_000))
        n_hi = int(min(_N_MAX, cn + 50_000))
        e_range_hint = f"approximately {e_lo:,}–{e_hi:,}"
        n_range_hint = f"approximately {n_lo:,}–{n_hi:,}"
    else:
        e_range_hint = "300,000–900,000 (ETRS89/UTM Zone 32N, Germany)"
        n_range_hint = "5,200,000–6,100,000 (ETRS89/UTM Zone 32N, Germany)"

    if side in ("top", "bottom"):
        edge = "TOP" if side == "top" else "BOTTOM"
        return f"""
You are reading the {edge} edge of a German engineering plan (north-up orientation).
The {edge} margin shows EASTING (X / Rechtswert) tick-mark labels -- horizontal
coordinates that increase LEFT → RIGHT.

This strip image is exactly {strip_w} pixels wide × {strip_h} pixels tall.
For each label, report its EXACT horizontal pixel position (pixel_x) in this image
(integer, 0 = far-left edge, {strip_w} = far-right edge).

Expected easting range: {e_range_hint}.
Abbreviated labels such as "395" mean 395000; "395.5" means 395500.
Read ONLY coordinate grid tick labels along the map frame border -- ignore title, scale bar, legend.

Return a JSON array (one entry per visible label):
[
  {{"value": <easting as float, fully expanded>, "label_text": "<exact text>", "pixel_x": <integer>}}
]
Return [] if nothing readable. Return ONLY the JSON array, no prose.
"""
    else:  # left / right
        edge = "LEFT" if side == "left" else "RIGHT"
        return f"""
You are reading the {edge} edge of a German engineering plan (north-up orientation).
The {edge} margin shows NORTHING (Y / Hochwert) tick-mark labels -- vertical
coordinates that increase BOTTOM → TOP (so larger values are near the top).

This strip image is exactly {strip_w} pixels wide × {strip_h} pixels tall.
For each label, report its EXACT vertical pixel position (pixel_y) in this image
(integer, 0 = top edge, {strip_h} = bottom edge).

IMPORTANT: read each label DIGIT BY DIGIT and report exactly what is printed.
Do NOT guess or round. Report the exact numeric string visible next to each tick mark.

Expected northing range: {n_range_hint}.
Abbreviated labels: "5658" means 5658000; "5658.5" means 5658500.
Ignore any inset map, legend, title block, page numbers -- only main map frame tick labels.

Return a JSON array (one entry per visible label):
[
  {{"value": <northing as float, fully expanded>, "label_text": "<exact digits>", "pixel_y": <integer>}}
]
Return [] if nothing readable. Return ONLY the JSON array, no prose.
"""

# Valid coordinate bounds for the active CRS — mutated by set_active_crs_preset().
# Initialized to wide defaults; _load_geopacks() will tighten these to the
# default preset's bounds when a geopack is installed.
_E_MIN: float = -10_000_000.0
_E_MAX: float =  10_000_000.0
_N_MIN: float = -10_000_000.0
_N_MAX: float =  10_000_000.0

# City lookup tables — populated at startup by _load_geopacks() from the
# city_lookup sections of installed geopacks.  Empty until a geopack is loaded.
CITY_NORTHING_UTM32: dict[str, float] = {}
CITY_EASTING_UTM32:  dict[str, float] = {}


def _load_geopacks() -> None:
    """
    Discover and load all installed geopacks.

    Merges CRS presets into CRS_PRESETS, city coordinates into
    CITY_NORTHING_UTM32 / CITY_EASTING_UTM32, and activates the first
    geopack preset that has "default": true (or the first preset found).
    """
    global CRS_PRESETS, CITY_NORTHING_UTM32, CITY_EASTING_UTM32
    try:
        from georef_core.geopack_manager import GeopackManager, default_store_dirs
        mgr = GeopackManager(default_store_dirs(_ROOT, OUTPUT_DIR))
        presets = mgr.load_all_presets()
        if presets:
            custom = CRS_PRESETS.pop("custom", None)
            CRS_PRESETS.update(presets)
            if custom is not None:
                CRS_PRESETS["custom"] = custom  # keep 'custom' at end

        # Merge city lookup for the active EPSG (and 25832 as common case)
        for _epsg in {25832, TARGET_EPSG}:
            e_d, n_d = mgr.load_city_lookup(_epsg)
            CITY_EASTING_UTM32.update(e_d)
            CITY_NORTHING_UTM32.update(n_d)

        # Activate the default preset from any geopack
        default_key = None
        for key, preset in CRS_PRESETS.items():
            if key == "custom":
                continue
            if preset.get("default"):
                default_key = key
                break
        if default_key is None:
            for key in CRS_PRESETS:
                if key != "custom":
                    default_key = key
                    break
        if default_key:
            set_active_crs_preset(default_key)
    except Exception as exc:
        print(f"[!] Geopack load error: {exc}")


_load_geopacks()


def ensure_project_address_template():
    """Create a blank project_address.json template if it doesn't exist yet."""
    if PROJECT_ADDRESS_FILE.exists():
        return
    template = {
        "enabled": False,
        "address": "",
        "notes": (
            "Set enabled=true and enter any address, place name, or area description. "
            "The address is geocoded via OpenStreetMap Nominatim and converted to "
            "EPSG:25832 UTM coordinates, then used as the geographic seed when the "
            "QGIS canvas position is unavailable. "
            "Example: \"Hauptstraße 5, 57462 Olpe, Deutschland\""
        ),
    }
    PROJECT_ADDRESS_FILE.write_text(
        json.dumps(template, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _utm_forward_math(lon_deg: float, lat_deg: float, lon0_deg: float) -> tuple:
    """
    Pure-Python UTM forward projection for any central meridian.
    Accurate to ~1 m — sufficient as a seed position fallback when pyproj
    is unavailable.  Only valid for UTM-family projections (ETRS89, WGS84).
    """
    import math
    lon_r  = math.radians(lon_deg)
    lat_r  = math.radians(lat_deg)
    lon0_r = math.radians(lon0_deg)
    k0  = 0.9996
    a   = 6_378_137.0
    e2  = 0.006_694_379_990_14
    e4  = e2 * e2
    e6  = e4 * e2
    ep2 = e2 / (1.0 - e2)
    N   = a / math.sqrt(1.0 - e2 * math.sin(lat_r) ** 2)
    T   = math.tan(lat_r) ** 2
    C   = ep2 * math.cos(lat_r) ** 2
    A   = (lon_r - lon0_r) * math.cos(lat_r)
    M   = a * (
        (1.0 - e2/4.0 - 3.0*e4/64.0 - 5.0*e6/256.0)  * lat_r
        - (3.0*e2/8.0 + 3.0*e4/32.0 + 45.0*e6/1024.0) * math.sin(2.0*lat_r)
        + (15.0*e4/256.0 + 45.0*e6/1024.0)             * math.sin(4.0*lat_r)
        - (35.0*e6/3072.0)                              * math.sin(6.0*lat_r)
    )
    easting = k0 * N * (
        A
        + (1.0 - T + C) * A**3 / 6.0
        + (5.0 - 18.0*T + T**2 + 72.0*C - 58.0*ep2) * A**5 / 120.0
    ) + 500_000.0
    northing = k0 * (
        M + N * math.tan(lat_r) * (
            A**2 / 2.0
            + (5.0 - T + 9.0*C + 4.0*C**2) * A**4 / 24.0
            + (61.0 - 58.0*T + T**2 + 600.0*C - 330.0*ep2) * A**6 / 720.0
        )
    )
    return float(easting), float(northing)


# Central meridians for common UTM-family EPSG codes (used by pure-Python fallback)
_UTM_LON0: dict[int, float] = {
    25829: -3.0,  25830: 3.0,   25831: 3.0,
    25832: 9.0,   25833: 15.0,  25834: 21.0,
    25835: 27.0,  25836: 33.0,
    31466: 6.0,   31467: 9.0,   31468: 12.0,  31469: 15.0,  # Gauss-Krüger
    32629: -9.0,  32630: -3.0,  32631: 3.0,
    32632: 9.0,   32633: 15.0,  32634: 21.0,
    32635: 27.0,  32636: 33.0,  32637: 39.0,
}


def _wgs84_to_projected(lon_deg: float, lat_deg: float, epsg: int | None = None) -> tuple:
    """
    Convert WGS84 lon/lat to the projected CRS given by `epsg` (defaults to
    the active TARGET_EPSG).  Uses pyproj when available; falls back to the
    pure-Python UTM formula for UTM-family zones.
    """
    _epsg = epsg if epsg is not None else TARGET_EPSG
    if HAS_PYPROJ:
        try:
            from pyproj import Transformer
            t = Transformer.from_crs("EPSG:4326", f"EPSG:{_epsg}", always_xy=True)
            e, n = t.transform(lon_deg, lat_deg)
            return float(e), float(n)
        except Exception:
            pass  # fall through to pure-Python

    lon0 = _UTM_LON0.get(_epsg)
    if lon0 is None:
        raise RuntimeError(
            f"pyproj is required for EPSG:{_epsg} coordinate conversion. "
            "Install it: pip install pyproj"
        )
    return _utm_forward_math(lon_deg, lat_deg, lon0)


def _wgs84_to_utm32n(lon_deg: float, lat_deg: float) -> tuple:
    """Backward-compat alias — always projects to EPSG:25832."""
    return _wgs84_to_projected(lon_deg, lat_deg, 25832)


def _normalize_text_token(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("ß", "ss")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _extract_station_text(text: str) -> str | None:
    if not text:
        return None
    m = re.search(
        r"(Stat\.?\s*[0-9+., ]+\s*(?:bis|-)\s*[0-9+., ]+|km\s*=\s*[0-9+., ]+)",
        text,
        re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else None


def _clean_city_name(value: str) -> str:
    value = re.sub(r"\s+", " ", value or "").strip(" ,")
    value = re.split(r"\b(?:Schild|Stat\.?|km|Datum|Ma[ßs]stab|Projekt|Unterlage)\b", value, maxsplit=1)[0]
    return re.sub(r"\s+", " ", value).strip(" ,")


def _extract_postal_address_candidates(text: str) -> list[dict]:
    if not text:
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    hits: list[dict] = []
    seen: set[str] = set()
    street_pat = re.compile(
        r"([A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-/ ]{2,60}?"
        r"(?:straße|strasse|str\.|weg|allee|platz|gasse|ufer|chaussee|ring|damm|pfad|kamp|wall|markt))"
        r"\s+(\d+[A-Za-z]?)",
        re.IGNORECASE,
    )
    city_pat = re.compile(r"\b(\d{5})\s+([A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-/ ]{2,50})")
    office_ctx_pat = re.compile(
        r"(regionalniederlassung|niederlassung|auftraggeber|landesbetrieb|verwaltung|büro|buero|anschrift|ruhr)",
        re.IGNORECASE,
    )
    for idx, line in enumerate(lines):
        sm = street_pat.search(line)
        if not sm:
            continue
        city_line = line
        cm = city_pat.search(city_line)
        if not cm and idx + 1 < len(lines):
            city_line = lines[idx + 1]
            cm = city_pat.search(city_line)
        if not cm:
            continue
        street = re.sub(r"\s+", " ", sm.group(1)).strip(" ,")
        house_no = sm.group(2).strip()
        postcode = cm.group(1)
        city = re.sub(r"\s+", " ", cm.group(2)).strip(" ,")
        query = f"{street} {house_no}, {postcode} {city}"
        ctx = " ".join(lines[max(0, idx - 2):min(len(lines), idx + 3)])
        ctx_lower = ctx.lower()
        _office_markers = (
            "telefon", "telefax", "fax", "e-mail", "email", "www.",
            "engineering", "gmbh", " ag ", " mbh", "hauptsitz", "zentrale",
        )
        _office_line = office_ctx_pat.search(ctx) or any(marker in ctx_lower for marker in _office_markers)
        role = "office" if _office_line else "site"
        key = _normalize_text_token(query)
        if key in seen:
            continue
        seen.add(key)
        hits.append({
            "query": query,
            "street": street,
            "house_number": house_no,
            "postcode": postcode,
            "city": city,
            "role": role,
            "context": ctx,
            "source": "postal_address",
        })
    return hits


def _extract_structured_location_hints(
    text: str,
    vision: dict | None = None,
    titleblock_meta: dict | None = None,
) -> dict:
    global LAST_STRUCTURED_LOCATION_HINTS
    hints = {
        "site_street": None,
        "site_house_number": None,
        "site_city": None,
        "site_postcode": None,
        "road_codes": [],
        "station_text": None,
        "parcel_refs": [],
        "landmarks": [],
        "site_addresses": [],
        "office_addresses": [],
        "client_address": None,
    }
    text = text or ""
    titleblock_meta = titleblock_meta or {}

    for addr in _extract_postal_address_candidates(text):
        if addr["role"] == "office":
            hints["office_addresses"].append(addr)
            if hints["client_address"] is None:
                hints["client_address"] = addr["query"]
        else:
            hints["site_addresses"].append(addr)
            if hints["site_street"] is None:
                hints["site_street"] = addr["street"]
                hints["site_house_number"] = addr["house_number"]
                hints["site_city"] = addr["city"]
                hints["site_postcode"] = addr["postcode"]

    street_city_pat = re.compile(
        r"([A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-/]+(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-/]+){0,4}\s+"
        r"(?:straße|strasse|allee|weg|platz|gasse|ufer|chaussee|ring|damm|pfad|kamp|wall|markt))"
        r"\s+(?:in|bei|nahe)\s+([A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-]+(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-]+)?)",
        re.IGNORECASE,
    )
    road_city_pat = re.compile(
        r"\b([ABLKS])\s*(\d{1,4}[a-z]?)\b[^\n]{0,100}?\s+(?:in|bei|nahe)\s+"
        r"([A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-]+(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-]+)?)",
        re.IGNORECASE,
    )
    standort_pat = re.compile(r"Standort\s*:?\s*([A-ZÄÖÜ][A-Za-zÄÖÜäöüß\- ]{2,40})", re.IGNORECASE)
    landmark_pat = re.compile(r'[Ss]child\s+[\'"]([^\'"]{3,80})[\'"]')
    parcel_pat = re.compile(
        r"(?:Gemarkung\s+([A-ZÄÖÜ][A-Za-zÄÖÜäöüß\- ]+))?[^\n]{0,40}?"
        r"Flur\s+(\d+)[^\n]{0,40}?(?:Flurstück|Flurst(?:ue|ü)ck)\s+([\d/]+)",
        re.IGNORECASE,
    )

    for m in street_city_pat.finditer(text):
        street = re.sub(r"\s+", " ", m.group(1)).strip(" ,")
        city = _clean_city_name(m.group(2))
        if hints["site_street"] is None:
            hints["site_street"] = street
        if hints["site_city"] is None:
            hints["site_city"] = city

    for m in road_city_pat.finditer(text):
        road = f"{m.group(1).upper()}{m.group(2)}"
        city = _clean_city_name(m.group(3))
        if road not in hints["road_codes"]:
            hints["road_codes"].append(road)
        if hints["site_city"] is None:
            hints["site_city"] = city

    if not hints["road_codes"]:
        hints["road_codes"] = _extract_road_codes(text)

    sm = standort_pat.search(text)
    if sm and not hints["site_city"]:
        hints["site_city"] = _clean_city_name(sm.group(1))

    for lm in landmark_pat.finditer(text):
        name = re.sub(r"\s+", " ", lm.group(1)).strip()
        if name and name not in hints["landmarks"]:
            hints["landmarks"].append(name)

    for pm in parcel_pat.finditer(text):
        gem = re.sub(r"\s+", " ", (pm.group(1) or "")).strip(" ,")
        flur = pm.group(2).strip()
        flst = pm.group(3).strip()
        parcel = f"{gem + ', ' if gem else ''}Flur {flur}, Flurstück {flst}"
        if parcel not in hints["parcel_refs"]:
            hints["parcel_refs"].append(parcel)

    hints["station_text"] = _extract_station_text(text)

    if titleblock_meta:
        if titleblock_meta.get("project_site_street") and not hints["site_street"]:
            hints["site_street"] = str(titleblock_meta["project_site_street"]).strip()
        if titleblock_meta.get("project_site_house_number") and not hints["site_house_number"]:
            hints["site_house_number"] = str(titleblock_meta["project_site_house_number"]).strip()
        if titleblock_meta.get("project_site_city") and not hints["site_city"]:
            hints["site_city"] = str(titleblock_meta["project_site_city"]).strip()
        if titleblock_meta.get("project_site_postcode") and not hints["site_postcode"]:
            hints["site_postcode"] = str(titleblock_meta["project_site_postcode"]).strip()
        if titleblock_meta.get("project_road_code"):
            _rc = re.sub(r"\s+", "", str(titleblock_meta["project_road_code"]).upper())
            if _rc and _rc not in hints["road_codes"]:
                hints["road_codes"].insert(0, _rc)
        if titleblock_meta.get("station_text") and not hints["station_text"]:
            hints["station_text"] = str(titleblock_meta["station_text"]).strip()
        if titleblock_meta.get("landmark_name"):
            _lm = str(titleblock_meta["landmark_name"]).strip()
            if _lm and _lm not in hints["landmarks"]:
                hints["landmarks"].insert(0, _lm)
        if titleblock_meta.get("client_address") and not hints["client_address"]:
            hints["client_address"] = str(titleblock_meta["client_address"]).strip()

    if vision and not hints["site_city"]:
        _vloc = str(vision.get("location_name") or "").strip()
        if _vloc:
            if "," in _vloc:
                hints["site_city"] = _vloc.split(",")[-1].strip()
                if not hints["site_street"]:
                    hints["site_street"] = _vloc.split(",")[0].strip()
            else:
                hints["site_city"] = _vloc

    LAST_STRUCTURED_LOCATION_HINTS = hints
    return hints


def _extract_project_city(project_address: str) -> str | None:
    return _location_extract_project_city(project_address)


def _address_is_specific(address: str) -> bool:
    return _location_address_is_specific(address)


def _classify_address_confidence(address: str) -> str:
    return _location_classify_address_confidence(address)


def _extract_road_codes(text: str) -> list[str]:
    return _location_extract_road_codes(text)


def _extract_structured_location_hints(
    text: str,
    vision: dict | None = None,
    titleblock_meta: dict | None = None,
) -> dict:
    global LAST_STRUCTURED_LOCATION_HINTS
    LAST_STRUCTURED_LOCATION_HINTS = _build_structured_location_hints(text or "", vision, titleblock_meta)
    return LAST_STRUCTURED_LOCATION_HINTS


_GEOCODE_CACHE: dict = {}   # (address, limit) → list[dict]; reset each module load
_NOMINATIM_LAST_CALL: float = 0.0   # epoch seconds of last HTTP request to Nominatim
_NOMINATIM_MIN_INTERVAL: float = 1.1  # Nominatim ToS: max 1 req/s; use 1.1 s to be safe


def geocode_address_candidates_utm32(address: str, limit: int = 5) -> list[dict]:
    """
    Geocode via Nominatim and project results into the active TARGET_EPSG.
    Results are cached; cache is cleared when set_active_crs_preset() changes
    the projection so stale coordinates are never returned.
    """
    global _NOMINATIM_LAST_CALL
    if not address:
        return []
    # Cache key includes EPSG so switching projection invalidates cached values
    cache_key = (address, int(limit), int(TARGET_EPSG))
    if cache_key in _GEOCODE_CACHE:
        return list(_GEOCODE_CACHE[cache_key] or [])
    _disk_key = _cache_key("geocode", address, int(limit), int(TARGET_EPSG))
    _disk_cached = _cache_read_json("geocode", _disk_key)
    if isinstance(_disk_cached, list):
        _GEOCODE_CACHE[cache_key] = list(_disk_cached)
        return list(_disk_cached)

    import time
    import urllib.request
    import urllib.parse

    # Enforce Nominatim rate limit (1 request per second)
    elapsed = time.monotonic() - _NOMINATIM_LAST_CALL
    if elapsed < _NOMINATIM_MIN_INTERVAL:
        time.sleep(_NOMINATIM_MIN_INTERVAL - elapsed)

    params = urllib.parse.urlencode({
        "q": address,
        "format": "jsonv2",
        "limit": max(1, min(int(limit), 10)),
        "addressdetails": 1,
    })
    url = f"https://nominatim.openstreetmap.org/search?{params}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "AutoGeoreferencer/1.0 (QGIS plugin; georef tool)"},
    )
    _NOMINATIM_LAST_CALL = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw_results = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"[!] Nominatim geocoding failed: {exc}")
        _GEOCODE_CACHE[cache_key] = []
        return []

    results: list[dict] = []
    for item in raw_results or []:
        try:
            lat = float(item["lat"])
            lon = float(item["lon"])
            e, n = _wgs84_to_projected(lon, lat)  # uses active TARGET_EPSG
        except Exception:
            continue
        addr = item.get("address") or {}
        results.append({
            "query": address,
            "display_name": item.get("display_name", ""),
            "class": item.get("class", ""),
            "type": item.get("type", ""),
            "importance": float(item.get("importance") or 0.0),
            "lat": lat,
            "lon": lon,
            "easting": float(e),
            "northing": float(n),
            "address": addr if isinstance(addr, dict) else {},
        })
    if results:
        print(f"[i] Geocoded '{address[:60]}' → {len(results)} candidate(s)")
    else:
        print(f"[!] Geocoding: no usable results for '{address}'")
    _GEOCODE_CACHE[cache_key] = list(results)
    _cache_write_json("geocode", _disk_key, results)
    return results


def _rank_geocode_candidates(
    candidates: list[dict],
    *,
    wanted_city: str | None = None,
    wanted_postcode: str | None = None,
    road_code: str | None = None,
    landmark: str | None = None,
    anchor_e: float | None = None,
    anchor_n: float | None = None,
) -> list[dict]:
    want_city_norm = _normalize_text_token(wanted_city)
    want_postcode = (wanted_postcode or "").strip()
    road_code_norm = _normalize_text_token(road_code)
    landmark_norm = _normalize_text_token(landmark)

    ranked: list[dict] = []
    for item in candidates:
        disp_norm = _normalize_text_token(item.get("display_name", ""))
        addr = item.get("address") or {}
        city_fields = [
            addr.get("city"), addr.get("town"), addr.get("village"),
            addr.get("municipality"), addr.get("county"), addr.get("state_district"),
        ]
        city_norms = [_normalize_text_token(v) for v in city_fields if v]
        postcode = str(addr.get("postcode") or "").strip()
        score = float(item.get("importance") or 0.0) * 2.0

        if want_city_norm:
            if any(want_city_norm == cn or want_city_norm in cn or cn in want_city_norm for cn in city_norms if cn):
                score += 6.0
            elif want_city_norm in disp_norm:
                score += 4.0
            else:
                score -= 2.0
        if want_postcode:
            if postcode == want_postcode:
                score += 6.0
            elif postcode:
                score -= 1.5
        if road_code_norm:
            if road_code_norm in disp_norm:
                score += 3.0
            elif item.get("type") in {"road", "highway"}:
                score += 1.0
        if landmark_norm:
            if landmark_norm in disp_norm:
                score += 4.0
            elif item.get("type") in {"amenity", "building", "office", "school", "hospital"}:
                score += 1.0
        if anchor_e is not None and anchor_n is not None:
            dist = ((float(item["easting"]) - anchor_e) ** 2 + (float(item["northing"]) - anchor_n) ** 2) ** 0.5
            item["_anchor_dist_m"] = dist
            if dist <= 300:
                score += 5.0
            elif dist <= 1000:
                score += 3.5
            elif dist <= 3000:
                score += 2.0
            elif dist <= 10000:
                score += 0.5
            elif dist > 25000:
                score -= min(8.0, dist / 7000.0)
        item["_score"] = score
        ranked.append(item)
    ranked.sort(key=lambda x: (x.get("_score", -999.0), -x.get("_anchor_dist_m", 0.0)), reverse=True)
    return ranked


def _projected_to_wgs84(easting: float, northing: float, epsg: int | None = None) -> tuple[float, float] | None:
    """
    Convert projected coordinates to WGS84 lon/lat.  Uses pyproj (required).
    Returns None if pyproj is unavailable or the transform fails.
    """
    _epsg = epsg if epsg is not None else TARGET_EPSG
    if HAS_PYPROJ:
        try:
            from pyproj import Transformer
            t = Transformer.from_crs(f"EPSG:{_epsg}", "EPSG:4326", always_xy=True)
            lon, lat = t.transform(float(easting), float(northing))
            return float(lon), float(lat)
        except Exception:
            return None
    return None


def _utm32n_to_wgs84(easting: float, northing: float) -> tuple[float, float] | None:
    """Backward-compat alias — always reprojects from EPSG:25832."""
    return _projected_to_wgs84(easting, northing, 25832)


def _overpass_query(query: str) -> dict | None:
    cache_key = _cache_key("overpass", query)
    cached = _cache_read_json("overpass", cache_key)
    if isinstance(cached, dict):
        return cached
    import urllib.request
    import urllib.parse
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    req = urllib.request.Request(
        "https://overpass-api.de/api/interpreter",
        data=data,
        headers={"User-Agent": "AutoGeoreferencer/1.0 (QGIS plugin; georef tool)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            _cache_write_json("overpass", cache_key, payload)
            return payload
    except Exception as exc:
        print(f"[~] Overpass lookup failed: {exc}")
        return None


def _snap_seed_to_osm(
    easting: float,
    northing: float,
    *,
    street: str | None = None,
    road_code: str | None = None,
    landmark: str | None = None,
) -> tuple[float, float, str] | None:
    global LAST_OSM_SNAP_INFO
    if not ENABLE_OSM_VECTOR_SNAPPING:
        return None
    lonlat = _utm32n_to_wgs84(easting, northing)
    if lonlat is None:
        return None
    lon, lat = lonlat
    radius = 450 if street or landmark else 900
    snippets: list[str] = []
    if road_code:
        rc = str(road_code).strip().replace('"', "")
        snippets.extend([
            f'way(around:{radius},{lat:.6f},{lon:.6f})["ref"="{rc}"];',
            f'way(around:{radius},{lat:.6f},{lon:.6f})["official_ref"="{rc}"];',
        ])
    if street:
        st = str(street).strip().replace('"', "")
        snippets.append(f'way(around:{radius},{lat:.6f},{lon:.6f})["name"="{st}"];')
    if landmark:
        lm = str(landmark).strip().replace('"', "")
        snippets.extend([
            f'node(around:{radius},{lat:.6f},{lon:.6f})["name"="{lm}"];',
            f'way(around:{radius},{lat:.6f},{lon:.6f})["name"="{lm}"];',
        ])
    if not snippets:
        return None
    query = f"[out:json][timeout:20];({''.join(snippets)});out geom center;"
    payload = _overpass_query(query)
    if not payload or not isinstance(payload.get("elements"), list):
        return None
    best = None
    best_dist = None
    best_axis = None
    for elem in payload["elements"]:
        if "lat" in elem and "lon" in elem:
            elat = float(elem["lat"])
            elon = float(elem["lon"])
        elif isinstance(elem.get("center"), dict) and "lat" in elem["center"] and "lon" in elem["center"]:
            elat = float(elem["center"]["lat"])
            elon = float(elem["center"]["lon"])
        else:
            continue
        try:
            ee, nn = _wgs84_to_utm32n(elon, elat)
        except Exception:
            continue
        dist = ((ee - easting) ** 2 + (nn - northing) ** 2) ** 0.5
        if best_dist is None or dist < best_dist:
            tags = elem.get("tags") or {}
            label = tags.get("name") or tags.get("ref") or elem.get("type", "osm")
            best = (float(ee), float(nn), str(label))
            best_dist = dist
            geom = elem.get("geometry")
            best_axis = None
            if isinstance(geom, list) and len(geom) >= 2:
                _axis_d = None
                _axis_a = None
                for i in range(len(geom) - 1):
                    p1 = geom[i]
                    p2 = geom[i + 1]
                    try:
                        e1, n1 = _wgs84_to_utm32n(float(p1["lon"]), float(p1["lat"]))
                        e2, n2 = _wgs84_to_utm32n(float(p2["lon"]), float(p2["lat"]))
                    except Exception:
                        continue
                    mx = (e1 + e2) / 2.0
                    my = (n1 + n2) / 2.0
                    seg_d = ((mx - easting) ** 2 + (my - northing) ** 2) ** 0.5
                    if _axis_d is None or seg_d < _axis_d:
                        _axis_d = seg_d
                        _axis_a = _normalize_axis_delta_deg(math.degrees(math.atan2((n2 - n1), (e2 - e1))))
                best_axis = _axis_a
    if best is None:
        return None
    max_snap = 180.0 if landmark else 250.0 if street else 400.0
    if best_dist is not None and best_dist <= max_snap:
        LAST_OSM_SNAP_INFO = {
            "easting": float(best[0]),
            "northing": float(best[1]),
            "label": best[2],
            "axis_deg": best_axis,
            "street": street,
            "road_code": road_code,
            "landmark": landmark,
        }
        print(f"[i] OSM snap: {best[2]}  Δ={best_dist:.0f} m")
        return best
    if GEOCODE_DEBUG and best_dist is not None:
        print(f"[~] OSM snap candidate too far ({best_dist:.0f} m) -- ignoring")
    return None


def _sample_way_points(geom: list, spacing_m: float = 80.0) -> list[tuple[float, float, float | None]]:
    pts: list[tuple[float, float, float | None]] = []
    if not isinstance(geom, list) or len(geom) < 2:
        return pts
    acc = 0.0
    try:
        prev_e, prev_n = _wgs84_to_utm32n(float(geom[0]["lon"]), float(geom[0]["lat"]))
    except Exception:
        return pts
    pts.append((float(prev_e), float(prev_n), None))
    for i in range(1, len(geom)):
        try:
            cur_e, cur_n = _wgs84_to_utm32n(float(geom[i]["lon"]), float(geom[i]["lat"]))
        except Exception:
            continue
        seg_len = ((cur_e - prev_e) ** 2 + (cur_n - prev_n) ** 2) ** 0.5
        if seg_len < 1e-6:
            continue
        axis = _normalize_axis_delta_deg(math.degrees(math.atan2((cur_n - prev_n), (cur_e - prev_e))))
        acc += seg_len
        if acc >= spacing_m:
            mx = (prev_e + cur_e) / 2.0
            my = (prev_n + cur_n) / 2.0
            pts.append((float(mx), float(my), float(axis)))
            acc = 0.0
        prev_e, prev_n = cur_e, cur_n
    return pts


def _road_segment_candidates(
    *,
    street: str | None,
    road_code: str | None,
    city: str | None,
    anchor_e: float | None,
    anchor_n: float | None,
    limit_points: int = 24,
) -> list[dict]:
    if not city:
        return []
    city_gc = geocode_address_to_utm32(city)
    if city_gc is None:
        return []
    ce, cn = city_gc
    lonlat = _utm32n_to_wgs84(ce, cn)
    if lonlat is None:
        return []
    lon, lat = lonlat
    radius = 3500.0
    snippets: list[str] = []
    if road_code:
        rc = str(road_code).strip().replace('"', "")
        snippets.extend([
            f'way(around:{radius},{lat:.6f},{lon:.6f})["ref"="{rc}"];',
            f'way(around:{radius},{lat:.6f},{lon:.6f})["official_ref"="{rc}"];',
        ])
    if street:
        st = str(street).strip().replace('"', "")
        snippets.append(f'way(around:{radius},{lat:.6f},{lon:.6f})["name"="{st}"];')
    if not snippets:
        return []
    payload = _overpass_query(f"[out:json][timeout:20];({''.join(snippets)});out geom;")
    if not payload or not isinstance(payload.get("elements"), list):
        return []
    results: list[dict] = []
    for elem in payload["elements"]:
        tags = elem.get("tags") or {}
        pts = _sample_way_points(elem.get("geometry"), spacing_m=80.0)
        for pe, pn, axis in pts:
            if anchor_e is not None and anchor_n is not None:
                dist = ((pe - anchor_e) ** 2 + (pn - anchor_n) ** 2) ** 0.5
                if dist > 2200.0:
                    continue
            results.append({
                "center_easting": pe,
                "center_northing": pn,
                "axis_deg": axis,
                "name": tags.get("name"),
                "ref": tags.get("ref") or tags.get("official_ref"),
            })
            if len(results) >= limit_points:
                break
        if len(results) >= limit_points:
            break
    return results


def _geometry_centroid_xy(geom: dict) -> tuple[float, float] | None:
    if not isinstance(geom, dict):
        return None
    coords = geom.get("coordinates")
    gtype = str(geom.get("type") or "")
    pts: list[tuple[float, float]] = []

    def _collect(obj):
        if isinstance(obj, (list, tuple)):
            if len(obj) >= 2 and all(isinstance(v, (int, float)) for v in obj[:2]):
                pts.append((float(obj[0]), float(obj[1])))
            else:
                for item in obj:
                    _collect(item)

    if gtype in {"Point", "MultiPoint", "LineString", "MultiLineString", "Polygon", "MultiPolygon"}:
        _collect(coords)
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _lookup_nrw_parcel_centroid(
    parcel_ref: str,
    *,
    anchor_e: float | None = None,
    anchor_n: float | None = None,
    city: str | None = None,
) -> tuple[float, float, dict] | None:
    """
    Resolve a parcel reference via the official NRW cadastral OGC API.
    Uses bbox narrowing plus client-side matching on queryable properties.
    Official API collection/queryable docs:
    - https://ogc-api.nrw.de/lika/v1/collections/flurstueck
    - https://ogc-api.nrw.de/lika/v1/collections/flurstueck/queryables
    """
    if not parcel_ref:
        return None
    m = re.search(
        r"(?:([A-ZÄÖÜ][A-Za-zÄÖÜäöüß\- ]+)\s*,\s*)?Flur\s*(\d+)\s*,?\s*Flurstück\s*([\d/]+)",
        parcel_ref,
        re.IGNORECASE,
    )
    if not m:
        return None
    gemarkung = re.sub(r"\s+", " ", (m.group(1) or "")).strip(" ,")
    flur = m.group(2).strip()
    flst = m.group(3).strip()
    if "/" in flst:
        zae, nen = flst.split("/", 1)
    else:
        zae, nen = flst, ""

    if anchor_e is None or anchor_n is None:
        if city:
            gc = geocode_address_to_utm32(city)
            if gc:
                anchor_e, anchor_n = gc
    if anchor_e is None or anchor_n is None:
        return None

    bbox_half = 2500.0
    params = {
        "f": "json",
        "bbox": f"{anchor_e-bbox_half:.1f},{anchor_n-bbox_half:.1f},{anchor_e+bbox_half:.1f},{anchor_n+bbox_half:.1f}",
        "bbox-crs": "http://www.opengis.net/def/crs/EPSG/0/25832",
        "limit": 100,
    }
    import urllib.parse
    import urllib.request
    url = "https://ogc-api.nrw.de/lika/v1/collections/flurstueck/items?" + urllib.parse.urlencode(params)
    cache_key = _cache_key("nrw-parcel", url)
    cached = _cache_read_json("nrw_parcel", cache_key)
    if isinstance(cached, dict):
        payload = cached
    else:
        req = urllib.request.Request(url, headers={"User-Agent": "AutoGeoreferencer/1.0 (QGIS plugin; georef tool)"})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            _cache_write_json("nrw_parcel", cache_key, payload)
        except Exception as exc:
            print(f"[~] NRW parcel lookup failed: {exc}")
            return None
    feats = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(feats, list):
        return None

    gem_norm = _normalize_text_token(gemarkung)
    city_norm = _normalize_text_token(city)
    best = None
    best_score = None
    for feat in feats:
        props = feat.get("properties") or {}
        score = 0.0
        if props.get("flur") and str(props.get("flur")).strip().lstrip("0") == flur.lstrip("0"):
            score += 6.0
        if props.get("flstnrzae") and str(props.get("flstnrzae")).strip().lstrip("0") == zae.lstrip("0"):
            score += 8.0
        if str(props.get("flstnrnen") or "").strip().lstrip("0") == nen.lstrip("0"):
            score += 4.0
        if gem_norm:
            feat_gem = _normalize_text_token(props.get("gemarkung"))
            if feat_gem == gem_norm or gem_norm in feat_gem or feat_gem in gem_norm:
                score += 6.0
        if city_norm:
            feat_city = _normalize_text_token(props.get("gemeinde"))
            if feat_city == city_norm or city_norm in feat_city or feat_city in city_norm:
                score += 3.0
        centroid = _geometry_centroid_xy(feat.get("geometry"))
        if centroid is None:
            continue
        dist = ((centroid[0] - anchor_e) ** 2 + (centroid[1] - anchor_n) ** 2) ** 0.5
        if dist <= 200:
            score += 5.0
        elif dist <= 1000:
            score += 2.0
        elif dist > 3000:
            score -= 3.0
        if best_score is None or score > best_score:
            best = (centroid[0], centroid[1], props)
            best_score = score
    if best is None or (best_score is not None and best_score < 12.0):
        return None
    print(
        f"[i] NRW parcel lookup: {parcel_ref} → "
        f"E={best[0]:.0f}  N={best[1]:.0f}  "
        f"(Gemarkung={best[2].get('gemarkung')}, Flur={best[2].get('flur')}, Flst={best[2].get('flstnrzae')}/{best[2].get('flstnrnen')})"
    )
    return float(best[0]), float(best[1]), dict(best[2])


def _normalize_axis_delta_deg(delta_deg: float) -> float:
    while delta_deg > 180.0:
        delta_deg -= 360.0
    while delta_deg <= -180.0:
        delta_deg += 360.0
    if delta_deg > 90.0:
        delta_deg -= 180.0
    elif delta_deg <= -90.0:
        delta_deg += 180.0
    return float(delta_deg)


def _estimate_plan_axis_deg(src_path: Path, crop_bbox: tuple | None = None) -> float | None:
    if not HAS_PIL:
        return None
    try:
        import numpy as np
        _img_base = Image.open(str(src_path)).convert("L")
        try:
            if crop_bbox:
                x1, y1, x2, y2 = crop_bbox
                _img_cropped = _img_base.crop((x1, y1, x2, y2))
                _img_base.close()
                img = _img_cropped
            else:
                img = _img_base
            img.thumbnail((1200, 1200), Image.LANCZOS)
            arr = np.array(img, dtype=np.uint8)
        finally:
            img.close()
        mask = arr < 235
        ys, xs = np.nonzero(mask)
        if len(xs) < 500:
            return None
        sample_step = max(1, len(xs) // 5000)
        xs = xs[::sample_step].astype(float)
        ys = ys[::sample_step].astype(float)
        # Convert image Y-down to math Y-up for axis estimation.
        pts = np.column_stack([xs - xs.mean(), -(ys - ys.mean())])
        cov = np.cov(pts, rowvar=False)
        vals, vecs = np.linalg.eigh(cov)
        vx, vy = vecs[:, int(np.argmax(vals))]
        angle = math.degrees(math.atan2(vy, vx))
        return _normalize_axis_delta_deg(angle)
    except Exception as exc:
        if GEOCODE_DEBUG:
            print(f"[~] Plan axis estimation failed: {exc}")
        return None


def _estimate_osm_way_axis_deg(
    easting: float,
    northing: float,
    *,
    street: str | None = None,
    road_code: str | None = None,
) -> float | None:
    lonlat = _utm32n_to_wgs84(easting, northing)
    if lonlat is None:
        return None
    lon, lat = lonlat
    radius = 350.0
    snippets: list[str] = []
    if road_code:
        rc = str(road_code).strip().replace('"', "")
        snippets.extend([
            f'way(around:{radius},{lat:.6f},{lon:.6f})["ref"="{rc}"];',
            f'way(around:{radius},{lat:.6f},{lon:.6f})["official_ref"="{rc}"];',
        ])
    if street:
        st = str(street).strip().replace('"', "")
        snippets.append(f'way(around:{radius},{lat:.6f},{lon:.6f})["name"="{st}"];')
    if not snippets:
        return None
    payload = _overpass_query(f"[out:json][timeout:20];({''.join(snippets)});out geom;")
    if not payload or not isinstance(payload.get("elements"), list):
        return None
    best_angle = None
    best_dist = None
    for elem in payload["elements"]:
        geom = elem.get("geometry")
        if not isinstance(geom, list) or len(geom) < 2:
            continue
        for i in range(len(geom) - 1):
            p1 = geom[i]
            p2 = geom[i + 1]
            try:
                e1, n1 = _wgs84_to_utm32n(float(p1["lon"]), float(p1["lat"]))
                e2, n2 = _wgs84_to_utm32n(float(p2["lon"]), float(p2["lat"]))
            except Exception:
                continue
            mx = (e1 + e2) / 2.0
            my = (n1 + n2) / 2.0
            dist = ((mx - easting) ** 2 + (my - northing) ** 2) ** 0.5
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_angle = _normalize_axis_delta_deg(math.degrees(math.atan2((n2 - n1), (e2 - e1))))
    if best_angle is not None and best_dist is not None and best_dist <= 250.0:
        return best_angle
    return None


def _derive_seed_rotation_deg(
    src_path: Path,
    crop_bbox: tuple | None,
    seed: dict,
    vision: dict,
) -> float | None:
    global LAST_OSM_SNAP_INFO
    try:
        seed_src = (CURRENT_SEED_SOURCE or str(seed.get("_source") or "")).strip().lower()
        seed_conf = (CURRENT_SEED_CONFIDENCE or str(seed.get("_seed_confidence") or "")).strip().lower()
        has_coord_anchors = (
            len(vision.get("easting_positions", []) or []) >= 2
            or len(vision.get("northing_positions", []) or []) >= 2
        )
        allow_vision_rotation = (
            has_coord_anchors
            or seed_src in ("manual_seed", "last_result", "ocr_coordinates", "nrw_parcel_lookup")
            or seed_conf in ("address", "parcel", "coordinates")
        )
        v_present = vision.get("north_arrow_present")
        v_north_up = vision.get("map_is_north_up")
        vrot = vision.get("north_arrow_direction_deg")
        # Only trust the Vision north-arrow when the model explicitly found one
        # on the plan itself. When canvas context is enabled the overview can
        # drift and produce a false orientation hint from unrelated context.
        if (
            USE_QGIS_CANVAS_VISION_CONTEXT
            and v_present is not True
        ):
            vrot = None
        if (
            isinstance(vrot, (int, float))
            and v_present is True
            and abs(float(vrot)) >= 0.2
        ):
            vrot_f = float(vrot)
            if not allow_vision_rotation:
                print(
                    f"[~] Ignoring north-arrow rotation {vrot_f:+.2f}° for low-confidence seed "
                    f"({(seed_src or seed_conf or 'unknown')[:40]}) until WMS confirms placement"
                )
                vrot = None
            # Only trust cardinal / near-cardinal rotations (0°, ±45°, ±90°, 180°).
            # Vision AI frequently hallucinates arbitrary angles that flip between
            # runs.  Angles that don't land near a cardinal direction are discarded
            # to prevent false rotation from defeating WMS refinement.
            if allow_vision_rotation:
                _CARDINALS = (0.0, 45.0, 90.0, 135.0, 180.0, -45.0, -90.0, -135.0, -180.0)
                _near_cardinal = any(abs(vrot_f - c) <= 10.0 for c in _CARDINALS)
                if not _near_cardinal:
                    print(f"[~] North arrow rotation {vrot_f:+.2f}° not near a cardinal direction -- ignoring")
                else:
                    print(f"[i] Seed rotation from north arrow: {vrot_f:+.2f}°")
                    return vrot_f
        if v_present is True and v_north_up is True and allow_vision_rotation:
            return 0.0
    except Exception:
        pass
    street = LAST_STRUCTURED_LOCATION_HINTS.get("site_street") if isinstance(LAST_STRUCTURED_LOCATION_HINTS, dict) else None
    road_codes = LAST_STRUCTURED_LOCATION_HINTS.get("road_codes") if isinstance(LAST_STRUCTURED_LOCATION_HINTS, dict) else None
    road_code = road_codes[0] if road_codes else None
    center_e = seed.get("center_easting")
    center_n = seed.get("center_northing")
    if center_e is None or center_n is None:
        return None
    plan_axis = _estimate_plan_axis_deg(src_path, crop_bbox=crop_bbox)
    world_axis = None
    if isinstance(seed.get("_rotation_world_axis"), (int, float)):
        world_axis = float(seed["_rotation_world_axis"])
        print(f"[i] Seed rotation using road-segment axis: {world_axis:+.1f}°")
    if isinstance(LAST_OSM_SNAP_INFO, dict):
        _cached_axis = LAST_OSM_SNAP_INFO.get("axis_deg")
        _cached_street = LAST_OSM_SNAP_INFO.get("street")
        _cached_road = LAST_OSM_SNAP_INFO.get("road_code")
        if world_axis is None and isinstance(_cached_axis, (int, float)) and (
            (_cached_street and street and str(_cached_street).lower() == str(street).lower())
            or (_cached_road and road_code and str(_cached_road).upper() == str(road_code).upper())
        ):
            world_axis = float(_cached_axis)
            print(f"[i] Seed rotation using cached OSM axis: {world_axis:+.1f}°")
    if world_axis is None:
        world_axis = _estimate_osm_way_axis_deg(float(center_e), float(center_n), street=street, road_code=road_code)
    if plan_axis is None or world_axis is None:
        return None
    rot = _normalize_axis_delta_deg(plan_axis - world_axis)
    if abs(rot) < 1.0:
        return None
    print(
        f"[i] Seed rotation from plan/OSM axis: plan={plan_axis:+.1f}°  "
        f"world={world_axis:+.1f}°  rot={rot:+.1f}°"
    )
    return rot


def geocode_address_to_utm32(address: str) -> tuple | None:
    """
    Geocode a free-text address via Nominatim (OpenStreetMap, no API key needed)
    and return (easting, northing) in the active TARGET_EPSG, or None on failure.
    Results are cached per (address, EPSG) so switching the active projection
    invalidates stale entries automatically.
    """
    results = geocode_address_candidates_utm32(address, limit=5)
    if not results:
        return None
    best = results[0]
    print(
        f"[i] Best geocode: '{address[:60]}'\n"
        f"    → lat={best['lat']:.5f}  lon={best['lon']:.5f}  ({best.get('display_name', '')[:80]})\n"
        f"    → E={best['easting']:.0f}  N={best['northing']:.0f}  (EPSG:{TARGET_EPSG})"
    )
    return float(best["easting"]), float(best["northing"])


def load_project_address() -> str | None:
    """
    Read project_address.json and return the address string if enabled,
    otherwise None.  Creates the template file on first call.
    """
    ensure_project_address_template()
    try:
        data = json.loads(PROJECT_ADDRESS_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[!] Could not read project address file: {exc}")
        return None
    if not data.get("enabled"):
        return None
    addr = (data.get("address") or "").strip()
    if not addr:
        print("[~] project_address.json enabled=true but address is empty -- ignoring")
        return None
    return addr


# Technical terms that should never be treated as place names
_NON_PLACE_TERMS = frozenset({
    "kranflache", "kranstellflachen", "kranstellflache", "montageflache",
    "blattlagerflache", "zuwegung", "hilfskranflache", "turmumfahrung",
    "lagerflache", "rustflache", "entwurfsplanung", "gelsenwasser",
    "gelsenkirchen", "karolingerstrasse", "zeichnung", "auftrags",
    "ubersichtsplan", "benennung", "zeichnungsnr", "auftraggeber",
    "nabenhohe", "telefon", "massiv", "future", "engineering",
    "datum", "index", "system", "acad", "telefax", "planverfasser",
    "laubholz", "nadelholz", "gewaesser", "bestand", "projekt",
    "planung", "bericht", "legende", "massstab", "north", "south",
    "east", "west", "anlage", "anhang", "bezeichnung", "datei",
    # Construction/engineering terms that geocode to wrong NRW locations
    "schacht", "mulde", "fahrbahn", "boschung", "gelander", "hohlen",
    "anschluss", "erhöhen", "erhohen", "toranlage", "steinmauer",
    "rigol", "kontrollschacht", "drosselorgan", "poller", "wildschutzzaun",
    "brucke", "steganlage", "umbau", "freianlagen", "okologische",
    "okologisch", "herstellung", "zuwegung", "teilnahmewettbewerb",
    "ausfertigung", "ausfertig", "masstab", "lagebezugsystem",
    "hohenbezugsystem", "gemarkung", "flurstueck", "flurstuck",
    "entwurf", "abschnitt", "legende", "strauss", "strasse", "wasser",
    # Traffic/infrastructure signs and facilities — geocode to unrelated NRW places
    "vorfahrt", "fahrradschild", "feuerwehrschild", "schild",
    # Generic height/elevation terms
    "hohen", "hohle", "niveau",
    # Titles/names that hit institution geocoding, not geography
    "furstin", "fuerstin", "christine", "franziska",
    # Sports/leisure facilities (not place names)
    "tennisplatz", "sportplatz", "spielplatz", "bolzplatz",
    # Utility-plan section identifiers
    "abschnitt", "lv", "trasse", "versorgung", "kanalisation",
    # Generic nouns that geocode to unrelated NRW features
    "fahrt", "muhle", "muhlen", "lasten", "industrie", "industriestrasse",
    "industriegebiet", "bahnhof", "messe", "hafen", "kanal", "bach",
    "burg", "turm", "halde", "zeche", "colonie", "siedlung",
    # Generic German administrative/formatting words
    "stadt", "kreis", "format", "rauxel",
    # Street-name suffixes that geocode as standalone addresses
    "wartburgstrasse", "rheinstrasse", "kirchstrasse",
    "hauptstrasse", "bahnhofstrasse", "schulstrasse",
})


def _choose_best_geocode_seed(
    query: str,
    *,
    source: str,
    scale: int,
    seed_confidence: str,
    wanted_city: str | None = None,
    wanted_postcode: str | None = None,
    street: str | None = None,
    road_code: str | None = None,
    landmark: str | None = None,
    anchor_e: float | None = None,
    anchor_n: float | None = None,
) -> dict | None:
    results = geocode_address_candidates_utm32(query, limit=7)
    if not results:
        return None
    ranked = _rank_geocode_candidates(
        results,
        wanted_city=wanted_city,
        wanted_postcode=wanted_postcode,
        road_code=road_code,
        landmark=landmark,
        anchor_e=anchor_e,
        anchor_n=anchor_n,
    )
    best = ranked[0]
    if GEOCODE_DEBUG:
        for idx, cand in enumerate(ranked[:3], start=1):
            print(
                f"    geocode[{idx}] score={cand.get('_score', 0.0):.2f}  "
                f"E={cand['easting']:.0f} N={cand['northing']:.0f}  {cand.get('display_name', '')[:120]}"
            )
    _snap = _snap_seed_to_osm(
        float(best["easting"]),
        float(best["northing"]),
        street=street,
        road_code=road_code,
        landmark=landmark,
    )
    if _snap:
        best["easting"], best["northing"] = _snap[0], _snap[1]
        best["_snap_label"] = _snap[2]
    print(
        f"[i] Structured geocode {source}: '{query}' → "
        f"E={best['easting']:.0f}  N={best['northing']:.0f}  score={best.get('_score', 0.0):.2f}"
    )
    return {
        "enabled": True,
        "_source": source,
        "_address": query,
        "center_easting": float(best["easting"]),
        "center_northing": float(best["northing"]),
        "scale_denominator": scale,
        "apply_after_crop": True,
        "_seed_confidence": seed_confidence,
    }


def _seed_from_structured_hints(
    hints: dict,
    scale: int,
    anchor_e: float | None = None,
    anchor_n: float | None = None,
) -> dict | None:
    if not hints:
        return None

    city = hints.get("site_city")
    postcode = hints.get("site_postcode")
    street = hints.get("site_street")
    house_no = hints.get("site_house_number")
    road_codes = list(hints.get("road_codes") or [])
    landmarks = list(hints.get("landmarks") or [])
    parcel_refs = list(hints.get("parcel_refs") or [])
    site_addresses = list(hints.get("site_addresses") or [])
    landmark_gc = None
    if landmarks and city:
        landmark_gc = geocode_address_to_utm32(f"{landmarks[0]}, {city}")

    if parcel_refs:
        for pref in parcel_refs[:3]:
            parcel_hit = _lookup_nrw_parcel_centroid(
                pref,
                anchor_e=anchor_e,
                anchor_n=anchor_n,
                city=city,
            )
            if parcel_hit:
                pe, pn, _props = parcel_hit
                return {
                    "enabled": True,
                    "_source": "nrw_parcel_lookup",
                    "_address": pref,
                    "center_easting": pe,
                    "center_northing": pn,
                    "scale_denominator": scale,
                    "apply_after_crop": True,
                    "_seed_confidence": "street",
                }

    if street and city and house_no:
        query = f"{street} {house_no}, {postcode + ' ' if postcode else ''}{city}"
        seed = _choose_best_geocode_seed(
            query,
            source="ocr_site_address",
            scale=scale,
            seed_confidence="street",
            wanted_city=city,
            wanted_postcode=postcode,
            street=street,
            road_code=road_codes[0] if road_codes else None,
            landmark=landmarks[0] if landmarks else None,
            anchor_e=anchor_e,
            anchor_n=anchor_n,
        )
        if seed:
            return seed

    if site_addresses:
        best_addr = site_addresses[0]
        seed = _choose_best_geocode_seed(
            best_addr["query"],
            source="ocr_site_address",
            scale=scale,
            seed_confidence="street",
            wanted_city=best_addr.get("city"),
            wanted_postcode=best_addr.get("postcode"),
            street=best_addr.get("street"),
            road_code=road_codes[0] if road_codes else None,
            landmark=landmarks[0] if landmarks else None,
            anchor_e=anchor_e,
            anchor_n=anchor_n,
        )
        if seed:
            return seed

    if street and city:
        query = f"{street}, {city}"
        seed = _choose_best_geocode_seed(
            query,
            source="ocr_street_city",
            scale=scale,
            seed_confidence="street",
            wanted_city=city,
            wanted_postcode=postcode,
            street=street,
            road_code=road_codes[0] if road_codes else None,
            landmark=landmarks[0] if landmarks else None,
            anchor_e=anchor_e,
            anchor_n=anchor_n,
        )
        if seed:
            segs = _road_segment_candidates(
                street=street,
                road_code=road_codes[0] if road_codes else None,
                city=city,
                anchor_e=seed["center_easting"],
                anchor_n=seed["center_northing"],
            )
            if segs:
                seed["_segment_candidate_count"] = len(segs)
                seed["_segment_ambiguous"] = len(segs) > 1
                if landmark_gc is not None:
                    best_seg = min(
                        segs,
                        key=lambda s: ((s["center_easting"] - landmark_gc[0]) ** 2 + (s["center_northing"] - landmark_gc[1]) ** 2) ** 0.5,
                    )
                    seed["center_easting"] = best_seg["center_easting"]
                    seed["center_northing"] = best_seg["center_northing"]
                    if isinstance(best_seg.get("axis_deg"), (int, float)):
                        seed["_rotation_world_axis"] = float(best_seg["axis_deg"])
                    seed["_segment_ambiguous"] = False
                    seed["_source"] = "ocr_street_city+landmark"
                    print(f"[i] Landmark resolved road segment among {len(segs)} candidate point(s)")
            return seed

    if road_codes and city:
        for road_code in road_codes[:3]:
            query = f"{road_code}, {city}"
            seed = _choose_best_geocode_seed(
                query,
                source="ocr_road_city",
                scale=scale,
                seed_confidence="street",
                wanted_city=city,
                wanted_postcode=postcode,
                road_code=road_code,
                landmark=landmarks[0] if landmarks else None,
                anchor_e=anchor_e,
                anchor_n=anchor_n,
            )
            if seed:
                segs = _road_segment_candidates(
                    street=street,
                    road_code=road_code,
                    city=city,
                    anchor_e=seed["center_easting"],
                    anchor_n=seed["center_northing"],
                )
                if segs:
                    seed["_segment_candidate_count"] = len(segs)
                    seed["_segment_ambiguous"] = len(segs) > 1
                return seed

    if landmarks and city:
        for landmark in landmarks[:3]:
            query = f"{landmark}, {city}"
            seed = _choose_best_geocode_seed(
                query,
                source="ocr_landmark_city",
                scale=scale,
                seed_confidence="street",
                wanted_city=city,
                wanted_postcode=postcode,
                road_code=road_codes[0] if road_codes else None,
                landmark=landmark,
                anchor_e=anchor_e,
                anchor_n=anchor_n,
            )
            if seed:
                return seed
    return None


def _geocode_ocr_place_names(
    ocr_text: str,
    scale: int,
    anchor_e: float | None = None,
    anchor_n: float | None = None,
) -> dict | None:
    """
    Extract candidate German place names from OCR text and geocode them via
    Nominatim.  Returns a seed dict centred on the average position of all
    successfully geocoded names, or None if fewer than 2 agree.

    This handles the common case where an overview plan prints village /
    district names as map labels but omits explicit UTM coordinates.

    Priority:
      0. "<street> in <city>" or "<road> in <city>" title-block patterns → street-level
      1. Explicit "Standort <City>" pattern → city-level
      2. Generic capitalized place-name tokens → feature-level centroid
    """
    global LAST_OCR_PLACES_CENTROID, LAST_OCR_PLACES_CENTROID_PRECISION
    import re as _re

    _structured_hints = _extract_structured_location_hints(ocr_text)
    _structured_seed = _seed_from_structured_hints(
        _structured_hints,
        scale,
        anchor_e=anchor_e,
        anchor_n=anchor_n,
    )
    if _structured_seed:
        _se = float(_structured_seed["center_easting"])
        _sn = float(_structured_seed["center_northing"])
        LAST_OCR_PLACES_CENTROID = (_se, _sn)
        LAST_OCR_PLACES_CENTROID_PRECISION = "street"
        return _structured_seed

    # ---- Priority -1: "<StreetName> in <City>" or "<Road> in <City>" ----
    # NRW Landesbetrieb plans always contain a project description line like:
    #   "L 663 Westabschnitt - Dortmunder Allee in Kamen"
    #   "Neubau der Ortsumgehung B 55n bei Hamm"
    #   "Radwegbau entlang K 14 in Selm-Bork"
    # This pattern gives street-level geocoding precision (~100 m) which is far
    # better than a city-centre geocode (±3-10 km).
    _STREET_SUFFIXES = (
        r"(?:stra(?:ße|sse)|allee|weg|platz|gasse|ring|damm|chaussee|ufer|pfad|kamp|wall|markt|gracht)"
    )
    _CITY_TOKEN = r"([A-ZÄÖÜ][A-Za-zäöüßÄÖÜ\-]+(?:\s+[A-ZÄÖÜ][A-Za-zäöüßÄÖÜ]+)?)"
    # Pattern A: "<StreetName...> in/bei <City>"
    _pat_street_in_city = _re.compile(
        r"([A-ZÄÖÜ][A-Za-zäöüßÄÖÜ\-]+\s+" + _STREET_SUFFIXES + r")"
        r"\s+(?:in|bei|nahe)\s+" + _CITY_TOKEN,
        _re.IGNORECASE,
    )
    # Pattern B: "<Road code e.g. L 663 / B 55 / A 42> [any text] in/bei <City>"
    _pat_road_in_city = _re.compile(
        r"\b([ABLKS]\s*\d{1,4}(?:[a-z]?)\b)"
        r"[^\n]{0,80}?\s+(?:in|bei|nahe)\s+" + _CITY_TOKEN,
        _re.IGNORECASE,
    )
    for _pat, _label in ((_pat_street_in_city, "street+city"), (_pat_road_in_city, "road+city")):
        for _m in _pat.finditer(ocr_text):
            _subject = _m.group(1).strip()
            _city    = _m.group(2).strip()
            # Filter out garbled/short tokens
            if len(_city) < 3 or _city.lower() in ("der", "die", "das", "den", "dem"):
                continue
            # Build street-level query
            if _label == "street+city":
                _query = f"{_subject}, {_city}"
            else:
                # Road + city: geocode "<road>, <city>" for a point on the road
                _query = f"{_subject.replace(' ', '')}, {_city}"
            _gc = geocode_address_to_utm32(_query)
            if _gc:
                _se, _sn = _gc
                if anchor_e is not None:
                    _dist = ((_se - anchor_e) ** 2 + (_sn - anchor_n) ** 2) ** 0.5
                    if _dist > 80_000:
                        print(f"[~] OCR {_label} '{_query}' geocoded but {_dist/1000:.0f} km from anchor -- skipping")
                        continue
                print(f"[i] OCR {_label}: '{_query}' → E={_se:.0f}  N={_sn:.0f}")
                # After finding the road-level geocode, try to refine it using
                # named landmarks from "Schild '...'" labels in the OCR.
                # Engineering plans mark on-plan signboards (Schild = sign).
                # "Schild 'Perthes-Zentrum'" means there is a physical sign for
                # that institution at this location — geocoding the institution
                # gives a precise point on the road section rather than the
                # road's OSM midpoint (which may be hundreds of metres off).
                for _lm in _re.finditer(r'[Ss]child\s+[\'"]([^\'"]{3,50})[\'"]', ocr_text):
                    _lname = _lm.group(1).strip()
                    _lquery = f"{_lname}, {_city}"
                    _lgc = geocode_address_to_utm32(_lquery)
                    if not _lgc:
                        _lgc = geocode_address_to_utm32(_lname)
                    if _lgc:
                        _le, _ln = _lgc
                        _ldist = ((_le - _se) ** 2 + (_ln - _sn) ** 2) ** 0.5
                        if _ldist <= 600:
                            print(f"[i] OCR landmark '{_lname}' refines road seed by {_ldist:.0f} m: E={_le:.0f}  N={_ln:.0f}")
                            _se, _sn = _le, _ln
                            _query = _lquery
                            break
                LAST_OCR_PLACES_CENTROID = (_se, _sn)
                LAST_OCR_PLACES_CENTROID_PRECISION = "street"
                return {
                    "enabled": True,
                    "_source": f"ocr_{_label.replace('+', '_')}",
                    "_address": _query,
                    "center_easting":  _se,
                    "center_northing": _sn,
                    "scale_denominator": scale,
                    "apply_after_crop": True,
                    "_seed_confidence": "street",
                }

    # ---- Priority 0: explicit "Standort <City>" pattern in OCR ----
    # Engineering plans often state the work site as "Standort Datteln" or
    # "Standort: Bottrop" in the title block.  This is more specific than any
    # generic place-name heuristic.
    for _sm in _re.finditer(r'Standort\s*:?\s*([A-ZÄÖÜ][a-zäöüßA-ZÄÖÜ\-]{2,})', ocr_text):
        _site_city = _sm.group(1).strip()
        _site_addr = _site_city
        _site_gc = geocode_address_to_utm32(_site_addr)
        if _site_gc:
            _se, _sn = _site_gc
            # Check against anchor if provided
            if anchor_e is not None:
                _dist = ((_se - anchor_e) ** 2 + (_sn - anchor_n) ** 2) ** 0.5
                if _dist > 50_000:
                    print(f"[~] OCR Standort '{_site_city}' geocoded but {_dist/1000:.0f} km from project address -- using as seed override")
            print(f"[i] OCR Standort site: '{_site_city}' → E={_se:.0f}  N={_sn:.0f}")
            LAST_OCR_PLACES_CENTROID = (_se, _sn)
            LAST_OCR_PLACES_CENTROID_PRECISION = "city"
            return {
                "enabled": True,
                "_source": "ocr_standort",
                "_address": _site_addr,
                "center_easting":  _se,
                "center_northing": _sn,
                "scale_denominator": scale,
                "apply_after_crop": True,
                "_seed_confidence": "city",
            }
    # Match title-case or ALL-CAPS words ≥ 5 characters that look like names.
    # We also try stripping a leading S/A/I artifact (common OCR noise before
    # a capital).
    raw_tokens = _re.findall(r'[A-ZÄÖÜ][A-ZÄÖÜa-zäöüß]{4,}', ocr_text)
    # Normalise: strip up to 2 leading all-caps noise chars before a capital
    tokens: list[str] = []
    for tok in raw_tokens:
        clean = _re.sub(r'^[A-Z]{1,2}([A-Z][a-zäöüß].+)$', r'\1', tok)
        tokens.append(clean)
    # Remove duplicates while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for t in tokens:
        low = t.lower().replace("ä", "a").replace("ö", "o").replace("ü", "u").replace("ß", "ss")
        if low not in seen and low not in _NON_PLACE_TERMS and len(t) >= 5:
            seen.add(low)
            unique.append(t)
    if not unique:
        return None

    # Try geocoding each candidate
    positions: list[tuple[float, float, str]] = []
    for name in unique[:12]:  # limit API calls
        candidate_addr = name
        gc = geocode_address_to_utm32(candidate_addr)
        if gc:
            e, n = gc
            positions.append((e, n, name))

    if len(positions) < 2:
        return None  # need at least 2 agreeing places

    # Cluster around the anchor (project address) when provided.
    # Two-pass radius: try tight (3 km) first so a precise postal-code geocode
    # gives a tight cluster; fall back to relaxed (5 km) for city-name geocodes
    # that land at the city centre several km from the actual plan location.
    # Scale both limits by plan scale so larger plans allow proportionally wider
    # searches (e.g. 1:5000 → 15 km / 25 km; 1:25000 → capped at 40 km).
    _tight_r  = min(40_000, max(3_000, scale * 3))
    _relax_r  = min(40_000, max(5_000, scale * 5))
    if anchor_e is not None and anchor_n is not None:
        cluster_e, cluster_n = anchor_e, anchor_n
    else:
        cluster_e = sorted(p[0] for p in positions)[len(positions) // 2]
        cluster_n = sorted(p[1] for p in positions)[len(positions) // 2]
    core = [(e, n, nm) for e, n, nm in positions
            if ((e - cluster_e) ** 2 + (n - cluster_n) ** 2) ** 0.5 < _tight_r]
    if len(core) < 2:
        # Tight radius gave too few places -- relax for city-level geocodes
        core = [(e, n, nm) for e, n, nm in positions
                if ((e - cluster_e) ** 2 + (n - cluster_n) ** 2) ** 0.5 < _relax_r]
    if len(core) < 2:
        return None

    avg_e = sum(pos[0] for pos in core) / len(core)
    avg_n = sum(pos[1] for pos in core) / len(core)
    names_used = ", ".join(nm for _, _, nm in core)
    print(f"[i] OCR place names geocoded ({len(core)} places: {names_used}): "
          f"E={avg_e:.0f}  N={avg_n:.0f}")
    LAST_OCR_PLACES_CENTROID = (avg_e, avg_n)
    LAST_OCR_PLACES_CENTROID_PRECISION = "feature"
    return {
        "enabled": True,
        "_source": "ocr_place_names",
        "_address": names_used,
        "center_easting":  avg_e,
        "center_northing": avg_n,
        "scale_denominator": scale,
        "apply_after_crop": True,
        "_seed_confidence": "feature",
    }


def derive_auto_seed(vision: dict, ocr_text: str, src_path: Path,
                     canvas_center: dict = None) -> dict:
    """
    Auto-derive a WMS starting seed from available context -- fully dynamic.

    Priority (highest → lowest):
      1. project_address.json geocoded via Nominatim OSM  [when enabled by user]
      2. QGIS canvas centre + last_result.json refinement when canvas is nearby
      3. Explicit UTM coordinate pair found in OCR text (Rechtswert / Hochwert)
      4. last_result.json alone (headless / no-canvas mode)
      5. Location name from OCR text or filename matched against city table

    project_address is highest priority because it is an EXPLICIT statement by the
    user of where the project is located.  The QGIS canvas position is just wherever
    the viewport happens to be panned and may point at a completely different area.
    If the address geocoding fails the seed falls back to the canvas position.

    Returns a seed dict or {} if no location could be determined.
    canvas_center: dict with keys center_easting / center_northing (EPSG:25832).
    """
    result: dict = {}

    # ---- scale (needed for all branches) --------------------------------
    # Priority: user override > Vision AI > OCR > default 5000
    scale = SCALE_OVERRIDE if (SCALE_OVERRIDE and 100 <= SCALE_OVERRIDE <= 100_000) else None
    if scale:
        print(f"[i] Using user-supplied scale 1:{scale}")
    if not scale:
        scale = vision.get("scale")
        if scale and not (100 <= scale <= 100_000):
            print(f"[!] Vision AI scale {scale} out of plausible range -- ignoring")
            scale = None
    if not scale and ocr_text:
        scale = _extract_best_scale(ocr_text)
    if not scale:
        scale = 5_000   # safe default

    _titleblock_meta = vision.get("title_block") if isinstance(vision.get("title_block"), dict) else {}
    _structured_hints = _extract_structured_location_hints(ocr_text or "", vision, _titleblock_meta)

    # ---- 1. project_address.json (highest-priority explicit user seed) --
    # The user explicitly states where the project is located -- this beats
    # the automatic canvas position which is just wherever QGIS happens to
    # be panned.  Canvas remains the fallback when no address is set.
    _proj_addr = load_project_address()
    if _proj_addr:
        _geocoded = geocode_address_to_utm32(_proj_addr)
        if _geocoded:
            _pa_e, _pa_n = _geocoded
            _proj_addr_conf = _classify_address_confidence(_proj_addr)
            result = {
                "enabled": True,
                "_source": "project_address",
                "_address": _proj_addr,
                "center_easting":  _pa_e,
                "center_northing": _pa_n,
                "scale_denominator": scale,
                "apply_after_crop": True,
                "_seed_confidence": _proj_addr_conf,
            }
            print(f"[i] Auto-seed from project address '{_proj_addr}': E={_pa_e:.0f}, N={_pa_n:.0f}")
            _structured_refine = _seed_from_structured_hints(
                _structured_hints,
                scale,
                anchor_e=_pa_e,
                anchor_n=_pa_n,
            )
            if _structured_refine:
                _sr_e = float(_structured_refine["center_easting"])
                _sr_n = float(_structured_refine["center_northing"])
                _sr_dist = ((_sr_e - _pa_e) ** 2 + (_sr_n - _pa_n) ** 2) ** 0.5
                _structured_cap = 250.0 if _address_is_specific(_proj_addr) else 12_000.0
                if _sr_dist <= _structured_cap:
                    print(f"[i] Structured OCR/title-block seed refined project address by {_sr_dist:.0f} m")
                    result.update({
                        "_source": _structured_refine.get("_source", "ocr_structured"),
                        "_address": _structured_refine.get("_address", _proj_addr),
                        "center_easting": _sr_e,
                        "center_northing": _sr_n,
                        "_seed_confidence": _structured_refine.get("_seed_confidence", "street"),
                    })
                    _pa_e, _pa_n = _sr_e, _sr_n
                else:
                    print(f"[~] Structured seed {_sr_dist:.0f} m from project address (cap={_structured_cap:.0f} m) -- keeping project address")
            # ---- road-code refinement: A42/B1/etc. in filename or OCR ----
            # If the plan filename or OCR contains a Bundesstraße/Autobahn code
            # (e.g. "A42", "B1"), geocoding "<road>, <city>" often returns a point
            # directly on the road rather than at the city centre.  This is more
            # useful as a WMS search seed for highway construction plans.
            # Extract city from project address: look for a word after a postal code
            # (e.g. "45141 Essen" → "Essen") or fall back to Vision AI location name.
            # Prefer the city from the explicit project address; only fall back to
            # Vision AI location if the address contains no city token.  Vision AI
            # often returns the client's office city (e.g. "Essen") rather than the
            # actual construction site, so we must not let it override a known address.
            _city_for_road = _extract_project_city(_proj_addr) or ""
            if not _city_for_road:
                _city_for_road = (vision.get("location_name") or "").strip()
            _project_addr_specific = _address_is_specific(_proj_addr)
            _road_search_text = src_path.stem + " " + (ocr_text[:1000] if ocr_text else "")
            _road_candidates = _extract_road_codes(_road_search_text)
            for _rc in _road_candidates:
                if not _city_for_road:
                    break
                _road_addr = f"{_rc}, {_city_for_road}"
                _road_gc = geocode_address_to_utm32(_road_addr)
                if _road_gc:
                    _re_e, _re_n = _road_gc
                    _rd = ((_re_e - _pa_e) ** 2 + (_re_n - _pa_n) ** 2) ** 0.5
                    _road_limit = 400.0 if _project_addr_specific else 5_000.0
                    if _rd <= _road_limit:
                        print(f"[i] Road '{_rc}' geocoded near {_city_for_road}: E={_re_e:.0f}  N={_re_n:.0f}  (Δ{_rd:.0f} m from project addr)")
                        result["center_easting"]  = _re_e
                        result["center_northing"] = _re_n
                        result["_source"] = "project_address+road"
                        if result.get("_seed_confidence") in ("city", "postcode"):
                            result["_seed_confidence"] = "feature"
                        _pa_e, _pa_n = _re_e, _re_n   # update anchor for OCR refine below
                        break
                    elif _project_addr_specific:
                        print(
                            f"[~] Road '{_rc}' geocoded but Δ{_rd:.0f} m from specific project address -- keeping address seed"
                        )
                    else:
                        print(f"[~] Road '{_rc}' geocoded but {_rd/1000:.0f} km from project address -- ignoring")
            # Try OCR place names / Standort to get a more specific seed.
            # OCR Standort overrides project address (it's the actual work site).
            # General place names are only accepted within 8 km of the project address.
            if ocr_text:
                _ocr_refine = _geocode_ocr_place_names(ocr_text, scale, anchor_e=_pa_e, anchor_n=_pa_n)
                if _ocr_refine:
                    _or_e = _ocr_refine["center_easting"]
                    _or_n = _ocr_refine["center_northing"]
                    _dist = ((_or_e - _pa_e) ** 2 + (_or_n - _pa_n) ** 2) ** 0.5
                    _is_standort = _ocr_refine.get("_source") == "ocr_standort"
                    if _is_standort:
                        # "Standort X" is useful when project_address is the client's HQ
                        # (a different city).  But if the Standort city is already present
                        # in the project_address string, the address is more specific
                        # (street-level) than the city-name Standort geocode — don't override.
                        _standort_city = (_ocr_refine.get("_address") or "").split(",")[0].strip().lower()
                        _addr_has_city = bool(_standort_city and _standort_city in _proj_addr.lower())
                        if _addr_has_city:
                            print(f"[~] OCR Standort city already in project address -- project address is more specific, keeping E={_pa_e:.0f}, N={_pa_n:.0f}")
                            _is_standort = False
                        else:
                            print(f"[i] OCR Standort overrides project address seed: E={_or_e:.0f}, N={_or_n:.0f}  (Δ{_dist:.0f} m from proj addr)")
                            result["center_easting"]  = _or_e
                            result["center_northing"] = _or_n
                            result["_source"] = "ocr_standort"
                    else:
                        # Limit how far OCR can move a project_address that is
                        # already precise.  A generic road-midpoint geocode
                        # (e.g. "Dortmunder Allee, Kamen") should not override a
                        # specific cross-street address (e.g. "Perthesstraße, Kamen")
                        # by several hundred metres.
                        #
                        # Tiers:
                        #   address has number + street  →  200 m cap (fine-tuning only)
                        #   address has street, no number →  400 m cap
                        #   address is postcode / city   → 8000 m cap (original behaviour)
                        _addr_has_street = bool(re.search(
                            r"(stra(?:ße|sse)|str\.|weg|allee|platz|gasse|ufer|"
                            r"chaussee|ring|damm|pfad|kamp|wall|markt)",
                            _proj_addr, re.IGNORECASE,
                        ))
                        if _project_addr_specific:
                            _ocr_dist_cap = 200.0
                        elif _addr_has_street:
                            _ocr_dist_cap = 400.0
                        else:
                            _ocr_dist_cap = 8000.0
                        if _dist <= _ocr_dist_cap:
                            print(f"[i] OCR place names refined seed by {_dist:.0f} m: E={_or_e:.0f}, N={_or_n:.0f}")
                            result["center_easting"]  = _or_e
                            result["center_northing"] = _or_n
                            result["_source"] = "project_address+ocr_places"
                            result["_seed_confidence"] = _ocr_refine.get("_seed_confidence", "feature")
                        else:
                            print(
                                f"[~] OCR place names {_dist:.0f} m from project address "
                                f"(cap={_ocr_dist_cap:.0f} m for {'specific' if _project_addr_specific else 'street-name'} address) -- keeping project address"
                            )
            return result
        else:
            print(f"[!] Project address geocoding failed -- trying OCR Standort, then Vision AI")
            # ---- Try OCR Standort before Vision AI when project_address fails ----
            if ocr_text:
                _ocr_standort_fb = _geocode_ocr_place_names(ocr_text, scale)
                if _ocr_standort_fb and _ocr_standort_fb.get("_source") == "ocr_standort":
                    _os_e = _ocr_standort_fb["center_easting"]
                    _os_n = _ocr_standort_fb["center_northing"]
                    print(f"[i] Auto-seed from OCR Standort (project addr failed): E={_os_e:.0f}, N={_os_n:.0f}")
                    return {
                        "enabled": True,
                        "_source": "ocr_standort",
                        "center_easting":  _os_e,
                        "center_northing": _os_n,
                        "scale_denominator": scale,
                        "apply_after_crop": True,
                    }
            # ---- Vision AI location name fallback ----
            _vis_loc = (vision.get("location_name") or "").strip()
            if _vis_loc:
                _vis_has_street = bool(re.search(
                    r"(?:stra(?:ße|sse)|allee|weg|platz|gasse|ring|damm|chaussee)",
                    _vis_loc, re.IGNORECASE,
                ))
                _vis_addr = _vis_loc
                _vis_gc = geocode_address_to_utm32(_vis_addr)
                if _vis_gc:
                    _vl_e, _vl_n = _vis_gc
                    _vis_conf = "street" if _vis_has_street else "city"
                    result = {
                        "enabled": True,
                        "_source": "vision_location",
                        "_address": _vis_addr,
                        "center_easting":  _vl_e,
                        "center_northing": _vl_n,
                        "scale_denominator": scale,
                        "apply_after_crop": True,
                        "_seed_confidence": _vis_conf,
                    }
                    print(f"[i] Auto-seed from Vision AI location '{_vis_loc}': E={_vl_e:.0f}, N={_vl_n:.0f}  confidence={_vis_conf}")
                    return result
                else:
                    print(f"[!] Vision AI location geocoding failed -- falling back to canvas position")
            else:
                print(f"[!] No Vision AI location available -- falling back to canvas position")
    else:
        # No project_address set — try OCR Standort first (plan names its own site),
        # then Vision AI location (title block), then canvas.

        _structured_seed = _seed_from_structured_hints(_structured_hints, scale)
        if _structured_seed:
            print(f"[i] Auto-seed from structured OCR/title-block extraction: "
                  f"E={_structured_seed['center_easting']:.0f}, N={_structured_seed['center_northing']:.0f}")
            return _structured_seed

        # ---- 1b-ocr. OCR title-block extraction (street+city, Standort, place names) ----
        # Priority: street+city patterns ("Dortmunder Allee in Kamen") are most
        # precise.  "Standort X" is second.  Generic place-name centroid is last.
        # All are more reliable than Vision AI for site location because they come
        # from the plan's own text, not from a thumbnail that may show client metadata.
        if ocr_text:
            _ocr_seed = _geocode_ocr_place_names(ocr_text, scale)
            if _ocr_seed:
                _os_src = _ocr_seed.get("_source", "")
                _os_e = _ocr_seed["center_easting"]
                _os_n = _ocr_seed["center_northing"]
                _os_conf = _ocr_seed.get("_seed_confidence", "city")
                print(f"[i] Auto-seed from OCR ({_os_src}): E={_os_e:.0f}, N={_os_n:.0f}  confidence={_os_conf}")
                return {
                    "enabled": True,
                    "_source": _os_src,
                    "_address": _ocr_seed.get("_address", ""),
                    "center_easting":  _os_e,
                    "center_northing": _os_n,
                    "scale_denominator": scale,
                    "apply_after_crop": True,
                    "_seed_confidence": _os_conf,
                }

        # ---- 1b-vis. Vision AI location name ----
        _vis_loc = (vision.get("location_name") or "").strip()
        if _vis_loc:
            # Vision may now return "Dortmunder Allee, Kamen" (street+city) or just "Kamen"
            _vis_has_street = bool(re.search(
                r"(?:stra(?:ße|sse)|allee|weg|platz|gasse|ring|damm|chaussee)",
                _vis_loc, re.IGNORECASE,
            ))
            _vis_addr = _vis_loc
            _vis_gc = geocode_address_to_utm32(_vis_addr)
            if _vis_gc:
                _vl_e, _vl_n = _vis_gc
                _vis_conf = "street" if _vis_has_street else "city"
                result = {
                    "enabled": True,
                    "_source": "vision_location",
                    "_address": _vis_addr,
                    "center_easting":  _vl_e,
                    "center_northing": _vl_n,
                    "scale_denominator": scale,
                    "apply_after_crop": True,
                    "_seed_confidence": _vis_conf,
                }
                print(f"[i] Auto-seed from Vision AI location '{_vis_loc}': E={_vl_e:.0f}, N={_vl_n:.0f}  confidence={_vis_conf}")
                return result

    # ---- 2. QGIS canvas viewport centre (+ last_result refinement) -----
    if canvas_center and canvas_center.get("center_easting") and canvas_center.get("center_northing"):
        ce = float(canvas_center["center_easting"])
        cn = float(canvas_center["center_northing"])
        if True:
            # If a previous run's result is close to the canvas centre, use it
            # as a more precise starting point (sub-pixel WMS-refined position).
            # "Close" = within WMS_MAX_SHIFT_M so WMS can still correct residuals.
            try:
                lr_data = json.loads(LAST_RESULT_FILE.read_text(encoding="utf-8"))
                if not _last_result_is_trusted(lr_data):
                    raise ValueError("last_result.json is missing trusted quality metadata")
                lr_e = float(lr_data.get("center_easting") or 0)
                lr_n = float(lr_data.get("center_northing") or 0)
                lr_dist = ((lr_e - ce) ** 2 + (lr_n - cn) ** 2) ** 0.5
                if lr_dist <= WMS_MAX_SHIFT_M:
                    result = {
                        "enabled": True,
                        "_source": "last_result",
                        "center_easting":  lr_e,
                        "center_northing": lr_n,
                        "scale_denominator": scale,
                        "apply_after_crop": True,
                    }
                    print(f"[i] Auto-seed from last result "
                          f"(canvas dist={lr_dist:.0f} m): E={lr_e:.0f}, N={lr_n:.0f}")
                    return result
            except Exception:
                pass  # no last_result.json or unreadable → fall through to canvas

            result = {
                "enabled": True,
                "_source": "qgis_canvas",
                "center_easting":  ce,
                "center_northing": cn,
                "scale_denominator": scale,
                "apply_after_crop": True,
            }
            print(f"[i] Auto-seed from QGIS canvas centre: E={ce:.0f}, N={cn:.0f}")
            return result

    # ---- 3. explicit coordinate pair in OCR ---------------------------
    # German plans often print Rechtswert / Hochwert in the title block
    _e_pat = re.compile(
        r'(?:Rechtswert|Ostwert|Easting|R\b|E\b)\D{0,8}(3\d{5}|4\d{5})',
        re.IGNORECASE)
    _n_pat = re.compile(
        r'(?:Hochwert|Nordwert|Northing|H\b|N\b)\D{0,8}(5[3-9]\d{5})',
        re.IGNORECASE)
    em = _e_pat.search(ocr_text or "")
    nm = _n_pat.search(ocr_text or "")
    if em and nm:
        result = {
            "enabled": True,
            "_source": "ocr_coordinates",
            "center_easting":  int(em.group(1)),
            "center_northing": int(nm.group(1)),
            "scale_denominator": scale,
            "apply_after_crop": True,
        }
        print(f"[i] Auto-seed from OCR coordinates: "
              f"E={result['center_easting']}, N={result['center_northing']}")
        return result

    # ---- 3b. OCR place names → geocode specific locations printed on plan ----
    # Plans often print village/district names as map labels even when they
    # omit explicit UTM coordinates.  Extract those names and geocode them to
    # get a seed that is much more precise than a city-centre geocode.
    if ocr_text:
        _ocr_place_seed = _geocode_ocr_place_names(ocr_text, scale)
        if _ocr_place_seed:
            print(f"[i] Auto-seed from OCR place names: "
                  f"E={_ocr_place_seed['center_easting']:.0f}  N={_ocr_place_seed['center_northing']:.0f}")
            return _ocr_place_seed

    # ---- 4. last_result.json fallback (no canvas available) -----------
    # Only reached when there is no live QGIS canvas.  The last result acts
    # as a reasonable default so standalone/headless runs still converge.
    try:
        lr_data = json.loads(LAST_RESULT_FILE.read_text(encoding="utf-8"))
        if not _last_result_is_trusted(lr_data):
            raise ValueError("last_result.json is missing trusted quality metadata")
        lr_e = float(lr_data.get("center_easting") or 0)
        lr_n = float(lr_data.get("center_northing") or 0)
        if lr_e != 0 or lr_n != 0:
            result = {
                "enabled": True,
                "_source": "last_result",
                "center_easting":  lr_e,
                "center_northing": lr_n,
                "scale_denominator": scale,
                "apply_after_crop": True,
            }
            print(f"[i] Auto-seed from last_result.json (no canvas): E={lr_e:.0f}, N={lr_n:.0f}")
            return result
    except Exception:
        pass

    # ---- 5. text / filename fallback (non-AI location) ---------------
    # Build a combined search string from OCR + filename only.
    candidates = [
        ocr_text or "",
        src_path.stem,
    ]
    combined = " ".join(candidates).lower()
    # Normalise umlauts so "ue/oe/ae" variants match
    combined = (combined
                .replace("ü", "u").replace("ue", "u")
                .replace("ö", "o").replace("oe", "o")
                .replace("ä", "a").replace("ae", "a")
                .replace("ß", "ss"))

    # Sort cities longest-first so "castrop-rauxel" beats "castrop"
    best_city = None
    for city in sorted(CITY_EASTING_UTM32, key=len, reverse=True):
        city_norm = (city
                     .replace("ü", "u").replace("ö", "o")
                     .replace("ä", "a").replace("ß", "ss"))
        if city_norm in combined:
            best_city = city
            break

    if best_city and best_city in CITY_NORTHING_UTM32 and TARGET_EPSG == 25832:
        result = {
            "enabled": True,
            "_source": f"city_fallback:{best_city}",
            "center_easting":  CITY_EASTING_UTM32[best_city],
            "center_northing": CITY_NORTHING_UTM32[best_city],
            "scale_denominator": scale,
            "apply_after_crop": True,
        }
        print(f"[i] Auto-seed fallback from text/filename '{best_city}': "
              f"E={result['center_easting']}, N={result['center_northing']}")
        return result

    print("[!] Auto-seed: could not determine location -- WMS will be skipped")
    return {}


# ---------------------------------------------------------------------------
# Plan type detection -- chooses the best WMS reference layer
# ---------------------------------------------------------------------------
def _normalize_search_text(text: str) -> str:
    text = (text or "").lower()
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii")


def detect_plan_type(img: "Image.Image", ocr_text: str, vision: dict, meta: dict | None = None, parsed: dict | None = None) -> str:
    """
    Classify the plan as one of:
      "aerial"  – NRW DOP orthophoto background  (match with NRW DOP WMS)
      "topo"    – topographic / overview map       (match with NRW DTK WMS)
      "vector"  – pure CAD / vector / line drawing (match with OSM WMS)

    Uses three signals (any strong signal wins):
      1. OCR keyword hints
      2. Vision AI notes
      3. Image pixel-variance (high = photographic, low = line art)
    """
    meta = meta or {}
    parsed = parsed or {}
    ocr_combined = _normalize_search_text(ocr_text or "")
    notes_combined = _normalize_search_text((vision.get("notes") or "") + " " + (vision.get("location_name") or ""))
    hint_chunks = []
    if meta.get("epsg_hint"):
        hint_chunks.append(f"epsg {meta['epsg_hint']}")
    if meta.get("scale_hint"):
        hint_chunks.append(f"scale 1:{meta['scale_hint']}")
    for _hint in parsed.get("crs_hints", []) or []:
        hint_chunks.append(str(_hint))
    if parsed.get("scale"):
        hint_chunks.append(f"parsed scale 1:{parsed['scale']}")
    hint_text = _normalize_search_text(" ".join(hint_chunks))
    combined = (ocr_combined + " " + notes_combined + " " + hint_text).strip()

    # -- keyword signals -------------------------------------------------
    aerial_kw = {
        "dop": 2.0,
        "orthophoto": 2.5,
        "luftbild": 2.5,
        "luftbildplan": 2.5,
        "nw_dop": 2.0,
        "aerial": 2.5,
        "aerial view": 3.0,
        "air photo": 2.5,
        "airphoto": 2.5,
        "ortho": 2.0,
    }
    topo_kw = {
        "dtk": 2.0,
        "topographisch": 2.0,
        "topographic": 2.0,
        "ubersichtskarte": 1.5,
        "uebersichtskarte": 1.5,
        "übersichtskarte": 1.5,
        "landkarte": 1.5,
        "nw_dtk": 2.0,
        "tk25": 1.5,
        "tk50": 1.5,
    }
    vector_kw = {
        "acad": 1.0,
        "autocad": 1.0,
        "cad": 1.0,
        "dwg": 1.0,
        "dxf": 1.0,
        "lageplan": 1.5,
        "ubersichtslageplan": 1.2,
        "uebersichtslageplan": 1.2,
        "ubersichtsplan": 1.2,
        "uebersichtsplan": 1.2,
        "übersichtslageplan": 1.2,
        "übersichtsplan": 1.2,
        "bebauungsplan": 1.5,
        "tiefbau": 2.5,
        "leitungsplan": 2.5,
        "rohrleitungsplan": 2.5,
        "entwasserungsplan": 2.5,
        "erschliessungsplan": 2.0,
        "entwässerungsplan": 2.5,
        "erschließungsplan": 2.0,
        "kanalplan": 2.0,
        "kanalnetz": 2.0,
    }

    kw_score = {"aerial": 0.0, "topo": 0.0, "vector": 0.0}

    def _score_keywords(text: str, keywords: dict, target: str, multiplier: float):
        for kw, weight in keywords.items():
            if kw in text:
                kw_score[target] += weight * multiplier

    _score_keywords(ocr_combined, aerial_kw, "aerial", 1.0)
    _score_keywords(ocr_combined, topo_kw, "topo", 1.0)
    _score_keywords(ocr_combined, vector_kw, "vector", 1.0)
    _score_keywords(notes_combined, aerial_kw, "aerial", 1.35)
    _score_keywords(notes_combined, topo_kw, "topo", 1.15)
    _score_keywords(notes_combined, vector_kw, "vector", 0.75)

    # -- image variance signal (photographic vs. line-art) ---------------
    variance  = None
    plan_type = "aerial"   # default
    if HAS_PIL and img is not None:
        try:
            import numpy as np
            small = img.copy()
            small.thumbnail((400, 400), Image.LANCZOS)
            arr = np.array(small.convert("L"), dtype=float)
            small.close()
            variance = float(arr.var())
        except Exception:
            variance = None

    if variance is not None:
        if variance >= 1_500:
            kw_score["aerial"] += 1.4
        elif variance >= 700:
            kw_score["topo"] += 0.9
        else:
            kw_score["vector"] += 0.9

    # If the model explicitly describes aerial imagery, do not let generic
    # engineering-plan terms force an OSM/vector reference.
    has_explicit_aerial = any(kw in notes_combined for kw in ("aerial", "aerial view", "orthophoto", "luftbild"))
    if has_explicit_aerial:
        kw_score["aerial"] += 2.0
        kw_score["vector"] -= 0.6

    # Hard override: decisive civil-engineering keywords in OCR / filename / notes
    # bypass the image-variance signal entirely and force "vector".
    _DECISIVE_VECTOR_KW = {
        "tiefbau", "leitungsplan", "rohrleitungsplan", "entwässerungsplan",
        "erschließungsplan", "kanalplan", "kanalnetz",
    }
    for _dkw in _DECISIVE_VECTOR_KW:
        if _dkw in combined:
            print(f"[i] Plan type forced to 'vector' (decisive keyword: '{_dkw}')")
            return "vector"

    # Engineering overview sheets with projected-CRS hints are usually sparse
    # vector/cadastral plans even when thumbnail variance looks photographic.
    # This stabilises classification for PDFs like "Übersichtsplan" with UTM/ETRS
    # text and avoids sending them down the orthophoto matcher.
    _vector_sheet_hints = 0
    if any(_kw in combined for _kw in (
        "lageplan", "übersichtsplan", "uebersichtsplan", "ubersichtsplan",
        "übersichtslageplan", "uebersichtslageplan", "abschnitte",
    )):
        _vector_sheet_hints += 1
    if any(_kw in combined for _kw in (
        "etrs", "utm", "utm32", "utm 32", "epsg:25832", "epsg 25832",
    )):
        _vector_sheet_hints += 1
    if re.search(r"\b1\s*[:]\s*(500|750|1000|1250|1500|2000)\b", combined):
        _vector_sheet_hints += 1
    if _vector_sheet_hints >= 2 and not has_explicit_aerial:
        print(f"[i] Plan type forced to 'vector' (engineering sheet heuristics: {_vector_sheet_hints})")
        return "vector"

    best_kw = max(kw_score, key=kw_score.get)
    sorted_scores = sorted(kw_score.items(), key=lambda kv: kv[1], reverse=True)
    best_score = sorted_scores[0][1]
    second_score = sorted_scores[1][1]

    # -- combine signals -------------------------------------------------
    if best_score <= 0:
        plan_type = "aerial" if variance is None else ("aerial" if variance >= 1_500 else "topo" if variance >= 700 else "vector")
        src = f"variance={variance:.0f}" if variance is not None else "default"
    elif has_explicit_aerial and kw_score["aerial"] >= kw_score["vector"] - 0.25:
        plan_type = "aerial"
        src = "vision+aerial"
    elif best_score - second_score < 0.75 and variance is not None:
        plan_type = "aerial" if variance >= 1_500 else "topo" if variance >= 700 else best_kw
        src = f"blended variance={variance:.0f}"
    else:
        plan_type = best_kw
        src = f"keywords {kw_score}"

    print(f"[i] Plan type detected: '{plan_type}'  (signal: {src})")
    return plan_type


def select_wms_config(plan_type: str):
    """Update the global WMS_URL / WMS_LAYER / WMS_FORMAT / WMS_VERSION / WMS_BGCOLOR to match the detected plan type."""
    global ACTIVE_WMS_CONFIG_KEY, WMS_URL, WMS_LAYER, WMS_FORMAT, WMS_VERSION, WMS_BGCOLOR
    # Merge built-ins with any user-added / dialog-injected configs.
    # WMS_CONFIGS_EXTRA wins over built-ins only for non-builtin keys.
    all_configs = {**WMS_CONFIGS, **WMS_CONFIGS_EXTRA}
    # "osm" is the canonical id for the OSM fallback; "vector_osm" is the
    # legacy alias kept for backward-compat with existing case bundles / logs.
    if plan_type not in all_configs and plan_type == "vector_osm":
        plan_type = "osm"
    cfg = all_configs.get(plan_type) or all_configs.get("aerial") or next(iter(all_configs.values()))
    ACTIVE_WMS_CONFIG_KEY = plan_type if plan_type in all_configs else "aerial"
    WMS_URL     = cfg["url"]
    WMS_LAYER   = cfg["layer"]
    WMS_FORMAT  = cfg.get("format", "image/jpeg")
    WMS_VERSION = cfg.get("wms_version", "1.3.0")
    WMS_BGCOLOR = cfg.get("bgcolor", "")
    print(f"[i] WMS selected: {cfg.get('label', plan_type)}")
    print(f"    URL  : {WMS_URL}")
    _layer_display = WMS_LAYER if len(WMS_LAYER) <= 60 else WMS_LAYER[:57] + "..."
    print(f"    Layer: {_layer_display}")
    if WMS_VERSION != "1.3.0":
        print(f"    WMS version: {WMS_VERSION}")


# ---------------------------------------------------------------------------
# QGIS canvas viewport reader
# ---------------------------------------------------------------------------
def get_qgis_canvas_info() -> dict:
    """
    Read the current QGIS map canvas viewport.

    Returns a dict with:
      center_easting   – map centre X in EPSG:25832
      center_northing  – map centre Y in EPSG:25832
      screenshot       – Path to a JPEG screenshot of the canvas (or None)

    Returns {} if not running inside QGIS or if the canvas is unavailable.
    """
    try:
        from qgis.utils import iface
        from qgis.core import (QgsCoordinateReferenceSystem,
                                QgsCoordinateTransform,
                                QgsProject, QgsPointXY)
        canvas = iface.mapCanvas()
        extent = canvas.extent()
        crs    = canvas.mapSettings().destinationCrs()

        center_x = (extent.xMinimum() + extent.xMaximum()) / 2.0
        center_y = (extent.yMinimum() + extent.yMaximum()) / 2.0

        # Transform canvas centre to the active projected CRS
        _active_crs_str = f"EPSG:{TARGET_EPSG}"
        target_crs = QgsCoordinateReferenceSystem(_active_crs_str)
        if crs.authid() != _active_crs_str:
            xform = QgsCoordinateTransform(crs, target_crs, QgsProject.instance())
            pt    = xform.transform(QgsPointXY(center_x, center_y))
            center_e, center_n = pt.x(), pt.y()
        else:
            center_e, center_n = center_x, center_y

        print(f"[i] QGIS canvas centre: E={center_e:.0f}  N={center_n:.0f}  ({_active_crs_str})")

        # Screenshot the canvas so we can send it to Vision AI for confirmation
        screenshot_path = None
        try:
            pixmap = canvas.grab()
            screenshot_path = _artifact_path("canvas_screenshot.jpg")
            pixmap.save(str(screenshot_path), "JPEG", 85)
            print(f"[i] Canvas screenshot saved → {screenshot_path}")
        except Exception as exc:
            print(f"[!] Canvas screenshot failed: {exc}")

        return {
            "center_easting":  center_e,
            "center_northing": center_n,
            "screenshot":      screenshot_path,
        }

    except ImportError:
        # Not inside QGIS – silently skip
        return {}
    except Exception as exc:
        print(f"[!] Could not read QGIS canvas: {exc}")
        return {}


def _repair_json_array(s: str) -> str:
    """
    Attempt to close a truncated JSON array by discarding everything after the
    last complete object (ends with '}') and appending ']'.
    Handles the common case where the model hits a token limit mid-response.
    """
    s = s.strip()
    if s.startswith('[') and s.endswith(']'):
        return s  # looks complete already
    last_obj_end = s.rfind('}')
    if last_obj_end < 0:
        return '[]'
    return s[:last_obj_end + 1] + ']'


def _img_to_b64(img: "Image.Image", quality: int = 88) -> str:
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.standard_b64encode(buf.getvalue()).decode()


def _ask_vision(client, prompt: str, img: "Image.Image",
                max_tokens: int = 1024,
                extra_imgs: list = None) -> str:
    """
    Send one (or more) images + a text prompt to the Vision model.
    extra_imgs: optional list of additional PIL images prepended before the main image
    (e.g. a QGIS canvas screenshot for location context).
    """
    content = []
    if extra_imgs:
        for ei in extra_imgs:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_img_to_b64(ei)}",
                              "detail": "low"},   # low detail = fewer tokens
            })
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{_img_to_b64(img)}",
                      "detail": "high"},
    })
    content.append({"type": "text", "text": prompt})
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": content}]
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw


def _extract_titleblock_metadata(client, img: "Image.Image") -> dict:
    W, H = img.width, img.height
    corners = [
        (int(W * 0.60), int(H * 0.60), W, H),
        (0, int(H * 0.60), int(W * 0.40), H),
        (int(W * 0.60), 0, W, int(H * 0.40)),
        (0, 0, int(W * 0.40), int(H * 0.40)),
    ]
    best: dict = {}
    best_score = -1
    for box in corners:
        try:
            tb = img.crop(box)
            raw = _ask_vision(client, TITLEBLOCK_SITE_PROMPT, tb, max_tokens=320)
            parsed = json.loads(raw)
        except Exception:
            continue
        if not isinstance(parsed, dict):
            continue
        score = sum(
            1 for key in (
                "project_site_street",
                "project_site_house_number",
                "project_site_city",
                "project_site_postcode",
                "project_road_code",
                "station_text",
                "landmark_name",
                "client_address",
            )
            if parsed.get(key)
        )
        if parsed.get("has_client_office_address"):
            score += 1
        if score > best_score:
            best = parsed
            best_score = score
    if best_score > 0:
        print(
            "[i] Title block metadata: "
            f"street={best.get('project_site_street')}  city={best.get('project_site_city')}  "
            f"road={best.get('project_road_code')}  office={'yes' if best.get('client_address') else 'no'}"
        )
    return best if isinstance(best, dict) else {}


def openai_vision_analysis(path: Path, meta: dict,
                           canvas_img: "Image.Image" = None,
                           ocr_text: str = "") -> dict:
    """
    Two-pass Vision analysis:
      Pass 1 – downscaled overview  → CRS, scale, location
              (canvas_img prepended when available so the AI sees the QGIS
               viewport as location context)
      Pass 2 – full-res margin crops → actual coordinate values
    """
    if not HAS_OPENAI:
        print("[!] Skipping Vision AI (no OpenAI SDK or API key)")
        return {}
    if not HAS_PIL:
        print("[!] Skipping Vision AI (Pillow needed)")
        return {}

    global LAST_VISION_RESULT
    _canvas_sig = None
    if canvas_img is not None:
        try:
            _canvas_sig = {
                "size": tuple(canvas_img.size),
                "hist": tuple(int(v) for v in canvas_img.convert("L").histogram()[::32][:8]),
            }
        except Exception:
            _canvas_sig = "present"
    cache_key = _cache_key(
        "vision",
        _path_fingerprint(path),
        meta or {},
        OPENAI_MODEL,
        _canvas_sig,
        hashlib.sha256((ocr_text or "").encode("utf-8")).hexdigest()[:16],
    )
    cached = _cache_read_json("vision", cache_key)
    if isinstance(cached, dict):
        LAST_VISION_RESULT = dict(cached)
        print("[i] Vision cache hit")
        return dict(cached)
    client = OpenAI(api_key=OPENAI_API_KEY)

    # ------------------------------------------------------------------
    # Load full image once (lazy – strips are sliced from this)
    # ------------------------------------------------------------------
    print("[~] Loading image for Vision analysis …")
    img = Image.open(str(path))
    if img.mode != "RGB":
        img = img.convert("RGB")
    W, H = img.width, img.height

    # ------------------------------------------------------------------
    # Pass 1: overview thumbnail → CRS / scale / location
    # ------------------------------------------------------------------
    thumb = img.copy()
    thumb.thumbnail((THUMB_MAX_PX, THUMB_MAX_PX), Image.LANCZOS)
    thumb_path = _artifact_path("thumb_for_ai.jpg")
    thumb.save(str(thumb_path), "JPEG", quality=85)
    print(f"[i] Overview thumbnail: {thumb.width}×{thumb.height} px")

    print("[~] Pass 1 – overview analysis …")
    # If a QGIS canvas screenshot is available, prepend it so the AI sees the
    # current map viewport as additional location context before analysing the plan.
    _canvas_extras = [canvas_img] if canvas_img is not None else None
    if _canvas_extras:
        print("[i]   (QGIS canvas screenshot included as location context)")
    raw1 = _ask_vision(client, OVERVIEW_PROMPT, thumb, max_tokens=512,
                       extra_imgs=_canvas_extras)
    try:
        overview = json.loads(raw1)
        overview["location_name"] = _resolve_location_name(overview.get("location_name"), path)
        if overview.get("north_arrow_direction_deg") is not None:
            try:
                deg = float(overview["north_arrow_direction_deg"])
                while deg > 180.0:
                    deg -= 360.0
                while deg <= -180.0:
                    deg += 360.0
                overview["north_arrow_direction_deg"] = deg
            except Exception:
                overview["north_arrow_direction_deg"] = None
        print(f"[✓] Overview: CRS={overview.get('coordinate_system')}  "
              f"EPSG={overview.get('epsg')}  Scale=1:{overview.get('scale')}  "
              f"Location={overview.get('location_name')}")
    except json.JSONDecodeError:
        print(f"[✗] Overview parse failed. Raw: {raw1[:200]}")
        overview = {}
        fallback_location = _resolve_location_name(None, path)
        if fallback_location:
            overview["location_name"] = fallback_location

    titleblock_meta = _extract_titleblock_metadata(client, img)
    if titleblock_meta:
        overview["title_block"] = titleblock_meta
        _tb_street = str(titleblock_meta.get("project_site_street") or "").strip()
        _tb_city = str(titleblock_meta.get("project_site_city") or "").strip()
        if _tb_street and _tb_city:
            overview["location_name"] = f"{_tb_street}, {_tb_city}"
        elif _tb_city and not overview.get("location_name"):
            overview["location_name"] = _tb_city

    if not overview.get("scale"):
        # Try all four corners -- German title blocks can be in any corner
        _corners = [
            (int(W * 0.60), int(H * 0.60), W, H),          # bottom-right (most common)
            (0, int(H * 0.60), int(W * 0.40), H),           # bottom-left
            (int(W * 0.60), 0, W, int(H * 0.40)),           # top-right
            (0, 0, int(W * 0.40), int(H * 0.40)),           # top-left
        ]
        for _box in _corners:
            try:
                tb = img.crop(_box)
                raw_scale = _ask_vision(client, TITLEBLOCK_SCALE_PROMPT, tb, max_tokens=128)
                scale_obj = json.loads(raw_scale)
                scale_val = scale_obj.get("scale")
                if scale_val and 100 <= int(scale_val) <= 100_000:
                    overview["scale"] = int(scale_val)
                    print(f"[i] Title block scale detected: 1:{overview['scale']}")
                    break
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Pass 2: margin strips → coordinate tick values via Tesseract OCR.
    # Tesseract image_to_data returns exact pixel bounding boxes for every
    # detected word -- no AI guesswork about positions needed here.
    # Vision AI is unreliable for pixel-coordinate estimation because it
    # interpolates / extrapolates instead of reading actual positions.
    # ------------------------------------------------------------------
    MARGIN_PX = max(300, int(min(W, H) * 0.05))

    # Narrow OCR coordinate expansion ranges to a ±100 km window around the
    # best available anchor position.  Priority matches derive_auto_seed:
    #   1. project_address geocoded position  (when address field is set)
    #   2. QGIS canvas centre
    #   3. Vision AI / OCR location name → city table lookup
    # This prevents coordinate label expansion from jumping to the wrong area.
    canvas_center = CANVAS_INFO_OVERRIDE if isinstance(CANVAS_INFO_OVERRIDE, dict) else None
    # Default coordinate expansion window — overridden to anchor±100km below
    # when an anchor is available.  The Germany-wide fallback is only used when
    # no anchor can be derived; provide a project address for non-German plans.
    e_min_loc, e_max_loc = _E_MIN, _E_MAX
    n_min_loc, n_max_loc = _N_MIN, _N_MAX

    # Try project_address first (highest-priority seed, same as derive_auto_seed)
    _ocr_anchor_e = _ocr_anchor_n = None
    _proj_addr_ocr = load_project_address()
    if _proj_addr_ocr:
        _gc = geocode_address_to_utm32(_proj_addr_ocr)
        if _gc:
            _ocr_anchor_e, _ocr_anchor_n = _gc
            print(f"[i] OCR coordinate ranges anchored to project address: "
                  f"E={_ocr_anchor_e:.0f}  N={_ocr_anchor_n:.0f}")

    # Prefer OCR/title-block-derived site metadata over Vision-only guesses.
    _structured_hints = _extract_structured_location_hints(ocr_text or "", overview, titleblock_meta)
    if _ocr_anchor_e is None:
        _site_seed = _seed_from_structured_hints(
            _structured_hints,
            overview.get("scale") or 5000,
        )
        if _site_seed:
            _ocr_anchor_e = float(_site_seed["center_easting"])
            _ocr_anchor_n = float(_site_seed["center_northing"])
            print(f"[i] OCR coordinate ranges anchored to structured site extraction: "
                  f"E={_ocr_anchor_e:.0f}  N={_ocr_anchor_n:.0f}")

    # Fall back to Vision AI location name (more reliable than canvas)
    if _ocr_anchor_e is None:
        _vis_loc_ocr = (overview.get("location_name") or "").strip()
        if _vis_loc_ocr:
            _vis_addr_ocr = _vis_loc_ocr
            _vis_gc_ocr = geocode_address_to_utm32(_vis_addr_ocr)
            if _vis_gc_ocr:
                _ocr_anchor_e, _ocr_anchor_n = _vis_gc_ocr
                print(f"[i] OCR coordinate ranges anchored to Vision location '{_vis_loc_ocr}': "
                      f"E={_ocr_anchor_e:.0f}  N={_ocr_anchor_n:.0f}")

    # Fall back to canvas centre when no address or vision anchor
    if _ocr_anchor_e is None and canvas_center and canvas_center.get("center_easting") and canvas_center.get("center_northing"):
        _ocr_anchor_e = float(canvas_center["center_easting"])
        _ocr_anchor_n = float(canvas_center["center_northing"])
        print(f"[i] OCR coordinate ranges anchored to canvas: "
              f"E={_ocr_anchor_e:.0f}  N={_ocr_anchor_n:.0f}")

    if _ocr_anchor_e is not None:
        e_min_loc = max(_E_MIN, _ocr_anchor_e - 100_000)
        e_max_loc = min(_E_MAX, _ocr_anchor_e + 100_000)
        n_min_loc = max(_N_MIN, _ocr_anchor_n - 100_000)
        n_max_loc = min(_N_MAX, _ocr_anchor_n + 100_000)
        print(f"    → E={e_min_loc:.0f}..{e_max_loc:.0f}  N={n_min_loc:.0f}..{n_max_loc:.0f}")
    elif TARGET_EPSG == 25832:
        # City lookup table only contains EPSG:25832 coordinates; skip for other CRS
        loc_key = (overview.get("location_name") or "").strip().lower()
        for city, e_c in CITY_EASTING_UTM32.items():
            if city in loc_key or loc_key in city:
                e_min_loc = max(_E_MIN, e_c - 100_000)
                e_max_loc = min(_E_MAX, e_c + 100_000)
                break
        for city, n_c in CITY_NORTHING_UTM32.items():
            if city in loc_key or loc_key in city:
                n_min_loc = max(_N_MIN, n_c - 100_000)
                n_max_loc = min(_N_MAX, n_c + 100_000)
                break

    def _expand_label(raw_val, vmin, vmax):
        """
        Expand an abbreviated coordinate label to its full UTM value.
        Examples: 395 → 395000, 5663 → 5663000, 395.5 → 395500, 395.8 → 395800
        Handles any decimal fraction, not just .5 half-increments.
        """
        if vmin <= raw_val <= vmax:
            return raw_val
        int_part = int(raw_val)
        frac = raw_val - int_part
        for mult in (1000, 100, 10):
            candidate = float(round(int_part * mult + frac * mult))
            if vmin <= candidate <= vmax:
                return candidate
        return None

    def _read_margin_ocr(strip_img, axis):
        """
        Run Tesseract on a margin strip, return (px_in_full_image, coord_value) pairs.
        axis: "easting" (top/bottom) or "northing" (left/right).

        Preprocessing pipeline:
          1. Convert to grayscale + LANCZOS upscale to improve small-digit recognition.
          2. Run Tesseract with digits-only whitelist in two PSM modes (11 + 6).
          3. For left/right strips also try the strip rotated 90° CW (some plans
             print northing labels vertically).
          4. Accept any Tesseract detection (conf ≥ 0); coordinate-range filter
             rejects non-label noise.
        """
        if not HAS_TESSERACT:
            return []
        vmin = e_min_loc if axis == "easting" else n_min_loc
        vmax = e_max_loc if axis == "easting" else n_max_loc
        sw, sh = strip_img.size

        # ---- preprocessing ------------------------------------------------
        gray = strip_img.convert("L")
        # Scale up narrow strips so digits are at least ~50 px tall
        min_text_side = min(sw, sh)
        if min_text_side < 600:
            scale = max(2, int(600 / min_text_side))
            scale = min(scale, 4)  # cap at 4× to avoid huge images
            gray = gray.resize((sw * scale, sh * scale), Image.LANCZOS)
            ocr_sw, ocr_sh = gray.size
        else:
            scale = 1
            ocr_sw, ocr_sh = sw, sh

        def _run_tess(img_pil, ocr_w, ocr_h, rotated=False):
            hits_local = []
            for psm in (11, 6):
                cfg = (r"--oem 3 --psm " + str(psm) +
                       r" -c tessedit_char_whitelist=0123456789.")
                try:
                    data = pytesseract.image_to_data(
                        img_pil, config=cfg,
                        output_type=pytesseract.Output.DICT,
                    )
                except Exception as exc:
                    print(f"    OCR error (psm {psm}): {exc}")
                    continue
                for i in range(len(data["text"])):
                    conf = int(data["conf"][i])
                    text = data["text"][i].strip()
                    if conf < 30 or len(text) < 2:
                        continue
                    clean = re.sub(r"[^0-9.]", "", text)
                    if len(clean) < 3 or clean.startswith("."):
                        continue
                    try:
                        raw_val = float(clean)
                    except ValueError:
                        continue
                    val = _expand_label(raw_val, vmin, vmax)
                    if val is None:
                        continue
                    cx = data["left"][i] + data["width"][i] / 2.0
                    cy = data["top"][i] + data["height"][i] / 2.0
                    if axis == "easting":
                        # cx is in the (possibly upscaled) strip → full-image x
                        hits_local.append((cx / ocr_w * W, val))
                    else:
                        if rotated:
                            # Strip was rotated 90° CW: original y = ocr_w - cx
                            orig_y = (ocr_w - cx) / ocr_w * H
                        else:
                            orig_y = cy / ocr_h * H
                        hits_local.append((orig_y, val))
            return hits_local

        hits = _run_tess(gray, ocr_sw, ocr_sh, rotated=False)

        # For northing strips: also try rotated 90° CW
        if axis == "northing" and len(hits) == 0:
            rotated = gray.rotate(-90, expand=True)
            rot_w, rot_h = rotated.size
            hits = _run_tess(rotated, rot_w, rot_h, rotated=True)
            if hits:
                print(f"    (rotated strip yielded {len(hits)} candidates)")

        # Debug: report raw hit count before dedup
        if hits:
            print(f"    raw OCR hits before dedup: {len(hits)}")
        return hits

    easting_positions  = []
    northing_positions = []

    margin_defs = [
        ("top",    img.crop((0,           0,          W,          MARGIN_PX)), "easting"),
        ("bottom", img.crop((0,           H-MARGIN_PX, W,         H)),         "easting"),
        ("left",   img.crop((0,           0,           MARGIN_PX, H)),         "northing"),
        ("right",  img.crop((W-MARGIN_PX, 0,           W,         H)),         "northing"),
    ]

    for side, strip, axis in margin_defs:
        sw, sh = strip.width, strip.height
        strip.save(str(_artifact_path(f"margin_{side}.jpg")), "JPEG", quality=90)
        print(f"[~] Pass 2 – reading {side} margin ({sw}×{sh}) via OCR …")
        hits = _read_margin_ocr(strip, axis)
        labels_found = []
        for px, val in hits:
            if axis == "easting":
                easting_positions.append((px, val))
            else:
                northing_positions.append((px, val))
            labels_found.append(str(int(val)) if val == int(val) else str(val))
        print(f"    {side}: {len(labels_found)} valid label(s) → {labels_found}")

    # ------------------------------------------------------------------
    # NOTE: Vision AI fallback for coordinate positions has been removed.
    # When OCR finds no labels the plan likely has no coordinate grid
    # (e.g. pure aerial-photo overlays).  In that case WMS image
    # correlation will localise the plan from the manual seed.
    # Vision AI cannot reliably estimate pixel positions -- it interpolates
    # uniformly and produces hallucinated coordinate values for plans
    # that have no grid, which corrupts the geotransform.
    # ------------------------------------------------------------------
    if len(easting_positions) < 2 and len(northing_positions) < 2:
        print("[i] OCR found no coordinate grid labels -- will rely on WMS + manual seed for localisation")

    # ------------------------------------------------------------------
    # Deduplicate then remove outliers via iterative residual filtering
    # ------------------------------------------------------------------
    def _dedup(positions):
        # Collect all pixel positions per coordinate value, then take the median.
        # This is more robust than first-occurrence when OCR reads the same label
        # multiple times from PSM 11 + PSM 6 passes or top/bottom strip overlap.
        groups: dict = {}
        for px, val in positions:
            groups.setdefault(val, []).append(px)
        return [(sorted(pxs)[len(pxs) // 2], val) for val, pxs in groups.items()]

    def _filter_outliers(positions, max_residual_m=500, label=""):
        """Iteratively remove the worst outlier until all residuals ≤ max_residual_m."""
        positions = list(positions)
        while len(positions) >= 3:
            px_vals = [p[0] for p in positions]
            co_vals = [p[1] for p in positions]
            intercept, slope = _linreg(px_vals, co_vals)
            residuals = [abs(v - (intercept + slope * x)) for x, v in positions]
            worst = max(residuals)
            if worst <= max_residual_m:
                break
            idx = residuals.index(worst)
            removed = positions.pop(idx)
            print(f"  [{label}] Removed outlier: px={removed[0]:.0f}, "
                  f"val={removed[1]:.0f}, residual={worst:.0f} m")
        return positions

    # max_residual_m=1500: allows ~2px error at the expected 0.66 m/px scale
    # (stricter than before, but not so tight that correct pixel_x readings fail)
    easting_positions  = _dedup(easting_positions)
    northing_positions = _dedup(northing_positions)

    # ------------------------------------------------------------------
    # Cross-axis scale validation for northings.
    # If the AI has read labels from an inset/legend (common on German plans),
    # the northing regression slope will be 2–4× larger than the easting slope
    # even though both represent square pixels.  Use RANSAC anchored to the
    # easting scale to select the internally-consistent northing subset before
    # the regular outlier filter runs.
    # ------------------------------------------------------------------
    if len(easting_positions) >= 3 and len(northing_positions) >= 3:
        _, de_ref = _linreg([px for px, _ in easting_positions], [e for _, e in easting_positions])
        _, dn_raw = _linreg([py for py, _ in northing_positions], [n for _, n in northing_positions])
        if abs(de_ref) > 0.01 and abs(dn_raw) > 0.01:
            ratio = abs(dn_raw / de_ref)
            if ratio > 2.0 or ratio < 0.5:
                ref_mpp = abs(de_ref)
                tol_m = 200.0   # metres -- tight enough to reject wrong-scale labels
                print(f"  [!] Northing/easting scale ratio={ratio:.2f} "
                      f"(ref mpp={ref_mpp:.4f}) -- applying cross-axis RANSAC filter")
                best_north: list = []
                for a_py, a_n in northing_positions:
                    # For a north-up map: n = a_n − ref_mpp × (py − a_py)
                    # Residual = (n − a_n) + ref_mpp × (py − a_py)
                    grp = [
                        (py, n) for py, n in northing_positions
                        if abs((n - a_n) + ref_mpp * (py - a_py)) < tol_m
                    ]
                    if len(grp) > len(best_north):
                        best_north = grp
                if len(best_north) >= 2 and len(best_north) < len(northing_positions):
                    print(f"  [~] Northing RANSAC: retained {len(best_north)} / "
                          f"{len(northing_positions)} points")
                    northing_positions = best_north

    # Dynamic residual tolerance: 30-pixel threshold derived from expected m/px.
    # Uses the already-resolved scale (SCALE_OVERRIDE > Vision AI > OCR > default).
    _scale_est = (SCALE_OVERRIDE if (SCALE_OVERRIDE and 100 <= SCALE_OVERRIDE <= 100_000) else None) or overview.get("scale") or 5000
    _dpi_est   = float(meta.get("dpi_x") or meta.get("dpi_y") or 300.0)
    _mpp_est   = _scale_est * 0.0254 / max(_dpi_est, 1.0)
    _max_res   = max(50.0, _mpp_est * 30)   # 30-pixel tolerance
    print(f"[i] Outlier filter tolerance: {_max_res:.1f} m  "
          f"(scale 1:{_scale_est}, DPI {_dpi_est:.0f}, mpp {_mpp_est:.4f})")
    easting_positions  = _filter_outliers(easting_positions,  max_residual_m=_max_res, label="easting")
    northing_positions = _filter_outliers(northing_positions, max_residual_m=_max_res, label="northing")

    print(f"[✓] Easting positions after filtering: {len(easting_positions)}, "
          f"Northing positions: {len(northing_positions)}")

    # Build grid_labels list for downstream GCP builder
    grid_labels = []
    for px_x, e in easting_positions:
        grid_labels.append({"value": e, "axis": "easting",
                             "px_x": px_x, "px_y": None})
    for px_y, n in northing_positions:
        grid_labels.append({"value": n, "axis": "northing",
                             "px_x": None, "px_y": px_y})

    result = {
        **overview,
        "width": W,
        "height": H,
        "dpi_x": meta.get("dpi_x"),
        "dpi_y": meta.get("dpi_y"),
        "easting_positions":  easting_positions,   # [(px_x, easting), ...]
        "northing_positions": northing_positions,  # [(px_y, northing), ...]
        "grid_labels": grid_labels,
    }
    if titleblock_meta:
        result["title_block"] = titleblock_meta
    LAST_VISION_RESULT = dict(result)
    _cache_write_json("vision", cache_key, result)
    return result


# ---------------------------------------------------------------------------
# STEP 5 – Fit affine geotransform by linear regression
# ---------------------------------------------------------------------------
def _linreg(xs, ys):
    """Simple linear regression: return (intercept, slope) for ys = intercept + slope*xs."""
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x*x for x in xs)
    sxy = sum(x*y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-9:
        # All x-values are identical -- return mean y, zero slope
        return sy / n, 0.0
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return intercept, slope


def _axis_quality(positions, rms):
    """Higher is better: reward more points and penalize high residuals."""
    if not positions:
        return 0.0
    return len(positions) / max(rms, 1.0)


def _fallback_location_name(path: Path | None) -> str | None:
    """Infer a likely location name from the source file/path when overview AI misses it."""
    if path is None:
        return None
    text = str(path).lower()
    for city in CITY_EASTING_UTM32.keys():
        if city in text:
            return city
    return None


def _resolve_location_name(ai_location: str | None, path: Path | None) -> str | None:
    """
    Prefer a filename/path-derived location when it conflicts with a weaker AI
    overview guess. The file name for these plans is usually more reliable.

    Special case: if the AI returned a rich "street, city" string and the
    filename-derived city is already contained within it, keep the richer AI
    result (the street name provides additional precision we should not discard).
    """
    fallback = _fallback_location_name(path)
    if not ai_location:
        return fallback
    loc = ai_location.strip().lower()
    if fallback and fallback not in loc:
        print(f"[!] Overview location '{ai_location}' conflicts with filename-derived '{fallback}'. Using '{fallback}'.")
        return fallback
    # AI location either matches the filename city or is a richer street+city string
    return ai_location


def ensure_manual_seed_template():
    if MANUAL_SEED_FILE.exists():
        return
    template = {
        "enabled": False,
        "notes": (
            "Set enabled=true and fill center_easting/center_northing (EPSG:25832) "
            "to hard-override the starting location. Leave disabled to use the "
            "live QGIS canvas position instead."
        ),
        "center_easting": None,
        "center_northing": None,
        "origin_easting": None,
        "origin_northing": None,
        "scale_denominator": None,
        "meters_per_pixel": None,
        "apply_after_crop": True,
    }
    MANUAL_SEED_FILE.write_text(json.dumps(template, indent=2), encoding="utf-8")


def load_manual_seed() -> dict | None:
    """
    Return the user-written manual seed if enabled.
    Auto-written entries (from previous runs) live in last_result.json,
    not here -- this function only returns genuine user overrides.
    """
    return None
    ensure_manual_seed_template()
    try:
        data = json.loads(MANUAL_SEED_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[!] Could not read manual seed file: {exc}")
        return None
    if not data.get("enabled"):
        return None
    # Guard: if this is an auto-written entry, migrate it to last_result.json
    # and treat manual_seed.json as disabled.
    _is_auto = (
        data.get("_source") in ("auto_result", "last_result", "qgis_canvas")
        or str(data.get("_note", "")).startswith("Auto-suggested from last successful run")
        or str(data.get("_note", "")).startswith("Auto-written after each")
    )
    if _is_auto:
        # Migrate to last_result.json if it has valid coordinates
        _lr_e = data.get("center_easting")
        _lr_n = data.get("center_northing")
        if _lr_e and _lr_n:
            try:
                _migrate = {
                    "_source": "auto_result",
                    "_note": "Migrated from manual_seed.json (legacy auto-write).",
                    "center_easting": float(_lr_e),
                    "center_northing": float(_lr_n),
                    "scale_denominator": data.get("scale_denominator"),
                }
                if not LAST_RESULT_FILE.exists():
                    LAST_RESULT_FILE.write_text(
                        json.dumps(_migrate, indent=2, ensure_ascii=False), encoding="utf-8"
                    )
                    print(f"[i] Migrated auto-seed from manual_seed.json → {LAST_RESULT_FILE}")
            except Exception:
                pass
        # Reset manual_seed.json to clean disabled template (write directly,
        # not via ensure_manual_seed_template which would no-op since file exists)
        _clean = {
            "enabled": False,
            "notes": (
                "Set enabled=true and fill center_easting/center_northing (EPSG:25832) "
                "to hard-override the starting location. Leave disabled to use the "
                "live QGIS canvas position instead."
            ),
            "center_easting": None,
            "center_northing": None,
            "origin_easting": None,
            "origin_northing": None,
            "scale_denominator": None,
            "meters_per_pixel": None,
            "apply_after_crop": True,
        }
        try:
            MANUAL_SEED_FILE.write_text(json.dumps(_clean, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[i] manual_seed.json reset to clean template (auto-result moved to last_result.json)")
        except Exception:
            pass
        return None
    # Require at least one coordinate to be set
    if not (data.get("center_easting") or data.get("origin_easting")):
        print("[~] manual_seed.json enabled=true but no coordinates set -- ignoring")
        return None
    print(f"[i] Manual seed enabled via {MANUAL_SEED_FILE}")
    return data


def build_manual_seed_geotransform(seed: dict, width_px: int, height_px: int, vision: dict) -> tuple | None:
    dpi_x = vision.get("dpi_x")
    dpi_y = vision.get("dpi_y")
    dpi = None
    if dpi_x and dpi_y and dpi_x > 0 and dpi_y > 0:
        dpi = (float(dpi_x) + float(dpi_y)) / 2.0

    mpp = seed.get("meters_per_pixel")
    if not mpp:
        # User override takes priority over everything
        if SCALE_OVERRIDE and 100 <= SCALE_OVERRIDE <= 100_000:
            scale_den = SCALE_OVERRIDE
        else:
            scale_den = seed.get("scale_denominator") or vision.get("scale")
        # Sanity-check: ignore nonsensical scales (e.g. 1:10 from a version string)
        if scale_den and not (100 <= float(scale_den) <= 100_000):
            print(f"[!] Seed scale_denominator {scale_den} out of plausible range -- trying Vision AI scale")
            scale_den = vision.get("scale")
            if scale_den and not (100 <= float(scale_den) <= 100_000):
                scale_den = None
        if scale_den and dpi:
            mpp = float(scale_den) * 0.0254 / dpi
    if not mpp:
        print("[!] Seed missing meters_per_pixel and no usable scale/DPI -- using 5000:1 default")
        # Fall back to a reasonable default so WMS refinement can still run
        if dpi:
            mpp = 5_000 * 0.0254 / dpi
        else:
            print("[!] No DPI available -- cannot compute mpp")
            return None
    mpp = float(mpp)

    origin_e = seed.get("origin_easting")
    origin_n = seed.get("origin_northing")
    center_e = seed.get("center_easting")
    center_n = seed.get("center_northing")
    rotation_deg = float(seed.get("_rotation_deg") or 0.0)

    if origin_e is None:
        if center_e is None:
            print("[!] Manual seed missing both origin_easting and center_easting")
            return None
        origin_e = float(center_e) - mpp * (width_px / 2.0)
    else:
        origin_e = float(origin_e)

    if origin_n is None:
        if center_n is None:
            print("[!] Manual seed missing both origin_northing and center_northing")
            return None
        origin_n = float(center_n) + mpp * (height_px / 2.0)
    else:
        origin_n = float(origin_n)

    if abs(rotation_deg) >= 0.2 and center_e is not None and center_n is not None:
        theta = math.radians(rotation_deg)
        gt1 = mpp * math.cos(theta)
        gt2 = -mpp * math.sin(theta)
        gt4 = -mpp * math.sin(theta)
        gt5 = -mpp * math.cos(theta)
        gt0 = float(center_e) - gt1 * (width_px / 2.0) - gt2 * (height_px / 2.0)
        gt3 = float(center_n) - gt4 * (width_px / 2.0) - gt5 * (height_px / 2.0)
        gt = (gt0, gt1, gt2, gt3, gt4, gt5)
    else:
        gt = (origin_e, mpp, 0.0, origin_n, 0.0, -mpp)
    coverage_m = mpp * max(width_px, height_px)
    print(f"[i] Seed geotransform: origin=({origin_e:.0f}, {origin_n:.0f})  "
          f"mpp={mpp:.4f} m/px  coverage≈{coverage_m/1000:.1f} km"
          + (f"  rot={rotation_deg:+.2f}°" if abs(rotation_deg) >= 0.2 else ""))
    return gt


def _reproject_geotransform(gt: tuple, width_px: int, height_px: int, src_epsg: int, dst_epsg: int) -> tuple:
    """
    Approximate an affine geotransform in dst_epsg by transforming three image
    corners from src_epsg. Suitable for the small plan extents handled here.
    """
    if src_epsg == dst_epsg or not HAS_PYPROJ:
        return gt
    try:
        tfm = Transformer.from_crs(f"EPSG:{src_epsg}", f"EPSG:{dst_epsg}", always_xy=True)
        tl_x, tl_y = tfm.transform(gt[0], gt[3])
        tr_src_x = gt[0] + gt[1] * width_px
        tr_src_y = gt[3] + gt[4] * width_px
        bl_src_x = gt[0] + gt[2] * height_px
        bl_src_y = gt[3] + gt[5] * height_px
        tr_x, tr_y = tfm.transform(tr_src_x, tr_src_y)
        bl_x, bl_y = tfm.transform(bl_src_x, bl_src_y)
        return (
            float(tl_x),
            float((tr_x - tl_x) / max(width_px, 1)),
            float((bl_x - tl_x) / max(height_px, 1)),
            float(tl_y),
            float((tr_y - tl_y) / max(width_px, 1)),
            float((bl_y - tl_y) / max(height_px, 1)),
        )
    except Exception as exc:
        print(f"[!] Could not reproject geotransform EPSG:{src_epsg}→EPSG:{dst_epsg}: {exc}")
        return gt


def compute_geotransform(vision: dict) -> tuple | None:
    """
    Fit a linear regression to easting and northing label positions
    and return the 6-element GDAL geotransform tuple, plus residuals.

    GT[0] = top-left easting   GT[1] = m/px (east)  GT[2] = 0 (no rotation)
    GT[3] = top-left northing  GT[4] = 0            GT[5] = m/px (north, negative)

    Two automatic corrections are applied if needed:
      1. Square-pixel: if |X_scale / Y_scale| differs by > 15%, the easting scale
         (more labels, higher confidence) is used for both axes.
      2. Location anchor: if the resulting northing origin deviates more than 15 km
         from the expected centre for the detected location name, the northing
         origin is shifted to match the lookup table value.
    """
    e_pos = vision.get("easting_positions", [])   # [(px_x, easting), ...]
    n_pos = vision.get("northing_positions", [])  # [(px_y, northing), ...]
    img_h = vision.get("height")

    if len(e_pos) < 2 or len(n_pos) < 2:
        print("[✗] Not enough positions for regression")
        return None

    # North-up maps must have decreasing northing as pixel Y increases.
    _, dn_probe = _linreg([py for py, _ in n_pos], [n for _, n in n_pos])
    _n_flip_applied = False
    if dn_probe > 0:
        if img_h is None:
            img_h = max(py for py, _ in n_pos)
        print("[!] Northing slope is positive; flipping Y positions for north-up map")
        n_pos = [(img_h - py, n) for py, n in n_pos]
        _n_flip_applied = True

    px_xs,  e_vals = zip(*e_pos)
    px_ys,  n_vals = zip(*n_pos)

    e0, de = _linreg(list(px_xs), list(e_vals))   # easting  = e0 + de * px_x
    n0, dn = _linreg(list(px_ys), list(n_vals))   # northing = n0 + dn * px_y

    # Residuals (RMS in metres)
    e_res = ((sum((e - (e0 + de*x))**2 for x, e in e_pos) / len(e_pos)) ** 0.5)
    n_res = ((sum((n - (n0 + dn*y))**2 for y, n in n_pos) / len(n_pos)) ** 0.5)

    print(f"  Easting  regression: origin={e0:.2f}, scale={de:.4f} m/px, RMS={e_res:.2f} m")
    print(f"  Northing regression: origin={n0:.2f}, scale={dn:.4f} m/px, RMS={n_res:.2f} m")

    if abs(de) < 0.01 or abs(dn) < 0.01:
        print("[✗] Regression produced near-zero pixel size -- labels likely wrong")
        return None

    e_quality = _axis_quality(e_pos, e_res)
    n_quality = _axis_quality(n_pos, n_res)
    print(f"  Axis quality: easting={e_quality:.2f}, northing={n_quality:.2f}")

    # ------------------------------------------------------------------
    # Correction 1: Force square pixels when X/Y scale ratio is off.
    # Use the better-constrained axis instead of always trusting easting.
    # ------------------------------------------------------------------
    ratio = abs(de / dn)
    if ratio < 0.85 or ratio > 1.15:
        print(f"[!] Pixel scale mismatch: X={de:.4f} m/px, Y={dn:.4f} m/px, ratio={ratio:.2f}")
        # Do not let a sparse axis dominate the whole fit.
        if len(n_pos) < 5 <= len(e_pos):
            chosen_scale = abs(de)
            print(f"    Northing has only {len(n_pos)} points; keeping easting scale ({chosen_scale:.4f} m/px)")
        elif len(e_pos) < 5 <= len(n_pos):
            chosen_scale = abs(dn)
            print(f"    Easting has only {len(e_pos)} points; keeping northing scale ({chosen_scale:.4f} m/px)")
        elif n_quality > e_quality:
            chosen_scale = abs(dn)
            print(f"    Using northing scale for square pixels ({chosen_scale:.4f} m/px)")
        else:
            chosen_scale = abs(de)
            print(f"    Using easting scale for square pixels ({chosen_scale:.4f} m/px)")

        de = chosen_scale
        dn = -chosen_scale   # must be negative for north-up

        # Recompute both origins using the surviving labels as anchors.
        e0_estimates = [e - de * px for px, e in e_pos]
        n0_estimates = [n - dn * py for py, n in n_pos]
        e0 = sorted(e0_estimates)[len(e0_estimates) // 2]
        n0 = sorted(n0_estimates)[len(n0_estimates) // 2]

        e_res2 = ((sum((e - (e0 + de*x))**2 for x, e in e_pos) / len(e_pos)) ** 0.5)
        n_res2 = ((sum((n - (n0 + dn*y))**2 for y, n in n_pos) / len(n_pos)) ** 0.5)
        print(f"    After square-pixel correction: E origin={e0:.2f}, X scale={de:.4f} m/px, RMS={e_res2:.2f} m")
        print(f"                                   N origin={n0:.2f}, Y scale={dn:.4f} m/px, RMS={n_res2:.2f} m")
    else:
        print(f"  Pixel scale ratio: {ratio:.3f} (square pixels ✓)")

    scale_den = vision.get("scale")
    dpi_x = vision.get("dpi_x")
    dpi_y = vision.get("dpi_y")
    dpi = None
    expected_mpp = None
    if dpi_x and dpi_y and dpi_x > 0 and dpi_y > 0:
        dpi = (float(dpi_x) + float(dpi_y)) / 2.0
    if scale_den and dpi:
        expected_mpp = float(scale_den) * 0.0254 / dpi
    # Only let the printed map scale override the grid fit when the
    # coordinate-derived solution is weak. If the extracted grid labels are
    # numerous and internally consistent, they are more trustworthy than the
    # model's read of the title-block scale.
    if scale_den and dpi and max(e_quality, n_quality) < 1.0:
        current_mpp = (abs(de) + abs(dn)) / 2.0
        deviation = abs(current_mpp - expected_mpp) / max(expected_mpp, 1e-9)
        print(f"  Scale check: plan 1:{int(scale_den)} @ {dpi:.0f} DPI -> expected {expected_mpp:.4f} m/px")
        if deviation > 0.20:
            print(f"[!] Pixel size deviates {deviation*100:.0f}% from printed scale; clamping to {expected_mpp:.4f} m/px")
            de = expected_mpp
            dn = -expected_mpp
            e0_estimates = [e - de * px for px, e in e_pos]
            n0_estimates = [n - dn * py for py, n in n_pos]
            e0 = sorted(e0_estimates)[len(e0_estimates) // 2]
            n0 = sorted(n0_estimates)[len(n0_estimates) // 2]
            e_res3 = ((sum((e - (e0 + de*x))**2 for x, e in e_pos) / len(e_pos)) ** 0.5)
            n_res3 = ((sum((n - (n0 + dn*y))**2 for y, n in n_pos) / len(n_pos)) ** 0.5)
            print(f"    After scale clamp: E origin={e0:.2f}, X scale={de:.4f} m/px, RMS={e_res3:.2f} m")
            print(f"                       N origin={n0:.2f}, Y scale={dn:.4f} m/px, RMS={n_res3:.2f} m")
    elif scale_den and dpi:
        print(f"  Scale check: plan 1:{int(scale_den)} @ {dpi:.0f} DPI -> expected {expected_mpp:.4f} m/px")
        print("  Keeping coordinate-grid fit; printed scale is advisory only for this run.")

    # ------------------------------------------------------------------
    # Correction 2: Location-based map-centre anchor check
    # If the location name was identified in the overview, compare the
    # computed image centre coordinates against the lookup tables.
    # Large deviations almost certainly mean the AI read digits incorrectly.
    # ------------------------------------------------------------------
    location = vision.get("location_name") or ""
    if location:
        loc_key = location.strip().lower()
        expected_centre_e = None
        expected_centre_n = None
        for city, e_centre in CITY_EASTING_UTM32.items():
            if city in loc_key or loc_key in city:
                expected_centre_e = e_centre
                break
        for city, n_centre in CITY_NORTHING_UTM32.items():
            if city in loc_key or loc_key in city:
                expected_centre_n = n_centre
                break

        img_w = vision.get("width")
        img_w_est = img_w if img_w is not None else max(px for px, _ in e_pos)
        img_h_est = img_h if img_h is not None else max(py for py, _ in n_pos)
        offset_e = None
        offset_m = None

        if expected_centre_e is not None:
            computed_centre_e = e0 + de * (img_w_est / 2)
            offset_e = computed_centre_e - expected_centre_e
            print(f"  Location '{location}': expected centre E≈{expected_centre_e:.0f}, "
                  f"computed centre E≈{computed_centre_e:.0f}, offset={offset_e:+.0f} m")

            if abs(offset_e) > 10_000:
                print(f"[!] Easting offset {abs(offset_e)/1000:.1f} km -- likely AI digit misread. "
                      f"Correcting origin by {-offset_e:+.0f} m.")
                e0 -= offset_e
                print(f"    Corrected easting origin: {e0:.2f}")
            elif abs(offset_e) > 3_000:
                print(f"[~] Easting offset {abs(offset_e)/1000:.1f} km -- minor. "
                      f"Keeping as-is (within acceptable range).")

        if expected_centre_n is not None:
            computed_centre_n = n0 + dn * (img_h_est / 2)
            offset_m = computed_centre_n - expected_centre_n
            print(f"  Location '{location}': expected centre N≈{expected_centre_n:.0f}, "
                  f"computed centre N≈{computed_centre_n:.0f}, offset={offset_m:+.0f} m")

            if abs(offset_m) > 5_000:
                print(f"[!] Northing offset {abs(offset_m)/1000:.1f} km -- likely AI digit misread. "
                      f"Correcting origin by {-offset_m:+.0f} m.")
                n0 -= offset_m   # shift to match expected location
                print(f"    Corrected northing origin: {n0:.2f}")
            elif abs(offset_m) > 3_000:
                print(f"[~] Northing offset {abs(offset_m)/1000:.1f} km -- minor. "
                      f"Keeping as-is (within acceptable range).")

        # If the grid fit is still far away from the anchored location, it is
        # almost certainly a synthetic/incorrect grid read. In that case,
        # fall back to the printed scale centred on the resolved location.
        if (
            expected_mpp is not None
            and expected_centre_e is not None
            and expected_centre_n is not None
            and (
                (offset_e is not None and abs(offset_e) > 8_000)
                or (offset_m is not None and abs(offset_m) > 8_000)
            )
        ):
            de = expected_mpp
            dn = -expected_mpp
            e0 = expected_centre_e - de * (img_w_est / 2)
            n0 = expected_centre_n - dn * (img_h_est / 2)
            print(f"[!] Grid fit remains far from '{location}'. Falling back to location + printed scale.")
            print(f"    Fallback geotransform: E origin={e0:.2f}, X scale={de:.4f} m/px")
            print(f"                          N origin={n0:.2f}, Y scale={dn:.4f} m/px")

    # If the Y-flip was applied, n0/dn were computed in the flipped coordinate
    # system (where pixel 0 = original bottom).  Convert back to original pixel
    # space so GT[3] = northing at original pixel (0,0) = top-left corner.
    #   N_flipped(py_flip) = n0 + dn * py_flip
    #   py_flip = img_h - py_orig  →  N(py_orig) = (n0 + dn*img_h) + (-dn)*py_orig
    if _n_flip_applied and img_h is not None:
        n0_orig = n0 + dn * img_h
        dn_orig = -dn
        print(f"  Y-flip back-conversion: N origin {n0:.2f}→{n0_orig:.2f}, "
              f"Y scale {dn:.4f}→{dn_orig:.4f} m/px (original pixel space)")
        n0, dn = n0_orig, dn_orig

    gt = (e0, de, 0.0, n0, 0.0, dn)
    return gt


# ---------------------------------------------------------------------------
# STEP 5b – Optional: refine geotransform via WMS phase correlation
# ---------------------------------------------------------------------------
def refine_geotransform_wms(src_path: Path, gt: tuple, epsg: int) -> tuple:
    """
    Download the official topographic WMS tile for the rough georeferenced
    extent, then use phase correlation against the plan to detect and correct
    any remaining translation offset.

    Works best when the plan contains a topo-map background (1:25000 – 1:5000).
    Returns the (possibly corrected) 6-element geotransform tuple.
    """
    if not ENABLE_WMS_REFINEMENT:
        return gt
    try:
        import numpy as np
    except ImportError:
        print("[!] numpy not available -- skipping WMS refinement")
        return gt
    if not (HAS_PIL and HAS_GDAL):
        return gt

    import io
    import ssl
    import urllib.request

    e0, de, _, n0, _, dn = gt
    ds_src = gdal.Open(str(src_path))
    W, H = ds_src.RasterXSize, ds_src.RasterYSize
    ds_src = None

    west  = e0
    east  = e0 + de * W
    north = n0
    south = n0 + dn * H   # dn is negative → south < north

    # Match the central part of the plan against a larger surrounding
    # reference tile, so refinement can search for the locally best fit.
    frac = 0.6
    cx, cy = (west + east) / 2, (south + north) / 2
    tpl_hw = (east - west)  * frac / 2
    tpl_hh = (north - south) * frac / 2
    w_c, e_c = cx - tpl_hw, cx + tpl_hw
    s_c, n_c = cy - tpl_hh, cy + tpl_hh

    search_hw = tpl_hw * WMS_SEARCH_EXPAND
    search_hh = tpl_hh * WMS_SEARCH_EXPAND
    w_s, e_s = cx - search_hw, cx + search_hw
    s_s, n_s = cy - search_hh, cy + search_hh

    ref_px = 384
    bbox   = f"{w_s},{s_s},{e_s},{n_s}"
    wms_req = (
        f"{WMS_URL}?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
        f"&LAYERS={WMS_LAYER}&STYLES=&CRS=EPSG:{epsg}"
        f"&BBOX={bbox}&WIDTH={ref_px}&HEIGHT={ref_px}&FORMAT=image/jpeg"
    )

    print(f"[~] WMS refinement: downloading reference tile …")
    try:
        data, _ = _fetch_url_bytes_cached(
            wms_req,
            namespace="wms_tiles",
            timeout=15,
            headers={"User-Agent": "auto_georeference/1.0"},
            suffix=".img",
        )
        ref_img = Image.open(io.BytesIO(data)).convert("L")
        ref_arr = np.array(ref_img, dtype=np.float32)
    except Exception as exc:
        print(f"[!] WMS download failed ({exc}) -- skipping refinement")
        return gt

    # Verify the WMS didn't return an error/blank image
    if ref_arr.std() < 2.0:
        print("[!] WMS returned a nearly blank image -- skipping refinement")
        return gt

    # Crop the same geographic region from the plan at the same pixel density
    px_x1 = max(0, int((w_c - e0) / de))
    px_x2 = min(W, int((e_c - e0) / de))
    px_y1 = max(0, int((n_c - n0) / dn))
    px_y2 = min(H, int((s_c - n0) / dn))

    # Read the plan through GDAL so VRT inputs work as well as TIFFs.
    ds_plan = gdal.Open(str(src_path))
    if ds_plan is None:
        print("[!] Could not open plan raster for WMS refinement -- skipping refinement")
        return gt

    read_w = max(1, px_x2 - px_x1)
    read_h = max(1, px_y2 - px_y1)
    band_count = ds_plan.RasterCount

    if band_count >= 3:
        r = ds_plan.GetRasterBand(1).ReadAsArray(px_x1, px_y1, read_w, read_h)
        g = ds_plan.GetRasterBand(2).ReadAsArray(px_x1, px_y1, read_w, read_h)
        b = ds_plan.GetRasterBand(3).ReadAsArray(px_x1, px_y1, read_w, read_h)
        if r is None or g is None or b is None:
            print("[!] Failed reading RGB plan crop -- skipping refinement")
            return gt
        rgb = np.dstack([r, g, b]).astype(np.uint8)
        plan_crop = Image.fromarray(rgb, mode="RGB").resize((ref_px, ref_px), Image.LANCZOS).convert("L")
    else:
        band = ds_plan.GetRasterBand(1)
        arr = band.ReadAsArray(px_x1, px_y1, read_w, read_h)
        if arr is None:
            print("[!] Failed reading grayscale plan crop -- skipping refinement")
            return gt
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        plan_crop = Image.fromarray(arr, mode="L").resize((ref_px, ref_px), Image.LANCZOS)

    ds_plan = None
    plan_arr = np.array(plan_crop, dtype=np.float32)

    # Save debug images
    ref_img.save(str(_artifact_path("wms_ref.jpg")))
    ref_img.close()
    plan_crop.save(str(_artifact_path("wms_plan_crop.jpg")))
    plan_crop.close()

    # Normalised phase correlation (translation-only)
    F1  = np.fft.fft2(ref_arr)
    F2  = np.fft.fft2(plan_arr)
    cp  = F1 * np.conj(F2)
    mag = np.abs(cp)
    mag[mag < 1e-10] = 1e-10
    corr = np.abs(np.fft.ifft2(cp / mag))

    peak_val = corr.max()
    peak_idx = np.unravel_index(np.argmax(corr), corr.shape)
    peak_confidence = peak_val / (corr.mean() + 1e-10)

    sy, sx = int(peak_idx[0]), int(peak_idx[1])
    if sy > ref_px // 2: sy -= ref_px
    if sx > ref_px // 2: sx -= ref_px

    # Convert pixel shift to metres
    m_per_px_x = (e_c - w_c) / ref_px
    m_per_px_y = (n_c - s_c) / ref_px
    shift_e =  sx * m_per_px_x
    shift_n = -sy * m_per_px_y   # image y increases downward, northing upward

    total_m = (shift_e**2 + shift_n**2) ** 0.5
    print(f"  Phase correlation: ΔE={shift_e:+.1f} m  ΔN={shift_n:+.1f} m  "
          f"total={total_m:.1f} m  confidence={peak_confidence:.1f}x")

    if total_m > WMS_MAX_SHIFT_M:
        print(f"[!] Shift {total_m:.0f} m exceeds limit {WMS_MAX_SHIFT_M} m "
              f"-- discarding (poor correlation)")
        return gt
    if peak_confidence < 3.0:
        print(f"[!] Phase correlation peak too weak (confidence={peak_confidence:.1f}x < 3.0) "
              f"-- discarding")
        return gt

    gt_refined = (gt[0] + shift_e, gt[1], gt[2], gt[3] + shift_n, gt[4], gt[5])
    print(f"[✓] WMS refinement applied: ΔE={shift_e:+.1f} m  ΔN={shift_n:+.1f} m")
    return gt_refined


# ---------------------------------------------------------------------------
# STEP 6 – Write georeferenced GeoTIFF directly (no warp, no resampling)
# ---------------------------------------------------------------------------
def detect_main_map_bbox(src_path: Path) -> tuple | None:
    """
    Detect the dominant non-white block on the page and return
    (x1, y1, x2, y2) in full-resolution pixel coordinates.
    """
    if not HAS_PIL:
        return None
    try:
        import numpy as np
    except ImportError:
        print("[!] numpy not available -- skipping main-map crop detection")
        return None

    img = Image.open(str(src_path)).convert("RGB")
    W, H = img.size
    thumb = img.copy()
    img.close()
    thumb.thumbnail((1200, 1200), Image.LANCZOS)
    sw, sh = thumb.size
    arr = np.array(thumb, dtype=np.uint8)
    thumb.close()

    gray = arr.mean(axis=2)
    color_span = arr.max(axis=2) - arr.min(axis=2)
    mask = (gray < 245) | (color_span > 18)
    if mask.sum() < 500:
        print("[!] Main-map crop detection found too little content -- skipping crop")
        return None

    def _longest_true_span(mask_1d):
        best = None
        start = None
        for i, ok in enumerate(mask_1d):
            if ok and start is None:
                start = i
            elif not ok and start is not None:
                if best is None or (i - start) > (best[1] - best[0]):
                    best = (start, i)
                start = None
        if start is not None:
            if best is None or (len(mask_1d) - start) > (best[1] - best[0]):
                best = (start, len(mask_1d))
        return best

    best = None
    best_score = -1

    if HAS_CV2:
        # Fast path: cv2 connected components (~100× faster than Python BFS)
        mask_u8 = mask.astype(np.uint8) * 255
        n_cc, _, stats_cc, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=4)
        for i in range(1, n_cc):
            bw_cc = int(stats_cc[i, cv2.CC_STAT_WIDTH])
            bh_cc = int(stats_cc[i, cv2.CC_STAT_HEIGHT])
            area_cc = int(stats_cc[i, cv2.CC_STAT_AREA])
            if bw_cc < sw * 0.15 or bh_cc < sh * 0.15:
                continue
            fill_cc = area_cc / max(bw_cc * bh_cc, 1)
            score_cc = area_cc * fill_cc
            if score_cc > best_score:
                best_score = score_cc
                x1_cc = int(stats_cc[i, cv2.CC_STAT_LEFT])
                y1_cc = int(stats_cc[i, cv2.CC_STAT_TOP])
                best = (x1_cc, y1_cc, x1_cc + bw_cc, y1_cc + bh_cc)
    else:
        # Fallback: pure-Python BFS
        visited = np.zeros(mask.shape, dtype=bool)
        for y in range(sh):
            for x in range(sw):
                if not mask[y, x] or visited[y, x]:
                    continue
                stack = [(x, y)]
                visited[y, x] = True
                area = 0
                minx = maxx = x
                miny = maxy = y
                while stack:
                    cx, cy = stack.pop()
                    area += 1
                    if cx < minx: minx = cx
                    if cx > maxx: maxx = cx
                    if cy < miny: miny = cy
                    if cy > maxy: maxy = cy
                    for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                        if 0 <= nx < sw and 0 <= ny < sh and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((nx, ny))
                bw = maxx - minx + 1
                bh = maxy - miny + 1
                bbox_area = bw * bh
                fill = area / max(bbox_area, 1)
                score = area * fill
                if bw >= sw * 0.15 and bh >= sh * 0.15 and score > best_score:
                    best_score = score
                    best = (minx, miny, maxx + 1, maxy + 1)

    if not best:
        print("[!] Main-map crop detection: no dominant block found -- skipping crop")
        return None

    x1, y1, x2, y2 = best
    scale_x = W / sw
    scale_y = H / sh
    pad_x = int(max(20, 0.01 * W))
    pad_y = int(max(20, 0.01 * H))
    fx1 = max(0, int(x1 * scale_x) - pad_x)
    fy1 = max(0, int(y1 * scale_y) - pad_y)
    fx2 = min(W, int(x2 * scale_x) + pad_x)
    fy2 = min(H, int(y2 * scale_y) + pad_y)

    if (fx2 - fx1) > 0.97 * W and (fy2 - fy1) > 0.97 * H:
        col_nonwhite = (gray < 242).mean(axis=0)
        row_nonwhite = (gray < 242).mean(axis=1)
        col_std = gray.std(axis=0)
        row_std = gray.std(axis=1)

        col_std_thr = max(8.0, float(np.percentile(col_std, 65)) * 0.75)
        row_std_thr = max(8.0, float(np.percentile(row_std, 65)) * 0.75)
        col_mask = (col_nonwhite > 0.18) & (col_std > col_std_thr)
        row_mask = (row_nonwhite > 0.18) & (row_std > row_std_thr)
        x_span = _longest_true_span(col_mask)
        y_span = _longest_true_span(row_mask)

        if x_span and y_span:
            tx1, tx2 = x_span
            ty1, ty2 = y_span
            span_w = tx2 - tx1
            span_h = ty2 - ty1
            if span_w >= sw * 0.55 and span_h >= sh * 0.55:
                fx1 = max(0, int(tx1 * scale_x) - pad_x)
                fy1 = max(0, int(ty1 * scale_y) - pad_y)
                fx2 = min(W, int(tx2 * scale_x) + pad_x)
                fy2 = min(H, int(ty2 * scale_y) + pad_y)
                if (fx2 - fx1) < 0.98 * W or (fy2 - fy1) < 0.98 * H:
                    print(
                        f"[i] Main-map crop refined from page-wide block: px=({fx1},{fy1})-({fx2},{fy2}) "
                        f"size={fx2-fx1}×{fy2-fy1}"
                    )
                    return fx1, fy1, fx2, fy2

        print("[~] Main-map crop covers almost entire page -- keeping full sheet")
        return None

    print(f"[i] Main-map crop: px=({fx1},{fy1})-({fx2},{fy2}) size={fx2-fx1}×{fy2-fy1}")
    return fx1, fy1, fx2, fy2


def _affine2x3_to_matrix3(affine_2x3):
    import numpy as np
    return np.array([
        [float(affine_2x3[0][0]), float(affine_2x3[0][1]), float(affine_2x3[0][2])],
        [float(affine_2x3[1][0]), float(affine_2x3[1][1]), float(affine_2x3[1][2])],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _compose_geotransform_from_local_affine(
    affine_2x3,
    *,
    px_x1,
    px_y1,
    read_w,
    read_h,
    tpl_px,
    photo_x1,
    photo_y1,
    tpl_px_base=None,
    tile_bounds,
    ref_width,
    ref_height,
):
    import numpy as np

    tw_s, te_s, ts_s, tn_s = tile_bounds
    if tpl_px_base is None:
        tpl_px_base = tpl_px
    photo_x1_scaled = float(photo_x1) * float(tpl_px) / max(float(tpl_px_base), 1.0)
    photo_y1_scaled = float(photo_y1) * float(tpl_px) / max(float(tpl_px_base), 1.0)
    # OpenCV keypoints / affine transforms use pixel-centre coordinates,
    # while GDAL geotransforms are defined at the top-left pixel corner.
    # Convert source centres -> cropped-template centres -> reference centres
    # -> map coordinates, then shift back to the output pixel corner basis.
    src_to_plan = np.array([
        [
            float(tpl_px) / float(read_w),
            0.0,
            -float(tpl_px) * (float(px_x1) + 0.5) / float(read_w) - photo_x1_scaled + 0.5,
        ],
        [
            0.0,
            float(tpl_px) / float(read_h),
            -float(tpl_px) * (float(px_y1) + 0.5) / float(read_h) - photo_y1_scaled + 0.5,
        ],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    ref_to_map = np.array([
        [
            (float(te_s) - float(tw_s)) / float(ref_width),
            0.0,
            float(tw_s) + 0.5 * (float(te_s) - float(tw_s)) / float(ref_width),
        ],
        [
            0.0,
            -(float(tn_s) - float(ts_s)) / float(ref_height),
            float(tn_s) - 0.5 * (float(tn_s) - float(ts_s)) / float(ref_height),
        ],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    local_aff = _affine2x3_to_matrix3(affine_2x3)
    matrix_center = ref_to_map @ local_aff @ src_to_plan
    centre_to_corner = np.array([
        [1.0, 0.0, -0.5],
        [0.0, 1.0, -0.5],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    matrix = matrix_center @ centre_to_corner
    gt = (
        float(matrix[0, 2]),
        float(matrix[0, 0]),
        float(matrix[0, 1]),
        float(matrix[1, 2]),
        float(matrix[1, 0]),
        float(matrix[1, 1]),
    )
    return gt, matrix


def _prepare_registration_image(gray_arr, plan_type: str):
    import numpy as np

    arr = np.clip(gray_arr, 0, 255).astype(np.uint8)
    if not HAS_CV2:
        return arr

    if plan_type == "aerial":
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        base = clahe.apply(arr)
        blur = cv2.GaussianBlur(base, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 120)
        prep = cv2.addWeighted(base, 0.82, edges, 0.55, 0)
        return prep

    if plan_type == "topo":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        base = clahe.apply(arr)
        edges = cv2.Canny(base, 45, 130)
        return cv2.addWeighted(base, 0.75, edges, 0.65, 0)

    # vector / CAD: emphasize sparse dark structure and corners
    blur = cv2.GaussianBlur(arr, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 7
    )
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    edges = cv2.Canny(binary, 30, 100)
    return np.maximum(binary, edges)


def _save_feature_debug(plan_img, ref_img, kp1, kp2, matches, inliers, out_path: Path):
    if not HAS_CV2:
        return
    try:
        keep = []
        if inliers is None:
            keep = matches[:40]
        else:
            mask = inliers.ravel().tolist()
            for m, ok in zip(matches, mask):
                if ok:
                    keep.append(m)
            keep = keep[:60]
        vis = cv2.drawMatches(
            plan_img, kp1, ref_img, kp2, keep, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imwrite(str(out_path), vis)
    except Exception as exc:
        print(f"[~] Could not write feature-match debug image: {exc}")


def _estimate_affine_feature_transform(plan_img, ref_img, plan_type: str):
    import numpy as np

    if not HAS_CV2:
        return None

    plan_prep = _prepare_registration_image(plan_img, plan_type)
    ref_prep = _prepare_registration_image(ref_img, plan_type)

    detector_specs = []
    try:
        detector_specs.append(("akaze", cv2.AKAZE_create(), cv2.NORM_HAMMING))
    except Exception:
        pass
    detector_specs.append((
        "orb",
        cv2.ORB_create(
            nfeatures=5000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            patchSize=31,
            fastThreshold=7,
        ),
        cv2.NORM_HAMMING,
    ))

    best = None
    for det_name, detector, norm in detector_specs:
        try:
            kp1, des1 = detector.detectAndCompute(plan_prep, None)
            kp2, des2 = detector.detectAndCompute(ref_prep, None)
        except Exception:
            continue
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            continue

        matcher = cv2.BFMatcher(norm, crossCheck=False)
        try:
            raw_matches = matcher.knnMatch(des1, des2, k=2)
        except Exception:
            continue
        good = []
        used_q = set()
        used_t = set()
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance >= 0.78 * n.distance:
                continue
            if m.queryIdx in used_q or m.trainIdx in used_t:
                continue
            good.append(m)
            used_q.add(m.queryIdx)
            used_t.add(m.trainIdx)
        if len(good) < 8:
            continue

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        if plan_type == "vector":
            affine, inliers = cv2.estimateAffine2D(
                src_pts, dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0,
                maxIters=5000,
                confidence=0.995,
                refineIters=40,
            )
            if affine is None:
                affine, inliers = cv2.estimateAffinePartial2D(
                    src_pts, dst_pts,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=5.0,
                    maxIters=5000,
                    confidence=0.995,
                    refineIters=40,
                )
        else:
            affine, inliers = cv2.estimateAffinePartial2D(
                src_pts, dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0,
                maxIters=5000,
                confidence=0.995,
                refineIters=40,
            )
        if affine is None or inliers is None:
            continue

        inlier_mask = inliers.ravel().astype(bool)
        inlier_count = int(inlier_mask.sum())
        if inlier_count < (10 if plan_type == "vector" else 8):
            continue

        src_in = src_pts[inlier_mask][:, 0, :]
        dst_in = dst_pts[inlier_mask][:, 0, :]
        pred = cv2.transform(src_in.reshape(-1, 1, 2), affine).reshape(-1, 2)
        residual_px = float(np.sqrt(np.mean(np.sum((pred - dst_in) ** 2, axis=1))))
        span_src = np.ptp(src_in, axis=0)
        coverage = float((span_src[0] * span_src[1]) / max(plan_img.shape[0] * plan_img.shape[1], 1))
        candidate = {
            "detector": det_name,
            "affine": affine,
            "matches": good,
            "inliers": inliers,
            "kp1": kp1,
            "kp2": kp2,
            "inlier_count": inlier_count,
            "match_count": len(good),
            "inlier_ratio": inlier_count / max(len(good), 1),
            "residual_px": residual_px,
            "coverage": coverage,
            "plan_prep": plan_prep,
            "ref_prep": ref_prep,
        }
        rank = (
            candidate["inlier_count"],
            candidate["coverage"],
            candidate["inlier_ratio"],
            -candidate["residual_px"],
        )
        if best is None or rank > (
            best["inlier_count"],
            best["coverage"],
            best["inlier_ratio"],
            -best["residual_px"],
        ):
            best = candidate

    return best


def _validate_affine_geotransform(gt_candidate, gt_seed, width_px, height_px, base_mpp, plan_type: str):
    sx = (gt_candidate[1] ** 2 + gt_candidate[4] ** 2) ** 0.5
    sy = (gt_candidate[2] ** 2 + gt_candidate[5] ** 2) ** 0.5
    if sx < 0.02 or sy < 0.02:
        return False, "near-zero scale"

    scale_ratio_x = sx / max(base_mpp, 1e-9)
    scale_ratio_y = sy / max(base_mpp, 1e-9)
    limit = 1.35 if plan_type == "aerial" else 1.60
    if not (1.0 / limit <= scale_ratio_x <= limit and 1.0 / limit <= scale_ratio_y <= limit):
        return False, f"scale ratio out of bounds ({scale_ratio_x:.2f}, {scale_ratio_y:.2f})"

    # All plan types: check that local refinement stays close to a similarity
    # transform. Large rotation/anisotropy/shear from patch-consensus or feature
    # matching indicates a false match, not a real improvement.
    anisotropy = max(sx, sy) / max(min(sx, sy), 1e-9)
    aniso_limit = 1.06 if plan_type == "aerial" else 1.08
    if anisotropy > aniso_limit:
        return False, f"anisotropic scale {anisotropy:.3f} exceeds limit"

    seed_angle = math.degrees(math.atan2(gt_seed[4], gt_seed[1]))
    cand_angle = math.degrees(math.atan2(gt_candidate[4], gt_candidate[1]))
    angle_delta = cand_angle - seed_angle
    while angle_delta > 180.0:
        angle_delta -= 360.0
    while angle_delta < -180.0:
        angle_delta += 360.0
    angle_delta = abs(angle_delta)
    rot_limit = 2.5 if plan_type == "aerial" else 5.0
    if angle_delta > rot_limit:
        return False, f"rotation delta {angle_delta:.2f}° exceeds limit"

    # Column and row vectors should stay near-orthogonal.
    dot = gt_candidate[1] * gt_candidate[2] + gt_candidate[4] * gt_candidate[5]
    ortho = abs(dot) / max(sx * sy, 1e-9)
    ortho_limit = 0.08 if plan_type == "aerial" else 0.12
    if ortho > ortho_limit:
        return False, f"shear {ortho:.3f} exceeds limit"

    seed_cx = gt_seed[0] + gt_seed[1] * (width_px / 2.0) + gt_seed[2] * (height_px / 2.0)
    seed_cy = gt_seed[3] + gt_seed[4] * (width_px / 2.0) + gt_seed[5] * (height_px / 2.0)
    cand_cx = gt_candidate[0] + gt_candidate[1] * (width_px / 2.0) + gt_candidate[2] * (height_px / 2.0)
    cand_cy = gt_candidate[3] + gt_candidate[4] * (width_px / 2.0) + gt_candidate[5] * (height_px / 2.0)
    center_shift = ((cand_cx - seed_cx) ** 2 + (cand_cy - seed_cy) ** 2) ** 0.5
    if center_shift > WMS_MAX_SHIFT_M:
        return False, f"centre shift {center_shift:.0f} m exceeds limit"

    return True, f"centre shift {center_shift:.0f} m"


def _local_refinement_shift_limit_m(plan_type: str) -> float:
    if plan_type == "aerial":
        base_limit = max(700.0, WMS_LOCAL_REFINEMENT_MAX_SHIFT_M)
    elif plan_type == "topo":
        base_limit = min(WMS_LOCAL_REFINEMENT_MAX_SHIFT_M, 800.0)
    else:
        base_limit = min(WMS_LOCAL_REFINEMENT_MAX_SHIFT_M, 650.0)

    seed_src = (CURRENT_SEED_SOURCE or "").strip().lower()
    seed_conf = (CURRENT_SEED_CONFIDENCE or "").strip().lower()

    if seed_src in ("manual_seed", "last_result", "ocr_coordinates"):
        return min(base_limit, 900.0)

    if seed_conf in ("city", "postcode") or seed_src in (
        "project_address",
        "project_address+road",
        "vision_location",
        "qgis_canvas",
        "city_name",
        "ocr_place_names",
    ):
        if plan_type == "aerial":
            return max(base_limit, 1_800.0)
        if plan_type == "topo":
            return max(base_limit, 1_600.0)
        return max(base_limit, 1_500.0)

    if seed_conf == "feature" or seed_src in (
        "project_address+ocr_places",
        "ocr_standort",
    ):
        if plan_type == "aerial":
            return max(base_limit, 1_300.0)
        return max(base_limit, 1_100.0)

    return base_limit


def _shifted_gt(gt_base: tuple, shift_e: float, shift_n: float) -> tuple:
    return (
        gt_base[0] + shift_e,
        gt_base[1],
        gt_base[2],
        gt_base[3] + shift_n,
        gt_base[4],
        gt_base[5],
    )


def _blend_candidate_cluster(candidates):
    """
    Blend nearby WMS candidates into one consensus shift so we do not let a
    single tile decide the final placement when several neighbours agree.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        result = dict(candidates[0])
        result.setdefault("cluster_spread_m", 0.0)
        return result

    def _weight(c):
        return max(0.001, (c.get("score", 0.0) + 1.0) * max(c.get("confidence", 1.0), 0.01) * (1.0 + max(c.get("score_gap", 0.0), 0.0) * 8.0))

    total_w = sum(_weight(c) for c in candidates)
    if total_w <= 0:
        return dict(candidates[0])

    ref = max(candidates, key=lambda c: (c.get("score", 0.0), c.get("confidence", 0.0), c.get("score_gap", 0.0)))
    blended = dict(ref)
    blended["shift_e"] = sum(c["shift_e"] * _weight(c) for c in candidates) / total_w
    blended["shift_n"] = sum(c["shift_n"] * _weight(c) for c in candidates) / total_w
    blended["total_m"] = (blended["shift_e"] ** 2 + blended["shift_n"] ** 2) ** 0.5
    blended["score"] = sum(c.get("score", 0.0) * _weight(c) for c in candidates) / total_w
    blended["confidence"] = sum(c.get("confidence", 1.0) * _weight(c) for c in candidates) / total_w
    blended["score_gap"] = sum(c.get("score_gap", 0.0) * _weight(c) for c in candidates) / total_w
    blended["cluster_size"] = len(candidates)
    blended["cluster_spread_m"] = max(
        (((c["shift_e"] - blended["shift_e"]) ** 2 + (c["shift_n"] - blended["shift_n"]) ** 2) ** 0.5)
        for c in candidates
    )
    return blended


def _ncc_score_global(search_arr, tpl_arr, step=1, x_range=None, y_range=None):
    import numpy as np

    sh, sw = search_arr.shape
    th_loc, tw_loc = tpl_arr.shape
    if sh < th_loc or sw < tw_loc:
        return None

    if x_range is None:
        xs = np.arange(0, sw - tw_loc + 1, step)
    else:
        x1, x2 = x_range
        xs = np.arange(max(0, x1), min(sw - tw_loc, x2) + 1, step)
    if y_range is None:
        ys = np.arange(0, sh - th_loc + 1, step)
    else:
        y1, y2 = y_range
        ys = np.arange(max(0, y1), min(sh - th_loc, y2) + 1, step)
    if xs.size == 0 or ys.size == 0:
        return None

    # ── cv2 windowed fast path ────────────────────────────────────────────────
    # When x_range/y_range constrain the search, extract the sub-region and
    # run cv2.matchTemplate on it instead of the slow Python loop.
    if HAS_CV2 and x_range is not None and y_range is not None:
        try:
            sub_x1 = int(xs[0])
            sub_y1 = int(ys[0])
            sub_x2 = int(xs[-1]) + tw_loc + 1
            sub_y2 = int(ys[-1]) + th_loc + 1
            sub_x2 = min(sub_x2, sw)
            sub_y2 = min(sub_y2, sh)
            sub = search_arr[sub_y1:sub_y2, sub_x1:sub_x2].astype(np.float32)
            tpl_cv = (tpl_arr.astype(np.float32) + 1.0)
            if sub.shape[0] >= tpl_cv.shape[0] and sub.shape[1] >= tpl_cv.shape[1]:
                scores = cv2.matchTemplate(sub, tpl_cv, cv2.TM_CCOEFF_NORMED)
                if step > 1:
                    scores = scores[::step, ::step]
                hs, ws = scores.shape
                flat = scores.ravel()
                best_i = int(np.argmax(flat))
                iy, ix = np.unravel_index(best_i, scores.shape)
                best_score_cv = float(flat[best_i])
                bx = int(ix) * step + sub_x1
                by = int(iy) * step + sub_y1
                sub_dx, sub_dy = 0.0, 0.0
                if step == 1 and 0 < ix < ws - 1:
                    denom_x = 2.0 * best_score_cv - float(scores[iy, ix - 1]) - float(scores[iy, ix + 1])
                    if abs(denom_x) > 1e-7:
                        sub_dx = float(np.clip(
                            (float(scores[iy, ix + 1]) - float(scores[iy, ix - 1])) / (2.0 * denom_x),
                            -0.5, 0.5,
                        ))
                if step == 1 and 0 < iy < hs - 1:
                    denom_y = 2.0 * best_score_cv - float(scores[iy - 1, ix]) - float(scores[iy + 1, ix])
                    if abs(denom_y) > 1e-7:
                        sub_dy = float(np.clip(
                            (float(scores[iy + 1, ix]) - float(scores[iy - 1, ix])) / (2.0 * denom_y),
                            -0.5, 0.5,
                        ))
                best_xy_cv = (float(bx) + sub_dx, float(by) + sub_dy)
                neigh = 6
                mask_cv = scores.copy()
                mask_cv[max(0, iy - neigh):iy + neigh + 1,
                        max(0, ix - neigh):ix + neigh + 1] = -2.0
                second_score_cv = float(mask_cv.max()) if mask_cv.size else -1.0
                return best_score_cv, second_score_cv, best_xy_cv
        except Exception:
            pass  # fall through to numpy/Python path
    # ── end cv2 windowed fast path ────────────────────────────────────────────

    tpl_arr = tpl_arr.astype(np.float32)
    tpl_arr = tpl_arr - tpl_arr.mean()
    tpl_norm = np.sqrt((tpl_arr * tpl_arr).sum()) + 1e-9
    best_score = -1e9
    second_score = -1e9
    best_xy = None
    for y in ys:
        for x in xs:
            patch = search_arr[y:y + th_loc, x:x + tw_loc]
            patch = patch - patch.mean()
            s = float((patch * tpl_arr).sum() / (np.sqrt((patch * patch).sum()) * tpl_norm + 1e-9))
            if s > best_score:
                second_score = best_score
                best_score = s
                best_xy = (int(x), int(y))
            elif s > second_score:
                second_score = s
    if best_xy is None:
        return None
    # Sub-pixel parabolic refinement using neighbour NCC evaluations.
    # Only applies when the search step is 1 (typical for patch consensus).
    if step == 1:
        bx, by = best_xy
        def _s(x, y):
            p = search_arr[y:y + th_loc, x:x + tw_loc]
            p = p - p.mean()
            return float((p * tpl_arr).sum() / (np.sqrt((p * p).sum()) * tpl_norm + 1e-9))
        sub_dx, sub_dy = 0.0, 0.0
        can_x = (bx > 0) and (bx + tw_loc < sw)
        can_y = (by > 0) and (by + th_loc < sh)
        if can_x:
            s_xp, s_xm = _s(bx + 1, by), _s(bx - 1, by)
            denom_x = 2.0 * best_score - s_xm - s_xp
            if abs(denom_x) > 1e-7:
                sub_dx = float(np.clip((s_xp - s_xm) / (2.0 * denom_x), -0.5, 0.5))
        if can_y:
            s_yp, s_ym = _s(bx, by + 1), _s(bx, by - 1)
            denom_y = 2.0 * best_score - s_ym - s_yp
            if abs(denom_y) > 1e-7:
                sub_dy = float(np.clip((s_yp - s_ym) / (2.0 * denom_y), -0.5, 0.5))
        best_xy = (float(bx) + sub_dx, float(by) + sub_dy)
    return best_score, second_score, best_xy


def _build_patch_specs(arr, plan_type: str):
    h, w = arr.shape[:2]
    if h < 32 or w < 32:
        return []

    frac = 0.34 if plan_type == "aerial" else 0.28
    pw = max(32, int(w * frac))
    ph = max(32, int(h * frac))
    xs = [0, max(0, (w - pw) // 2), max(0, w - pw)]
    ys = [0, max(0, (h - ph) // 2), max(0, h - ph)]
    raw = []
    for y in ys:
        for x in xs:
            patch = arr[y:y + ph, x:x + pw]
            if patch.shape[0] < ph or patch.shape[1] < pw:
                continue
            if float(patch.std()) < (10.0 if plan_type == "aerial" else 7.0):
                continue
            raw.append((x, y, pw, ph, float(patch.std())))
    raw.sort(key=lambda item: item[4], reverse=True)
    return [(x, y, pw, ph) for x, y, pw, ph, _ in raw[:6]]


def _estimate_affine_from_patch_consensus(
    plan_img,
    ref_img,
    *,
    photo_x1,
    photo_y1,
    tpl_px_current,
    tpl_px_base,
    plan_type: str,
):
    import numpy as np

    plan_prep = _prepare_registration_image(plan_img, plan_type)
    ref_prep = _prepare_registration_image(ref_img, plan_type)
    patches = _build_patch_specs(plan_prep, plan_type)
    if len(patches) < 3:
        print(f"      [pc-dbg] only {len(patches)} patch(es) above std threshold -- too sparse")
        return None

    photo_x_scaled = photo_x1 * tpl_px_current / max(tpl_px_base, 1)
    photo_y_scaled = photo_y1 * tpl_px_current / max(tpl_px_base, 1)
    exp_base_x = (ref_prep.shape[1] - tpl_px_current) / 2.0 + photo_x_scaled
    exp_base_y = (ref_prep.shape[0] - tpl_px_current) / 2.0 + photo_y_scaled
    # Vector/CAD plans have sparse linework; a larger search window tolerates the
    # reduced NCC signal without missing the correct match peak.
    if plan_type == "vector":
        search_radius = 40 if ref_prep.shape[0] <= 256 else 60
    else:
        search_radius = 26 if ref_prep.shape[0] <= 256 else 42

    src_pts = []
    dst_pts = []
    patch_results = []
    for x, y, pw, ph in patches:
        patch = plan_prep[y:y + ph, x:x + pw]
        if patch.size == 0:
            continue
        ncc = _ncc_score_global(
            ref_prep,
            patch,
            step=1,
            x_range=(int(exp_base_x + x - search_radius), int(exp_base_x + x + search_radius)),
            y_range=(int(exp_base_y + y - search_radius), int(exp_base_y + y + search_radius)),
        )
        if ncc is None:
            continue
        score, second, best_xy = ncc
        conf = (score + 1.0) / max(second + 1.0, 1e-6)
        gap = score - second
        # Vector/ALKIS plans have sparse edge content -- NCC peaks are shallower.
        _min_conf = 1.005 if plan_type == "vector" else 1.01
        _min_gap  = 0.003 if plan_type == "vector" else 0.005
        if conf < _min_conf or gap < _min_gap:
            continue
        src_pts.append([x + pw / 2.0, y + ph / 2.0])
        dst_pts.append([best_xy[0] + pw / 2.0, best_xy[1] + ph / 2.0])
        patch_results.append({
            "box": (x, y, pw, ph),
            "score": float(score),
            "confidence": float(conf),
            "gap": float(gap),
            "best_xy": best_xy,
        })

    if len(src_pts) < 3:
        _total = len(patches)
        print(f"      [pc-dbg] {_total} patch(es) tested, {len(src_pts)} passed NCC conf/gap -- need ≥3")
        return None

    src_pts = np.float32(src_pts).reshape(-1, 1, 2)
    dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)
    affine, inliers = cv2.estimateAffinePartial2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=4.0 if plan_type == "aerial" else 5.0,
        maxIters=5000,
        confidence=0.995,
        refineIters=30,
    )
    if affine is None or inliers is None or int(inliers.sum()) < 3:
        return None

    # Compute residual on INLIERS only -- including outliers inflates the metric
    # when only 3 of 6 patches agree, masking a bad fit.
    _inlier_mask = inliers.ravel().astype(bool)
    pred_in = cv2.transform(src_pts[_inlier_mask], affine)
    residual_px = float(np.sqrt(np.mean(np.sum(
        (pred_in[:, 0, :] - dst_pts[_inlier_mask, 0, :]) ** 2, axis=1
    ))))

    full_affine = affine.copy()
    offset_vec = np.array([photo_x_scaled, photo_y_scaled], dtype=np.float32)
    full_affine[:, 2] = affine[:, 2] - affine[:, :2] @ offset_vec

    return {
        "affine": full_affine,
        "inliers": inliers,
        "patches": patch_results,
        "residual_px": residual_px,
        "plan_prep": plan_prep,
        "ref_prep": ref_prep,
        "patch_count": len(patch_results),
        "inlier_count": int(inliers.sum()),
    }


def _refine_affine_ecc(plan_img, ref_img, affine_init, plan_type: str = "aerial"):
    import numpy as np
    if not HAS_CV2:
        return affine_init, None
    try:
        warp = affine_init.astype("float32").copy()
        plan_f = plan_img.astype("float32") / 255.0
        ref_f = ref_img.astype("float32") / 255.0
        motion_model = cv2.MOTION_EUCLIDEAN if plan_type == "aerial" else cv2.MOTION_AFFINE
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            60,
            1e-5,
        )
        if motion_model == cv2.MOTION_EUCLIDEAN:
            # Reduce the similarity transform to rotation + translation only.
            a = float(warp[0, 0])
            b = float(warp[0, 1])
            theta = math.atan2(b, a)
            c = math.cos(theta)
            s = math.sin(theta)
            warp = np.array([
                [c, s, warp[0, 2]],
                [-s, c, warp[1, 2]],
            ], dtype="float32")
        cc, warp = cv2.findTransformECC(
            ref_f,
            plan_f,
            warp,
            motion_model,
            criteria,
            None,
            5,
        )
        return warp.astype("float64"), float(cc)
    except Exception:
        return affine_init, None


def _isolate_achromatic_layer(rgb_arr):
    """
    Extract only the achromatic (gray/black/white) pixels from an RGB plan image.

    Vector/CAD plans typically show *existing* infrastructure in neutral grays and
    *proposed* works in bright colours (pink, blue, green, orange hatching).  A
    simple RGB→gray conversion maps those saturated colours to mid-gray values that
    are visually indistinguishable from the structural linework, ruining NCC matching
    against a monochrome cadastral WMS reference.

    This function masks out any pixel whose HSV saturation exceeds a threshold (i.e.
    it belongs to a coloured overlay) and replaces it with white (255 = background).
    The result is a uint8 grayscale image containing only the structural/existing
    layer — building outlines, roads, parcel boundaries — which can be directly
    matched against ALKIS or DTK WMS tiles.

    Parameters : rgb_arr  shape (H, W, 3)  dtype uint8   RGB
    Returns    :          shape (H, W)     dtype uint8   grayscale, non-achromatic → 255
    """
    import numpy as _np

    if not HAS_CV2:
        # Fallback: plain luminance (original behaviour)
        from PIL import Image as _PIL_Image
        return _np.array(
            _PIL_Image.fromarray(rgb_arr, mode="RGB").convert("L"), dtype=_np.uint8
        )

    hsv = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1]   # 0-255 in OpenCV
    v = hsv[:, :, 2]   # 0-255 in OpenCV
    gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)

    # Achromatic: low saturation (gray tones) or very dark (black lines,
    # regardless of nominal hue).  Threshold tuned for CAD plan printing.
    achromatic = (s < 40) | (v < 25)
    result = _np.where(achromatic, gray, _np.uint8(255)).astype(_np.uint8)

    # Sparseness check: if colored overlays cover most of the plan the strict
    # threshold leaves < 4% dark pixels -- far too little for NCC matching.
    # Widen the saturation cutoff in two relaxation steps before giving up.
    dark_frac = float((result < 220).mean())
    if dark_frac < 0.04:
        # Step 1: raise sat. threshold to keep "lightly tinted" content
        # (e.g. faded fills, light-coloured area markings).
        achromatic_r1 = (s < 80) | (v < 40)
        result_r1 = _np.where(achromatic_r1, gray, _np.uint8(255)).astype(_np.uint8)
        dark_frac_r1 = float((result_r1 < 220).mean())
        if dark_frac_r1 >= dark_frac * 1.5:
            print(f"[i] Achromatic isolation: raised sat. threshold 40→80 "
                  f"({dark_frac:.1%} → {dark_frac_r1:.1%} dark pixels)")
            result = result_r1
            dark_frac = dark_frac_r1
        if dark_frac < 0.04:
            # Step 2: keep everything except pure vivid saturated overlay patches;
            # also retain semi-dark colored pixels (dark-colored roads/water).
            achromatic_r2 = (s < 130) | (v < 60)
            result_r2 = _np.where(achromatic_r2, gray, _np.uint8(255)).astype(_np.uint8)
            dark_frac_r2 = float((result_r2 < 220).mean())
            if dark_frac_r2 >= dark_frac * 1.5:
                print(f"[i] Achromatic isolation: raised sat. threshold →130 "
                      f"({dark_frac:.1%} → {dark_frac_r2:.1%} dark pixels)")
                result = result_r2
            else:
                # All thresholds still sparse: fall back to plain luminance so the
                # template at least contains the full structural context.
                print(f"[~] Achromatic isolation: plan too densely overlaid "
                      f"({dark_frac:.1%} dark) -- falling back to plain luminance")
                result = gray

    return result


def refine_geotransform_wms_v2(src_path: Path, gt: tuple, epsg: int, *,
                                has_coord_anchors: bool = True,
                                seed_confidence: str = "") -> tuple:
    """
    Improved local fit: match a central plan template against a larger
    surrounding reference tile using edge-enhanced template matching.
    """
    if not ENABLE_WMS_REFINEMENT:
        return gt
    global LAST_GEOREF_QUALITY, CURRENT_SEED_SOURCE, CURRENT_SEED_CONFIDENCE
    LAST_GEOREF_QUALITY = {
        "stage": "wms_refinement",
        "accepted": False,
        "acceptance_reason": "not_run",
        "has_coord_anchors": bool(has_coord_anchors),
        "seed_confidence": seed_confidence,
        "patch_consensus_succeeded": False,
        "feature_refinement_succeeded": False,
    }
    try:
        import numpy as np
    except ImportError:
        print("[!] numpy not available -- skipping WMS refinement")
        return gt
    if not (HAS_PIL and HAS_GDAL):
        return gt

    import io
    import urllib.request

    e0, de, _, n0, _, dn = gt
    has_seed_rotation = abs(gt[2]) > 1e-6 or abs(gt[4]) > 1e-6
    if has_seed_rotation:
        print("[i] Rotated seed geotransform detected -- skipping WMS refinement because north-up template extraction is not rotation-safe")
        return gt
    ds_src = gdal.Open(str(src_path))
    if ds_src is None:
        return gt
    W, H = ds_src.RasterXSize, ds_src.RasterYSize
    ds_src = None
    base_mpp = (abs(de) + abs(dn)) / 2.0

    west = e0
    east = e0 + de * W
    north = n0
    south = n0 + dn * H

    frac = 0.6
    cx, cy = (west + east) / 2, (south + north) / 2
    tpl_hw = (east - west) * frac / 2
    tpl_hh = (north - south) * frac / 2
    w_c, e_c = cx - tpl_hw, cx + tpl_hw
    s_c, n_c = cy - tpl_hh, cy + tpl_hh

    search_hw = tpl_hw * WMS_SEARCH_EXPAND
    search_hh = tpl_hh * WMS_SEARCH_EXPAND

    ref_px = WMS_REF_PX
    # Template must be sized to match reference resolution exactly so NCC does
    # not introduce a systematic scale-mismatch offset.
    # tpl_px = ref_px / expand ensures 1 template-px = 1 reference-px in meters.
    tpl_px = max(64, int(ref_px / WMS_SEARCH_EXPAND))
    tpl_px_coarse = tpl_px  # saved so fine-pass expected_x/y formula can scale photo_x1/y1

    print(f"[~] WMS refinement: searching surrounding imagery tiles …")

    px_x1 = max(0, int((w_c - e0) / de))
    px_x2 = min(W, int((e_c - e0) / de))
    px_y1 = max(0, int((n_c - n0) / dn))
    px_y2 = min(H, int((s_c - n0) / dn))

    ds_plan = gdal.Open(str(src_path))
    if ds_plan is None:
        print("[!] Could not open plan raster for WMS refinement -- skipping refinement")
        return gt

    try:
        read_w = max(1, px_x2 - px_x1)
        read_h = max(1, px_y2 - px_y1)
        band_count = ds_plan.RasterCount

        if band_count >= 3:
            r = ds_plan.GetRasterBand(1).ReadAsArray(px_x1, px_y1, read_w, read_h)
            g = ds_plan.GetRasterBand(2).ReadAsArray(px_x1, px_y1, read_w, read_h)
            b = ds_plan.GetRasterBand(3).ReadAsArray(px_x1, px_y1, read_w, read_h)
            if r is None or g is None or b is None:
                print("[!] Failed reading RGB plan crop -- skipping refinement")
                return gt
            rgb = np.dstack([r, g, b]).astype(np.uint8)
            # For non-aerial plans (vector/CAD with coloured overlays) isolate the
            # achromatic layer first so coloured hatching doesn't corrupt the template.
            if ACTIVE_WMS_CONFIG_KEY != "aerial":
                _achr = _isolate_achromatic_layer(rgb)
                plan_crop = Image.fromarray(_achr, mode="L").resize((tpl_px, tpl_px), Image.LANCZOS)
                print(f"[i] Achromatic layer isolated for '{ACTIVE_WMS_CONFIG_KEY}' plan template")
            else:
                plan_crop = Image.fromarray(rgb, mode="RGB").resize((tpl_px, tpl_px), Image.LANCZOS).convert("L")
        else:
            band = ds_plan.GetRasterBand(1)
            arr = band.ReadAsArray(px_x1, px_y1, read_w, read_h)
            if arr is None:
                print("[!] Failed reading grayscale plan crop -- skipping refinement")
                return gt
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            plan_crop = Image.fromarray(arr, mode="L").resize((tpl_px, tpl_px), Image.LANCZOS)
    finally:
        ds_plan = None
    plan_arr = np.array(plan_crop, dtype=np.float32)
    orig_th, orig_tw = plan_arr.shape

    # Detect the dominant aerial/photo rectangle inside the cropped sheet.
    # This connected-component scan finds the large photo block embedded in a
    # white-border sheet.  It is meaningful ONLY for aerial/topo plans where a
    # single dense rectangular image sits inside a white margin.  For vector/CAD
    # plans the sparse achromatic content forms isolated clusters, and the scan
    # mistakes the densest cluster for the "photo block", cropping away most of
    # the template.  Skip for non-aerial plans.
    gray_mask = plan_arr < 235
    best_bbox = None
    best_area = -1
    if HAS_CV2 and ACTIVE_WMS_CONFIG_KEY == "aerial":
        # Fast path: cv2 connected components
        gm_u8 = gray_mask.astype(np.uint8) * 255
        n_cc2, _, stats2, _ = cv2.connectedComponentsWithStats(gm_u8, connectivity=4)
        for i in range(1, n_cc2):
            bw_i = int(stats2[i, cv2.CC_STAT_WIDTH])
            bh_i = int(stats2[i, cv2.CC_STAT_HEIGHT])
            area_i = int(stats2[i, cv2.CC_STAT_AREA])
            if bw_i < orig_tw * 0.35 or bh_i < orig_th * 0.25:
                continue
            if bw_i / max(bh_i, 1) < 1.2:
                continue
            if area_i > best_area:
                best_area = area_i
                x1_i = int(stats2[i, cv2.CC_STAT_LEFT])
                y1_i = int(stats2[i, cv2.CC_STAT_TOP])
                best_bbox = (x1_i, y1_i, x1_i + bw_i, y1_i + bh_i)
    elif ACTIVE_WMS_CONFIG_KEY == "aerial":
        visited = np.zeros(gray_mask.shape, dtype=bool)
        for y in range(orig_th):
            for x in range(orig_tw):
                if not gray_mask[y, x] or visited[y, x]:
                    continue
                stack = [(x, y)]
                visited[y, x] = True
                minx = maxx = x
                miny = maxy = y
                area = 0
                while stack:
                    px, py = stack.pop()
                    area += 1
                    if px < minx: minx = px
                    if px > maxx: maxx = px
                    if py < miny: miny = py
                    if py > maxy: maxy = py
                    for nx, ny in ((px - 1, py), (px + 1, py), (px, py - 1), (px, py + 1)):
                        if 0 <= nx < orig_tw and 0 <= ny < orig_th and gray_mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((nx, ny))
                bw = maxx - minx + 1
                bh = maxy - miny + 1
                if bw < orig_tw * 0.35 or bh < orig_th * 0.25:
                    continue
                if bw / max(bh, 1) < 1.2:
                    continue
                if area > best_area:
                    best_area = area
                    best_bbox = (minx, miny, maxx + 1, maxy + 1)

    def _longest_true_span(mask):
        best = None
        start = None
        for i, ok in enumerate(mask):
            if ok and start is None:
                start = i
            elif not ok and start is not None:
                if best is None or (i - start) > (best[1] - best[0]):
                    best = (start, i)
                start = None
        if start is not None:
            if best is None or (len(mask) - start) > (best[1] - best[0]):
                best = (start, len(mask))
        return best

    photo_x1 = photo_y1 = 0
    if best_bbox:
        bx1, by1, bx2, by2 = best_bbox
        pad = 6
        bx1 = max(0, bx1 - pad)
        by1 = max(0, by1 - pad)
        bx2 = min(orig_tw, bx2 + pad)
        by2 = min(orig_th, by2 + pad)
        photo_x1, photo_y1 = bx1, by1
        plan_arr = plan_arr[by1:by2, bx1:bx2]
        plan_crop = plan_crop.crop((bx1, by1, bx2, by2))
        print(f"[i] Template aerial bbox: px=({bx1},{by1})-({bx2},{by2}) size={bx2-bx1}×{by2-by1}")

    # Trim off white borders / legend strips using content fraction + texture.
    if plan_arr.size:
        col_nonwhite = (plan_arr < 242).mean(axis=0)
        row_nonwhite = (plan_arr < 242).mean(axis=1)
        col_std = plan_arr.std(axis=0)
        row_std = plan_arr.std(axis=1)

        col_mask = (col_nonwhite > 0.55) & (col_std > max(8.0, float(np.median(col_std)) * 0.55))
        row_mask = (row_nonwhite > 0.55) & (row_std > max(8.0, float(np.median(row_std)) * 0.55))
        x_span = _longest_true_span(col_mask)
        y_span = _longest_true_span(row_mask)
        if x_span and y_span:
            tx1, tx2 = x_span
            ty1, ty2 = y_span
            span_w = tx2 - tx1
            span_h = ty2 - ty1
            if (
                span_w >= plan_arr.shape[1] * 0.70
                and span_h >= plan_arr.shape[0] * 0.60
                and span_w >= 100
                and span_h >= 100
            ):
                photo_x1 += tx1
                photo_y1 += ty1
                plan_arr = plan_arr[ty1:ty2, tx1:tx2]
                plan_crop = plan_crop.crop((tx1, ty1, tx2, ty2))
                print(f"[i] Template texture bbox: px=({tx1},{ty1})-({tx2},{ty2}) size={tx2-tx1}×{ty2-ty1}")
            else:
                print(f"[~] Ignoring texture bbox {span_w}×{span_h} -- too small / unstable")

        col_score = col_nonwhite + np.clip(col_std / max(float(np.percentile(col_std, 90)), 1.0), 0, 1)
        row_score = row_nonwhite + np.clip(row_std / max(float(np.percentile(row_std, 90)), 1.0), 0, 1)
        x_span2 = _longest_true_span(col_score > 0.35)
        y_span2 = _longest_true_span(row_score > 0.35)
        if x_span2 and y_span2:
            tx1, tx2 = x_span2
            ty1, ty2 = y_span2
            span_w2 = tx2 - tx1
            span_h2 = ty2 - ty1
            if (
                span_w2 >= plan_arr.shape[1] * 0.78
                and span_h2 >= plan_arr.shape[0] * 0.72
                and (tx1 > 0 or ty1 > 0 or tx2 < plan_arr.shape[1] or ty2 < plan_arr.shape[0])
            ):
                photo_x1 += tx1
                photo_y1 += ty1
                plan_arr = plan_arr[ty1:ty2, tx1:tx2]
                plan_crop = plan_crop.crop((tx1, ty1, tx2, ty2))
                print(f"[i] Template border trim: px=({tx1},{ty1})-({tx2},{ty2}) size={span_w2}×{span_h2}")

    plan_crop.save(str(_artifact_path("wms_plan_crop.jpg")))

    def _edgeify(arr):
        arr = arr.astype(np.float32)
        gx = np.zeros_like(arr)
        gy = np.zeros_like(arr)
        gx[:, 1:-1] = arr[:, 2:] - arr[:, :-2]
        gy[1:-1, :] = arr[2:, :] - arr[:-2, :]
        mag = np.hypot(gx, gy)
        p90 = np.percentile(mag, 90)
        if p90 > 1e-6:
            mag = np.clip(mag / p90, 0, 1)
        return mag

    def _rotate_edge_template(edge_arr, angle_deg):
        angle_deg = float(angle_deg)
        if abs(angle_deg) < 1e-6:
            return edge_arr.astype(np.float32), (0, 0)
        edge_u8 = (np.clip(edge_arr, 0, 1) * 255).astype(np.uint8)
        edge_img = Image.fromarray(edge_u8, mode="L")
        rot_img = edge_img.rotate(
            angle_deg,
            resample=Image.BICUBIC,
            expand=True,
            fillcolor=0,
        )
        rot_arr = np.array(rot_img, dtype=np.float32) / 255.0
        mask_img = Image.new("L", edge_img.size, color=255).rotate(
            angle_deg,
            resample=Image.NEAREST,
            expand=True,
            fillcolor=0,
        )
        mask = np.array(mask_img, dtype=np.uint8) > 0
        if mask.any():
            ys, xs = np.where(mask)
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            rot_arr = rot_arr[y1:y2, x1:x2]
            return rot_arr.astype(np.float32), (x1, y1)
        return rot_arr.astype(np.float32), (0, 0)

    def _ncc_score(search_arr, tpl_arr, step=1, x_range=None, y_range=None):
        sh, sw = search_arr.shape
        th_loc, tw_loc = tpl_arr.shape
        if sh < th_loc or sw < tw_loc:
            return None

        # ── cv2 fast path (FFT-based, ~1000x faster than numpy naive) ────────
        # Only used when there are no explicit range constraints; falls through
        # to the numpy path for rotation-search windowed calls.
        if HAS_CV2 and x_range is None and y_range is None:
            try:
                # cv2.TM_CCOEFF_NORMED subtracts means internally; adding a
                # constant to the pre-mean-subtracted template is harmless.
                src_cv = search_arr.astype(np.float32)
                tpl_cv = (tpl_arr + 1.0).astype(np.float32)
                scores = cv2.matchTemplate(src_cv, tpl_cv, cv2.TM_CCOEFF_NORMED)
                # scores shape: (sh - th_loc + 1, sw - tw_loc + 1)
                if step > 1:
                    scores = scores[::step, ::step]
                hs, ws = scores.shape
                flat = scores.ravel()
                best_i = int(np.argmax(flat))
                iy, ix = np.unravel_index(best_i, scores.shape)
                best_score = float(flat[best_i])
                bx = int(ix) * step
                by = int(iy) * step
                # Sub-pixel parabolic interpolation
                sub_dx, sub_dy = 0.0, 0.0
                if step == 1 and 0 < ix < ws - 1:
                    denom_x = 2.0 * best_score - float(scores[iy, ix - 1]) - float(scores[iy, ix + 1])
                    if abs(denom_x) > 1e-7:
                        sub_dx = float(np.clip(
                            (float(scores[iy, ix + 1]) - float(scores[iy, ix - 1])) / (2.0 * denom_x),
                            -0.5, 0.5,
                        ))
                if step == 1 and 0 < iy < hs - 1:
                    denom_y = 2.0 * best_score - float(scores[iy - 1, ix]) - float(scores[iy + 1, ix])
                    if abs(denom_y) > 1e-7:
                        sub_dy = float(np.clip(
                            (float(scores[iy + 1, ix]) - float(scores[iy - 1, ix])) / (2.0 * denom_y),
                            -0.5, 0.5,
                        ))
                best_xy = (float(bx) + sub_dx, float(by) + sub_dy)
                neigh = 6
                mask = scores.copy()
                mask[max(0, iy - neigh):iy + neigh + 1,
                     max(0, ix - neigh):ix + neigh + 1] = -2.0
                second_score = float(mask.max()) if mask.size else -1.0
                return best_score, second_score, best_xy
            except Exception:
                pass  # fall through to numpy path
        # ── end cv2 fast path ─────────────────────────────────────────────────

        if x_range is None:
            xs = np.arange(0, sw - tw_loc + 1, step)
        else:
            x1, x2 = x_range
            xs = np.arange(max(0, x1), min(sw - tw_loc, x2) + 1, step)
        if y_range is None:
            ys = np.arange(0, sh - th_loc + 1, step)
        else:
            y1, y2 = y_range
            ys = np.arange(max(0, y1), min(sh - th_loc, y2) + 1, step)
        if xs.size == 0 or ys.size == 0:
            return None

        mem_est = len(ys) * len(xs) * th_loc * tw_loc * 4
        tpl_arr = tpl_arr.astype(np.float32)
        tpl_arr = tpl_arr - tpl_arr.mean()
        tpl_norm = np.sqrt((tpl_arr * tpl_arr).sum()) + 1e-9
        if mem_est < 120_000_000:
            from numpy.lib.stride_tricks import as_strided as _ast
            shape = (len(ys), len(xs), th_loc, tw_loc)
            strides = (search_arr.strides[0]*step, search_arr.strides[1]*step,
                       search_arr.strides[0],      search_arr.strides[1])
            patches = _ast(search_arr[ys[0]:ys[-1] + th_loc, xs[0]:xs[-1] + tw_loc], shape=shape, strides=strides).copy()
            pmeans = patches.mean(axis=(2, 3), keepdims=True)
            pz = patches - pmeans
            num = (pz * tpl_arr).sum(axis=(2, 3))
            denom = np.sqrt((pz * pz).sum(axis=(2, 3))) * tpl_norm + 1e-9
            scores = num / denom
            flat = scores.ravel()
            best_i = int(np.argmax(flat))
            iy, ix = np.unravel_index(best_i, scores.shape)
            best_score = float(flat[best_i])
            # Sub-pixel parabolic interpolation: fits a parabola through the
            # 3-point neighbourhood around the integer peak in each axis to
            # find the fractional offset that maximises the NCC surface.
            # This halves the effective position quantisation error.
            sub_dx, sub_dy = 0.0, 0.0
            if step == 1 and 0 < ix < scores.shape[1] - 1:
                denom_x = 2.0 * best_score - float(scores[iy, ix - 1]) - float(scores[iy, ix + 1])
                if abs(denom_x) > 1e-7:
                    sub_dx = float(np.clip(
                        (float(scores[iy, ix + 1]) - float(scores[iy, ix - 1])) / (2.0 * denom_x),
                        -0.5, 0.5,
                    ))
            if step == 1 and 0 < iy < scores.shape[0] - 1:
                denom_y = 2.0 * best_score - float(scores[iy - 1, ix]) - float(scores[iy + 1, ix])
                if abs(denom_y) > 1e-7:
                    sub_dy = float(np.clip(
                        (float(scores[iy + 1, ix]) - float(scores[iy - 1, ix])) / (2.0 * denom_y),
                        -0.5, 0.5,
                    ))
            best_xy = (float(xs[ix]) + sub_dx, float(ys[iy]) + sub_dy)
            neigh = 6
            mask = scores.copy()
            mask[max(0, iy-neigh):iy+neigh+1,
                 max(0, ix-neigh):ix+neigh+1] = -1e9
            second_score = float(mask.max()) if mask.size else -1.0
            return best_score, second_score, best_xy

        best_score = -1e9
        second_score = -1e9
        best_xy = None
        for y in ys:
            for x in xs:
                patch = search_arr[y:y + th_loc, x:x + tw_loc]
                patch = patch - patch.mean()
                s = float((patch * tpl_arr).sum() / (np.sqrt((patch * patch).sum()) * tpl_norm + 1e-9))
                if s > best_score:
                    second_score = best_score
                    best_score = s
                    best_xy = (int(x), int(y))
                elif s > second_score:
                    second_score = s
        if best_xy is None:
            return None
        return best_score, second_score, best_xy

    template = _edgeify(plan_arr)

    # Save the pre-road-band plan image and photo offsets for patch-consensus.
    # Road-band crops plan_crop to a single horizontal stripe, reducing the
    # patch grid from ~3×3 = 9 patches to ~4×1 = 4, most of which fall in
    # empty areas and are rejected.  Using the full image gives more spatially
    # distributed patches and reliable 3-patch consensus.
    _pc_plan_arr   = plan_arr.copy()
    _pc_photo_x1   = photo_x1
    _pc_photo_y1   = photo_y1
    _full_plan_feature_img = np.clip(plan_arr, 0, 255).astype(np.uint8)
    _full_template_edge = template.copy()
    _full_photo_x1 = photo_x1
    _full_photo_y1 = photo_y1
    _road_band_applied = False
    _feature_use_full_plan = False

    # Road-band reduction crops the template to the horizontally densest strip.
    # For aerial/topo plans this can help focus NCC on road structure, but for
    # vector/ALKIS plans it shrinks the template from ~260px tall to ~96px,
    # keeping only a thin stripe of diagonal features (e.g. canal curves) that
    # appear at multiple locations along the same waterway — causing systematic
    # false positives 1-2 km from the true location.
    # Road-band is therefore limited to aerial plans only.
    # Guard: skip road-band entirely when the edge map is already sparse.
    # A heavily overlaid plan after achromatic isolation may have < 5% edge
    # pixels; cropping to 34% of that leaves an unusable template.
    _edge_density = float((template > 0.1).mean())
    _road_band_viable = (_edge_density >= 0.05)
    if not _road_band_viable:
        print(f"[~] Template edge density {_edge_density:.1%} -- skipping road-band crop")
    if template.size and ACTIVE_WMS_CONFIG_KEY == "aerial" and _road_band_viable:
        row_energy = template.mean(axis=1)
        band_h = max(40, int(template.shape[0] * 0.34))
        # Center-biased weights: rows near the vertical center score higher so the
        # selected band stays near the map body rather than drifting to an edge.
        _h = template.shape[0]
        _mid = (_h - 1) / 2.0
        weights = (1.0 + 0.3 * (1.0 - np.abs(np.arange(_h, dtype=np.float32) - _mid) / max(_mid, 1.0))).astype(np.float32)
        best_band = None
        best_score = -1e9
        limit = max(1, template.shape[0] - band_h + 1)
        for y0 in range(limit):
            score = float((row_energy[y0:y0 + band_h] * weights[y0:y0 + band_h]).sum())
            if score > best_score:
                best_score = score
                best_band = (y0, y0 + band_h)
        if best_band:
            ry1, ry2 = best_band
            band_h = ry2 - ry1
            # Only apply band crop if it keeps at least 30% of template height.
            # A very thin template produces a flat NCC surface (confidence ≈ 1).
            if band_h >= max(30, int(template.shape[0] * 0.30)):
                photo_y1 += ry1
                template = template[ry1:ry2, :]
                plan_crop = plan_crop.crop((0, ry1, plan_crop.width, ry2))
                _road_band_applied = True
                print(f"[i] Template road band: rows={ry1}-{ry2} height={band_h}")
            else:
                print(f"[~] Road band too narrow ({band_h}px) -- using full template")
    elif template.size:
        print(f"[i] {ACTIVE_WMS_CONFIG_KEY.capitalize()} reference -- keeping full template height (road-band disabled)")

    plan_feature_img = np.array(plan_crop, dtype=np.uint8)
    th, tw = template.shape
    template_edge_raw = template.copy()   # save un-normalised edge map for fine-pass resize
    template = template - template.mean()

    # Save template debug image so mismatches can be diagnosed visually.
    try:
        _tpl_dbg = (np.clip(template_edge_raw, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(_tpl_dbg, mode="L").save(str(_artifact_path("wms_template.png")))
        print(f"[i] Template debug image saved: {tw}×{th}px → _work/wms_template.png")
    except Exception as _e:
        print(f"[~] Could not save template debug image: {_e}")

    ssl_ctx = ssl._create_unverified_context()

    def _download_ref_tile(tile_cx, tile_cy):
        tile_w = search_hw * 2
        tile_h = search_hh * 2
        tw_s, te_s = tile_cx - tile_w / 2, tile_cx + tile_w / 2
        ts_s, tn_s = tile_cy - tile_h / 2, tile_cy + tile_h / 2
        # WMS 1.1.1 uses SRS and always E,N axis order.
        # WMS 1.3.0 uses CRS and follows the CRS axis definition:
        #   for EPSG:25832 the OGC axis order is (Northing, Easting) → BBOX = minN,minE,maxN,maxE.
        # Using 1.1.1 for services (e.g. ALKIS) that strictly enforce the 1.3.0 axis convention.
        _bgcolor_param = f"&BGCOLOR={WMS_BGCOLOR}" if WMS_BGCOLOR else ""
        # WMS LAYERS can be comma-separated; STYLES must match layer count (empty=default)
        _n_layers = WMS_LAYER.count(",") + 1
        _styles_param = ",".join([""] * _n_layers)
        if WMS_VERSION == "1.1.1":
            tile_bbox = f"{tw_s},{ts_s},{te_s},{tn_s}"   # E,N order (always in 1.1.1)
            tile_req = (
                f"{WMS_URL}?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap"
                f"&LAYERS={WMS_LAYER}&STYLES={_styles_param}&SRS=EPSG:{epsg}"
                f"&BBOX={tile_bbox}&WIDTH={ref_px}&HEIGHT={ref_px}"
                f"&FORMAT={WMS_FORMAT}&TRANSPARENT=FALSE{_bgcolor_param}"
            )
        else:
            tile_bbox = f"{tw_s},{ts_s},{te_s},{tn_s}"   # E,N order (NRW servers accept this)
            tile_req = (
                f"{WMS_URL}?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
                f"&LAYERS={WMS_LAYER}&STYLES={_styles_param}&CRS=EPSG:{epsg}"
                f"&BBOX={tile_bbox}&WIDTH={ref_px}&HEIGHT={ref_px}"
                f"&FORMAT={WMS_FORMAT}&TRANSPARENT=FALSE{_bgcolor_param}"
            )
        tile_data, _ = _fetch_url_bytes_cached(
            tile_req,
            namespace="wms_tiles",
            timeout=tile_timeout_s,
            headers={"User-Agent": "auto_georeference/1.0"},
            context=ssl_ctx,
            suffix=".img",
        )
        try:
            _tile_img_raw = Image.open(io.BytesIO(tile_data)).convert("L")
        except Exception:
            # Server returned non-image data (e.g. WMS exception XML) -- surface the message
            preview = tile_data[:300].decode("utf-8", errors="replace").replace("\n", " ")
            raise RuntimeError(f"WMS returned non-image response: {preview!r}")
        tile_arr = np.array(_tile_img_raw, dtype=np.float32)
        tile_img = _tile_img_raw  # kept alive only for callers that save it as debug artifact
        return tile_img, tile_arr, (tw_s, te_s, ts_s, tn_s)

    def _read_local_reference_tile(tile_cx, tile_cy):
        ds_ref = gdal.Open(str(LOCAL_REFERENCE_RASTER))
        if ds_ref is None:
            raise RuntimeError("local reference raster cannot be opened")
        try:
            gt_ref = ds_ref.GetGeoTransform()
            ref_w = ds_ref.RasterXSize
            ref_h = ds_ref.RasterYSize
            ref_west = gt_ref[0]
            ref_north = gt_ref[3]
            ref_east = ref_west + gt_ref[1] * ref_w
            ref_south = ref_north + gt_ref[5] * ref_h

            tile_w = search_hw * 2
            tile_h = search_hh * 2
            tw_s, te_s = tile_cx - tile_w / 2, tile_cx + tile_w / 2
            ts_s, tn_s = tile_cy - tile_h / 2, tile_cy + tile_h / 2

            if te_s <= ref_west or tw_s >= ref_east or tn_s <= ref_south or ts_s >= ref_north:
                raise RuntimeError("local reference raster does not cover requested search tile")

            px1 = max(0, int((tw_s - ref_west) / gt_ref[1]))
            px2 = min(ref_w, int((te_s - ref_west) / gt_ref[1]))
            py1 = max(0, int((tn_s - ref_north) / gt_ref[5]))
            py2 = min(ref_h, int((ts_s - ref_north) / gt_ref[5]))
            read_w = max(1, px2 - px1)
            read_h = max(1, py2 - py1)

            if ds_ref.RasterCount >= 3:
                r = ds_ref.GetRasterBand(1).ReadAsArray(px1, py1, read_w, read_h)
                g = ds_ref.GetRasterBand(2).ReadAsArray(px1, py1, read_w, read_h)
                b = ds_ref.GetRasterBand(3).ReadAsArray(px1, py1, read_w, read_h)
                arr = np.dstack([r, g, b]).astype(np.uint8)
                tile_img = Image.fromarray(arr, mode="RGB").resize((ref_px, ref_px), Image.LANCZOS).convert("L")
            else:
                arr = ds_ref.GetRasterBand(1).ReadAsArray(px1, py1, read_w, read_h)
                arr = np.clip(arr, 0, 255).astype(np.uint8)
                tile_img = Image.fromarray(arr, mode="L").resize((ref_px, ref_px), Image.LANCZOS)
            tile_arr = np.array(tile_img, dtype=np.float32)
        finally:
            ds_ref = None
        return tile_img, tile_arr, (tw_s, te_s, ts_s, tn_s)

    use_local_reference = LOCAL_REFERENCE_RASTER.exists()
    if use_local_reference:
        print(f"[i] Using local reference raster for refinement: {LOCAL_REFERENCE_RASTER}")
    else:
        print(f"[i] No local reference raster found. To avoid WMS issues, export NRW DOP to: {LOCAL_REFERENCE_RASTER}")
    tile_timeout_s = WMS_TILE_TIMEOUT
    tile_workers = WMS_PARALLEL_WORKERS
    if ACTIVE_WMS_CONFIG_KEY == "vector":
        tile_timeout_s = max(WMS_TILE_TIMEOUT, 18)
        tile_workers = min(WMS_PARALLEL_WORKERS, 3)
        print(f"[i] Vector WMS tuning: timeout={tile_timeout_s}s workers={tile_workers}")
    tile_step_x = search_hw * 1.15
    tile_step_y = search_hh * 1.15

    def _score_tile_grid(offset_pairs):
        best_local = None
        successful_local = 0
        first_local = None
        top_candidates = []
        fetch_failures_local = 0
        timeout_failures_local = 0
        def _candidate_rank(c):
            return (
                c["score"],
                c["confidence"],
                c["score_gap"],
                -c["total_m"],
            )
        # ---- parallel tile downloads ------------------------------------
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _fetch(oy_ox):
            if _CANCEL_FLAG.is_set():
                raise RuntimeError("cancelled by user")
            oy, ox = oy_ox
            tcx = cx + ox * tile_step_x
            tcy = cy + oy * tile_step_y
            if use_local_reference:
                return _read_local_reference_tile(tcx, tcy)
            return _download_ref_tile(tcx, tcy)

        n_workers = min(tile_workers, len(offset_pairs))
        fetched = {}   # (oy, ox) → (tile_img, tile_arr, tile_bounds)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_fetch, k): k for k in offset_pairs}
            for fut in as_completed(futures):
                key = futures[fut]
                oy, ox = key
                try:
                    fetched[key] = fut.result()
                except Exception as exc:
                    fetch_failures_local += 1
                    if "timed out" in str(exc).lower():
                        timeout_failures_local += 1
                    print(f"    tile ({ox},{oy}) skipped: {exc}")
        print(f"[i] {len(fetched)}/{len(offset_pairs)} tiles fetched")

        # ---- score each tile -----------------------------------------
        for (oy, ox), (tile_img, tile_arr, tile_bounds) in fetched.items():
            tile_cx = cx + ox * tile_step_x
            tile_cy = cy + oy * tile_step_y

            successful_local += 1
            if first_local is None:
                first_local = tile_img
                tile_img.save(str(_artifact_path("wms_ref_first.jpg")))

            tile_std = float(tile_arr.std())
            tile_mean = float(tile_arr.mean())
            tile_max  = float(tile_arr.max())
            if tile_std < 1.0 or (tile_mean < 2.0 and tile_max < 5.0):
                print(f"    tile ({ox},{oy}) ignored: low variance ({tile_std:.2f})")
                continue
            search = _edgeify(tile_arr)
            sh, sw = search.shape
            if sh < th or sw < tw:
                print(f"    tile ({ox},{oy}) too small ({sw}×{sh}) for template ({tw}×{th})")
                continue

            ncc = _ncc_score(search, template, step=1)
            if ncc is None:
                continue
            tile_best_score, tile_second_score, tile_best_xy = ncc

            if tile_best_xy is None:
                continue

            best_x, best_y = tile_best_xy
            # tpl_px is the current-pass template scale (142 coarse, 284 fine).
            # photo_x1/y1 are in tpl_px_coarse coordinates, so scale them up
            # for the fine pass to avoid a systematic offset of (tpl_px-tpl_px_coarse)/2 pixels.
            _photo_x1_scaled = photo_x1 * tpl_px / tpl_px_coarse
            _photo_y1_scaled = photo_y1 * tpl_px / tpl_px_coarse
            expected_x = (sw - tpl_px) / 2 + _photo_x1_scaled
            expected_y = (sh - tpl_px) / 2 + _photo_y1_scaled
            sx = best_x - expected_x
            sy = best_y - expected_y
            tw_s, te_s, ts_s, tn_s = tile_bounds
            m_per_px_x = (te_s - tw_s) / sw
            m_per_px_y = (tn_s - ts_s) / sh
            shift_e = (tile_cx - cx) + sx * m_per_px_x
            shift_n = (tile_cy - cy) - sy * m_per_px_y
            total_m = (shift_e**2 + shift_n**2) ** 0.5
            peak_confidence = (tile_best_score + 1.0) / max(tile_second_score + 1.0, 1e-6)
            score_gap = tile_best_score - tile_second_score

            candidate = {
                "score": tile_best_score,
                "confidence": peak_confidence,
                "score_gap": score_gap,
                "shift_e": shift_e,
                "shift_n": shift_n,
                "total_m": total_m,
                "tile": (ox, oy),
                "img": tile_img,
                "search": search,
                "tile_bounds": tile_bounds,
                "best_xy": tile_best_xy,
            }
            top_candidates.append({
                "score": tile_best_score,
                "confidence": peak_confidence,
                "score_gap": score_gap,
                "shift_e": shift_e,
                "shift_n": shift_n,
                "total_m": total_m,
                "tile": (ox, oy),
                "img": tile_img,
                "search": search,
                "tile_bounds": tile_bounds,
                "best_xy": tile_best_xy,
            })
            if best_local is None or _candidate_rank(candidate) > _candidate_rank(best_local):
                best_local = candidate
        top_candidates.sort(key=_candidate_rank, reverse=True)
        return (
            best_local,
            successful_local,
            first_local,
            top_candidates[:5],
            {
                "requested": len(offset_pairs),
                "fetched": len(fetched),
                "failed": fetch_failures_local,
                "timeouts": timeout_failures_local,
            },
        )

    def _run_tile_search(pass_label: str = ""):
        nonlocal best, successful_tiles, ref_px, tpl_px, template, th, tw, cx, cy, tile_step_x, tile_step_y
        tile_step_x = search_hw * 1.15
        tile_step_y = search_hh * 1.15
        coarse_offsets = [
            (oy, ox)
            for oy in range(-WMS_COARSE_RANGE, WMS_COARSE_RANGE + 1)
            for ox in range(-WMS_COARSE_RANGE, WMS_COARSE_RANGE + 1)
        ]
        best_local, successful_local, _, coarse_top_local, coarse_metrics_local = _score_tile_grid(coarse_offsets)

        _saved_wms_local = None
        _effective_plan_type_local = ACTIVE_WMS_CONFIG_KEY
        if ACTIVE_WMS_CONFIG_KEY == "vector" and best_local is None:
            _requested = max(1, coarse_metrics_local.get("requested", len(coarse_offsets)))
            _fetched = coarse_metrics_local.get("fetched", 0)
            _timeouts = coarse_metrics_local.get("timeouts", 0)
            _unstable_fetch = (
                _fetched < max(5, _requested // 3)
                or _timeouts >= max(3, _requested // 5)
            )
            if successful_local == 0 or _unstable_fetch:
                _fallback_keys = ["vector_osm", "topo"] if _unstable_fetch else ["topo"]
                _g = globals()
                _saved_wms_local = {k: _g[k] for k in ("WMS_URL", "WMS_LAYER", "WMS_FORMAT", "WMS_VERSION", "WMS_BGCOLOR")}
                for _fb_key in _fallback_keys:
                    _fb = WMS_CONFIGS.get(_fb_key, {})
                    if not _fb:
                        continue
                    _g["WMS_URL"] = _fb["url"]
                    _g["WMS_LAYER"] = _fb["layer"]
                    _g["WMS_FORMAT"] = _fb.get("format", "image/jpeg")
                    _g["WMS_VERSION"] = _fb.get("wms_version", "1.3.0")
                    _g["WMS_BGCOLOR"] = _fb.get("bgcolor", "")
                    _effective_plan_type_local = _fb_key
                    if _unstable_fetch:
                        print(
                            f"[!] ALKIS tile fetch unstable ({_fetched}/{_requested} fetched, {_timeouts} timeout(s)) -- retrying with {_fb.get('label', _fb_key)}"
                        )
                    else:
                        print(f"[!] ALKIS returned 0 usable tiles -- retrying with {_fb.get('label', _fb_key)}")
                    (
                        best_local,
                        successful_local,
                        _,
                        coarse_top_local,
                        coarse_metrics_local,
                    ) = _score_tile_grid(coarse_offsets)
                    if best_local is not None or successful_local > 0:
                        break

        final_candidates_local = list(coarse_top_local or [])
        if coarse_top_local:
            print(f"[i] Top coarse candidates{pass_label}:")
            for cand in coarse_top_local[:3]:
                print(
                    f"    tile={cand['tile']}  score={cand['score']:.3f}  gap={cand['score_gap']:.3f}  conf={cand['confidence']:.2f}x  "
                    f"ΔE={cand['shift_e']:+.0f}  ΔN={cand['shift_n']:+.0f}  total={cand['total_m']:.0f} m"
                )

        _seed_precise = has_coord_anchors or seed_confidence in ("address", "street", "feature", "manual_seed", "last_result")
        _can_skip_fine = (
            best_local is not None
            and _seed_precise
            and best_local.get("score", 0.0) >= 0.42
            and best_local.get("confidence", 0.0) >= 1.12
            and best_local.get("score_gap", 0.0) >= 0.05
            and best_local.get("total_m", WMS_MAX_SHIFT_M + 1.0) <= min(WMS_MAX_SHIFT_M, 800.0)
            and (
                len(coarse_top_local) < 2
                or best_local.get("score", 0.0) - coarse_top_local[1].get("score", 0.0) >= 0.03
            )
        )
        if _can_skip_fine:
            print(
                f"[i] Coarse WMS match already decisive{pass_label} -- skipping fine tile search "
                f"(score={best_local['score']:.3f} gap={best_local['score_gap']:.3f} conf={best_local['confidence']:.2f}x)"
            )
            return best_local, successful_local, coarse_top_local, final_candidates_local, _saved_wms_local, _effective_plan_type_local

        if _CANCEL_FLAG.is_set():
            print("[!] WMS refinement cancelled by user -- skipping refinement")
            return best_local, successful_local, coarse_top_local, final_candidates_local, _saved_wms_local, _effective_plan_type_local

        # ── Weak-NCC DTK fallback ─────────────────────────────────────────────
        # ALKIS (cadastral) NCC scores < 0.08 are noise-level for fine-scale CAD
        # road plans: ALKIS shows sparse property-boundary lines that have little
        # visual overlap with the plan's construction geometry.  DTK (topographic)
        # renders the road network, buildings, and terrain in a form much closer
        # to what the plan references, giving reliably higher NCC scores.
        # Trigger: ALKIS was selected automatically (no manual WMS override) AND
        # all coarse top-3 scores are below the threshold AND WMS_CONFIG_OVERRIDE
        # is not set by the user.
        _alkis_weak_ncc = (
            ACTIVE_WMS_CONFIG_KEY == "vector"
            and WMS_CONFIG_OVERRIDE is None          # respect manual overrides
            and _saved_wms_local is None             # didn't already fall back
            and best_local is not None               # tiles did fetch (not a network issue)
            and coarse_top_local
            and coarse_top_local[0].get("score", 1.0) < 0.08
        )
        if _alkis_weak_ncc:
            _dtk_cfg = WMS_CONFIGS.get("topo", {})
            if _dtk_cfg:
                print(
                    f"[i] ALKIS NCC weak (best={coarse_top_local[0]['score']:.3f}) -- retrying coarse with DTK topographic"
                )
                _g = globals()
                _saved_wms_local = {k: _g[k] for k in ("WMS_URL", "WMS_LAYER", "WMS_FORMAT", "WMS_VERSION", "WMS_BGCOLOR")}
                _g["WMS_URL"]     = _dtk_cfg["url"]
                _g["WMS_LAYER"]   = _dtk_cfg["layer"]
                _g["WMS_FORMAT"]  = _dtk_cfg.get("format", "image/jpeg")
                _g["WMS_VERSION"] = _dtk_cfg.get("wms_version", "1.3.0")
                _g["WMS_BGCOLOR"] = _dtk_cfg.get("bgcolor", "")
                _effective_plan_type_local = "topo"
                (
                    _dtk_best, _dtk_success, _, _dtk_top, _
                ) = _score_tile_grid(coarse_offsets)
                if _dtk_top:
                    print(f"[i] DTK coarse candidates:")
                    for _dc in _dtk_top[:3]:
                        print(
                            f"    tile={_dc['tile']}  score={_dc['score']:.3f}  gap={_dc['score_gap']:.3f}  conf={_dc['confidence']:.2f}x  "
                            f"ΔE={_dc['shift_e']:+.0f}  ΔN={_dc['shift_n']:+.0f}  total={_dc['total_m']:.0f} m"
                        )
                # Keep DTK result only if it improves on ALKIS
                _alkis_top_score = coarse_top_local[0].get("score", 0.0)
                _dtk_top_score   = _dtk_top[0].get("score", 0.0) if _dtk_top else 0.0
                if _dtk_top_score > _alkis_top_score + 0.005:
                    print(f"[i] DTK improves NCC ({_alkis_top_score:.3f} → {_dtk_top_score:.3f}) -- using DTK for fine pass")
                    best_local         = _dtk_best
                    successful_local  += _dtk_success
                    coarse_top_local   = _dtk_top
                    final_candidates_local = list(_dtk_top or [])
                else:
                    print(
                        f"[~] DTK did not improve NCC ({_alkis_top_score:.3f} vs {_dtk_top_score:.3f}) -- reverting to ALKIS result"
                    )
                    # Restore ALKIS globals
                    for k, v in _saved_wms_local.items():
                        _g[k] = v
                    _saved_wms_local = None
                    _effective_plan_type_local = "vector"

        if best_local is not None:
            def _candidate_rank(c):
                return (c["score"], c["confidence"], c["score_gap"], -c["total_m"])

            best_local["img"].save(str(_artifact_path("wms_ref_coarse_best.jpg")))
            tile_step_x = search_hw * WMS_FINE_STEP
            tile_step_y = search_hh * WMS_FINE_STEP
            fine_offsets = [
                (oy, ox)
                for oy in range(-WMS_FINE_RANGE, WMS_FINE_RANGE + 1)
                for ox in range(-WMS_FINE_RANGE, WMS_FINE_RANGE + 1)
            ]

            ref_px = min(512, WMS_REF_PX * 2)
            tpl_px = max(64, int(ref_px / WMS_SEARCH_EXPAND))
            _scale = tpl_px / max(tw, th)
            _new_w = max(1, int(tw * _scale))
            _new_h = max(1, int(th * _scale))
            _tpl_img = Image.fromarray(
                (np.clip(template_edge_raw, 0, 1) * 255).astype(np.uint8)
            ).resize((_new_w, _new_h), Image.LANCZOS)
            template = np.array(_tpl_img, dtype=np.float32) / 255.0
            template = template - template.mean()
            th, tw = template.shape
            print(f"[i] Fine pass{pass_label}: ref={ref_px}px  template={tw}×{th}px")

            fine_global_best = None
            fine_global_top = []
            for coarse_seed in coarse_top_local[:WMS_COARSE_TOP_K]:
                coarse_shift_e = coarse_seed["shift_e"]
                coarse_shift_n = coarse_seed["shift_n"]
                old_cx, old_cy = cx, cy
                cx = old_cx + coarse_shift_e
                cy = old_cy + coarse_shift_n
                best_fine, successful_fine, _, fine_top, _ = _score_tile_grid(fine_offsets)
                successful_local += successful_fine
                for cand in fine_top:
                    fine_global_top.append({
                        **cand,
                        "coarse_tile": coarse_seed["tile"],
                        "shift_e": coarse_shift_e + cand["shift_e"],
                        "shift_n": coarse_shift_n + cand["shift_n"],
                        "total_m": ((coarse_shift_e + cand["shift_e"])**2 + (coarse_shift_n + cand["shift_n"])**2) ** 0.5,
                    })
                if best_fine is not None:
                    merged = {
                        **best_fine,
                        "shift_e": coarse_shift_e + best_fine["shift_e"],
                        "shift_n": coarse_shift_n + best_fine["shift_n"],
                        "total_m": ((coarse_shift_e + best_fine["shift_e"])**2 + (coarse_shift_n + best_fine["shift_n"])**2) ** 0.5,
                        "tile": ("fine", coarse_seed["tile"], best_fine["tile"]),
                    }
                    if fine_global_best is None or _candidate_rank(merged) > _candidate_rank(fine_global_best):
                        fine_global_best = merged
                cx, cy = old_cx, old_cy

            if fine_global_top:
                fine_global_top.sort(key=_candidate_rank, reverse=True)
                final_candidates_local = fine_global_top
                print(f"[i] Top fine candidates{pass_label}:")
                for cand in fine_global_top[:5]:
                    print(
                        f"    coarse={cand.get('coarse_tile')} fine={cand['tile']}  "
                        f"score={cand['score']:.3f}  gap={cand['score_gap']:.3f}  conf={cand['confidence']:.2f}x  "
                        f"ΔE={cand['shift_e']:+.0f}  ΔN={cand['shift_n']:+.0f}  total={cand['total_m']:.0f} m"
                    )

            if fine_global_best is not None:
                fine_global_best["img"].save(str(_artifact_path("wms_ref_fine_best.jpg")))
                best_local = fine_global_best
            elif coarse_top_local:
                final_candidates_local = coarse_top_local

        return best_local, successful_local, coarse_top_local, final_candidates_local, _saved_wms_local, _effective_plan_type_local

    best = None
    successful_tiles = 0
    _saved_wms = None
    _effective_plan_type = ACTIVE_WMS_CONFIG_KEY
    best, successful_tiles, coarse_top, final_candidates, _saved_wms, _effective_plan_type = _run_tile_search()

    _need_full_template_retry = False
    if _road_band_applied and best is not None:
        _cluster_probe = [c for c in final_candidates[:5] if abs(c["score"] - best["score"]) <= 0.02]
        _shift_spread = 0.0
        if len(_cluster_probe) >= 2:
            for _cand in _cluster_probe[1:]:
                _delta = ((_cand["shift_e"] - best["shift_e"])**2 + (_cand["shift_n"] - best["shift_n"])**2) ** 0.5
                _shift_spread = max(_shift_spread, _delta)
        _need_full_template_retry = (
            best["score"] < 0.22
            or best["confidence"] < 1.05
            or _shift_spread >= 150.0
        )
    if _road_band_applied and _need_full_template_retry and ACTIVE_WMS_CONFIG_KEY == "aerial":
        print(
            "[i] Road-band template looks ambiguous/weak -- retrying WMS search with full-height template"
        )
        photo_x1 = _full_photo_x1
        photo_y1 = _full_photo_y1
        template_edge_raw = _full_template_edge.copy()
        template = template_edge_raw - template_edge_raw.mean()
        th, tw = template.shape
        ref_px = WMS_REF_PX
        tpl_px = max(64, int(ref_px / WMS_SEARCH_EXPAND))
        try:
            _tpl_dbg = (np.clip(template_edge_raw, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(_tpl_dbg, mode="L").save(str(_artifact_path("wms_template_full_retry.png")))
        except Exception:
            pass
        best_retry, successful_retry, coarse_top_retry, final_candidates_retry, _saved_wms_retry, _effective_plan_type_retry = _run_tile_search(" (full template retry)")
        if best_retry is not None and (
            best is None
            or best_retry["score"] > best["score"] + 0.015
            or (best_retry["score"] >= best["score"] - 0.005 and best_retry["confidence"] > best["confidence"] + 0.03)
        ):
            print(
                f"[i] Full-template retry selected: score={best_retry['score']:.3f} conf={best_retry['confidence']:.2f}x "
                f"(road-band score={best['score']:.3f} conf={best['confidence']:.2f}x)"
            )
            best = best_retry
            successful_tiles = successful_retry
            coarse_top = coarse_top_retry
            final_candidates = final_candidates_retry
            _saved_wms = _saved_wms_retry
            _effective_plan_type = _effective_plan_type_retry
            _feature_use_full_plan = True
        else:
            print("[i] Full-template retry did not improve the match -- keeping original NCC search")

    # Restore WMS globals if we temporarily switched to DTK for ALKIS fallback.
    if _saved_wms is not None:
        globals().update(_saved_wms)

    if best is None:
        if successful_tiles == 0:
            print("[!] Template matching found no valid location -- no reference tiles downloaded")
        else:
            print(f"[!] Template matching found no valid location -- {successful_tiles} tile(s) downloaded but none usable")
        print("[!] Template matching found no valid location -- skipping refinement")
        return gt

    # Patch-consensus and feature refinement match plan sub-patches against a
    # north-up WMS reference.  When the plan is significantly rotated the patches
    # are misoriented relative to the reference and matching always fails.
    # Skip both stages for rotated plans; tile-level NCC accuracy is the best
    # achievable without rotation-aware sub-pixel search.
    # Use the vision north-arrow hint (available now) rather than rotation_deg
    # (computed later in the rotation-search block below).
    _vis_rot = None
    try:
        _v = LAST_VISION_RESULT.get("north_arrow_direction_deg") if LAST_VISION_RESULT else None
        if isinstance(_v, (int, float)):
            _vis_rot = float(_v)
    except Exception:
        pass
    _plan_rotation_significant = (_vis_rot is not None and abs(_vis_rot) >= 15.0)
    if _plan_rotation_significant:
        print(f"[i] Plan rotation hint {_vis_rot:+.1f}° -- skipping patch-consensus and feature refinement")

    # _phase_corr_ultrafinement must be defined BEFORE any code that calls it.
    # Python treats any name that appears as a def target anywhere in a function
    # as a local variable for the entire function scope; calling it before the
    # def line raises UnboundLocalError.
    def _phase_corr_ultrafinement(gt_in, patch_px=1024):
        """
        Ultra-fine translation correction: native-resolution ORB matching between
        a plan patch and a WMS patch of the same area.  Operates at base_mpp
        (e.g. 0.77 m/px) instead of the ~16 m/px thumbnail used by coarse ORB,
        yielding sub-metre translation accuracy.  Rotation is deliberately NOT
        applied — north-up is preserved.
        """
        print(f"[~] Ultra-fine ORB refinement starting (patch={patch_px}px @ {base_mpp:.4f} m/px)")
        if not HAS_CV2:
            print("[~] Ultra-fine skipped: cv2 not available")
            return gt_in
        try:
            cx = gt_in[0] + gt_in[1] * (W / 2.0)
            cy = gt_in[3] + gt_in[5] * (H / 2.0)
            half = patch_px * base_mpp / 2.0          # metres half-extent
            tw_s = cx - half;  te_s = cx + half
            ts_s = cy - half;  tn_s = cy + half
            _styles = ",".join([""] * (WMS_LAYER.count(",") + 1))
            if WMS_VERSION == "1.1.1":
                _bbox = f"{tw_s},{ts_s},{te_s},{tn_s}"
                _url = (f"{WMS_URL}?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap"
                        f"&LAYERS={WMS_LAYER}&STYLES={_styles}&SRS=EPSG:{epsg}"
                        f"&BBOX={_bbox}&WIDTH={patch_px}&HEIGHT={patch_px}"
                        f"&FORMAT={WMS_FORMAT}&TRANSPARENT=FALSE")
            else:
                _bbox = f"{tw_s},{ts_s},{te_s},{tn_s}"
                _url = (f"{WMS_URL}?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
                        f"&LAYERS={WMS_LAYER}&STYLES={_styles}&CRS=EPSG:{epsg}"
                        f"&BBOX={_bbox}&WIDTH={patch_px}&HEIGHT={patch_px}"
                        f"&FORMAT={WMS_FORMAT}&TRANSPARENT=FALSE")
            _wms_data, _ = _fetch_url_bytes_cached(
                _url,
                namespace="wms_tiles",
                timeout=WMS_TILE_TIMEOUT,
                headers={"User-Agent": "auto_georeference/1.0"},
                context=ssl_ctx,
                suffix=".img",
            )
            _ref_tile = Image.open(io.BytesIO(_wms_data)).convert("L")
            ref_u8 = np.array(_ref_tile, dtype=np.uint8)
            _ref_tile.close()
            _px_cx = int((cx - gt_in[0]) / max(gt_in[1], 1e-9))
            _px_cy = int((cy - gt_in[3]) / min(gt_in[5], -1e-9))
            _x1 = max(0, _px_cx - patch_px // 2)
            _y1 = max(0, _px_cy - patch_px // 2)
            _x2 = min(W, _x1 + patch_px)
            _y2 = min(H, _y1 + patch_px)
            _rw = _x2 - _x1;  _rh = _y2 - _y1
            if _rw < patch_px // 4 or _rh < patch_px // 4:
                print(f"[~] Ultra-fine skipped: plan patch too small ({_rw}×{_rh} px, need ≥{patch_px//4})")
                return gt_in
            ds_full = gdal.Open(str(src_path))
            if ds_full is None:
                print(f"[~] Ultra-fine skipped: could not open plan raster {src_path}")
                return gt_in
            try:
                if ds_full.RasterCount >= 3:
                    _r = ds_full.GetRasterBand(1).ReadAsArray(_x1, _y1, _rw, _rh).astype(np.float32)
                    _g = ds_full.GetRasterBand(2).ReadAsArray(_x1, _y1, _rw, _rh).astype(np.float32)
                    _b = ds_full.GetRasterBand(3).ReadAsArray(_x1, _y1, _rw, _rh).astype(np.float32)
                    plan_gray = (0.299 * _r + 0.587 * _g + 0.114 * _b)
                else:
                    plan_gray = ds_full.GetRasterBand(1).ReadAsArray(_x1, _y1, _rw, _rh).astype(np.float32)
            finally:
                ds_full = None
            plan_u8 = np.clip(plan_gray, 0, 255).astype(np.uint8)
            if plan_u8.shape != (patch_px, patch_px):
                _p = np.full((patch_px, patch_px), 255, dtype=np.uint8)
                _p[:_rh, :_rw] = plan_u8
                plan_u8 = _p
                _r2 = np.full((patch_px, patch_px), 128, dtype=np.uint8)
                _r2[:min(_rh, ref_u8.shape[0]), :min(_rw, ref_u8.shape[1])] = (
                    ref_u8[:min(_rh, ref_u8.shape[0]), :min(_rw, ref_u8.shape[1])]
                )
                ref_u8 = _r2
            _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            plan_eq = _clahe.apply(plan_u8)
            ref_eq  = _clahe.apply(ref_u8)
            _orb_uf = cv2.ORB_create(nfeatures=3000, scaleFactor=1.2, nlevels=8,
                                      edgeThreshold=15, patchSize=31)
            kp1, des1 = _orb_uf.detectAndCompute(plan_eq, None)
            kp2, des2 = _orb_uf.detectAndCompute(ref_eq, None)
            if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
                print(f"[~] Ultra-fine skipped: too few keypoints "
                      f"(plan={len(kp1) if kp1 else 0}, ref={len(kp2) if kp2 else 0})")
                return gt_in
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            raw = bf.knnMatch(des1, des2, k=2)
            good = [m for m, n in raw if m.distance < 0.75 * n.distance]
            if len(good) < 8:
                print(f"[~] Ultra-fine skipped: only {len(good)} good matches after ratio test")
                return gt_in
            pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
            M, inlier_mask = cv2.estimateAffinePartial2D(
                pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3.0,
                maxIters=2000, confidence=0.99
            )
            if M is None or inlier_mask is None:
                print("[~] Ultra-fine skipped: RANSAC failed")
                return gt_in
            n_inliers = int(inlier_mask.sum())
            if n_inliers < 6:
                print(f"[~] Ultra-fine skipped: only {n_inliers} RANSAC inliers")
                return gt_in
            tx_px = M[0, 2]
            ty_px = M[1, 2]
            shift_e =  tx_px * base_mpp
            shift_n = -ty_px * base_mpp
            max_uf_shift = 40.0
            if n_inliers > 0:
                _pts1_t = cv2.transform(pts1.reshape(-1, 1, 2), M).reshape(-1, 2)
                _mask = inlier_mask.ravel() == 1
                residual_px = float(np.mean(np.linalg.norm(_pts1_t[_mask] - pts2[_mask], axis=1)))
            else:
                residual_px = 999.0
            residual_m = residual_px * base_mpp
            if abs(shift_e) > max_uf_shift or abs(shift_n) > max_uf_shift:
                print(f"[~] Ultra-fine rejected: shift too large "
                      f"ΔE={shift_e:+.1f} m  ΔN={shift_n:+.1f} m  "
                      f"(limit {max_uf_shift:.0f} m)  inliers={n_inliers}")
                return gt_in
            print(f"[✓] Ultra-fine ORB: ΔE={shift_e:+.1f} m  ΔN={shift_n:+.1f} m  "
                  f"inliers={n_inliers}/{len(good)}  residual={residual_m:.2f} m  "
                  f"patch={patch_px}px @ {base_mpp:.3f} m/px")
            return (gt_in[0] + shift_e, gt_in[1], gt_in[2],
                    gt_in[3] + shift_n, gt_in[4], gt_in[5])
        except Exception as _uf_exc:
            print(f"[~] Ultra-fine ORB skipped: {_uf_exc}")
            return gt_in

    _refinement_attempted = False
    _patch_refinement_succeeded = False
    _feature_refinement_succeeded = False
    # Tracks whether patch-consensus rejected any candidate due to geometric
    # validation failure (rotation/anisotropy/shear). When True the NCC location
    # is geometrically inconsistent with the plan's known orientation, and the
    # tight-consensus bypass should NOT override this evidence.
    _pc_geo_rejected = False
    if HAS_CV2 and final_candidates and ACTIVE_WMS_CONFIG_KEY in ("aerial", "topo", "vector") and not _plan_rotation_significant:
        _refinement_attempted = True
        try:
            print("[~] Patch-consensus refinement: evaluating top reference candidates …")
            patch_best = None
            _patch_limit_base = 6
            if len(final_candidates) >= 2 and (
                final_candidates[0].get("score", 0.0) - final_candidates[1].get("score", 0.0) >= 0.05
                and final_candidates[0].get("confidence", 0.0) >= 1.10
            ):
                _patch_limit_base = 3
            patch_limit = min(_patch_limit_base, len(final_candidates))
            # Use the pre-road-band full plan image: larger spatial extent gives
            # a 3×3 patch grid instead of 4×1, greatly improving the chance of
            # finding 3 spatially distinct matching patches.
            current_plan_img = np.clip(_pc_plan_arr, 0, 255).astype(np.uint8)
            _pc_photo_x1_cur = _pc_photo_x1
            _pc_photo_y1_cur = _pc_photo_y1
            if tpl_px != tpl_px_coarse:
                scale = tpl_px / max(tpl_px_coarse, 1)
                new_w = max(1, int(current_plan_img.shape[1] * scale))
                new_h = max(1, int(current_plan_img.shape[0] * scale))
                current_plan_img = np.array(
                    Image.fromarray(current_plan_img, mode="L").resize((new_w, new_h), Image.LANCZOS),
                    dtype=np.uint8,
                )
                _pc_photo_x1_cur = int(round(_pc_photo_x1 * scale))
                _pc_photo_y1_cur = int(round(_pc_photo_y1 * scale))
            for cand in final_candidates[:patch_limit]:
                ref_gray = np.array(cand["img"], dtype=np.uint8)
                gt_anchor = _shifted_gt(gt, cand["shift_e"], cand["shift_n"])
                patch_fit = _estimate_affine_from_patch_consensus(
                    current_plan_img,
                    ref_gray,
                    photo_x1=_pc_photo_x1_cur,
                    photo_y1=_pc_photo_y1_cur,
                    tpl_px_current=tpl_px,
                    tpl_px_base=tpl_px_coarse,
                    plan_type=_effective_plan_type,
                )
                if not patch_fit:
                    print(f"    tile={cand['tile']}  no stable patch consensus")
                    continue
                affine_candidate, ecc_score = _refine_affine_ecc(
                    patch_fit["plan_prep"],
                    patch_fit["ref_prep"],
                    patch_fit["affine"],
                    _effective_plan_type,
                )
                gt_candidate, _ = _compose_geotransform_from_local_affine(
                    affine_candidate,
                    px_x1=px_x1,
                    px_y1=px_y1,
                    read_w=read_w,
                    read_h=read_h,
                    tpl_px=tpl_px,
                    tpl_px_base=tpl_px_coarse,
                    photo_x1=photo_x1,
                    photo_y1=photo_y1,
                    tile_bounds=cand["tile_bounds"],
                    ref_width=ref_gray.shape[1],
                    ref_height=ref_gray.shape[0],
                )
                ref_mpp_x = (cand["tile_bounds"][1] - cand["tile_bounds"][0]) / max(ref_gray.shape[1], 1)
                ref_mpp_y = (cand["tile_bounds"][3] - cand["tile_bounds"][2]) / max(ref_gray.shape[0], 1)
                residual_m = patch_fit["residual_px"] * ((abs(ref_mpp_x) + abs(ref_mpp_y)) / 2.0)
                # Patch-consensus only contributes a local correction. Like the
                # ORB path below, validate the north-up geotransform we would
                # actually write instead of rejecting a good translation due to
                # small affine rotation/shear noise in the intermediate fit.
                _gc_e = gt_candidate[0] + gt_candidate[1] * (W / 2.0) + gt_candidate[2] * (H / 2.0)
                _gc_n = gt_candidate[3] + gt_candidate[4] * (W / 2.0) + gt_candidate[5] * (H / 2.0)
                gt_northup = (
                    _gc_e - base_mpp * (W / 2.0),
                    base_mpp, 0.0,
                    _gc_n + base_mpp * (H / 2.0),
                    0.0, -base_mpp,
                )
                ok, reason = _validate_affine_geotransform(
                    gt_northup, gt_anchor, W, H, base_mpp, ACTIVE_WMS_CONFIG_KEY
                )
                ecc_text = f"  ecc={ecc_score:.4f}" if ecc_score is not None else ""
                print(
                    f"    tile={cand['tile']}  patches={patch_fit['patch_count']}  "
                    f"inliers={patch_fit['inlier_count']}  residual={residual_m:.2f} m{ecc_text}  {reason}"
                )
                # Residual limit: at 1:1000 scale (~1 m/px reference) a correct
                # patch match should stay within ~10 m; allow up to 20 m for
                # sparser topo/vector plans.  Larger residuals mean the inliers
                # themselves disagree and the affine is unreliable.
                _max_residual_m = 10.0 if _effective_plan_type == "aerial" else 20.0
                if not ok:
                    # Geometric validation failure (rotation/anisotropy/shear) is
                    # strong evidence the NCC location is wrong -- flag this so the
                    # tight-consensus bypass cannot override it.
                    _pc_geo_rejected = True
                if not ok or patch_fit["inlier_count"] < 3 or residual_m > _max_residual_m:
                    continue
                rank = (
                    patch_fit["inlier_count"],
                    -(residual_m),
                    ecc_score if ecc_score is not None else -1.0,
                    cand["score"],
                )
                if patch_best is None or rank > patch_best["rank"]:
                    patch_best = {
                        "rank": rank,
                        "gt": gt_northup,
                        "gt_anchor": gt_anchor,
                        "cand": cand,
                        "fit": patch_fit,
                        "ecc": ecc_score,
                        "residual_m": residual_m,
                    }

            if patch_best is not None:
                try:
                    dbg = Image.new(
                        "L",
                        (
                            patch_best["fit"]["plan_prep"].shape[1] + patch_best["fit"]["ref_prep"].shape[1],
                            max(patch_best["fit"]["plan_prep"].shape[0], patch_best["fit"]["ref_prep"].shape[0]),
                        ),
                        color=0,
                    )
                    dbg.paste(Image.fromarray(patch_best["fit"]["plan_prep"], mode="L"), (0, 0))
                    dbg.paste(Image.fromarray(patch_best["fit"]["ref_prep"], mode="L"), (patch_best["fit"]["plan_prep"].shape[1], 0))
                    dbg.save(str(_artifact_path("patch_consensus_debug.jpg")))
                except Exception:
                    pass
                center_seed_e = patch_best["gt_anchor"][0] + patch_best["gt_anchor"][1] * (W / 2.0) + patch_best["gt_anchor"][2] * (H / 2.0)
                center_seed_n = patch_best["gt_anchor"][3] + patch_best["gt_anchor"][4] * (W / 2.0) + patch_best["gt_anchor"][5] * (H / 2.0)
                center_cand_e = patch_best["gt"][0] + patch_best["gt"][1] * (W / 2.0) + patch_best["gt"][2] * (H / 2.0)
                center_cand_n = patch_best["gt"][3] + patch_best["gt"][4] * (W / 2.0) + patch_best["gt"][5] * (H / 2.0)
                center_shift = ((center_cand_e - center_seed_e) ** 2 + (center_cand_n - center_seed_n) ** 2) ** 0.5
                if center_shift > _local_refinement_shift_limit_m(ACTIVE_WMS_CONFIG_KEY):
                    print(
                        f"[!] Patch-consensus candidate rejected: center_shift={center_shift:.1f} m exceeds local limit"
                    )
                else:
                    # Check total displacement from the original seed geotransform.
                    # Patch-consensus returns early and bypasses the WMS_MAX_SHIFT_M
                    # guard in the acceptance block, so we must enforce it here.
                    # Additionally: when the best NCC score is weak (< 0.10) and the
                    # seed is already at street precision (±100-500 m), a large shift
                    # almost certainly reflects a noise peak in the correlation surface
                    # rather than a genuine geographic improvement.  Inliers as few as
                    # 3 (RANSAC minimum) are too fragile to override a good geocode.
                    _orig_cx = gt[0] + gt[1] * (W / 2.0) + gt[2] * (H / 2.0)
                    _orig_cy = gt[3] + gt[4] * (W / 2.0) + gt[5] * (H / 2.0)
                    _total_from_seed = ((center_cand_e - _orig_cx) ** 2 + (center_cand_n - _orig_cy) ** 2) ** 0.5
                    _best_ncc = patch_best["cand"].get("score", 1.0)
                    _pc_inliers = patch_best["fit"]["inlier_count"]
                    # Weak-NCC shift cap: for street-level seeds, only allow large
                    # shifts when the NCC score is strong enough to be trustworthy.
                    _weak_ncc_shift_cap = (
                        _best_ncc < 0.10
                        and seed_confidence in ("street", "feature")
                        and _total_from_seed > WMS_MAX_SHIFT_M * 0.25
                        and _pc_inliers < 5
                    )
                    # No-anchor strict floor: without OCR coordinate grid labels the
                    # seed may be off by many km (city-level precision).  A patch
                    # consensus with only the RANSAC minimum of 3 inliers and an NCC
                    # score below the strict no-anchor floor (0.38) is very likely a
                    # spurious match (e.g. rotated plan matched against a wrong tile).
                    # Require ≥ 4 inliers OR NCC ≥ 0.35 to accept in this regime.
                    _no_anchor_weak_patch = (
                        not has_coord_anchors
                        and _best_ncc < 0.35
                        and _pc_inliers < 4
                    )
                    if _total_from_seed > WMS_MAX_SHIFT_M:
                        print(
                            f"[!] Patch-consensus: total shift from seed {_total_from_seed:.0f} m "
                            f"exceeds WMS_MAX_SHIFT_M ({WMS_MAX_SHIFT_M:.0f} m) -- keeping seed"
                        )
                    elif _weak_ncc_shift_cap:
                        print(
                            f"[!] Patch-consensus: NCC score {_best_ncc:.3f} too weak to justify "
                            f"{_total_from_seed:.0f} m shift from {seed_confidence}-precision seed "
                            f"(inliers={_pc_inliers} < 5) -- keeping seed position"
                        )
                    elif _no_anchor_weak_patch:
                        print(
                            f"[!] Patch-consensus: NCC score {_best_ncc:.3f} < 0.35 with only "
                            f"{_pc_inliers} inliers -- insufficient evidence without coordinate anchors"
                        )
                    else:
                        print(
                            f"[✓] Patch-consensus refinement applied: tile={patch_best['cand']['tile']}  "
                            f"patches={patch_best['fit']['patch_count']}  inliers={_pc_inliers}  "
                            f"residual={patch_best['residual_m']:.2f} m  center_shift={center_shift:.1f} m"
                        )
                        _patch_refinement_succeeded = True
                        LAST_GEOREF_QUALITY.update({
                            "accepted": True,
                            "acceptance_reason": "patch_consensus",
                            "patch_consensus_succeeded": True,
                            "feature_refinement_succeeded": False,
                            "patch_tile": str(patch_best["cand"]["tile"]),
                            "patch_inliers": int(patch_best["fit"]["inlier_count"]),
                            "patch_residual_m": round(float(patch_best["residual_m"]), 3),
                            "patch_center_shift_m": round(float(center_shift), 3),
                        })
                        _pc_gt = _enforce_isotropic_geotransform(patch_best["gt"], W, H, base_mpp)
                        return _phase_corr_ultrafinement(_pc_gt)
        except Exception as exc:
            print(f"[!] Patch-consensus refinement failed at runtime: {exc}")

    if HAS_CV2 and final_candidates and not _plan_rotation_significant:
        try:
            _feature_plan_img = _full_plan_feature_img if _feature_use_full_plan else plan_feature_img
            if _feature_use_full_plan:
                print("[i] Feature refinement using full-height plan image")
            print("[~] Feature refinement: evaluating top reference candidates …")
            feature_best = None
            _feature_limit_base = 6
            if len(final_candidates) >= 2 and (
                final_candidates[0].get("score", 0.0) - final_candidates[1].get("score", 0.0) >= 0.05
                and final_candidates[0].get("confidence", 0.0) >= 1.10
            ):
                _feature_limit_base = 3
            feature_limit = min(_feature_limit_base, len(final_candidates))
            # Scale the residual limit to the reference tile pixel size.
            # At coarse resolution (~8 m/px) a 2-pixel residual is already ~16 m,
            # so a hard 6 m absolute limit would reject all valid matches.
            _ref_mpp_est = (search_hw + search_hh) / max(ref_px, 1)
            residual_limit_m = max(
                6.0 if ACTIVE_WMS_CONFIG_KEY == "aerial" else 9.0,
                _ref_mpp_est * 2.5,
            )
            min_inliers = 10 if ACTIVE_WMS_CONFIG_KEY == "vector" else 8
            for cand in final_candidates[:feature_limit]:
                try:
                    ref_gray = np.array(cand["img"], dtype=np.uint8)
                except Exception:
                    continue
                gt_anchor = _shifted_gt(gt, cand["shift_e"], cand["shift_n"])
                feat = _estimate_affine_feature_transform(_feature_plan_img, ref_gray, ACTIVE_WMS_CONFIG_KEY)
                if not feat:
                    print(f"    tile={cand['tile']}  no usable feature transform")
                    continue

                gt_candidate, _ = _compose_geotransform_from_local_affine(
                    feat["affine"],
                    px_x1=px_x1,
                    px_y1=px_y1,
                    read_w=read_w,
                    read_h=read_h,
                    tpl_px=tpl_px,
                    tpl_px_base=tpl_px_coarse,
                    photo_x1=photo_x1,
                    photo_y1=photo_y1,
                    tile_bounds=cand["tile_bounds"],
                    ref_width=ref_gray.shape[1],
                    ref_height=ref_gray.shape[0],
                )
                ref_mpp_x = (cand["tile_bounds"][1] - cand["tile_bounds"][0]) / max(ref_gray.shape[1], 1)
                ref_mpp_y = (cand["tile_bounds"][3] - cand["tile_bounds"][2]) / max(ref_gray.shape[0], 1)
                residual_m = feat["residual_px"] * ((abs(ref_mpp_x) + abs(ref_mpp_y)) / 2.0)
                # Extract centre from the full (possibly rotated) affine result and
                # build a north-up geotransform immediately.  We validate this north-up
                # version -- shear/rotation in the raw affine is irrelevant because we
                # never apply it.  This prevents good translations from being discarded
                # solely because of ORB shear noise.
                _gc_e = gt_candidate[0] + gt_candidate[1] * (W / 2.0) + gt_candidate[2] * (H / 2.0)
                _gc_n = gt_candidate[3] + gt_candidate[4] * (W / 2.0) + gt_candidate[5] * (H / 2.0)
                gt_northup = (
                    _gc_e - base_mpp * (W / 2.0),
                    base_mpp, 0.0,
                    _gc_n + base_mpp * (H / 2.0),
                    0.0, -base_mpp,
                )
                rot_deg_raw = math.degrees(math.atan2(gt_candidate[4], gt_candidate[1]))
                ok, reason = _validate_affine_geotransform(
                    gt_northup, gt_anchor, W, H, base_mpp, ACTIVE_WMS_CONFIG_KEY
                )
                print(
                    f"    tile={cand['tile']}  detector={feat['detector']}  "
                    f"inliers={feat['inlier_count']}/{feat['match_count']}  "
                    f"coverage={feat['coverage']:.2f}  residual={residual_m:.2f} m  {reason}"
                )
                if not ok or feat["inlier_count"] < min_inliers or residual_m > residual_limit_m:
                    continue

                center_seed_e = gt_anchor[0] + gt_anchor[1] * (W / 2.0) + gt_anchor[2] * (H / 2.0)
                center_seed_n = gt_anchor[3] + gt_anchor[4] * (W / 2.0) + gt_anchor[5] * (H / 2.0)
                center_cand_e = _gc_e
                center_cand_n = _gc_n
                center_shift = ((center_cand_e - center_seed_e) ** 2 + (center_cand_n - center_seed_n) ** 2) ** 0.5
                # High-confidence ORB matches (many inliers + low residual) are
                # trusted even with large center_shift — they can correct a NCC
                # false match that landed in the wrong geographic area.
                _base_limit = _local_refinement_shift_limit_m(ACTIVE_WMS_CONFIG_KEY)
                _HIGH_INL = 80
                _HIGH_RES = 20.0  # metres
                if feat["inlier_count"] >= _HIGH_INL and residual_m <= _HIGH_RES:
                    _shift_limit = max(_base_limit, WMS_MAX_SHIFT_M)
                else:
                    _shift_limit = _base_limit
                if center_shift > _shift_limit:
                    _cand_e_str = f"E={center_cand_e:.0f}  N={center_cand_n:.0f}"
                    print(f"    tile={cand['tile']}  rejected: center_shift={center_shift:.1f} m "
                          f"exceeds limit ({_shift_limit:.0f} m)  candidate at {_cand_e_str}")
                    continue
                # Score = weighted combination: residual accuracy + spatial coverage
                # + inlier ratio. Raw inlier count is NOT the primary key -- a wide
                # spatial coverage with low residual is more trustworthy than a high
                # inlier count concentrated in one corner of the plan.
                _feat_score = (
                    feat["coverage"] * 0.50       # 0-1: fraction of plan area spanned
                    + feat["inlier_ratio"] * 0.30  # 0-1: % of matches that are inliers
                    - residual_m / 50.0 * 0.20    # residual penalty (normalised to 50m)
                )
                rank = (
                    round(_feat_score, 4),
                    feat["inlier_count"],
                    -residual_m,
                )
                if feature_best is None or rank > feature_best["rank"]:
                    feature_best = {
                        "rank": rank,
                        "gt": gt_northup,        # already north-up, rotation stripped
                        "rot_deg": rot_deg_raw,  # original ORB rotation (for logging only)
                        "cand": cand,
                        "feat": feat,
                        "residual_m": residual_m,
                        "center_shift": center_shift,
                    }

            if feature_best is None and _refinement_attempted:
                print(
                    "[!] Both patch-consensus and feature-refinement failed for all candidates.\n"
                    "    NCC result accepted at low confidence (score={:.3f} gap={:.3f} conf={:.2f}x).\n"
                    "    Check _work/wms_template.png and _work/wms_ref_coarse_best.jpg to verify\n"
                    "    the template matches the reference. If placement is wrong, set manual_seed.json.".format(
                        best.get("score", 0), best.get("score_gap", 0), best.get("confidence", 0)
                    )
                )
            if feature_best is not None:
                gt = feature_best["gt"]   # already north-up (built in loop)
                feat = feature_best["feat"]
                cand = feature_best["cand"]
                _feature_refinement_succeeded = True
                _save_feature_debug(
                    feat["plan_prep"],
                    feat["ref_prep"],
                    feat["kp1"],
                    feat["kp2"],
                    feat["matches"],
                    feat["inliers"],
                    _artifact_path("feature_match_debug.jpg"),
                )
                print(
                    f"[✓] Feature refinement applied: tile={cand['tile']}  "
                    f"inliers={feat['inlier_count']}/{feat['match_count']}  "
                    f"residual={feature_best['residual_m']:.2f} m  "
                    f"center_shift={feature_best['center_shift']:.1f} m  "
                    f"scale=1:{int(round(base_mpp*300/0.0254))}"
                )
                # Ultra-fine pass: native-resolution phase correlation refines
                # the translation from ~20 m (ORB at 16 m/px thumbnail) to
                # sub-metre accuracy (phase correlation at 0.77 m/px native).
                LAST_GEOREF_QUALITY.update({
                    "accepted": True,
                    "acceptance_reason": "feature_refinement",
                    "patch_consensus_succeeded": bool(_patch_refinement_succeeded),
                    "feature_refinement_succeeded": True,
                    "feature_tile": str(cand["tile"]),
                    "feature_inliers": int(feat["inlier_count"]),
                    "feature_match_count": int(feat["match_count"]),
                    "feature_residual_m": round(float(feature_best["residual_m"]), 3),
                    "feature_center_shift_m": round(float(feature_best["center_shift"]), 3),
                })
                gt = _phase_corr_ultrafinement(gt)
                return gt
        except Exception as exc:
            print(f"[!] Feature refinement failed at runtime: {exc}")

    cluster_consensus = False
    cluster_size = 1
    near_tied_count = 1
    cluster_dominance = 0.0
    if final_candidates:
        final_candidates = sorted(
            final_candidates,
            key=lambda c: (c["score"], c["confidence"], c["score_gap"], -c["total_m"]),
            reverse=True,
        )
        best_global = final_candidates[0]
        ambiguous = []
        for cand in final_candidates[1:5]:
            shift_delta = ((cand["shift_e"] - best_global["shift_e"])**2 + (cand["shift_n"] - best_global["shift_n"])**2) ** 0.5
            if cand["score"] >= best_global["score"] - 0.010 and shift_delta >= 80.0:
                ambiguous.append((cand, shift_delta))
        if ambiguous:
            print("[!] Template match ambiguous -- near-tied candidates disagree on shift:")
            for cand, shift_delta in ambiguous[:3]:
                print(
                    f"    tile={cand['tile']}  score={cand['score']:.3f}  gap={cand['score_gap']:.3f}  "
                    f"conf={cand['confidence']:.2f}x  ΔE={cand['shift_e']:+.0f}  ΔN={cand['shift_n']:+.0f}  "
                    f"delta_vs_best={shift_delta:.0f} m"
                )
            # Cluster all near-tied candidates geographically.  If one geographic
            # cluster dominates (strictly more members than any other), use the
            # highest-gap candidate from that cluster instead of discarding.
            _CLUSTER_R = 600.0  # metres -- candidates within this radius are "same location"
            near_tied = [best_global] + [c for c, _ in ambiguous]
            near_tied_count = len(near_tied)
            clusters: list[list] = []
            for _c in near_tied:
                _placed = False
                for _cl in clusters:
                    _ce = sum(x["shift_e"] for x in _cl) / len(_cl)
                    _cn = sum(x["shift_n"] for x in _cl) / len(_cl)
                    if ((_c["shift_e"] - _ce)**2 + (_c["shift_n"] - _cn)**2) ** 0.5 < _CLUSTER_R:
                        _cl.append(_c)
                        _placed = True
                        break
                if not _placed:
                    clusters.append([_c])
            clusters.sort(key=lambda _cl: (-len(_cl), -max(x["score"] for x in _cl)))
            if len(clusters) == 1 or len(clusters[0]) > len(clusters[1]):
                # Dominant cluster found.
                # If the cluster spread is large (candidates split into two sub-groups),
                # find the tightest sub-cluster (within 50 m) and use that instead of
                # blending across sub-groups -- the midpoint between two competing
                # positions is not a reliable estimate of either.
                cluster_candidates = clusters[0]
                _blended_all = _blend_candidate_cluster(cluster_candidates)
                _spread_all = _blended_all.get("cluster_spread_m", 0.0) or 0.0
                _SUB_R = 50.0
                if _spread_all > _SUB_R and len(cluster_candidates) >= 2:
                    # Build tight sub-clusters from the near-tied candidates
                    _sub_cls: list[list] = []
                    for _cc in cluster_candidates:
                        _placed = False
                        for _sc in _sub_cls:
                            _sce = sum(x["shift_e"] for x in _sc) / len(_sc)
                            _scn = sum(x["shift_n"] for x in _sc) / len(_sc)
                            if ((_cc["shift_e"] - _sce)**2 + (_cc["shift_n"] - _scn)**2) ** 0.5 < _SUB_R:
                                _sc.append(_cc)
                                _placed = True
                                break
                        if not _placed:
                            _sub_cls.append([_cc])
                    # Augment each sub-cluster with any high-scoring candidate from
                    # final_candidates that was excluded from near_tied (because it was
                    # within 80 m of the best and not "ambiguous"), but is within _SUB_R
                    # of this sub-cluster's centroid.  This recovers supporting evidence
                    # for the best group that the ambiguity filter dropped.
                    _near_tied_ids = {id(c) for c in near_tied}
                    for _sc in _sub_cls:
                        _sce = sum(x["shift_e"] for x in _sc) / len(_sc)
                        _scn = sum(x["shift_n"] for x in _sc) / len(_sc)
                        for _fc in final_candidates:
                            if id(_fc) in _near_tied_ids:
                                continue  # already in near_tied
                            if _fc.get("score", 0) < best_global["score"] - 0.020:
                                continue  # too low score to be supporting evidence
                            if any(id(_fc) == id(x) for x in _sc):
                                continue  # already in this sub-cluster
                            _d = ((_fc["shift_e"] - _sce)**2 + (_fc["shift_n"] - _scn)**2) ** 0.5
                            if _d < _SUB_R:
                                _sc.append(_fc)
                    # Sort: when sub-clusters are near-equal in size (within 1 of largest),
                    # prefer highest max score — quality over marginal count advantage.
                    # Only prefer by count when one cluster is clearly larger (>1 gap).
                    _max_sub_size = max(len(_sc) for _sc in _sub_cls)
                    _sub_cls.sort(key=lambda _sc: (
                        0 if len(_sc) < _max_sub_size - 1 else -1,  # -1=near-max (wins)
                        -max(x["score"] for x in _sc),               # tiebreak: higher score first
                    ))
                    _best_sub = _sub_cls[0]
                    _winner_sub = _blend_candidate_cluster(_best_sub)
                    _spread_sub = _winner_sub.get("cluster_spread_m", 0.0) or 0.0
                    print(
                        f"[i] Sub-cluster refinement: best sub-cluster "
                        f"{len(_best_sub)}/{len(cluster_candidates)} candidates, "
                        f"spread={_spread_sub:.0f} m  "
                        f"ΔE={_winner_sub['shift_e']:+.0f}  ΔN={_winner_sub['shift_n']:+.0f}"
                    )
                    cluster_consensus = True
                    cluster_size = len(_best_sub)
                    cluster_dominance = len(_best_sub) / max(len(near_tied), 1)
                    _winner = _winner_sub
                else:
                    cluster_consensus = True
                    cluster_size = len(cluster_candidates)
                    cluster_dominance = len(cluster_candidates) / max(len(near_tied), 1)
                    _winner = _blended_all
                print(
                    f"[i] Cluster consensus: {len(clusters[0])}/{len(near_tied)} candidates in dominant cluster  "
                    f"ΔE={_winner['shift_e']:+.0f}  ΔN={_winner['shift_n']:+.0f}"
                )
                best_global = _winner
            else:
                # Two equally-sized geographic clusters -- normally discard.
                # Exception: if OCR place-name geocoding produced a centroid
                # (e.g. Wartburg/Henrichenburg), use it to pick the cluster
                # whose implied plan position is closer to that centroid.
                _ocr_c = LAST_OCR_PLACES_CENTROID
                if _ocr_c is not None and len(clusters) == 2:
                    _sr_e = gt[0] + gt[1] * (W / 2) + gt[2] * (H / 2)
                    _sr_n = gt[3] + gt[4] * (W / 2) + gt[5] * (H / 2)
                    _dists = []
                    for _cl in clusters:
                        _bl = _blend_candidate_cluster(_cl)
                        _impl_e = _sr_e + _bl["shift_e"]
                        _impl_n = _sr_n + _bl["shift_n"]
                        _dists.append((
                            ((_impl_e - _ocr_c[0]) ** 2 + (_impl_n - _ocr_c[1]) ** 2) ** 0.5,
                            _bl,
                            _cl,
                        ))
                    _dists.sort(key=lambda x: x[0])
                    _ocr_dist_best, _winner_bl, _winner_cl = _dists[0]
                    _ocr_dist_other = _dists[1][0]
                    if _ocr_dist_best < _ocr_dist_other * 0.5:
                        # Winning cluster is at least 2× closer to OCR centroid.
                        print(
                            f"[i] Ambiguity resolved by OCR place-name centroid "
                            f"(dist={_ocr_dist_best:.0f} m vs {_ocr_dist_other:.0f} m): "
                            f"ΔE={_winner_bl['shift_e']:+.0f}  ΔN={_winner_bl['shift_n']:+.0f}"
                        )
                        cluster_consensus = True
                        cluster_size = len(_winner_cl)
                        cluster_dominance = len(_winner_cl) / max(len(near_tied), 1)
                        best_global = _winner_bl
                    else:
                        print("[!] Discarding refinement because the best shift is not unique")
                        return gt
                else:
                    print("[!] Discarding refinement because the best shift is not unique")
                    return gt
        else:
            # Even when there is no ambiguity warning, blend the best nearby
            # tiles so the final shift reflects multi-tile evidence.
            _CONSENSUS_R = 220.0
            nearby = [best_global]
            for cand in final_candidates[1:5]:
                if cand["score"] < best_global["score"] - 0.020:
                    continue
                shift_delta = ((cand["shift_e"] - best_global["shift_e"])**2 + (cand["shift_n"] - best_global["shift_n"])**2) ** 0.5
                if shift_delta <= _CONSENSUS_R:
                    nearby.append(cand)
            if len(nearby) >= 2:
                cluster_consensus = True
                cluster_size = len(nearby)
                near_tied_count = len(nearby)
                cluster_dominance = 1.0
                best_global = _blend_candidate_cluster(nearby)
                print(
                    f"[i] Multi-tile consensus: blended {len(nearby)} nearby candidates  "
                    f"shift=({best_global['shift_e']:+.1f}, {best_global['shift_n']:+.1f}) m  "
                    f"spread={best_global.get('cluster_spread_m', 0.0):.0f} m"
                )
        best = best_global

    rotation_deg = 0.0
    vision_rotation_hint = None
    vision_north_up = None
    try:
        vision_rotation_hint = LAST_VISION_RESULT.get("north_arrow_direction_deg")
        vision_north_up = LAST_VISION_RESULT.get("map_is_north_up")
    except Exception:
        pass
    # Run rotation search when:
    #   (a) Vision did not confirm north-up, OR
    #   (b) The best NCC score is suspiciously weak even with a good seed — this is
    #       a reliable indicator that the template orientation doesn't match the
    #       reference, usually because the plan is rotated 90°/270° (landscape vs
    #       portrait paper orientation) or slightly tilted.
    # Always includes 90° and 270° as explicit test angles in addition to the
    # ±12° fine search, because landscape→portrait orientation flips are common
    # in German engineering plans printed on A0/A1 paper.
    _ncc_score_weak = best.get("score", 1.0) < 0.12
    _run_rotation_search = (
        ACTIVE_WMS_CONFIG_KEY in ("vector", "topo")
        and best.get("search") is not None
        and (vision_north_up is not True or _ncc_score_weak)
    )
    if _ncc_score_weak and vision_north_up is True:
        print(f"[i] NCC score {best.get('score', 0):.3f} is very weak despite north-up assertion -- "
              f"running rotation search to check for 90°/180°/270° orientation mismatch")
    if _run_rotation_search:
        search = best["search"]
        base_best_x, base_best_y = best["best_xy"]
        if isinstance(vision_rotation_hint, (int, float)) and abs(float(vision_rotation_hint)) >= 0.2:
            hint = float(vision_rotation_hint)
            coarse_vals = []
            for center in sorted({round(hint, 2), round(-hint, 2)}):
                _rot_limit = WMS_LARGE_ROTATION_DEG if abs(center) >= 30.0 else WMS_MAX_ROTATION_DEG
                lo = max(-_rot_limit, center - 3.0)
                hi = min(_rot_limit, center + 3.0)
                coarse_vals.extend(np.arange(lo, hi + 0.1, 0.5, dtype=np.float32).tolist())
            coarse_angles = np.array(sorted({round(v, 3) for v in coarse_vals}), dtype=np.float32)
            print(f"[i] Rotation search guided by north arrow hint: {hint:+.1f}°")
        else:
            # Standard ±12° fine search
            _fine_angles = np.arange(-WMS_MAX_ROTATION_DEG, WMS_MAX_ROTATION_DEG + 0.1, 1.0, dtype=np.float32)
            # Always add 90°, 180°, 270° coarse candidates — a plan printed in landscape
            # orientation but correctly georeferenced north-up will score well at 90° or
            # 270° if the road runs N-S (or E-W on the plan canvas but N-S in reality).
            _cardinal_angles = np.array([87.0, 90.0, 93.0, 177.0, 180.0, 183.0,
                                          267.0, 270.0, 273.0], dtype=np.float32)
            coarse_angles = np.array(
                sorted(set(round(float(a), 1) for a in np.concatenate([_fine_angles, _cardinal_angles]))),
                dtype=np.float32,
            )
        rot_best = None
        for angle in coarse_angles:
            rot_tpl, rot_offset = _rotate_edge_template(template_edge_raw, float(angle))
            if rot_tpl.size == 0:
                continue
            th_r, tw_r = rot_tpl.shape
            x_expected = int(round(base_best_x - rot_offset[0]))
            y_expected = int(round(base_best_y - rot_offset[1]))
            ncc = _ncc_score(
                search,
                rot_tpl,
                step=1,
                x_range=(x_expected - 16, x_expected + 16),
                y_range=(y_expected - 16, y_expected + 16),
            )
            if ncc is None:
                continue
            score_r, second_r, best_xy_r = ncc
            conf_r = (score_r + 1.0) / max(second_r + 1.0, 1e-6)
            score_gap_r = score_r - second_r
            candidate = {
                "angle": float(angle),
                "score": float(score_r),
                "confidence": float(conf_r),
                "gap": float(score_gap_r),
                "best_xy": best_xy_r,
                "rot_shape": (tw_r, th_r),
                "offset": rot_offset,
            }
            if rot_best is None or (candidate["score"], candidate["confidence"], candidate["gap"]) > (rot_best["score"], rot_best["confidence"], rot_best["gap"]):
                rot_best = candidate

        if rot_best is not None and abs(rot_best["angle"]) >= 0.2:
            fine_angles = np.arange(rot_best["angle"] - 0.8, rot_best["angle"] + 0.81, 0.2, dtype=np.float32)
            for angle in fine_angles:
                rot_tpl, rot_offset = _rotate_edge_template(template_edge_raw, float(angle))
                if rot_tpl.size == 0:
                    continue
                th_r, tw_r = rot_tpl.shape
                x_expected = int(round(rot_best["best_xy"][0] - (rot_offset[0] - rot_best["offset"][0])))
                y_expected = int(round(rot_best["best_xy"][1] - (rot_offset[1] - rot_best["offset"][1])))
                ncc = _ncc_score(
                    search,
                    rot_tpl,
                    step=1,
                    x_range=(x_expected - 10, x_expected + 10),
                    y_range=(y_expected - 10, y_expected + 10),
                )
                if ncc is None:
                    continue
                score_r, second_r, best_xy_r = ncc
                conf_r = (score_r + 1.0) / max(second_r + 1.0, 1e-6)
                score_gap_r = score_r - second_r
                candidate = {
                    "angle": float(angle),
                    "score": float(score_r),
                    "confidence": float(conf_r),
                    "gap": float(score_gap_r),
                    "best_xy": best_xy_r,
                    "rot_shape": (tw_r, th_r),
                    "offset": rot_offset,
                }
                if (candidate["score"], candidate["confidence"], candidate["gap"]) > (rot_best["score"], rot_best["confidence"], rot_best["gap"]):
                    rot_best = candidate

        if rot_best is not None and rot_best["score"] >= best["score"] + 0.01 and rot_best["confidence"] >= max(best["confidence"], 1.02):
            rot_offset_x, rot_offset_y = rot_best["offset"]
            _photo_x1_scaled = photo_x1 * tpl_px / tpl_px_coarse
            _photo_y1_scaled = photo_y1 * tpl_px / tpl_px_coarse
            expected_x = (search.shape[1] - tpl_px) / 2 + _photo_x1_scaled - rot_offset_x
            expected_y = (search.shape[0] - tpl_px) / 2 + _photo_y1_scaled - rot_offset_y
            sx = rot_best["best_xy"][0] - expected_x
            sy = rot_best["best_xy"][1] - expected_y
            tw_s, te_s, ts_s, tn_s = best["tile_bounds"]
            m_per_px_x = (te_s - tw_s) / search.shape[1]
            m_per_px_y = (tn_s - ts_s) / search.shape[0]
            tile_cx_ref = (tw_s + te_s) / 2.0
            tile_cy_ref = (ts_s + tn_s) / 2.0
            rotation_deg = rot_best["angle"]
            best["shift_e"] = (tile_cx_ref - cx) + sx * m_per_px_x
            best["shift_n"] = (tile_cy_ref - cy) - sy * m_per_px_y
            best["total_m"] = (best["shift_e"]**2 + best["shift_n"]**2) ** 0.5
            best["score"] = rot_best["score"]
            best["confidence"] = rot_best["confidence"]
            best["score_gap"] = rot_best["gap"]
            print(f"[i] Rotation refinement: angle={rotation_deg:+.2f}°  score={rot_best['score']:.3f}  gap={rot_best['gap']:.3f}  conf={rot_best['confidence']:.2f}x")
    elif not _run_rotation_search:
        print("[i] Rotation refinement skipped: Vision confirmed north-up and NCC score is adequate")

    shift_e = best["shift_e"]
    shift_n = best["shift_n"]
    total_m = best["total_m"]
    peak_confidence = best["confidence"]
    best_score = best["score"]
    score_gap = best.get("score_gap", 0.0)
    ref_img = best["img"]
    ref_img.save(str(_artifact_path("wms_ref.jpg")))
    ref_img.close()
    print(f"  Template match: ΔE={shift_e:+.1f} m  ΔN={shift_n:+.1f} m  "
          f"total={total_m:.1f} m  score={best_score:.3f}  gap={score_gap:.3f}  confidence={peak_confidence:.2f}x  tile={best['tile']}")

    _high_conf_ncc = best_score > 0.45 and score_gap > 0.30
    _effective_shift_limit = WMS_MAX_SHIFT_M
    if _high_conf_ncc and total_m > WMS_MAX_SHIFT_M:
        # Extraordinarily strong NCC evidence (score>0.45, gap>0.30): the match is
        # near-certain regardless of seed quality.  The geocoder may have returned a
        # city/road centroid far from the actual site, so the nominal per-confidence
        # limit is too conservative.  Expand to a hard 15 km ceiling (well within
        # Germany) rather than discarding a reliable match.
        _effective_shift_limit = 15_000.0
        print(
            f"[i] High-confidence NCC (score={best_score:.3f} gap={score_gap:.3f}) -- "
            f"shift limit relaxed to {_effective_shift_limit/1000:.0f} km"
        )
    if total_m > _effective_shift_limit:
        print(f"[!] Shift {total_m:.0f} m exceeds limit {_effective_shift_limit:.0f} m -- discarding")
        return gt

    min_confidence = WMS_MIN_CONFIDENCE
    min_score = WMS_MIN_SCORE
    min_score_gap = WMS_MIN_SCORE_GAP
    strict_no_anchor_floor = {
        "min_confidence": max(WMS_MIN_CONFIDENCE, 1.25),
        "min_score": max(WMS_MIN_SCORE, 0.38),
        "min_score_gap": max(WMS_MIN_SCORE_GAP, 0.06),
    }
    if not has_coord_anchors:
        # Without OCR coordinate anchors the seed location may be wrong.
        # Require a stronger NCC match so spurious cross-correlations at the
        # wrong geographic location are rejected rather than accepted.
        min_confidence = strict_no_anchor_floor["min_confidence"]
        min_score      = strict_no_anchor_floor["min_score"]
        min_score_gap  = strict_no_anchor_floor["min_score_gap"]
        print(
            f"[i] No OCR coordinate anchors -- WMS thresholds tightened: "
            f"score≥{min_score:.2f}  gap≥{min_score_gap:.3f}  conf≥{min_confidence:.2f}x"
        )
    cluster_spread_m = float(best.get("cluster_spread_m", 999999.0) or 999999.0)
    if cluster_consensus and (
        (cluster_size >= 3 and near_tied_count >= 3) or
        (cluster_size >= 2 and near_tied_count >= 3 and cluster_dominance >= 0.50)
    ):
        min_confidence = min(min_confidence, 0.98)   # slight headroom for NCC rounding
        min_score_gap = min(min_score_gap, 0.001)
        if cluster_spread_m <= 350.0:
            min_score = min(min_score, 0.080)
        elif cluster_spread_m <= 500.0:
            min_score = min(min_score, 0.110)
    # Snapshot thresholds after cluster relaxation, before any re-application of
    # the no-anchor floor.  Used by the tight-consensus bypass below.
    _cr_conf  = min_confidence
    _cr_score = min_score
    _cr_gap   = min_score_gap
    # Re-apply the no-anchor floor AFTER cluster relaxation, but allow a tight
    # consensus to override it:
    #   • ≥3 tiles with spread ≤20 m, OR
    #   • ≥5 tiles with spread ≤30 m (large consensus tolerates slightly wider
    #     spread because each individual tile contributes less noise)
    #   • ≥2 tiles (best sub-cluster) with spread ≤20 m and dominance ≥0.40
    # Sub-30-metre spread across independent tiles is strong geographic evidence
    # even when absolute NCC scores are low (CAD plans vs topo/ortho reference).
    if not has_coord_anchors:
        # Spread limit scales with cluster size: more tiles → lower per-tile noise
        # → tolerate slightly wider spread before rejecting the consensus.
        _tight_spread_limit = 30.0 if cluster_size >= 5 else 20.0
        _tight_consensus = cluster_consensus and cluster_spread_m <= _tight_spread_limit and (
            cluster_size >= 3
            or (cluster_size >= 2 and cluster_dominance >= 0.40)
        )
        # If patch-consensus explicitly rejected all candidates due to geometric
        # validation (rotation/anisotropy/shear), the NCC location is geometrically
        # inconsistent with the plan's known orientation.  Tight NCC consensus is
        # merely N tiles agreeing on the same wrong location -- do not override.
        if _tight_consensus and _pc_geo_rejected:
            print(
                "[!] Tight-consensus bypass suppressed: patch-consensus detected geometric "
                "inconsistency (rotation/anisotropy) at this NCC location -- NCC result untrusted"
            )
            _tight_consensus = False
        _allow_consensus_bypass = (
            _patch_refinement_succeeded or _feature_refinement_succeeded
        )
        _street_seed_bypass = (
            not STRICT_WMS_VALIDATION
            and seed_confidence == "street"
            and cluster_size >= 5
            and cluster_spread_m <= 50.0
            and not _pc_geo_rejected
            and total_m <= WMS_MAX_SHIFT_M
        )
        if _tight_consensus and not _allow_consensus_bypass and not _street_seed_bypass:
            print(
                "[!] Tight-consensus bypass suppressed: no hard anchors and no successful geometric refinement; "
                "multi-tile NCC agreement alone is insufficient"
            )
            _tight_consensus = False
        elif _tight_consensus and _street_seed_bypass and not _allow_consensus_bypass:
            print(
                f"[i] Non-strict street-seed bypass: cluster={cluster_size} tiles, "
                f"spread={cluster_spread_m:.1f} m, shift={total_m:.0f} m"
            )
        if not _tight_consensus:
            min_confidence = strict_no_anchor_floor["min_confidence"]
            min_score      = strict_no_anchor_floor["min_score"]
            min_score_gap  = strict_no_anchor_floor["min_score_gap"]
        else:
            # Tight geographic consensus overrides the no-anchor floor.
            # Use min(cluster_relaxed, base) so we preserve any cluster relaxation
            # that already ran, while still undoing the no-anchor tightening.
            min_confidence = min(_cr_conf,  WMS_MIN_CONFIDENCE)
            min_score      = min(_cr_score, WMS_MIN_SCORE)
            # Gap threshold: tight geographic consensus (≥5 tiles / ≤30 m spread)
            # is the real quality signal -- NCC score gap is a secondary metric.
            # Set to 0 to avoid floating-point edge cases where gap=0.001 is stored
            # as 0.0009999... and incorrectly fails the `< 0.001` check.
            min_score_gap  = 0.0
            print(
                f"[i] Tight multi-tile consensus ({cluster_size} tiles, spread={cluster_spread_m:.1f} m) "
                f"-- no-anchor floor bypassed  "
                f"(conf≥{min_confidence:.2f}  score≥{min_score:.3f}  gap≥{min_score_gap:.3f})"
            )
    # Very strong consensus (≥5 tiles, ≤5 m spread, no geometric rejection):
    # multiple independent tile comparisons all agree on the same shift to within
    # a few metres.  This is near-certain even when NCC scores are low (CAD/schematic
    # plans vs aerial orthophoto produce inherently low absolute scores).
    _very_strong_consensus = (
        cluster_consensus
        and cluster_size >= 5
        and cluster_spread_m <= 5.0
        and not _pc_geo_rejected
    )
    if _very_strong_consensus and not has_coord_anchors and not (
        _patch_refinement_succeeded or _feature_refinement_succeeded
    ):
        # An ultra-tight spatial agreement across ≥5 independent tiles is its own
        # geometric validation: random NCC peaks cannot cluster to within a few metres
        # across non-overlapping tiles by chance.  Two tiers:
        #   • spread ≤ 3 m: accept when score ≥ 0.30 (strong signal)
        #   • spread = 0 m: accept when score ≥ 0.10 (even a barely-above-noise
        #     correlation is meaningful when 5 tiles perfectly agree — this handles
        #     1:250 CAD plans matched against aerial DOP where absolute NCC is low)
        _min_score_for_spread = 0.10 if cluster_spread_m == 0.0 else 0.30
        if cluster_spread_m <= 3.0 and best_score >= _min_score_for_spread:
            print(
                f"[i] Very-strong-consensus: ultra-tight spread ({cluster_spread_m:.1f} m) "
                f"across {cluster_size} tiles accepted without OCR anchors "
                f"(score={best_score:.3f}  gap={score_gap:.3f}  min_score≥{_min_score_for_spread:.2f})"
            )
        else:
            print(
                "[!] Very-strong-consensus shortcut suppressed: no OCR anchors and no successful geometric refinement"
            )
            _very_strong_consensus = False
    if _very_strong_consensus:
        print(
            f"[i] Very strong consensus ({cluster_size} tiles, spread={cluster_spread_m:.1f} m) "
            f"-- accepting NCC result (conf={peak_confidence:.2f}x  score={best_score:.3f})"
        )
    elif peak_confidence < min_confidence or best_score < min_score or score_gap < min_score_gap:
        _why = []
        if peak_confidence < min_confidence: _why.append(f"conf={peak_confidence:.2f}x < {min_confidence:.2f}x")
        if best_score      < min_score:      _why.append(f"score={best_score:.3f} < {min_score:.3f}")
        if score_gap       < min_score_gap:  _why.append(f"gap={score_gap:.3f} < {min_score_gap:.3f}")
        _cluster_info = (
            f"  cluster={cluster_size}/{near_tied_count} spread={cluster_spread_m:.0f}m"
            if cluster_consensus else ""
        )
        print(f"[!] Template match too weak ({'; '.join(_why)}){_cluster_info} -- discarding")
        return gt

    # OCR place-name centroid validation: when OCR place names (e.g. street names
    # found in the plan's PDF text) were geocoded, the plan centre must lie within
    # ~1.5 × plan_width of that centroid.  This catches false-positive NCC matches
    # where the same highway/canal shape occurs at multiple locations (e.g. A2 along
    # its whole length) and the tight-consensus criterion accepted the wrong one.
    # Only applies when no hard coordinate anchors exist (those make NCC reliable).
    if LAST_OCR_PLACES_CENTROID is not None and not has_coord_anchors:
        _opc_e, _opc_n = LAST_OCR_PLACES_CENTROID
        _plan_w_m = base_mpp * W   # physical plan width in metres
        # City-level centroid (Standort geocode) is accurate to ±5 km — use a
        # generous limit so the plan can be placed anywhere within the municipality.
        # Feature-level centroid (multiple geocoded place labels) is ±1–2 km — use
        # the tight 1.5 × plan_width limit to catch false-positive NCC matches.
        if LAST_OCR_PLACES_CENTROID_PRECISION == "city":
            _ocr_val_limit = 5_000.0
        else:
            _ocr_val_limit = _plan_w_m * 1.5
        _impl_e = (gt[0] + shift_e) + gt[1] * (W / 2.0) + gt[2] * (H / 2.0)
        _impl_n = (gt[3] + shift_n) + gt[4] * (W / 2.0) + gt[5] * (H / 2.0)
        _ocr_dist_m = ((_impl_e - _opc_e) ** 2 + (_impl_n - _opc_n) ** 2) ** 0.5
        if _ocr_dist_m > _ocr_val_limit:
            print(
                f"[!] WMS result rejected by OCR centroid check: "
                f"implied plan centre is {_ocr_dist_m:.0f} m from OCR place centroid "
                f"(limit {_ocr_val_limit:.0f} m, precision={LAST_OCR_PLACES_CENTROID_PRECISION}) -- discarding"
            )
            return gt

    if abs(rotation_deg) >= 0.2:
        theta = math.radians(rotation_deg)
        # GDAL geotransform for CW rotation θ (image "up" tilted θ CW from north):
        #   GT[1] = +mpp*cos(θ)  (ΔE per col right)
        #   GT[2] = -mpp*sin(θ)  (ΔE per row down)   ← must be negative for CW
        #   GT[4] = -mpp*sin(θ)  (ΔN per col right)  ← must be negative for CW
        #   GT[5] = -mpp*cos(θ)  (ΔN per row down)
        rot_gt1 =  base_mpp * math.cos(theta)
        rot_gt2 = -base_mpp * math.sin(theta)
        rot_gt4 = -base_mpp * math.sin(theta)
        rot_gt5 = -base_mpp * math.cos(theta)
        center_e = (gt[0] + shift_e) + gt[1] * (W / 2.0) + gt[2] * (H / 2.0)
        center_n = (gt[3] + shift_n) + gt[4] * (W / 2.0) + gt[5] * (H / 2.0)
        rot_gt0 = center_e - rot_gt1 * (W / 2.0) - rot_gt2 * (H / 2.0)
        rot_gt3 = center_n - rot_gt4 * (W / 2.0) - rot_gt5 * (H / 2.0)
        gt_refined = (rot_gt0, rot_gt1, rot_gt2, rot_gt3, rot_gt4, rot_gt5)
        print(f"[✓] WMS refinement applied: ΔE={shift_e:+.1f} m  ΔN={shift_n:+.1f} m  rot={rotation_deg:+.2f}°")
    else:
        gt_refined = (gt[0] + shift_e, gt[1], gt[2], gt[3] + shift_n, gt[4], gt[5])
        print(f"[✓] WMS refinement applied: ΔE={shift_e:+.1f} m  ΔN={shift_n:+.1f} m")
    LAST_GEOREF_QUALITY.update({
        "accepted": True,
        "acceptance_reason": "ncc_thresholds",
        "patch_consensus_succeeded": bool(_patch_refinement_succeeded),
        "feature_refinement_succeeded": bool(_feature_refinement_succeeded),
        "ncc_score": round(float(best_score), 4),
        "ncc_score_gap": round(float(score_gap), 4),
        "ncc_confidence": round(float(peak_confidence), 4),
        "cluster_consensus": bool(cluster_consensus),
        "cluster_size": int(cluster_size),
        "cluster_spread_m": round(float(cluster_spread_m), 3),
        "rotation_deg": round(float(rotation_deg), 3),
        "shift_total_m": round(float(total_m), 3),
    })
    return _enforce_isotropic_geotransform(
        gt_refined, W, H, base_mpp, preserve_rotation=abs(rotation_deg) >= 15.0
    )


def _enforce_isotropic_geotransform(gt: tuple, W: int, H: int, base_mpp: float, preserve_rotation: bool = False) -> tuple:
    """
    For north-up plans, enforce square pixels AND zero rotation.
    If the computed X and Y pixel sizes differ by more than 3 %, snap the
    worse axis to the better one (whichever is closest to base_mpp).
    Rotation terms (gt[2], gt[4]) are always stripped -- north-up output
    matches QGIS Georeferencer "Linear" mode.
    The map centre is held fixed so downstream shift metrics stay valid.
    """
    mpp_x = abs(gt[1])
    mpp_y = abs(gt[5])
    rot_x = gt[2]
    rot_y = gt[4]
    has_rotation = abs(rot_x) > 1e-6 or abs(rot_y) > 1e-6
    if max(mpp_x, mpp_y) < 1e-9:
        return gt
    aniso = abs(mpp_x - mpp_y) / max(mpp_x, mpp_y)
    if aniso <= 0.03 and (preserve_rotation or not has_rotation):
        return gt
    mpp_iso = mpp_x if abs(base_mpp - mpp_x) <= abs(base_mpp - mpp_y) else mpp_y
    cx_e = gt[0] + gt[1] * (W / 2.0) + rot_x * (H / 2.0)
    cx_n = gt[3] + rot_y * (W / 2.0) + gt[5] * (H / 2.0)
    if preserve_rotation and has_rotation:
        theta = math.atan2(-rot_x, gt[1] if abs(gt[1]) > 1e-12 else 1e-12)
        new_gt1 = mpp_iso * math.cos(theta)
        new_gt2 = -mpp_iso * math.sin(theta)
        new_gt4 = -mpp_iso * math.sin(theta)
        new_gt5 = -mpp_iso * math.cos(theta)
        new_gt = (
            cx_e - new_gt1 * (W / 2.0) - new_gt2 * (H / 2.0),
            new_gt1, new_gt2,
            cx_n - new_gt4 * (W / 2.0) - new_gt5 * (H / 2.0),
            new_gt4, new_gt5,
        )
        _rot_msgs = []
        if aniso > 0.03:
            _rot_msgs.append(f"X={mpp_x:.4f} Y={mpp_y:.4f} -> {mpp_iso:.4f} m/px (aniso {aniso*100:.1f}%)")
        _rot_msgs.append("rotation preserved")
        print(f"[i] Isotropic/rotated correction: {';  '.join(_rot_msgs)}")
        return new_gt
    new_gt = (
        cx_e - mpp_iso * (W / 2.0),
        mpp_iso, 0.0,
        cx_n + mpp_iso * (H / 2.0),
        0.0, -mpp_iso,
    )
    msgs = []
    if aniso > 0.03:
        msgs.append(f"X={mpp_x:.4f} Y={mpp_y:.4f} → {mpp_iso:.4f} m/px (aniso {aniso*100:.1f}%)")
    if has_rotation:
        msgs.append(f"rotation stripped (rot_x={rot_x:.4f} rot_y={rot_y:.4f})")
    print(f"[i] Isotropic/north-up correction: {';  '.join(msgs)}")
    return new_gt


def georeference(src_path: Path, vision: dict, epsg: int,
                 auto_seed: dict = None) -> Path:
    """
    Apply the computed affine geotransform directly to the TIFF.
    No warping -- every pixel stays in its original position.

    Seed priority (highest → lowest):
      1. manual_seed.json with enabled=true  (explicit user override — user-written only)
      2. auto_seed from derive_auto_seed():
           a. last_result.json if within WMS_MAX_SHIFT_M of canvas centre
           b. QGIS canvas centre (live user viewport)
           c. OCR explicit Rechtswert/Hochwert coordinates
           d. City/location name from OCR text or filename
      3. compute_geotransform from OCR-detected coordinate grid labels
    """
    if not HAS_GDAL:
        print("[!] GDAL not available – skipping georeferencing")
        return None
    global LAST_GEOREF_QUALITY, CURRENT_SEED_SOURCE, CURRENT_SEED_CONFIDENCE
    LAST_GEOREF_QUALITY = {
        "stage": "georeference",
        "accepted": False,
        "acceptance_reason": "not_started",
    }

    manual_seed = load_manual_seed()
    # manual > auto_seed; empty dict from derive_auto_seed() is falsy → None path
    effective_seed = manual_seed or auto_seed
    CURRENT_SEED_SOURCE = str((effective_seed or {}).get("_source") or "")
    CURRENT_SEED_CONFIDENCE = str((effective_seed or {}).get("_seed_confidence") or "")
    working_epsg = int(epsg)
    _pre_wms_quality = None

    gt = None if effective_seed else compute_geotransform(vision)
    if gt is None and not effective_seed:
        print("[!] No coordinate grid labels and no location seed -- cannot georeference")
        return None
    if gt is not None and not effective_seed:
        LAST_GEOREF_QUALITY.update({
            "accepted": True,
            "acceptance_reason": "ocr_grid_regression",
            "seed_source": None,
            "has_coord_anchors": True,
        })

    crop_bbox = detect_main_map_bbox(src_path)
    src_for_output = str(src_path)
    crop_width = vision.get("width")
    crop_height = vision.get("height")
    if crop_bbox:
        x1, y1, x2, y2 = crop_bbox
        crop_width = x2 - x1
        crop_height = y2 - y1
        if gt is not None:
            gt = (
                gt[0] + gt[1] * x1,
                gt[1],
                gt[2],
                gt[3] + gt[5] * y1,
                gt[4],
                gt[5],
            )
        crop_vrt = _artifact_path("cropped_map.vrt")
        gdal.Translate(
            str(crop_vrt),
            str(src_path),
            format="VRT",
            srcWin=[x1, y1, x2 - x1, y2 - y1],
        )
        src_for_output = str(crop_vrt)
        print("[i] Using cropped map block for refinement/output")

    if effective_seed:
        _seed_source = effective_seed.get("_source", "manual_seed.json")
        _seed_label = "Manual override" if manual_seed else f"Auto seed ({_seed_source})"
        _se = effective_seed.get("center_easting") or effective_seed.get("origin_easting")
        _sn = effective_seed.get("center_northing") or effective_seed.get("origin_northing")
        if _se is not None and _sn is not None:
            print(f"[i] {_seed_label}: E={float(_se):.0f}  N={float(_sn):.0f}")
        else:
            print(f"[i] {_seed_label}")
        apply_after_crop = effective_seed.get("apply_after_crop", True)
        target_w = crop_width if apply_after_crop and crop_width else vision.get("width")
        target_h = crop_height if apply_after_crop and crop_height else vision.get("height")
        if "_rotation_deg" not in effective_seed and not manual_seed:
            _seed_rot = _derive_seed_rotation_deg(src_path, crop_bbox, effective_seed, vision)
            if isinstance(_seed_rot, (int, float)) and abs(float(_seed_rot)) >= 0.2:
                effective_seed = dict(effective_seed)
                effective_seed["_rotation_deg"] = float(_seed_rot)
        gt = build_manual_seed_geotransform(effective_seed, int(target_w), int(target_h), vision)
        if gt is None:
            return None
        LAST_GEOREF_QUALITY.update({
            "accepted": True,
            "acceptance_reason": "seed_geotransform",
            "seed_source": _seed_source,
            "seed_confidence": effective_seed.get("_seed_confidence"),
            "rotation_deg": effective_seed.get("_rotation_deg"),
            "segment_ambiguous": bool(effective_seed.get("_segment_ambiguous")),
            "segment_candidate_count": int(effective_seed.get("_segment_candidate_count") or 0),
            "has_coord_anchors": False,
        })
        _pre_wms_quality = dict(LAST_GEOREF_QUALITY)

    # Optional: refine with WMS phase correlation
    _has_anchors = (
        len(vision.get("easting_positions", [])) >= 2
        or len(vision.get("northing_positions", [])) >= 2
    )
    if effective_seed and not _has_anchors and working_epsg != 25832:
        print(
            f"[i] No hard coordinates available -- using EPSG:25832 as internal matching CRS "
            f"and converting final geotransform to EPSG:{working_epsg}"
        )
        working_epsg = 25832
    # Widen coarse grid when the seed has city-level accuracy (vision_location or
    # qgis_canvas source without last_result refinement).  A city-centre geocode
    # can be ±5-10 km from the project, so we expand from 5×5 to 7×7 tiles.
    # "project_address" / "last_result" / "ocr_coordinates" are kept at default
    # range (higher spatial precision).
    _seed_src = (effective_seed or {}).get("_source", "")
    _seed_conf = (effective_seed or {}).get("_seed_confidence", "")
    _low_conf_sources = {
        "vision_location", "qgis_canvas", "city_name",
        "ocr_place_names",
    }
    _seed_is_low_conf = (_seed_src in _low_conf_sources or _seed_conf == "city")
    if _seed_src == "project_address+ocr_places" and _seed_conf == "feature":
        _seed_is_low_conf = False
    # Street-level seeds (street+city pattern, direct address) are precise to
    # ~100-500 m -- tighten the WMS max-shift window so the matcher cannot
    # wander to a visually similar structure several km away.
    # Confidence tier → max allowed WMS shift from seed:
    #   street   : ±500 m  (geocoded street+city — sub-500 m precision)
    #   feature  : ±2 km   (multiple geocoded place names)
    #   city     : ±10 km  (city-centre geocode — keep current default)
    #   postcode : ±12 km  (postcode centroid — wider than city)
    _orig_max_shift = globals().get("WMS_MAX_SHIFT_M", WMS_MAX_SHIFT_M)
    _shift_by_conf = {
        "street":   500.0,
        "feature":  2_000.0,
        "city":     10_000.0,
        "postcode": 12_000.0,
    }
    _effective_max_shift = _shift_by_conf.get(_seed_conf, _orig_max_shift)
    # last_result and OCR-coordinate seeds are inherently high-precision
    if _seed_src in ("last_result", "ocr_coordinates", "manual_seed"):
        _effective_max_shift = min(_effective_max_shift, 2_000.0)
    _shift_changed = abs(_effective_max_shift - _orig_max_shift) > 1.0
    if _shift_changed:
        globals()["WMS_MAX_SHIFT_M"] = _effective_max_shift
        print(f"[i] WMS max shift set to {_effective_max_shift:.0f} m  "
              f"(seed confidence={_seed_conf or _seed_src})")
    _orig_coarse_range = globals().get("WMS_COARSE_RANGE", 2)
    _coarse_boosted = False
    if _seed_is_low_conf:
        globals()["WMS_COARSE_RANGE"] = max(_orig_coarse_range, 3)
        _coarse_boosted = True
        print(f"[i] Low-confidence seed ({_seed_src or _seed_conf}) -- widening coarse search to "
              f"{globals()['WMS_COARSE_RANGE']*2+1}×{globals()['WMS_COARSE_RANGE']*2+1} tiles")
    try:
        gt = refine_geotransform_wms_v2(
            Path(src_for_output), gt, working_epsg,
            has_coord_anchors=_has_anchors,
            seed_confidence=_seed_conf,
        )
    finally:
        if _coarse_boosted:
            globals()["WMS_COARSE_RANGE"] = _orig_coarse_range
        if _shift_changed:
            globals()["WMS_MAX_SHIFT_M"] = _orig_max_shift
    if _pre_wms_quality is not None and LAST_GEOREF_QUALITY.get("accepted") is not True:
        LAST_GEOREF_QUALITY = {
            **_pre_wms_quality,
            "wms_quality": dict(LAST_GEOREF_QUALITY or {}),
        }
        if not _has_anchors:
            LAST_GEOREF_QUALITY["provisional"] = True
            LAST_GEOREF_QUALITY["acceptance_reason"] = "seed_geotransform_provisional"
            print("[!] WMS validation did not confirm the seed placement -- output is provisional")
    # Final safety: enforce isotropic pixel size regardless of refinement path
    _out_w = crop_width or vision.get("width") or 1
    _out_h = crop_height or vision.get("height") or 1
    _base_mpp_final = (abs(gt[1]) + abs(gt[5])) / 2.0
    gt = _enforce_isotropic_geotransform(
        gt,
        int(_out_w),
        int(_out_h),
        _base_mpp_final,
        preserve_rotation=(abs(gt[2]) > 1e-6 or abs(gt[4]) > 1e-6),
    )
    if working_epsg != epsg:
        gt = _reproject_geotransform(gt, int(_out_w), int(_out_h), working_epsg, epsg)
        print(f"[i] Final geotransform converted from EPSG:{working_epsg} to EPSG:{epsg}")

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    wkt = srs.ExportToWkt()

    out_path = OUTPUT_DIR / (src_path.stem + "_georef.tif")

    # Suppress GDAL's automatic .aux.xml creation for a clean output folder.
    gdal.SetConfigOption("GDAL_PAM_ENABLED", "NO")

    # Copy to output and set geotransform + projection
    print(f"[~] Writing georeferenced GeoTIFF …  (EPSG:{epsg})")
    ds_src = gdal.Open(src_for_output)
    driver = gdal.GetDriverByName("GTiff")
    ds_out = driver.CreateCopy(
        str(out_path), ds_src,
        options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"]
    )
    ds_out.SetGeoTransform(gt)
    ds_out.SetProjection(wkt)
    ds_out.FlushCache()
    ds_out = None
    ds_src = None

    # Re-enable PAM so other QGIS operations aren't affected.
    gdal.SetConfigOption("GDAL_PAM_ENABLED", "YES")

    # Remove any .aux.xml that GDAL may have written before we disabled PAM.
    _aux = out_path.with_suffix(".tif.aux.xml")
    if _aux.exists():
        try:
            _aux.unlink()
        except Exception:
            pass

    # Compute and report the effective map scale (assumes 96 DPI screen, informational only)
    # For rotated plans GT[1]=mpp·cos(θ) and GT[2]=-mpp·sin(θ), so the actual pixel
    # size in metres is √(GT[1]²+GT[2]²), not GT[1] alone.
    px_size_m = (math.sqrt(gt[1] ** 2 + gt[2] ** 2) + math.sqrt(gt[4] ** 2 + gt[5] ** 2)) / 2.0
    ds_info = gdal.Open(src_for_output)
    img_w_px = ds_info.RasterXSize
    ds_info = None
    dpi_candidates = []
    dpi_x = vision.get("dpi_x")
    dpi_y = vision.get("dpi_y")
    if dpi_x and dpi_y and dpi_x > 0 and dpi_y > 0:
        dpi_candidates.append((float(dpi_x) + float(dpi_y)) / 2.0)
    dpi_candidates.extend([300, 400, 600, 800, 1000, 1200])
    seen = set()
    for dpi_hint in dpi_candidates:
        if dpi_hint in seen:
            continue
        seen.add(dpi_hint)
        paper_mm = img_w_px / (dpi_hint / 25.4)
        inferred_scale = int(round(px_size_m * dpi_hint / 0.0254, -2))
        if 500 <= inferred_scale <= 100_000:
            print(f"    Maßstab  : ~1:{inferred_scale:,}  "
                  f"(at {dpi_hint:.0f} DPI scan, paper width ≈{paper_mm:.0f} mm)")
            break

    print(f"[✓] Georeferenced TIFF → {out_path}")
    print(f"    Origin   : ({gt[0]:.2f}, {gt[3]:.2f})")
    _disp_mpp_x = math.sqrt(gt[1] ** 2 + gt[2] ** 2)
    _disp_mpp_y = math.sqrt(gt[4] ** 2 + gt[5] ** 2)
    _rot_deg = math.degrees(math.atan2(-gt[2], gt[1])) if (gt[2] or gt[1]) else 0.0
    _rot_suffix = f"   rot={_rot_deg:+.2f}°" if abs(_rot_deg) > 0.05 else ""
    print(f"    Pixel sz : {_disp_mpp_x:.4f} m/px{_rot_suffix}")
    try:
        _result_meta_path = OUTPUT_DIR / f"{src_path.stem}_georef_meta.json"
        _result_meta = {
            "output_path": str(out_path),
            "epsg": int(epsg),
            "seed_source": (effective_seed or {}).get("_source"),
            "seed_confidence": (effective_seed or {}).get("_seed_confidence"),
            "has_coord_anchors": bool(_has_anchors),
            "quality": dict(LAST_GEOREF_QUALITY or {}),
        }
        _result_meta_path.write_text(
            json.dumps(_result_meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[i] Result metadata saved → {_result_meta_path}")
    except Exception as _meta_exc:
        print(f"[~] Could not save result metadata: {_meta_exc}")

    # Save result centre to last_result.json so the next run in the same
    # area starts from a precise WMS-refined position rather than the raw
    # canvas centre.  manual_seed.json is left untouched (user-intent only).
    try:
        _w_out = int(_out_w)
        _h_out = int(_out_h)
        _cx = gt[0] + gt[1] * (_w_out / 2.0) + gt[2] * (_h_out / 2.0)
        _cy = gt[3] + gt[4] * (_w_out / 2.0) + gt[5] * (_h_out / 2.0)
        _scale_out = int(round(
            (SCALE_OVERRIDE or vision.get("scale") or 5000)
        ))
        _quality = dict(LAST_GEOREF_QUALITY or {})
        _persist_ok = (
            _quality.get("acceptance_reason") in {
                "ocr_grid_regression",
                "patch_consensus",
                "feature_refinement",
            }
            or (
                _quality.get("acceptance_reason") == "ncc_thresholds"
                and _quality.get("has_coord_anchors") is True
            )
        )
        if _persist_ok:
            _last_result = {
                "_source": "auto_result",
                "_note": "Auto-written after trusted georeferencing. Used as precision seed when canvas is nearby.",
                "center_easting": round(_cx, 1),
                "center_northing": round(_cy, 1),
                "scale_denominator": _scale_out,
                "quality": _quality,
            }
            LAST_RESULT_FILE.write_text(
                json.dumps(_last_result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"[i] Last result written → {LAST_RESULT_FILE}")
            print(f"    Centre: E={_cx:.0f}  N={_cy:.0f}")
        else:
            print(
                f"[~] Skipping last_result.json write: result is not trusted enough "
                f"(reason={_quality.get('acceptance_reason', 'unknown')})"
            )
    except Exception as _seed_exc:
        print(f"[~] Could not write last result: {_seed_exc}")

    return out_path


# ---------------------------------------------------------------------------
# STEP 7 – Load result in QGIS (if running inside QGIS)
# ---------------------------------------------------------------------------
def load_in_qgis(path: Path):
    path_str = str(path).replace("\\", "/")
    try:
        from qgis.core import QgsRasterLayer, QgsProject
        layer = QgsRasterLayer(path_str, path.stem)
        if not layer.isValid():
            print(f"[✗] Layer invalid – check the output file: {path}")
            return
        QgsProject.instance().addMapLayer(layer)
        # Zoom to the new layer
        try:
            from qgis.utils import iface
            iface.mapCanvas().setExtent(layer.extent())
            iface.mapCanvas().refresh()
        except Exception:
            pass
        print(f"[✓] Layer added to QGIS project: {path.stem}")
    except ImportError:
        # Not inside QGIS – print a one-liner the user can paste into QGIS console
        print("[i] Running outside QGIS. To load the layer, paste this into the QGIS Python Console:")
        print(f'    iface.addRasterLayer(r"{path_str}", "{path.stem}")')


def get_raster_georef_info(path: Path) -> dict:
    if not HAS_GDAL:
        raise RuntimeError("GDAL is required to inspect raster georeferencing")
    ds = gdal.Open(str(path))
    if ds is None:
        raise FileNotFoundError(path)
    gt = ds.GetGeoTransform()
    width = ds.RasterXSize
    height = ds.RasterYSize
    center_e = gt[0] + gt[1] * (width / 2.0) + gt[2] * (height / 2.0)
    center_n = gt[3] + gt[4] * (width / 2.0) + gt[5] * (height / 2.0)
    info = {
        "path": str(path),
        "width": width,
        "height": height,
        "geotransform": list(gt),
        "projection": ds.GetProjection(),
        "center_easting": center_e,
        "center_northing": center_n,
    }
    ds = None
    return info


def save_reviewed_geotiff(
    preview_path: Path,
    *,
    corrected_center_easting: float,
    corrected_center_northing: float,
    output_path: Path | None = None,
) -> dict:
    if not HAS_GDAL:
        raise RuntimeError("GDAL is required to write a reviewed GeoTIFF")
    preview_info = get_raster_georef_info(preview_path)
    gt = tuple(preview_info["geotransform"])
    delta_e = float(corrected_center_easting) - float(preview_info["center_easting"])
    delta_n = float(corrected_center_northing) - float(preview_info["center_northing"])
    reviewed_gt = (
        gt[0] + delta_e,
        gt[1],
        gt[2],
        gt[3] + delta_n,
        gt[4],
        gt[5],
    )
    target = output_path or preview_path.with_name(preview_path.stem + "_reviewed.tif")
    gdal.SetConfigOption("GDAL_PAM_ENABLED", "NO")
    ds_src = gdal.Open(str(preview_path))
    driver = gdal.GetDriverByName("GTiff")
    ds_out = driver.CreateCopy(
        str(target), ds_src,
        options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"]
    )
    ds_out.SetGeoTransform(reviewed_gt)
    ds_out.SetProjection(ds_src.GetProjection())
    ds_out.FlushCache()
    ds_out = None
    ds_src = None
    gdal.SetConfigOption("GDAL_PAM_ENABLED", "YES")
    _aux = target.with_suffix(".tif.aux.xml")
    if _aux.exists():
        try:
            _aux.unlink()
        except Exception:
            pass
    reviewed_info = get_raster_georef_info(target)
    meta_path = target.with_name(f"{target.stem}_meta.json")
    meta_payload = {
        "preview_path": str(preview_path),
        "reviewed_path": str(target),
        "original_geotransform": preview_info["geotransform"],
        "corrected_geotransform": reviewed_info["geotransform"],
        "original_center_easting": preview_info["center_easting"],
        "original_center_northing": preview_info["center_northing"],
        "corrected_center_easting": reviewed_info["center_easting"],
        "corrected_center_northing": reviewed_info["center_northing"],
        "delta_easting": delta_e,
        "delta_northing": delta_n,
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[i] Reviewed GeoTIFF saved → {target}")
    print(f"    Shift: ΔE={delta_e:.2f}  ΔN={delta_n:.2f}")
    return {
        **meta_payload,
        "meta_path": str(meta_path),
    }


def persist_case_bundle_for_run(
    *,
    input_path: Path,
    work_path: Path,
    is_pdf: bool,
    meta: dict,
    ocr_text: str,
    parsed: dict,
    vision: dict,
    auto_seed: dict | None,
    runtime_candidates: list[dict] | None,
    selected_candidate_id: str | None,
    output_path: Path | None,
) -> Path | None:
    try:
        from georef_core.models import (
            CandidateEvidence,
            GeoCandidate,
            GeorefCaseBundle,
            GeorefResult,
            IngestResult,
            OCRResult,
            PipelineArtifacts,
            PipelineContext,
            StructuredHints,
            ValidationResult,
            VisionResult,
        )
        from georef_core.persistence import save_case_bundle
        from georef_core.review import review_status_from_validation
    except Exception as exc:
        print(f"[~] Case bundle persistence unavailable: {exc}")
        return None

    try:
        hints_raw = _extract_structured_location_hints(
            ocr_text or "",
            vision or {},
            vision.get("title_block") if isinstance(vision.get("title_block"), dict) else {},
        )
        structured = StructuredHints.from_dict(hints_raw)
        candidate_list = []
        if runtime_candidates:
            for raw in runtime_candidates:
                candidate_list.append(
                    GeoCandidate(
                        candidate_id=str(raw.get("candidate_id") or "candidate"),
                        source=str(raw.get("source") or "unknown"),
                        confidence_tier=str(raw.get("confidence_tier") or "fallback"),
                        center_easting=float(raw["center_easting"]) if raw.get("center_easting") is not None else None,
                        center_northing=float(raw["center_northing"]) if raw.get("center_northing") is not None else None,
                        search_radius_m=float(raw.get("search_radius_m") or WMS_MAX_SHIFT_M),
                        label=str(raw.get("label") or raw.get("source") or "Candidate"),
                        rank_score=float(raw.get("rank_score") or 0.0),
                        evidence=[
                            CandidateEvidence(
                                source=str(ev.get("source") or "candidate"),
                                text=ev.get("text"),
                                weight=float(ev.get("weight") or 0.0),
                                details=dict(ev.get("details") or {}),
                            )
                            for ev in (raw.get("evidence") or [])
                        ],
                        metadata=dict(raw.get("metadata") or {}),
                        conflicts=list(raw.get("conflicts") or []),
                    )
                )
        elif auto_seed and auto_seed.get("center_easting") is not None and auto_seed.get("center_northing") is not None:
            candidate_list.append(
                GeoCandidate(
                    candidate_id="legacy-run-auto-seed",
                    source=str(auto_seed.get("_source") or "auto_seed"),
                    confidence_tier=str(auto_seed.get("_seed_confidence") or "fallback"),
                    center_easting=float(auto_seed["center_easting"]),
                    center_northing=float(auto_seed["center_northing"]),
                    search_radius_m=float(WMS_MAX_SHIFT_M),
                    label=str(auto_seed.get("_address") or auto_seed.get("_source") or "Auto seed"),
                    rank_score=0.5,
                    evidence=[
                        CandidateEvidence(
                            source="legacy_run_seed",
                            text=str(auto_seed.get("_address") or ""),
                            weight=0.5,
                            details=dict(auto_seed),
                        )
                    ],
                    metadata=dict(auto_seed),
                )
            )
        _selected_candidate_obj = next(
            (c for c in candidate_list if c.candidate_id == selected_candidate_id),
            candidate_list[0] if candidate_list else None,
        )
        try:
            from georef_core.decision_engine import evaluate_quality as _eval_quality
            _decision = _eval_quality(
                dict(LAST_GEOREF_QUALITY or {}),
                evidence=(_selected_candidate_obj.evidence if _selected_candidate_obj else []),
                conflicts=(_selected_candidate_obj.conflicts if _selected_candidate_obj else []),
            )
            _confidence = _decision.confidence
        except Exception:
            _confidence = (
                0.9 if LAST_GEOREF_QUALITY.get("accepted") and not LAST_GEOREF_QUALITY.get("provisional")
                else 0.6 if LAST_GEOREF_QUALITY.get("accepted")
                else 0.2
            )
        validation = ValidationResult(
            candidate_id=selected_candidate_id or (candidate_list[0].candidate_id if candidate_list else "direct-run"),
            accepted=bool(LAST_GEOREF_QUALITY.get("accepted")),
            provisional=bool(LAST_GEOREF_QUALITY.get("provisional")),
            confidence=_confidence,
            acceptance_reason=LAST_GEOREF_QUALITY.get("acceptance_reason"),
            metrics=dict(LAST_GEOREF_QUALITY or {}),
            notes=[],
        )
        georef_result = GeorefResult(
            output_path=output_path,
            epsg=int(vision.get("epsg") or meta.get("epsg_hint") or TARGET_EPSG),
            selected_candidate_id=validation.candidate_id if candidate_list else None,
            quality=dict(LAST_GEOREF_QUALITY or {}),
            validation=validation,
        )
        bundle = GeorefCaseBundle(
            context=PipelineContext(
                input_path=input_path,
                is_pdf=bool(is_pdf),
                output_dir=Path(OUTPUT_DIR),
                artifact_dir=_artifact_dir(),
            ),
            artifacts=PipelineArtifacts(
                ingest=IngestResult(
                    source_path=input_path,
                    working_path=work_path,
                    is_pdf=bool(is_pdf),
                    metadata=dict(meta or {}),
                    pdf_text="",
                    rendered_from_pdf=bool(is_pdf and work_path != input_path),
                ),
                ocr=OCRResult(
                    text=ocr_text or "",
                    parsed=dict(parsed or {}),
                    text_source="pdf+ocr" if is_pdf else "ocr",
                ),
                structured_hints=structured,
                vision=VisionResult(
                    overview=dict(vision or {}),
                    title_block=dict(vision.get("title_block") or {}) if isinstance(vision, dict) else {},
                ),
                candidates=candidate_list,
                validations=[validation],
                georef_result=georef_result,
            ),
            review=review_status_from_validation(validation),
        )
        case_path = _artifact_dir() / f"{input_path.stem}_case_bundle.json"
        save_case_bundle(bundle, case_path)
        print(f"[i] Case bundle saved → {case_path}")
        return case_path
    except Exception as exc:
        print(f"[~] Could not save case bundle: {exc}")
        return None


# ---------------------------------------------------------------------------
# MAIN WORKER – all heavy work lives here, safe to run in a thread
# ---------------------------------------------------------------------------
def run_georef(input_path: Path | None = None, input_is_pdf: bool | None = None):
    LOG_FILE.write_text("", encoding="utf-8")  # clear log for fresh run
    # Clean up stale intermediates from the previous run so _work/ doesn't grow
    # indefinitely.  Only non-essential debug images are removed; GeoTIFFs are
    # kept in OUTPUT_DIR (not _work/) and are never touched here.
    try:
        _wdir = _artifact_dir()
        for _f in _wdir.iterdir():
            if _f.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff") and _f.is_file():
                _f.unlink(missing_ok=True)
    except Exception:
        pass

    active_input_path = Path(input_path) if input_path is not None else INPUT_PATH
    active_input_is_pdf = bool(input_is_pdf) if input_is_pdf is not None else (
        active_input_path is not None and active_input_path.suffix.lower() == ".pdf"
    )

    if active_input_path is None:
        print(f"[✗] No PDF or TIFF found in {PLAN_FOLDER}")
        return

    print("=" * 60)
    print(f"Auto-Georeferencing: {active_input_path.name}")
    print("=" * 60)
    try:
        backend_path = Path(__file__).resolve()
        backend_mtime = backend_path.stat().st_mtime
        print(f"[i] Backend: {backend_path}")
        print(f"[i] Backend mtime: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(backend_mtime))}")
        _addr_flag = load_project_address() or ""
        print(
            "[i] Backend flags: "
            f"patch_consensus=on  road_band_vector_only=on  "
            f"canvas_vision_context={'on' if USE_QGIS_CANVAS_VISION_CONTEXT else 'off'}  "
            f"strict_wms={'on' if STRICT_WMS_VALIDATION else 'off'}  "
            f"osm_snap={'on' if ENABLE_OSM_VECTOR_SNAPPING else 'off'}  "
            f"seed_priority={'project_address' if _addr_flag else 'canvas'}"
        )
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 0. Read QGIS canvas viewport (used as fallback seed; project_address
    #    takes priority when set — see derive_auto_seed)
    # ------------------------------------------------------------------
    # canvas.grab() is a Qt UI call and MUST run on the main thread.
    # When called from the plugin dialog the dialog pre-captures this on
    # the main thread and stores the result in CANVAS_INFO_OVERRIDE before
    # starting the worker thread.  We never call get_qgis_canvas_info()
    # ourselves here to avoid the thread-safety crash.
    canvas_info = CANVAS_INFO_OVERRIDE if CANVAS_INFO_OVERRIDE is not None \
                  else {}
    canvas_img  = None
    if canvas_info.get("screenshot") and HAS_PIL:
        try:
            canvas_img = Image.open(str(canvas_info["screenshot"])).convert("RGB")
            # Downscale so it doesn't eat too many tokens
            canvas_img.thumbnail((800, 800), Image.LANCZOS)
        except Exception as exc:
            print(f"[!] Could not load canvas screenshot: {exc}")
            canvas_img = None

    # ------------------------------------------------------------------
    # PDF path: render page → PNG/TIFF, extract text layer for OCR step
    # ------------------------------------------------------------------
    if active_input_is_pdf:
        if not HAS_FITZ:
            print("[✗] pymupdf not installed – cannot render PDF.  Run: pip install pymupdf")
            return
        print(f"[~] Rendering PDF at {PDF_RENDER_DPI} DPI …")
        rendered_tif = _artifact_path(active_input_path.stem + "_rendered.tif")
        cached_render = render_pdf_to_tiff_cached(active_input_path, dest_path=rendered_tif)
        try:
            with Image.open(str(cached_render)) as _rendered_img:
                print(f"[✓] PDF rendered → {rendered_tif}  ({_rendered_img.width}×{_rendered_img.height} px)")
        except Exception:
            print(f"[✓] PDF rendered → {rendered_tif}")
        work_path = rendered_tif

        # 1. Metadata from PDF (no GDAL needed for the rendered TIFF yet)
        meta = pdf_metadata(active_input_path)

        # 2. OCR: use the PDF text layer first (exact machine-readable text);
        #    also merge rendered-image OCR so raster-only map labels are not lost.
        pdf_text = extract_pdf_text(active_input_path) or ""
        if len(pdf_text.strip()) > 50:
            print(f"[✓] PDF text layer extracted ({len(pdf_text)} chars)")
            print("[~] Running supplemental image OCR to recover raster-only map labels …")
            img_ocr_text = ocr_extract_text(work_path)
            ocr_text = _merge_text_sources(pdf_text, img_ocr_text)
            print(f"[i] Merged PDF text + image OCR → {len(ocr_text)} chars")
        else:
            print("[~] PDF text layer sparse -- running Tesseract on rendered image …")
            ocr_text = ocr_extract_text(work_path)
    else:
        work_path = active_input_path
        # 1. Metadata
        meta = read_tiff_metadata(work_path)
        # 2. OCR
        ocr_text = ocr_extract_text(work_path)

    if ocr_text:
        _ocr_out = _artifact_path("ocr_text.txt")
        _ocr_out.write_text(ocr_text, encoding="utf-8")
        print(f"[i] OCR text saved → {_ocr_out}")
        _structured_preview = _extract_structured_location_hints(ocr_text or "")
        _artifact_path("structured_location_hints.json").write_text(
            json.dumps(_structured_preview, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if _structured_preview.get("parcel_refs"):
            print(f"[i] Parsed parcel references: {', '.join(_structured_preview['parcel_refs'][:3])}")

    # 3. Parse coordinates from OCR / PDF text
    parsed = parse_coordinates(ocr_text) if ocr_text else {
        "eastings": [], "northings": [], "pairs": [], "scale": None, "crs_hints": []
    }
    # Merge scale/EPSG hints from PDF metadata into parsed
    if active_input_is_pdf:
        if not parsed.get("scale") and meta.get("scale_hint"):
            parsed["scale"] = meta["scale_hint"]
        if meta.get("epsg_hint") and "25832" not in parsed.get("crs_hints", []):
            parsed.setdefault("crs_hints", []).append(str(meta["epsg_hint"]))

    # 4. Vision AI. The QGIS canvas screenshot used to be passed as primary
    # location context, but that made the model anchor to the current viewport
    # instead of the plan itself. Keep it disabled by default and only use the
    # plan image for overview inference.
    vision = openai_vision_analysis(
        work_path,
        meta,
        canvas_img=canvas_img if USE_QGIS_CANVAS_VISION_CONTEXT else None,
        ocr_text=ocr_text or "",
    )
    if canvas_img is not None:
        canvas_img.close()
        canvas_img = None
    if vision:
        (_artifact_path("vision_result.json")).write_text(
            json.dumps(vision, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[i] Vision result saved → {_artifact_path('vision_result.json')}")

    # 4b. Detect plan type and select the appropriate WMS reference layer.
    #     WMS_CONFIG_OVERRIDE (set by the plugin dialog) takes priority over
    #     auto-detection so the user can manually choose the reference map.
    if WMS_CONFIG_OVERRIDE:
        print(f"[i] WMS override active: {WMS_CONFIG_OVERRIDE} (manual/plugin selection)")
        select_wms_config(WMS_CONFIG_OVERRIDE)
        plan_type = WMS_CONFIG_OVERRIDE
        # Warn when aerial WMS is chosen for a very fine scale plan — NCC will be
        # weak because the CAD linework doesn't correlate well with aerial imagery.
        _override_scale = (
            SCALE_OVERRIDE or vision.get("scale") or meta.get("scale_hint") or 0
        )
        if WMS_CONFIG_OVERRIDE == "aerial" and 0 < _override_scale <= 1000:
            print(
                f"[!] WMS override 'aerial' selected for 1:{_override_scale} plan. "
                f"CAD plans at this scale match poorly against aerial orthophotos. "
                f"Consider switching to 'vector' (ALKIS cadastral) for better NCC scores."
            )
    else:
        _plan_img = None
        if HAS_PIL:
            try:
                _plan_img = Image.open(str(work_path)).convert("RGB")
            except Exception:
                pass
        plan_type = detect_plan_type(
            _plan_img,
            (ocr_text or "") + "\n" + active_input_path.stem,
            vision,
            meta=meta,
            parsed=parsed,
        )
        if _plan_img is not None:
            _plan_img.close()
            _plan_img = None
        # For very fine scale plans (1:250 – 1:1000), prefer vector/ALKIS WMS even
        # if the image variance would suggest aerial.  At these scales the plan shows
        # individual road lanes and parcel geometry that matches ALKIS far better than
        # orthophotos.  Scale 0 means unknown — do not override in that case.
        _auto_scale = SCALE_OVERRIDE or vision.get("scale") or meta.get("scale_hint") or 0
        if plan_type == "aerial" and 0 < _auto_scale <= 1000:
            print(
                f"[i] Scale 1:{_auto_scale} is very fine — overriding auto-detected 'aerial' "
                f"to 'vector' (ALKIS cadastral) for better NCC match quality."
            )
            plan_type = "vector"
        select_wms_config(plan_type)

    # 5. Determine EPSG  (PDF hint > Vision AI > OCR hint > default)
    epsg = vision.get("epsg") or meta.get("epsg_hint") or TARGET_EPSG
    if "31467" in parsed.get("crs_hints", []) or "Gauss" in parsed.get("crs_hints", []):
        epsg = 31467
    print(f"[i] Using EPSG:{epsg}")

    manual_seed_requested = False

    # 5b. Derive auto-seed from plan context.
    #     The QGIS canvas centre is only a fallback; plan-derived cues are
    #     more reliable and match the pre-plugin behaviour better.
    auto_seed = derive_auto_seed(vision, ocr_text or "", active_input_path,
                                 canvas_center=canvas_info or None)
    runtime_candidates = []
    selected_candidate_id = None
    try:
        import importlib, sys as _sys
        # Force reload of candidate_generation so that fixes to _norm_city and
        # other scoring functions take effect immediately in QGIS without needing
        # a full QGIS restart (QGIS reloads auto_georeference.py but not submodules).
        if "georef_core.candidate_generation" in _sys.modules:
            try:
                importlib.reload(_sys.modules["georef_core.candidate_generation"])
            except Exception:
                pass
        from georef_core.candidate_generation import build_candidates
        from georef_core.extract_text import extract_structured_hints
        from georef_core.models import VisionResult

        _runtime_hints = extract_structured_hints(
            ocr_text or "",
            vision,
            vision.get("title_block") if isinstance(vision.get("title_block"), dict) else {},
        )
        _runtime_candidates = build_candidates(
            src_path=active_input_path,
            ocr_text=ocr_text or "",
            hints=_runtime_hints,
            vision=VisionResult(
                overview=dict(vision or {}),
                title_block=dict(vision.get("title_block") or {}) if isinstance(vision, dict) else {},
            ),
            parsed=parsed,
            library_path=GEOREF_LIBRARY_FILE,
        )
        runtime_candidates = [
            {
                "candidate_id": item.candidate_id,
                "source": item.source,
                "confidence_tier": item.confidence_tier,
                "center_easting": item.center_easting,
                "center_northing": item.center_northing,
                "search_radius_m": item.search_radius_m,
                "label": item.label,
                "rank_score": item.rank_score,
                "evidence": [
                    {
                        "source": ev.source,
                        "text": ev.text,
                        "weight": ev.weight,
                        "details": dict(ev.details or {}),
                    }
                    for ev in item.evidence
                ],
                "metadata": dict(item.metadata or {}),
                "conflicts": list(item.conflicts or []),
            }
            for item in _runtime_candidates
        ]
        for _cand in _runtime_candidates[:5]:
            print(
                f"[i] Candidate: {_cand.candidate_id}  src={_cand.source}  "
                f"tier={_cand.confidence_tier}  rank={_cand.rank_score:.2f}  "
                f"E={(_cand.center_easting or 0):.0f} N={(_cand.center_northing or 0):.0f}"
            )

        # Keep ranked candidates for diagnostics and training, but do not let
        # them override the legacy live seed unless no legacy seed exists.
        # The live runtime should still be driven by image/OCR/WMS refinement,
        # not by an unvalidated candidate prior.
        try:
            from georef_core.ranker import load_ranker_model, default_ranker_model_path
            _ranker_model = load_ranker_model(default_ranker_model_path())
            if _ranker_model is not None:
                _scored = sorted(
                    _runtime_candidates,
                    key=lambda c: _ranker_model.score_candidate(c),
                    reverse=True,
                )
            else:
                _scored = sorted(
                    _runtime_candidates,
                    key=lambda c: (c.rank_score, -(c.search_radius_m or 0)),
                    reverse=True,
                )
            _best = next(
                (c for c in _scored if c.center_easting is not None and c.center_northing is not None),
                None,
            )
            if _best is not None:
                _scale_for_seed = SCALE_OVERRIDE or vision.get("scale") or parsed.get("scale")
                # A library hit (rank=1.5, operator-verified ground truth) or any
                # candidate that clearly outranks the legacy OCR/Vision seed should
                # take over.  "library" source means we previously accepted/corrected
                # this exact plan — always trust it.  For non-library sources we only
                # override when rank_score >= 1.0 (coordinates tier) AND no legacy
                # seed exists, keeping the conservative behaviour for ambiguous cases.
                _is_library = _best.source == "library"
                _use_best = not auto_seed or _is_library
                if _use_best:
                    selected_candidate_id = _best.candidate_id
                    auto_seed = _best.to_seed_dict(
                        scale_denominator=int(_scale_for_seed) if _scale_for_seed else None
                    )
                    _reason = "library ground-truth override" if _is_library else "fallback seed"
                    print(
                        f"[i] Using ranked candidate as {_reason}: {_best.candidate_id} "
                        f"(src={_best.source}, tier={_best.confidence_tier}, "
                        f"rank={_best.rank_score:.2f})  "
                        f"E={_best.center_easting:.0f} N={_best.center_northing:.0f}"
                    )
                else:
                    # Legacy seed wins — but only record it as selected if the
                    # legacy seed source matches a candidate; otherwise leave
                    # selected_candidate_id as-is (None or already set above).
                    print(
                        f"[i] Top ranked candidate (advisory): {_best.candidate_id} "
                        f"(src={_best.source}, tier={_best.confidence_tier}, "
                        f"rank={_best.rank_score:.2f})  "
                        f"E={_best.center_easting:.0f} N={_best.center_northing:.0f}; "
                        f"keeping legacy seed {str(auto_seed.get('_source') or 'auto_seed')[:40]}"
                    )
            else:
                print("[~] No valid ranked candidate -- keeping legacy auto-seed")
        except Exception as _rank_exc:
            print(f"[~] Candidate ranking failed -- keeping legacy auto-seed: {_rank_exc}")

    except Exception as _candidate_exc:
        print(f"[~] Structured candidate extraction unavailable -- continuing with legacy auto-seed: {_candidate_exc}")

    # Warn when OCR and Vision AI disagree on scale by more than 20 %
    _ocr_scale = parsed.get("scale")
    _vis_scale  = vision.get("scale")
    if _ocr_scale and _vis_scale and abs(_ocr_scale - _vis_scale) / max(float(_vis_scale), 1) > 0.20:
        _active_scale = SCALE_OVERRIDE or _vis_scale
        print(
            f"[!] Scale conflict: OCR 1:{_ocr_scale}  Vision AI 1:{_vis_scale}"
            + (f"  SCALE_OVERRIDE={SCALE_OVERRIDE}" if SCALE_OVERRIDE else f"  → using 1:{_active_scale}")
        )

    # 6. Georeference
    out = None
    if HAS_GDAL:
        out = georeference(work_path, vision, epsg, auto_seed=auto_seed)

        persist_case_bundle_for_run(
            input_path=active_input_path,
            work_path=work_path,
            is_pdf=active_input_is_pdf,
            meta=meta,
            ocr_text=ocr_text or "",
            parsed=parsed,
            vision=vision or {},
            auto_seed=auto_seed,
            runtime_candidates=runtime_candidates,
            selected_candidate_id=selected_candidate_id,
            output_path=out,
        )

        # 7. Load in QGIS -- skip when the plugin dialog handles it on the
        #    main thread (calling Qt UI ops from a worker thread freezes QGIS)
        if out and not SKIP_AUTO_LAYER_LOAD:
            load_in_qgis(out)
    else:
        print("[!] GDAL missing – cannot georeference")

    print("=" * 60)
    print("Done. Check the 'output/' folder for results.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# MAIN – uses QThread inside QGIS so the UI never freezes
# ---------------------------------------------------------------------------
def main():
    try:
        # Running inside QGIS → offload to a background thread
        from qgis.PyQt.QtCore import QThread, pyqtSignal, QObject

        class _Worker(QObject):
            finished = pyqtSignal()

            def run(self):
                try:
                    run_georef()
                except Exception as exc:
                    print(f"[✗] Error: {exc}")
                finally:
                    self.finished.emit()

        # Keep references alive for the duration of the job
        main._thread = QThread()
        main._worker = _Worker()
        main._worker.moveToThread(main._thread)
        main._thread.started.connect(main._worker.run)
        main._worker.finished.connect(main._thread.quit)
        main._worker.finished.connect(main._worker.deleteLater)
        main._thread.finished.connect(main._thread.deleteLater)
        main._thread.start()
        print("[i] Running in background thread – QGIS stays responsive.")

    except ImportError:
        # Running outside QGIS (OSGeo4W Shell etc.) → run directly
        run_georef()


if __name__ == "__main__":
    main()

