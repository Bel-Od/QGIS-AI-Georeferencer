"""
geopack_manager.py — load, install and manage regional geopack configuration files.

A geopack (.geopack.json) defines everything the georeferencer needs for a
specific region: CRS presets, WMS endpoints, coordinate bounds, city lookup
tables, and optional parcel API config.

Store locations (searched in order, later dirs have higher priority):
  1. plugin_root/geopacks/    — packs shipped with the plugin
  2. output_dir/geopacks/     — user-installed packs (writable)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

GEOPACK_EXT        = ".geopack.json"
SCHEMA_VERSION     = "1.0"
_REQUIRED_FIELDS   = {"id", "name", "schema_version", "crs_presets"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate(data: dict) -> None:
    missing = _REQUIRED_FIELDS - set(data.keys())
    if missing:
        raise ValueError(f"Geopack missing required fields: {missing}")
    sv = str(data.get("schema_version", ""))
    if not sv.startswith("1."):
        raise ValueError(f"Unsupported geopack schema_version: {sv!r}")


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class GeopackManager:
    """
    Manages the discovery, loading, installation and removal of geopacks.

    Parameters
    ----------
    store_dirs : list of Path
        Directories to search for .geopack.json files, in order.
        The LAST directory is used as the install target for user packs.
    """

    def __init__(self, store_dirs: list[Path]) -> None:
        self.store_dirs = [Path(d) for d in store_dirs]

    @property
    def install_dir(self) -> Path:
        """User-writable directory where new packs are installed."""
        d = self.store_dirs[-1]
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover_all(self) -> list[dict]:
        """
        Scan all store_dirs and return a list of raw geopack dicts.
        Later directories override earlier ones when IDs collide.
        """
        found: dict[str, dict] = {}
        for store in self.store_dirs:
            if not store.is_dir():
                continue
            for path in sorted(store.glob(f"*{GEOPACK_EXT}")):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    _validate(data)
                    data["_path"] = str(path)
                    data["_builtin"] = (store != self.install_dir)
                    found[str(data["id"])] = data
                except Exception as exc:
                    # Non-fatal — bad file is skipped
                    print(f"[!] Geopack load error ({path.name}): {exc}")
        return list(found.values())

    # ------------------------------------------------------------------
    # Preset loading
    # ------------------------------------------------------------------

    def load_all_presets(self) -> dict:
        """
        Merge CRS presets from all installed geopacks.
        Returns a dict compatible with auto_georeference.CRS_PRESETS.
        """
        presets: dict = {}
        for pack in self.discover_all():
            pack_id  = str(pack.get("id", "unknown"))
            for key, preset in (pack.get("crs_presets") or {}).items():
                # Annotate with source pack so the UI can show provenance
                entry = dict(preset)
                entry.setdefault("_pack_id",   pack_id)
                entry.setdefault("_pack_name", str(pack.get("name", pack_id)))
                presets[key] = entry
        return presets

    # ------------------------------------------------------------------
    # City lookup
    # ------------------------------------------------------------------

    def load_city_lookup(self, epsg: int) -> tuple[dict[str, float], dict[str, float]]:
        """
        Return (easting_by_name, northing_by_name) for cities whose
        city_lookup_epsg matches the requested EPSG.
        """
        e_dict: dict[str, float] = {}
        n_dict: dict[str, float] = {}
        for pack in self.discover_all():
            if int(pack.get("city_lookup_epsg") or 0) != epsg:
                continue
            for city, coords in (pack.get("city_lookup") or {}).items():
                city_key = str(city).strip().lower()
                if "e" in coords and "n" in coords:
                    e_dict[city_key] = float(coords["e"])
                    n_dict[city_key] = float(coords["n"])
        return e_dict, n_dict

    # ------------------------------------------------------------------
    # Parcel APIs
    # ------------------------------------------------------------------

    def load_parcel_apis(self) -> dict:
        """Return all parcel API configs from installed packs, keyed by api_id."""
        apis: dict = {}
        for pack in self.discover_all():
            for api_id, cfg in (pack.get("parcel_apis") or {}).items():
                apis[api_id] = dict(cfg)
        return apis

    # ------------------------------------------------------------------
    # Install / remove
    # ------------------------------------------------------------------

    def install(self, source_path: Path) -> dict:
        """
        Validate and install a geopack file into the user install directory.

        Raises ValueError if the file is invalid.
        Returns the parsed geopack dict on success.
        """
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Geopack file not found: {source_path}")
        data = json.loads(source_path.read_text(encoding="utf-8"))
        _validate(data)

        pack_id  = str(data["id"])
        dest     = self.install_dir / f"{pack_id}{GEOPACK_EXT}"
        shutil.copy2(source_path, dest)
        data["_path"]    = str(dest)
        data["_builtin"] = False
        return data

    def uninstall(self, pack_id: str) -> bool:
        """
        Remove a user-installed geopack.  Built-in (shipped) packs cannot
        be removed via this method — returns False.
        """
        target = self.install_dir / f"{pack_id}{GEOPACK_EXT}"
        if target.exists():
            target.unlink()
            return True
        return False

    def find(self, pack_id: str) -> dict | None:
        """Return the geopack with the given id, or None if not found."""
        for pack in self.discover_all():
            if str(pack.get("id")) == pack_id:
                return pack
        return None


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def default_store_dirs(plugin_root: Path, output_dir: Path) -> list[Path]:
    """
    Return the standard geopack store directory list for this plugin.

    Built-in packs live alongside the plugin; user packs live in
    output_dir/geopacks/ so they survive plugin reinstalls.
    """
    dirs = []
    builtin = Path(plugin_root) / "geopacks"
    user    = Path(output_dir)  / "geopacks"
    if builtin.is_dir() or True:          # always include even if empty
        dirs.append(builtin)
    dirs.append(user)
    return dirs
