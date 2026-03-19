from __future__ import annotations

import importlib


def load_auto_georeference():
    """
    Import the main auto_georeference module in either packaged-plugin mode
    or legacy top-level/dev mode.
    """
    candidates = [
        "auto_georef_plugin.auto_georeference",
        "auto_georeference",
    ]
    last_exc: Exception | None = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise ModuleNotFoundError("Could not import auto_georeference")
