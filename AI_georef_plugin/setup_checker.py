"""
setup_checker.py
----------------
Checks all runtime dependencies required by Auto Georeferencer and returns
structured status objects so the UI can render a dependency status panel.

Safe to import anywhere — has no heavy dependencies of its own.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Map from Python import name → PyPI distribution name.
# Only entries that differ from the import name are needed.
_IMPORT_TO_DIST: dict[str, str] = {
    "PIL":  "Pillow",
    "cv2":  "opencv-python-headless",
    "fitz": "pymupdf",
}

_OPENCV_PIP_SPEC = "opencv-python-headless>=4.10.0.84"
_CONFLICTING_OPENCV_DISTS = [
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
]


def _python_exe() -> str:
    """
    Return the real Python interpreter path.

    Inside QGIS, sys.executable points to qgis.exe (the launcher), not python.exe.
    Running subprocess with qgis.exe causes QGIS to interpret each argument as a
    data source path.  We resolve the actual python.exe via sys.exec_prefix instead.
    """
    candidates = [
        Path(sys.exec_prefix) / "python.exe",
        Path(sys.exec_prefix) / "python3.exe",
        Path(sys.exec_prefix) / "bin" / "python.exe",
        Path(sys.exec_prefix) / "bin" / "python3.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # Fallback: may be wrong inside QGIS but better than nothing
    return sys.executable


def _site_roots() -> list[Path]:
    roots: list[Path] = []
    for raw in sys.path:
        if not raw:
            continue
        try:
            roots.append(Path(raw).resolve())
        except Exception:
            continue
    return roots


def _path_is_under(path: Path, roots: list[Path]) -> bool:
    try:
        path_r = path.resolve()
    except Exception:
        return False
    for root in roots:
        try:
            path_r.relative_to(root)
            return True
        except Exception:
            continue
    return False


def _shadowing_site_packages(module_name: str) -> tuple[Path | None, str]:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None, ""
    origin = getattr(spec, "origin", None)
    if not origin:
        return None, ""
    try:
        origin_path = Path(origin).resolve()
    except Exception:
        return None, ""
    user_root = Path.home().resolve()
    qgis_root = Path(sys.exec_prefix).resolve()
    if _path_is_under(origin_path, [user_root]) and not _path_is_under(origin_path, [qgis_root]):
        return origin_path, str(origin_path)
    return None, str(origin_path)


def _run_subprocess(cmd: list[str], timeout: int = 180) -> tuple[bool, str]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        output = (result.stdout + result.stderr).strip()
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, f"Command timed out after {timeout} seconds: {' '.join(cmd)}"
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Requirement:
    name: str               # Human-readable display name
    key: str                # Unique internal id
    status: str = "unknown" # "ok" | "missing" | "warning"
    version: str = ""       # Installed version string (empty if missing)
    message: str = ""       # Short user-facing message shown in the UI
    install_cmd: str = ""   # pip install command that fixes this (empty if N/A)
    fix_url: str = ""       # External download URL (e.g. Tesseract installer)
    required: bool = True   # False = optional / nice-to-have


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _try_import(import_name: str, version_attr: str = "__version__") -> tuple[str, str]:
    """Return (status, version) for a package.

    Uses importlib.metadata for the install check so we never touch sys.modules.
    This means:
      - C extensions (numpy, cv2, PIL) are never reloaded mid-process.
      - Status is accurate immediately after pip install/uninstall.
      - For submodule paths (e.g. "osgeo.gdal") we fall back to a plain import
        since those are QGIS core bundles with no dist-info.
    """
    # Submodule paths are QGIS core — just import them.
    if "." in import_name:
        try:
            mod = importlib.import_module(import_name)
            return "ok", str(getattr(mod, version_attr, "installed"))
        except ImportError:
            return "missing", ""

    # Use the PyPI distribution name for the metadata lookup.
    dist_name = _IMPORT_TO_DIST.get(import_name, import_name)
    try:
        importlib.invalidate_caches()
        ver = importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return "missing", ""

    # Package is on disk. Prefer the version string from the live module if loaded.
    mod = sys.modules.get(import_name)
    if mod is not None:
        live_ver = getattr(mod, version_attr, None)
        if isinstance(live_ver, str) and live_ver:
            ver = live_ver

    return "ok", ver


def _try_live_import(import_name: str, version_attr: str = "__version__") -> tuple[str, str, str]:
    """Return (status, version, error_message) for an actual import attempt."""
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, version_attr, "")
        if not isinstance(ver, str):
            ver = str(ver) if ver is not None else ""
        return "ok", ver, ""
    except ImportError as exc:
        return "missing", "", str(exc)
    except Exception as exc:
        return "warning", "", str(exc)


def _check_gdal() -> Requirement:
    status, ver = _try_import("osgeo.gdal", "__version__")
    if status == "ok":
        msg = "Bundled with QGIS."
    else:
        msg = "GDAL not found — make sure you are running inside QGIS."
    return Requirement(
        name="GDAL", key="gdal", status=status, version=ver,
        message=msg, required=True,
    )


def _check_pillow() -> Requirement:
    status, ver = _try_import("PIL", "__version__")
    if status == "ok" and (not ver or ver == "installed"):
        # PIL.__version__ is sometimes in PIL.Image rather than PIL.__init__
        mod = sys.modules.get("PIL")
        if mod is not None:
            ver = getattr(mod, "__version__", ver)
    return Requirement(
        name="Pillow (PIL)", key="pillow", status=status, version=ver,
        message="Image processing." if status == "ok" else "Required for all image operations.",
        install_cmd=f'"{_python_exe()}" -m pip install Pillow',
        required=True,
    )


def _check_numpy() -> Requirement:
    status, ver = _try_import("numpy")
    shadow_path, origin = _shadowing_site_packages("numpy")
    if status == "ok" and shadow_path is not None:
        return Requirement(
            name="NumPy", key="numpy", status="warning", version=ver,
            message=(
                "NumPy is loading from a user/site-packages path instead of the QGIS runtime. "
                f"Current module: {origin}. Remove pip-installed NumPy from the QGIS/user environment."
            ),
            install_cmd="",
            required=True,
        )
    return Requirement(
        name="NumPy", key="numpy", status=status, version=ver,
        message="Bundled with QGIS; do not manage via plugin pip installer." if status == "ok"
                else "Required by QGIS itself. Repair the QGIS installation instead of pip-installing NumPy.",
        install_cmd="",
        required=True,
    )


def _check_pytesseract() -> Requirement:
    status, ver = _try_import("pytesseract")
    return Requirement(
        name="pytesseract", key="pytesseract", status=status, version=ver,
        message="Python–Tesseract bridge." if status == "ok" else "Required for OCR step.",
        install_cmd=f'"{_python_exe()}" -m pip install pytesseract',
        required=True,
    )


def _check_tesseract_binary() -> Requirement:
    """Check that the Tesseract executable is reachable."""
    candidates: list[str] = []

    # Match the runtime resolver used by auto_georeference.py as closely as possible.
    env_cmd = os.environ.get("TESSERACT_CMD", "").strip()
    if env_cmd:
        candidates.append(env_cmd)
    candidates.extend([
        shutil.which("tesseract") or "",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"D:\Program Files\Tesseract\tesseract.exe",
        r"D:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files\Tesseract\tesseract.exe",
        r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(
            os.environ.get("USERNAME", "")
        ),
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
    ])

    # On Windows, PATH lookup inside QGIS can differ from the shell. `where`
    # is what the runtime uses as an extra fallback, so include it here too.
    try:
        where_result = subprocess.run(
            ["where", "tesseract"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        candidates.extend(
            line.strip() for line in where_result.stdout.splitlines() if line.strip()
        )
    except Exception:
        pass

    # If pytesseract is installed and already configured with an explicit binary
    # path, honor that as well.
    try:
        import pytesseract  # type: ignore
        configured = getattr(pytesseract.pytesseract, "tesseract_cmd", "") or ""
        configured = str(configured).strip()
        if configured:
            candidates.insert(0, configured)
    except Exception:
        pass

    seen: set[str] = set()
    path = None
    for candidate in candidates:
        candidate = str(candidate).strip().strip('"')
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isfile(candidate):
            path = candidate
            break

    if path is None:
        return Requirement(
            name="Tesseract OCR (binary)", key="tesseract_bin",
            status="missing", version="",
            message="Not found. Download installer from github.com/UB-Mannheim/tesseract/wiki"
                    " — tick German language data during install, then restart QGIS.",
            fix_url="https://github.com/UB-Mannheim/tesseract/wiki",
            required=True,
        )

    try:
        result = subprocess.run(
            [path, "--version"], capture_output=True, text=True, timeout=5
        )
        raw = (result.stdout or result.stderr or "").strip()
        ver_line = raw.splitlines()[0] if raw else "unknown"
        # Check German language data
        list_result = subprocess.run(
            [path, "--list-langs"], capture_output=True, text=True, timeout=5
        )
        langs_raw = (list_result.stdout or list_result.stderr or "")
        has_deu = "deu" in langs_raw
        status = "ok" if has_deu else "warning"
        msg = f"{ver_line}  |  path: {path}"
        if not has_deu:
            msg += "\n⚠ German language data (deu) not installed — OCR accuracy will be reduced."
        return Requirement(
            name="Tesseract OCR (binary)", key="tesseract_bin",
            status=status, version=ver_line, message=msg,
            fix_url="https://github.com/UB-Mannheim/tesseract/wiki",
            required=True,
        )
    except Exception as exc:
        return Requirement(
            name="Tesseract OCR (binary)", key="tesseract_bin",
            status="warning", version="", message=f"Found at {path} but check failed: {exc}",
            fix_url="https://github.com/UB-Mannheim/tesseract/wiki",
            required=True,
        )


def _check_opencv() -> Requirement:
    status, ver, err = _try_live_import("cv2", "__version__")
    if status == "ok":
        msg = "Fast image matching available."
    else:
        msg = (
            "Optional - install a QGIS/Python-compatible OpenCV build for faster image matching. "
            "The plugin installer will remove conflicting OpenCV wheels before installing."
        )
        if err:
            msg = f"{msg} Current import error: {err}"
    return Requirement(
        name="OpenCV (cv2)", key="opencv", status=status, version=ver,
        message=msg,
        install_cmd=f'"{_python_exe()}" -m pip install {_OPENCV_PIP_SPEC}',
        required=False,
    )


def _check_openai() -> Requirement:
    status, ver = _try_import("openai", "__version__")
    if status == "ok":
        # openai >= 1.x may store version in openai.version.VERSION instead
        mod = sys.modules.get("openai")
        if mod is not None:
            v = getattr(mod, "__version__", None)
            if not isinstance(v, str):
                try:
                    v = mod.version.VERSION  # type: ignore[attr-defined]
                except AttributeError:
                    v = str(v) if v is not None else ver
            ver = v or ver
    return Requirement(
        name="openai", key="openai", status=status, version=ver or "?",
        message="GPT-4o Vision AI step." if status == "ok"
                else "Required for Vision AI (location, CRS, scale detection).",
        install_cmd=f'"{_python_exe()}" -m pip install openai',
        required=True,
    )


def _check_pymupdf() -> Requirement:
    status, ver = _try_import("pymupdf", "__version__")
    if status == "missing":
        status, ver = _try_import("fitz", "__version__")
    return Requirement(
        name="PyMuPDF (PDF support)", key="pymupdf", status=status, version=ver,
        message="PDF rendering." if status == "ok"
                else "Optional — required only for PDF input files.",
        install_cmd=f'"{_python_exe()}" -m pip install pymupdf',
        required=False,
    )


def _check_api_key() -> Requirement:
    key = os.environ.get("OPENAI_API_KEY", "")
    # Also check .env file in common locations
    if not key:
        for env_path in [
            Path(__file__).parent / ".env",
            Path(__file__).parent.parent / ".env",
            Path.home() / ".auto_georef.env",
        ]:
            if env_path.exists():
                try:
                    for line in env_path.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if line.startswith("OPENAI_API_KEY="):
                            key = line.split("=", 1)[1].strip().strip('"').strip("'")
                            break
                except Exception:
                    pass
                if key:
                    break

    if key and key.startswith("sk-") and len(key) > 20:
        return Requirement(
            name="OPENAI_API_KEY", key="api_key", status="ok",
            message=f"Key set  (…{key[-6:]})",
            required=True,
        )
    elif key:
        return Requirement(
            name="OPENAI_API_KEY", key="api_key", status="warning",
            message="Variable is set but does not look like a valid sk-… key.",
            required=True,
        )
    else:
        return Requirement(
            name="OPENAI_API_KEY", key="api_key", status="missing",
            message="Not set — Vision AI will be skipped. "
                    "Set it as a system environment variable or enter it below.",
            required=True,
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def check_all() -> list[Requirement]:
    """Run all dependency checks and return a list of Requirement objects."""
    return [
        _check_gdal(),
        _check_pillow(),
        _check_numpy(),
        _check_pytesseract(),
        _check_tesseract_binary(),
        _check_opencv(),
        _check_openai(),
        _check_pymupdf(),
        _check_api_key(),
    ]


def any_required_missing(reqs: list[Requirement]) -> bool:
    return any(r.required and r.status == "missing" for r in reqs)


def pip_installable_missing(reqs: list[Requirement]) -> list[Requirement]:
    """Return missing requirements that can be fixed with pip (have install_cmd, no fix_url)."""
    return [r for r in reqs if r.status in ("missing", "warning") and r.install_cmd and not r.fix_url]


def _verify_opencv_install() -> tuple[bool, str]:
    code = (
        "import sys; "
        "import numpy; "
        "import cv2; "
        "print('python=' + sys.version.split()[0]); "
        "print('numpy=' + getattr(numpy, '__version__', '?')); "
        "print('numpy_file=' + getattr(numpy, '__file__', '?')); "
        "print('cv2=' + getattr(cv2, '__version__', '?')); "
        "print('cv2_file=' + getattr(cv2, '__file__', '?'))"
    )
    return _run_subprocess([_python_exe(), "-c", code], timeout=60)


def install_missing(reqs: list[Requirement]) -> tuple[bool, str]:
    """
    Run a single pip install for all pip-installable missing packages.
    Returns (success, output_text).
    """
    targets = pip_installable_missing(reqs)
    if not targets:
        return True, "Nothing to install via pip."

    log_lines: list[str] = []
    packages: list[str] = []
    need_opencv = False
    for r in targets:
        if r.key == "opencv":
            need_opencv = True
            continue
        after_pip_install = r.install_cmd.split("pip install", 1)[-1].strip()
        packages.extend(tok.strip('"\'') for tok in after_pip_install.split())

    if packages:
        cmd = [_python_exe(), "-m", "pip", "install"] + packages
        ok, output = _run_subprocess(cmd, timeout=180)
        log_lines.append("$ " + " ".join(cmd))
        if output:
            log_lines.append(output)
        if not ok:
            return False, "\n".join(log_lines)

    if need_opencv:
        uninstall_cmd = [_python_exe(), "-m", "pip", "uninstall", "-y"] + _CONFLICTING_OPENCV_DISTS
        ok_uninstall, uninstall_output = _run_subprocess(uninstall_cmd, timeout=180)
        log_lines.append("$ " + " ".join(uninstall_cmd))
        if uninstall_output:
            log_lines.append(uninstall_output)
        if not ok_uninstall:
            return False, "\n".join(log_lines)

        install_cmd = [_python_exe(), "-m", "pip", "install", "--no-cache-dir", _OPENCV_PIP_SPEC]
        ok_install, install_output = _run_subprocess(install_cmd, timeout=240)
        log_lines.append("$ " + " ".join(install_cmd))
        if install_output:
            log_lines.append(install_output)
        if not ok_install:
            return False, "\n".join(log_lines)

        ok_verify, verify_output = _verify_opencv_install()
        log_lines.append("$ verify cv2 import against QGIS Python runtime")
        if verify_output:
            log_lines.append(verify_output)
        if not ok_verify:
            rollback_cmd = [_python_exe(), "-m", "pip", "uninstall", "-y", "opencv-python-headless"]
            ok_rollback, rollback_output = _run_subprocess(rollback_cmd, timeout=180)
            log_lines.append("$ " + " ".join(rollback_cmd))
            if rollback_output:
                log_lines.append(rollback_output)
            if not ok_rollback:
                log_lines.append("Rollback failed; remove OpenCV manually from the QGIS Python environment.")
            log_lines.append(
                "OpenCV installation failed verification and was removed. "
                "The plugin can still run without OpenCV; restart QGIS and refresh Setup."
            )
            return False, "\n".join(log_lines)

        log_lines.append("OpenCV installed and verified. Restart QGIS before running the plugin.")

    return True, "\n".join(log_lines) if log_lines else "Nothing to install via pip."


def save_api_key_to_env(key: str) -> tuple[bool, str]:
    """
    Write or remove OPENAI_API_KEY in the .env file beside the plugin.
    Pass an empty string to remove the key entirely.
    Changes take effect immediately in the current process.
    """
    env_path = Path(__file__).parent / ".env"
    try:
        lines: list[str] = []
        if env_path.exists():
            lines = env_path.read_text(encoding="utf-8").splitlines()
        # Strip any existing key line regardless
        lines = [l for l in lines if not l.startswith("OPENAI_API_KEY=")]
        if key:
            lines.append(f'OPENAI_API_KEY="{key}"')
            os.environ["OPENAI_API_KEY"] = key
            action = f"Key saved to {env_path}"
        else:
            # Removing — also clear from environment
            os.environ.pop("OPENAI_API_KEY", None)
            action = f"Key removed from {env_path}"
        # Write back (omit if file would be empty)
        content = "\n".join(l for l in lines if l.strip()) + "\n"
        if content.strip():
            env_path.write_text(content, encoding="utf-8")
        elif env_path.exists():
            env_path.unlink()
        return True, action
    except Exception as exc:
        return False, f"Could not write {env_path}: {exc}"


def get_tesseract_cmd() -> str:
    """Return the saved Tesseract executable path from the .env file, or ''."""
    env_path = Path(__file__).parent / ".env"
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("TESSERACT_CMD="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return os.environ.get("TESSERACT_CMD", "")


def save_tesseract_cmd(path: str) -> tuple[bool, str]:
    """
    Write or remove TESSERACT_CMD in the .env file beside the plugin.
    Also applies the path immediately to the current process so OCR works
    without a restart:
      - sets os.environ["TESSERACT_CMD"]
      - sets pytesseract.pytesseract.tesseract_cmd (if pytesseract is loaded)
    Pass an empty string to remove the saved path.
    """
    env_path = Path(__file__).parent / ".env"
    try:
        lines: list[str] = []
        if env_path.exists():
            lines = env_path.read_text(encoding="utf-8").splitlines()
        lines = [l for l in lines if not l.startswith("TESSERACT_CMD=")]
        if path:
            lines.append(f'TESSERACT_CMD="{path}"')
            os.environ["TESSERACT_CMD"] = path
            action = f"Tesseract path saved to {env_path}"
        else:
            os.environ.pop("TESSERACT_CMD", None)
            action = f"Tesseract path removed from {env_path}"
        content = "\n".join(l for l in lines if l.strip()) + "\n"
        if content.strip():
            env_path.write_text(content, encoding="utf-8")
        elif env_path.exists():
            env_path.unlink()
        # Apply immediately so current session OCR works without restart
        try:
            import pytesseract  # type: ignore
            pytesseract.pytesseract.tesseract_cmd = path or "tesseract"
        except Exception:
            pass
        return True, action
    except Exception as exc:
        return False, f"Could not write {env_path}: {exc}"


# ---------------------------------------------------------------------------
# Uninstall helpers
# ---------------------------------------------------------------------------

# Packages this plugin installs via pip.
# numpy is intentionally excluded — it is a QGIS core dependency and must
# not be removed, or QGIS itself will stop working.
PLUGIN_PACKAGES = [
    "Pillow",
    "pytesseract",
    "openai",
    "pymupdf",
    "opencv-python-headless",
]

# Plugin-generated data files kept in the output directory.
# These are safe to delete on a clean uninstall.
PLUGIN_DATA_FILES = [
    "georef_library.json",
    "candidate_ranker_model.json",
    "last_result.json",
    "training_dataset.jsonl",
    "candidate_labels.jsonl",
    "transform_labels.jsonl",
    "map_sources.json",
    "run.log",
]

# User-config files — listed separately so the UI can let the user choose
# whether to keep them.
USER_CONFIG_FILES = [
    "manual_seed.json",
    "project_address.json",
]


def uninstall_pip_packages() -> tuple[bool, str]:
    """
    pip-uninstall all packages this plugin installed.
    numpy is intentionally skipped — it is shared with QGIS core.
    Returns (success, output_text).
    """
    cmd = [_python_exe(), "-m", "pip", "uninstall", "-y"] + PLUGIN_PACKAGES
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        output = (result.stdout + result.stderr).strip()
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, "pip uninstall timed out."
    except Exception as exc:
        return False, str(exc)


def remove_plugin_data(output_dir: Path, *, keep_user_config: bool = True) -> list[str]:
    """
    Delete plugin-generated data files from output_dir.
    Returns list of paths that were deleted.
    If keep_user_config is True, manual_seed.json and project_address.json are preserved.
    """
    deleted: list[str] = []
    targets = list(PLUGIN_DATA_FILES)
    if not keep_user_config:
        targets += USER_CONFIG_FILES

    for name in targets:
        p = output_dir / name
        if p.exists():
            try:
                p.unlink()
                deleted.append(str(p))
            except Exception:
                pass
    return deleted


def remove_work_directory(output_dir: Path) -> tuple[bool, str]:
    """Delete the _work/ subdirectory inside output_dir."""
    work = output_dir / "_work"
    if not work.exists():
        return True, "_work directory does not exist."
    try:
        shutil.rmtree(work)
        return True, f"Deleted: {work}"
    except Exception as exc:
        return False, f"Could not delete {work}: {exc}"


def remove_env_file() -> tuple[bool, str]:
    """Delete the .env file containing the saved API key."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return True, ".env file does not exist."
    try:
        env_path.unlink()
        os.environ.pop("OPENAI_API_KEY", None)
        return True, f"Deleted: {env_path}"
    except Exception as exc:
        return False, f"Could not delete {env_path}: {exc}"


def get_plugin_install_path() -> Path:
    """Return the path where this plugin is installed."""
    return Path(__file__).parent


def get_qgis_plugin_dir() -> Path:
    """Return the QGIS user plugin directory."""
    import os
    appdata = os.environ.get("APPDATA", str(Path.home()))
    return Path(appdata) / "QGIS" / "QGIS3" / "profiles" / "default" / "python" / "plugins"
