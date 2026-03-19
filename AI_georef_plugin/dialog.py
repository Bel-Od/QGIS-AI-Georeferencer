"""
dialog.py – Run dialog for the Auto Georeferencer QGIS plugin.

Layout
------
  ┌──────────────────────────────────────────────────────────┐
  │  Input file:      [path]                      [Browse…]  │
  │  Output folder:   [path]                      [Browse…]  │
  │  Reference map:   [ Auto-detect          ▾ ]             │
  │  Scale  1:        [________]                             │
  │  Project address: [_______________________________]      │
  │  Projection:      [ EPSG:25832 — Germany/NRW    ▾ ]     │
  ├──────────────────────────────────────────────────────────┤
  │  [████████████░░░░░░░░░░]  Step 2 / 6 – Running OCR…    │
  ├──────────────────────────────────────────────────────────┤
  │  ┌ Log ─────────────────────────────────────────────┐   │
  │  │  (live log streamed from run.log)                │   │
  │  └──────────────────────────────────────────────────┘   │
  ├──────────────────────────────────────────────────────────┤
  │  [▶ Run]   [Clear Log]          [Open Output]  [Close]   │
  └──────────────────────────────────────────────────────────┘
"""
import sys
import re
import json
import shutil
import importlib
from pathlib import Path

from qgis.PyQt.QtCore    import QThread, QObject, pyqtSignal, QTimer, QSettings, Qt
from qgis.PyQt.QtGui     import QFont, QTextCursor, QColor, QPalette
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QTextEdit, QProgressBar,
    QFileDialog, QComboBox, QSizePolicy, QFrame, QLineEdit, QCheckBox, QMessageBox,
    QTabWidget, QWidget, QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea,
    QAbstractItemView,
)

try:
    from .adjustment_tool import GeorefAdjustmentTool, _AdjustmentPanel
except Exception:
    GeorefAdjustmentTool = None
    _AdjustmentPanel = None

try:
    from . import setup_checker
except Exception:
    setup_checker = None


def _import_plugin_module(module_name: str):
    """
    Import a plugin module in packaged QGIS installs and in legacy top-level/dev
    layouts used by local tests.
    """
    candidates: list[str] = []
    if __package__:
        package_root = __package__.split(".", 1)[0]
        candidates.append(f"{package_root}.{module_name}")
    candidates.append(module_name)

    last_exc: Exception | None = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise ModuleNotFoundError(f"Could not import plugin module: {module_name}")


# ---------------------------------------------------------------------------
# Locate the plugin root (auto_georeference.py lives beside this file)
# ---------------------------------------------------------------------------
def _find_script_dir() -> Path:
    # Standard install (ZIP or dev junction): auto_georeference.py is beside dialog.py
    plugin_dir = Path(__file__).resolve().parent
    if (plugin_dir / "auto_georeference.py").exists():
        return plugin_dir

    # Fallback: scan sys.path (should not be needed in normal installs)
    for p in sys.path:
        d = Path(p)
        if (d / "auto_georeference.py").exists():
            return d

    raise FileNotFoundError(
        "auto_georeference.py not found inside the plugin folder.\n"
        "Re-install the plugin via Plugins > Manage and Install Plugins > Install from ZIP."
    )


# ---------------------------------------------------------------------------
# Step labels used to drive the progress bar.
# Each tuple: (regex pattern to match in log line, step index 1-based, label)
# ---------------------------------------------------------------------------
_STEPS = [
    (r"Rendering PDF",                                                                  1, "Rendering PDF…"),
    (r"PDF text layer|Running Tesseract|OCR|Extracting text",                          2, "Extracting text (OCR)…"),
    (r"Pass 1.*overview|Vision AI|openai_vision|Vision result saved",                  3, "Vision AI – overview…"),
    (r"Plan type detected|WMS override|WMS config|Backend flags",                      4, "Detecting plan type…"),
    (r"Candidate:|Building candidate|extract_structured|Structured candidate",         4, "Building location candidates…"),
    (r"Selected seed from ranked|Using ranked candidate|legacy auto-seed|Auto-seed from|could not determine",
                                                                                       4, "Selecting best candidate…"),
    (r"WMS refinement|coarse.*grid|fine.*grid",                                        5, "WMS – searching reference tiles…"),
    (r"Writing georeferenced",                                                          6, "Writing georeferenced TIFF…"),
    (r"Georeferenced TIFF",                                                             6, "Done"),
]
_TOTAL_STEPS = 6


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------
class _Worker(QObject):
    finished = pyqtSignal()

    def __init__(self, ag_module, input_path=None, input_is_pdf=None):
        super().__init__()
        self._ag = ag_module
        self._input_path = input_path
        self._input_is_pdf = input_is_pdf

    def run(self):
        try:
            self._ag.run_georef(
                input_path=self._input_path,
                input_is_pdf=self._input_is_pdf,
            )
        except Exception as exc:
            self._ag.print(f"[✗] Unhandled error: {exc}")
        finally:
            self.finished.emit()


class _CleanupWorker(QObject):
    """Runs pip uninstall and file deletion off the main thread."""
    log_line = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, checker, options: dict):
        super().__init__()
        self._checker = checker
        self._opts = options   # keys: pip, key, work, data, config, output_dir

    def run(self):
        try:
            opts = self._opts
            sc = self._checker
            output_dir = opts.get("output_dir")

            if opts.get("pip"):
                self.log_line.emit("[ pip uninstall ]")
                ok, out = sc.uninstall_pip_packages()
                self.log_line.emit(out or ("OK" if ok else "Failed"))
                self.log_line.emit("")

            if opts.get("key"):
                self.log_line.emit("[ Remove API key ]")
                ok, msg = sc.remove_env_file()
                self.log_line.emit(msg)
                self.log_line.emit("")

            if opts.get("work") and output_dir:
                self.log_line.emit("[ Delete _work/ ]")
                ok, msg = sc.remove_work_directory(output_dir)
                self.log_line.emit(msg)
                self.log_line.emit("")

            if opts.get("data") and output_dir:
                self.log_line.emit("[ Delete plugin data files ]")
                deleted = sc.remove_plugin_data(output_dir, keep_user_config=True)
                self.log_line.emit(
                    "\n".join(f"  Deleted: {p}" for p in deleted) or "  Nothing to delete."
                )
                self.log_line.emit("")

            if opts.get("config") and output_dir:
                self.log_line.emit("[ Delete user config files ]")
                deleted = sc.remove_plugin_data(output_dir, keep_user_config=False)
                config_names = set(sc.USER_CONFIG_FILES)
                config_deleted = [p for p in deleted if Path(p).name in config_names]
                self.log_line.emit(
                    "\n".join(f"  Deleted: {p}" for p in config_deleted) or "  Nothing to delete."
                )
                self.log_line.emit("")

            self.log_line.emit("Cleanup complete.")
            self.log_line.emit("")
            self.log_line.emit("To remove the plugin itself:")
            self.log_line.emit("  QGIS: Plugins > Manage and Install Plugins > Installed > uncheck Auto Georeferencer")
            self.log_line.emit(
                f"  Then delete: {sc.get_qgis_plugin_dir() / 'auto_georef_plugin'}"
            )
        finally:
            self.finished.emit()


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------
class GeorefDialog(QDialog):
    def __init__(self, iface, script_dir: Path, parent=None):
        super().__init__(parent)
        self.iface = iface
        self._settings = QSettings("TestDash", "AutoGeoreferencer")

        self.setWindowTitle("Auto Georeferencer")
        self.setMinimumSize(740, 560)

        # State — _ag filled in by _deferred_import (runs after dialog shows)
        self._ag           = None
        self._thread       = None
        self._worker       = None
        self._log_pos      = 0
        self._output_dir   = Path.home() / "AutoGeoref" / "output"  # fallback (overridden by _deferred_import)
        self._orig_out_dir = self._output_dir
        self._current_step = 0
        self._selected_input_path = None
        self._selected_input_is_pdf = None
        self._current_run_input_path = None
        self._current_run_input_is_pdf = None
        self._expected_result_name = None
        self._latest_case_bundle_path = None
        self._latest_result_path = None

        # Map source registry (populated in _deferred_import / _load_map_sources)
        self._map_sources: dict = {}   # id -> MapSource

        # Adjustment tool state
        self._adj_tool     = None
        self._adj_prev_tool = None
        self._W_adj        = None
        self._H_adj        = None
        self._orig_adj_gt  = None
        self._pending_result_work_path = None
        self._pending_result_dest_path = None
        self._pending_result_meta_src = None
        self._pending_result_meta_dest = None
        self._adj_panel_widget = None
        self._close_when_finished = False

        self.setObjectName("autoGeorefDialog")
        self._apply_styles()
        self._build_ui()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_log)

        # Defer the heavy import (GDAL/PIL/Tesseract detection) so the dialog
        # appears immediately instead of hanging for several seconds.
        QTimer.singleShot(50, self._deferred_import)

    def _deferred_import(self):
        """Run after the dialog is visible — imports auto_georeference."""
        self._lbl_status.setText("Loading plugin…")
        self._progress.setRange(0, 0)   # pulse
        try:
            # Reload georef_core submodules so QGIS picks up any code changes
            # made since the last time the module was imported into sys.modules.
            try:
                _ms = _import_plugin_module("georef_core.map_sources")
                importlib.reload(_ms)
                _gc = _import_plugin_module("georef_core")
                importlib.reload(_gc)
                sys.modules.setdefault("georef_core", _gc)
                sys.modules.setdefault("georef_core.map_sources", _ms)
            except Exception:
                pass
            _ag = _import_plugin_module("auto_georeference")
            self._ag           = importlib.reload(_ag)
            sys.modules.setdefault("auto_georeference", self._ag)
            if self._selected_input_path is not None:
                self._ag.INPUT_PATH = self._selected_input_path
                self._ag.INPUT_IS_PDF = bool(self._selected_input_is_pdf)
            self._output_dir   = Path(_ag.OUTPUT_DIR)
            self._orig_out_dir = Path(_ag.OUTPUT_DIR)
            self._lbl_output.setText(str(self._output_dir))
            self.refresh_input_label()
            # Load map source registry from the output dir (includes user-added sources)
            self._load_map_sources()
            self._refresh_wms_combo()
            self._refresh_crs_combo()
            # Pre-populate project address field from saved file
            _addr_file = self._orig_out_dir / "project_address.json"
            if _addr_file.exists():
                try:
                    _addr_data = json.loads(_addr_file.read_text(encoding="utf-8"))
                    if _addr_data.get("enabled") and _addr_data.get("address"):
                        self._edit_address.setText(_addr_data["address"])
                except Exception:
                    pass
            self._progress.setRange(0, _TOTAL_STEPS)
            self._progress.setValue(0)
            self._lbl_status.setText("Ready")
        except Exception as exc:
            self._lbl_status.setText(f"⚠️  Load error: {exc}")
            self._progress.setRange(0, 1)
            self._progress.setValue(0)

        # Refresh setup tab and auto-switch to it if required deps are missing
        reqs = self._refresh_setup_tab()
        if setup_checker and setup_checker.any_required_missing(reqs):
            self._tabs.setCurrentWidget(self._tab_setup)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _apply_styles(self):
        self.setStyleSheet("")

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(14, 14, 14, 14)

        title = QLabel("Auto Georeferencer")
        title.setObjectName("dialogTitle")
        subtitle = QLabel(
            "Run georeferencing, review the result, and refine placement from the same modal."
        )
        subtitle.setObjectName("dialogSubtitle")
        root.addWidget(title)
        root.addWidget(subtitle)

        # ── form rows ──────────────────────────────────────────────────
        form = QFormLayout()
        form.setSpacing(6)
        form.setHorizontalSpacing(14)

        # Input file
        row_in = QHBoxLayout()
        self._lbl_input = QLabel()
        self._lbl_input.setWordWrap(True)
        self._lbl_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        btn_in = QPushButton("Browse…")
        btn_in.setFixedWidth(80)
        btn_in.clicked.connect(self._browse_input)
        row_in.addWidget(self._lbl_input, 1)
        row_in.addWidget(btn_in)
        form.addRow("<b>Input file:</b>", row_in)

        # Output folder
        row_out = QHBoxLayout()
        self._lbl_output = QLabel(str(self._output_dir))
        self._lbl_output.setWordWrap(True)
        self._lbl_output.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        btn_out = QPushButton("Browse…")
        btn_out.setFixedWidth(80)
        btn_out.clicked.connect(self._browse_output)
        row_out.addWidget(self._lbl_output, 1)
        row_out.addWidget(btn_out)
        form.addRow("<b>Output folder:</b>", row_out)

        # Reference map selector (populated dynamically from map_sources registry)
        wms_row = QHBoxLayout()
        self._combo_wms = QComboBox()
        self._combo_wms.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._combo_wms.setToolTip(
            "Auto-detect: the script picks the best reference map based on plan content.\n"
            "Manual: force a specific map source for WMS template matching.\n"
            "OSM is always available as a fallback.\n"
            "Click 'Manage…' to add your own WMS sources."
        )
        # Populate with built-ins immediately so the combo is usable before
        # _deferred_import finishes; _refresh_wms_combo() will re-populate later.
        self._refresh_wms_combo()
        wms_row.addWidget(self._combo_wms, 1)
        btn_manage_wms = QPushButton("Manage…")
        btn_manage_wms.setFixedWidth(75)
        btn_manage_wms.setToolTip("Add or remove WMS/map sources")
        btn_manage_wms.clicked.connect(self._open_map_sources_manager)
        wms_row.addWidget(btn_manage_wms)
        form.addRow("<b>Reference map:</b>", wms_row)

        # Scale override
        scale_row = QHBoxLayout()
        self._edit_scale = QLineEdit()
        self._edit_scale.setPlaceholderText("Auto-detect  (e.g. 1000 for 1:1000)")
        self._edit_scale.setMaximumWidth(260)
        self._edit_scale.setToolTip(
            "Leave blank to auto-detect scale from Vision AI / OCR title block.\n"
            "Type just the denominator, e.g. 500 for 1:500, 1000 for 1:1000."
        )
        scale_row.addWidget(self._edit_scale)
        scale_row.addStretch()
        form.addRow("<b>Scale  1:</b>", scale_row)

        # Project address (geocoded seed when canvas is not in the right area)
        self._edit_address = QLineEdit()
        self._edit_address.setPlaceholderText(
            "e.g.  Hauptstraße 5, 57462 Olpe, Deutschland  (leave blank to use canvas position)"
        )
        self._edit_address.setToolTip(
            "Optional project address used as the geographic seed location.\n"
            "Geocoded via OpenStreetMap Nominatim (no API key needed).\n"
            "When set, TAKES PRIORITY over the canvas position — use this when\n"
            "the canvas is not pointing at the plan area.\n"
            "Leave blank to use the canvas position or previous run result.\n"
            "Tip: use a specific address (e.g. 'Hauptstraße 5, 57462 Olpe') for\n"
            "best accuracy; a city name gives only the city-centre position."
        )
        form.addRow("<b>Project address:</b>", self._edit_address)

        # CRS / projection selector
        crs_row = QHBoxLayout()
        self._combo_crs = QComboBox()
        self._combo_crs.setToolTip(
            "Select the projected coordinate system for the output GeoTIFF.\n"
            "Each preset sets the output EPSG and loads the matching WMS reference service.\n"
            "Use 'Custom EPSG…' to enter any EPSG code manually.\n"
            "pyproj is required for non-UTM projections (Netherlands, France, UK…)."
        )
        # Populated once the ag module is available (_deferred_import will call
        # _refresh_crs_combo).  Fill with a placeholder now so the row exists.
        self._combo_crs.addItem("EPSG:25832 — Germany/NRW (ETRS89 / UTM Zone 32N)", userData="epsg_25832")
        crs_row.addWidget(self._combo_crs, 1)

        self._edit_custom_epsg = QLineEdit()
        self._edit_custom_epsg.setPlaceholderText("EPSG code, e.g. 31467")
        self._edit_custom_epsg.setMaximumWidth(140)
        self._edit_custom_epsg.setVisible(False)
        self._edit_custom_epsg.setToolTip("Enter the numeric EPSG code for the target CRS.")
        crs_row.addWidget(self._edit_custom_epsg)
        form.addRow("<b>Projection:</b>", crs_row)

        btn_geopacks = QPushButton("Geopacks…")
        btn_geopacks.setFixedWidth(90)
        btn_geopacks.setToolTip(
            "Install, view and remove regional geopack configuration files.\n"
            "Each geopack adds one or more CRS presets with WMS services."
        )
        btn_geopacks.clicked.connect(self._open_geopack_manager)
        crs_row.addWidget(btn_geopacks)

        self._combo_crs.currentIndexChanged.connect(self._on_crs_changed)
        self._edit_custom_epsg.editingFinished.connect(self._on_custom_epsg_changed)

        self._chk_canvas_context = QCheckBox("Use canvas image as Vision context")
        self._chk_canvas_context.setChecked(
            self._settings.value("vision/use_canvas_context", False, type=bool)
        )
        self._chk_canvas_context.setToolTip(
            "Off by default. Enable only when the plan itself has almost no readable location text."
        )
        form.addRow("<b>Vision context:</b>", self._chk_canvas_context)

        self._chk_strict_wms = QCheckBox("Require geometric confirmation for weak WMS matches")
        self._chk_strict_wms.setChecked(
            self._settings.value("wms/strict_validation", True, type=bool)
        )
        self._chk_strict_wms.setToolTip(
            "Recommended. Prevents weak NCC-only consensus from being accepted without patch/feature confirmation."
        )
        form.addRow("<b>WMS strict:</b>", self._chk_strict_wms)

        self._chk_osm_snap = QCheckBox("Snap text-derived seeds to nearby OSM roads/features")
        self._chk_osm_snap.setChecked(
            self._settings.value("seed/osm_snap", True, type=bool)
        )
        self._chk_osm_snap.setToolTip(
            "Uses Overpass/OSM to refine road, street, and landmark seeds to nearby mapped features."
        )
        form.addRow("<b>OSM snapping:</b>", self._chk_osm_snap)

        self._chk_geocode_debug = QCheckBox("Write detailed geocode candidate logs")
        self._chk_geocode_debug.setChecked(
            self._settings.value("seed/geocode_debug", False, type=bool)
        )
        self._chk_geocode_debug.setToolTip(
            "Useful for calibration. Writes top candidate scores and snap decisions into the run log."
        )
        form.addRow("<b>Seed debug:</b>", self._chk_geocode_debug)

        self._edit_reviewer = QLineEdit()
        self._edit_reviewer.setPlaceholderText("Optional reviewer name")
        form.addRow("<b>Reviewer:</b>", self._edit_reviewer)

        setup_box = QGroupBox("Run Setup")
        setup_layout = QVBoxLayout(setup_box)
        setup_layout.setContentsMargins(10, 10, 10, 10)
        setup_layout.addLayout(form)
        root.addWidget(setup_box)

        # ── separator ─────────────────────────────────────────────────
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        root.addWidget(line)

        # ── progress bar + status label ───────────────────────────────
        status_card = QFrame()
        status_card.setObjectName("statusCard")
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(10, 10, 10, 10)
        prog_row = QHBoxLayout()
        self._progress = QProgressBar()
        self._progress.setRange(0, _TOTAL_STEPS)
        self._progress.setValue(0)
        self._progress.setFixedHeight(18)
        self._progress.setTextVisible(False)
        prog_row.addWidget(self._progress, 1)

        self._lbl_status = QLabel("Ready")
        self._lbl_status.setMinimumWidth(260)
        prog_row.addWidget(self._lbl_status)
        status_layout.addLayout(prog_row)
        root.addWidget(status_card)

        self._tabs = QTabWidget()
        self._tab_run = QWidget()
        self._tab_adjust = QWidget()
        self._tab_setup = QWidget()
        self._tabs.addTab(self._tab_run, "Run")
        self._tabs.addTab(self._tab_adjust, "Adjustment")
        self._tabs.addTab(self._tab_setup, "⚙ Setup")
        self._tabs.setTabEnabled(1, False)
        root.addWidget(self._tabs, 1)

        run_outer_layout = QVBoxLayout(self._tab_run)
        run_outer_layout.setContentsMargins(0, 0, 0, 0)
        run_outer_layout.setSpacing(0)
        self._run_scroll = QScrollArea(self._tab_run)
        self._run_scroll.setWidgetResizable(True)
        self._run_scroll.setFrameShape(QFrame.NoFrame)
        self._run_content = QWidget()
        run_layout = QVBoxLayout(self._run_content)
        run_layout.setContentsMargins(0, 0, 0, 0)
        run_layout.setSpacing(8)
        run_layout.addWidget(setup_box)
        run_layout.addWidget(line)
        run_layout.addWidget(status_card)
        self._run_scroll.setWidget(self._run_content)
        run_outer_layout.addWidget(self._run_scroll, 1)

        adjust_layout = QVBoxLayout(self._tab_adjust)
        adjust_layout.setContentsMargins(12, 12, 12, 12)
        adjust_layout.setSpacing(10)
        self._adjust_scroll = QScrollArea(self._tab_adjust)
        self._adjust_scroll.setWidgetResizable(True)
        self._adjust_scroll.setFrameShape(QFrame.NoFrame)
        self._adjust_content = QWidget()
        self._adjust_content_layout = QVBoxLayout(self._adjust_content)
        self._adjust_content_layout.setContentsMargins(0, 0, 0, 0)
        self._adjust_content_layout.setSpacing(10)
        self._adjust_placeholder = QLabel(
            "Adjustment controls appear here while placement adjustment is active. Center, rotation, and scale can be typed directly."
        )
        self._adjust_placeholder.setObjectName("adjustPlaceholder")
        self._adjust_placeholder.setWordWrap(True)
        self._adjust_content_layout.addWidget(self._adjust_placeholder)
        self._adjust_content_layout.addStretch(1)
        self._adjust_scroll.setWidget(self._adjust_content)
        adjust_layout.addWidget(self._adjust_scroll, 1)

        # ── setup tab ─────────────────────────────────────────────────
        self._build_setup_tab()

        # ── log area ───────────────────────────────────────────────────
        self._log = QTextEdit()
        self._log.setObjectName("runLog")
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Courier New", 9))
        pal = self._log.palette()
        pal.setColor(QPalette.Base, QColor("#1e1e1e"))
        pal.setColor(QPalette.Text, QColor("#d4d4d4"))
        self._log.setPalette(pal)
        run_layout.addWidget(self._log, 1)

        review_row = QHBoxLayout()
        self._btn_accept = QPushButton("Accept Result")
        self._btn_accept.clicked.connect(self._mark_accepted)
        self._btn_accept.setEnabled(False)
        self._btn_review = QPushButton("Needs Review")
        self._btn_review.clicked.connect(self._mark_review_required)
        self._btn_review.setEnabled(False)
        self._btn_adjust = QPushButton("↕ Adjust Placement")
        self._btn_adjust.setToolTip(
            "Interactively drag and rotate the georeferenced TIFF on the map canvas.\n"
            "Left-drag: translate  |  green handle / right-drag: rotate  |  Shift+wheel: fine rotation\n"
            "Use the Adjustment tab for typed center, absolute rotation, scale, opacity, and Fast/Full TIFF display.\n"
            "Accept to write the corrected geotransform back to the file."
        )
        self._btn_adjust.setObjectName("accentButton")
        self._btn_adjust.setEnabled(False)
        self._btn_adjust.clicked.connect(self._launch_adjustment_tool)
        self._btn_export = QPushButton("Export Dataset")
        self._btn_export.setToolTip(
            "Export reviewed case bundles to training_dataset.jsonl plus lightweight\n"
            "candidate_labels.jsonl and transform_labels.jsonl supervision files.\n"
            "Run this before 'Retrain Ranker' to include the latest reviews."
        )
        self._btn_export.clicked.connect(self._export_training_dataset)
        self._btn_retrain = QPushButton("Retrain Ranker")
        self._btn_retrain.setToolTip(
            "Retrain the candidate-ranking model from the exported training dataset.\n"
            "Click 'Export Dataset' first, then retrain.\n"
            "Requires at least 5 reviewed cases."
        )
        self._btn_retrain.setObjectName("primaryButton")
        self._btn_retrain.clicked.connect(self._retrain_ranker)
        review_row.addWidget(self._btn_accept)
        review_row.addWidget(self._btn_review)
        review_row.addWidget(self._btn_adjust)
        review_row.addStretch()
        review_row.addWidget(self._btn_export)
        review_row.addWidget(self._btn_retrain)
        run_layout.addLayout(review_row)

        # ── button row ────────────────────────────────────────────────
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setFrameShadow(QFrame.Sunken)
        root.addWidget(line2)

        row_btn = QHBoxLayout()
        self._btn_run = QPushButton("▶  Run")
        self._btn_run.setObjectName("primaryButton")
        self._btn_run.setFixedHeight(32)
        self._btn_run.setMinimumWidth(100)
        self._btn_run.clicked.connect(self._run)

        self._btn_stop = QPushButton("⏹  Stop")
        self._btn_stop.setObjectName("dangerButton")
        self._btn_stop.setFixedHeight(32)
        self._btn_stop.setMinimumWidth(80)
        self._btn_stop.setEnabled(False)
        self._set_review_buttons_enabled(self._latest_case_bundle_path is not None)
        self._btn_stop.clicked.connect(self._stop)

        btn_clear = QPushButton("Clear Log")
        btn_clear.clicked.connect(self._log.clear)

        self._btn_open = QPushButton("Open Output Folder")
        self._btn_open.clicked.connect(self._open_output)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)

        row_btn.addWidget(self._btn_run)
        row_btn.addWidget(self._btn_stop)
        row_btn.addWidget(btn_clear)
        row_btn.addStretch()
        row_btn.addWidget(self._btn_open)
        row_btn.addWidget(btn_close)
        root.addLayout(row_btn)

        self.refresh_input_label()

    # ------------------------------------------------------------------
    # Setup tab
    # ------------------------------------------------------------------
    def _build_setup_tab(self):
        """Build the static skeleton of the Setup tab (rows filled by _refresh_setup_tab)."""
        outer_layout = QVBoxLayout(self._tab_setup)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        self._setup_scroll = QScrollArea(self._tab_setup)
        self._setup_scroll.setWidgetResizable(True)
        self._setup_scroll.setFrameShape(QFrame.NoFrame)
        self._setup_content = QWidget()
        layout = QVBoxLayout(self._setup_content)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        hdr = QLabel("<b>System Requirements</b>")
        sub = QLabel(
            "All required components must show a green checkmark before running. "
            "Click <i>Install Missing Packages</i> to fix pip-installable items automatically."
        )
        sub.setWordWrap(True)
        layout.addWidget(hdr)
        layout.addWidget(sub)

        # Status table
        self._setup_table = QTableWidget(0, 4)
        self._setup_table.setHorizontalHeaderLabels(["Component", "Status", "Version", "Notes"])
        self._setup_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._setup_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._setup_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._setup_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self._setup_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._setup_table.setSelectionMode(QAbstractItemView.NoSelection)
        self._setup_table.verticalHeader().setVisible(False)
        self._setup_table.setAlternatingRowColors(True)
        layout.addWidget(self._setup_table)

        # API key entry
        key_box = QGroupBox("OpenAI API Key")
        key_layout = QVBoxLayout(key_box)
        key_note = QLabel(
            "Required for the Vision AI step (GPT-4o). "
            "Get a key at <a href='https://platform.openai.com/api-keys'>platform.openai.com/api-keys</a>. "
            "Saved to a local <b>.env</b> file beside the plugin — never sent anywhere except OpenAI."
        )
        key_note.setWordWrap(True)
        key_note.setOpenExternalLinks(True)
        key_layout.addWidget(key_note)

        # Current key status banner (updated by _refresh_setup_tab)
        self._setup_key_banner = QLabel("")
        self._setup_key_banner.setWordWrap(True)
        self._setup_key_banner.setVisible(False)
        key_layout.addWidget(self._setup_key_banner)

        key_row = QHBoxLayout()
        self._setup_key_edit = QLineEdit()
        self._setup_key_edit.setPlaceholderText("Paste your key here:  sk-…")
        self._setup_key_edit.setEchoMode(QLineEdit.Password)
        key_row.addWidget(self._setup_key_edit, 1)

        self._btn_save_key = QPushButton("Save Key")
        self._btn_save_key.setFixedWidth(90)
        self._btn_save_key.setObjectName("primaryButton")
        self._btn_save_key.clicked.connect(self._setup_save_key)
        key_row.addWidget(self._btn_save_key)

        self._btn_show_key = QPushButton("Show")
        self._btn_show_key.setFixedWidth(55)
        self._btn_show_key.setCheckable(True)
        self._btn_show_key.toggled.connect(
            lambda checked: self._setup_key_edit.setEchoMode(
                QLineEdit.Normal if checked else QLineEdit.Password
            )
        )
        key_row.addWidget(self._btn_show_key)

        self._btn_clear_key = QPushButton("Remove Key")
        self._btn_clear_key.setFixedWidth(90)
        self._btn_clear_key.setObjectName("dangerButton")
        self._btn_clear_key.setToolTip("Delete the saved API key from the .env file")
        self._btn_clear_key.clicked.connect(self._setup_remove_key)
        self._btn_clear_key.setVisible(False)
        key_row.addWidget(self._btn_clear_key)

        key_layout.addLayout(key_row)
        self._setup_key_status = QLabel("")
        self._setup_key_status.setWordWrap(True)
        key_layout.addWidget(self._setup_key_status)
        layout.addWidget(key_box)

        # Tesseract path
        tess_box = QGroupBox("Tesseract OCR — Executable Path")
        tess_layout = QVBoxLayout(tess_box)
        tess_note = QLabel(
            "Browse to <b>tesseract.exe</b> if it was not auto-detected above. "
            "Download from <a href='https://github.com/UB-Mannheim/tesseract/wiki'>"
            "github.com/UB-Mannheim/tesseract/wiki</a> — tick <b>German language data</b> during install."
        )
        tess_note.setWordWrap(True)
        tess_note.setOpenExternalLinks(True)
        tess_layout.addWidget(tess_note)

        tess_row = QHBoxLayout()
        self._setup_tess_edit = QLineEdit()
        self._setup_tess_edit.setPlaceholderText(
            r"e.g. C:\Program Files\Tesseract-OCR\tesseract.exe"
        )
        tess_row.addWidget(self._setup_tess_edit, 1)

        btn_tess_browse = QPushButton("Browse…")
        btn_tess_browse.setFixedWidth(75)
        btn_tess_browse.clicked.connect(self._setup_browse_tesseract)
        tess_row.addWidget(btn_tess_browse)

        btn_tess_save = QPushButton("Save Path")
        btn_tess_save.setFixedWidth(80)
        btn_tess_save.setObjectName("primaryButton")
        btn_tess_save.clicked.connect(self._setup_save_tesseract)
        tess_row.addWidget(btn_tess_save)

        btn_tess_clear = QPushButton("Clear")
        btn_tess_clear.setFixedWidth(55)
        btn_tess_clear.clicked.connect(self._setup_clear_tesseract)
        tess_row.addWidget(btn_tess_clear)

        tess_layout.addLayout(tess_row)
        self._setup_tess_status = QLabel("")
        self._setup_tess_status.setWordWrap(True)
        tess_layout.addWidget(self._setup_tess_status)
        layout.addWidget(tess_box)

        # Install button row
        btn_row = QHBoxLayout()
        self._btn_install_deps = QPushButton("Install Missing Packages (pip)")
        self._btn_install_deps.setObjectName("primaryButton")
        self._btn_install_deps.clicked.connect(self._setup_install_deps)
        btn_refresh_setup = QPushButton("Refresh")
        btn_refresh_setup.clicked.connect(lambda: self._refresh_setup_tab())
        btn_row.addWidget(self._btn_install_deps)
        btn_row.addWidget(btn_refresh_setup)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Install log
        self._setup_log = QTextEdit()
        self._setup_log.setReadOnly(True)
        self._setup_log.setMaximumHeight(110)
        self._setup_log.setFont(QFont("Courier New", 8))
        self._setup_log.setPlaceholderText("pip output appears here …")
        layout.addWidget(self._setup_log)

        # ── Uninstall / Cleanup section ───────────────────────────────
        uninstall_box = QGroupBox("Cleanup && Uninstall")
        uninstall_layout = QVBoxLayout(uninstall_box)

        uninstall_note = QLabel(
            "Select what to remove, then click <b>Run Cleanup</b>.  "
            "The plugin files themselves are removed via <i>Plugins &gt; Manage and Install Plugins</i>.<br>"
            "<b>Note:</b> NumPy is never uninstalled — it is a QGIS core dependency.  "
            "Tesseract must be removed via its own uninstaller (Windows Add/Remove Programs)."
        )
        uninstall_note.setWordWrap(True)
        uninstall_layout.addWidget(uninstall_note)

        self._chk_uninstall_pip = QCheckBox(
            "Uninstall pip packages installed by this plugin  "
            "(Pillow, pytesseract, openai, pymupdf, opencv-python-headless)"
        )
        self._chk_uninstall_pip.setChecked(True)
        uninstall_layout.addWidget(self._chk_uninstall_pip)

        self._chk_remove_key = QCheckBox("Remove saved API key  (.env file)")
        self._chk_remove_key.setChecked(True)
        uninstall_layout.addWidget(self._chk_remove_key)

        self._chk_remove_work = QCheckBox(
            "Delete intermediate work files  (output/_work/)"
        )
        self._chk_remove_work.setChecked(True)
        uninstall_layout.addWidget(self._chk_remove_work)

        self._chk_remove_data = QCheckBox(
            "Delete plugin data files  (run log, training data, model, map sources)"
        )
        self._chk_remove_data.setChecked(False)
        uninstall_layout.addWidget(self._chk_remove_data)

        self._chk_remove_config = QCheckBox(
            "Delete user config files  (manual_seed.json, project_address.json)"
        )
        self._chk_remove_config.setChecked(False)
        uninstall_layout.addWidget(self._chk_remove_config)

        uninstall_btn_row = QHBoxLayout()
        self._btn_run_cleanup = QPushButton("Run Cleanup")
        self._btn_run_cleanup.setObjectName("dangerButton")
        self._btn_run_cleanup.setFixedHeight(28)
        self._btn_run_cleanup.clicked.connect(self._setup_run_cleanup)
        uninstall_btn_row.addWidget(self._btn_run_cleanup)
        uninstall_btn_row.addStretch()
        uninstall_layout.addLayout(uninstall_btn_row)

        self._setup_uninstall_log = QTextEdit()
        self._setup_uninstall_log.setReadOnly(True)
        self._setup_uninstall_log.setMaximumHeight(90)
        self._setup_uninstall_log.setFont(QFont("Courier New", 8))
        self._setup_uninstall_log.setPlaceholderText("Cleanup output appears here …")
        uninstall_layout.addWidget(self._setup_uninstall_log)

        layout.addWidget(uninstall_box)
        layout.addStretch(1)
        self._setup_scroll.setWidget(self._setup_content)
        outer_layout.addWidget(self._setup_scroll, 1)

    def _refresh_setup_tab(self) -> list:
        """Re-run all dependency checks and redraw the status table. Returns requirement list."""
        if setup_checker is None:
            return []

        reqs = setup_checker.check_all()

        STATUS_ICON  = {"ok": "✔", "warning": "⚠", "missing": "✘", "unknown": "?"}
        STATUS_COLOR = {
            "ok":      "#2d7a2d",
            "warning": "#996600",
            "missing": "#c0392b",
            "unknown": "#666666",
        }

        tbl = self._setup_table
        tbl.setRowCount(len(reqs))
        for row, req in enumerate(reqs):
            icon = STATUS_ICON.get(req.status, "?")
            color = STATUS_COLOR.get(req.status, "#666666")
            label = req.name + ("" if req.required else "  (optional)")

            name_item = QTableWidgetItem(label)
            status_item = QTableWidgetItem(f"{icon}  {req.status.upper()}")
            status_item.setForeground(QColor(color))
            version_item = QTableWidgetItem(req.version or "—")
            notes_parts = [req.message] if req.message else []
            if req.status == "missing":
                if req.install_cmd and not req.fix_url:
                    notes_parts.append(f"Fix: {req.install_cmd}")
                elif req.fix_url:
                    notes_parts.append(f"Download: {req.fix_url}")
            notes_item = QTableWidgetItem("  ".join(notes_parts))

            for item in (name_item, status_item, version_item, notes_item):
                item.setFlags(item.flags() & ~0x2)  # not editable

            tbl.setItem(row, 0, name_item)
            tbl.setItem(row, 1, status_item)
            tbl.setItem(row, 2, version_item)
            tbl.setItem(row, 3, notes_item)

        tbl.resizeRowsToContents()

        # Update install button state
        has_pip_targets = bool(setup_checker.pip_installable_missing(reqs))
        self._btn_install_deps.setEnabled(has_pip_targets)
        if not has_pip_targets:
            self._btn_install_deps.setText("Install Missing Packages (pip)")

        # Update API key banner based on current status
        api_req = next((r for r in reqs if r.key == "api_key"), None)
        if api_req:
            if api_req.status == "ok":
                self._setup_key_banner.setText(
                    f"<b style='color:#2d7a2d'>Key configured:</b>  {api_req.message}  "
                    "— paste a new key below to replace it."
                )
                self._setup_key_banner.setVisible(True)
                self._btn_clear_key.setVisible(True)
                self._setup_key_edit.setPlaceholderText("Paste a new key to replace the existing one  (sk-…)")
            elif api_req.status == "warning":
                self._setup_key_banner.setText(
                    f"<b style='color:#996600'>Warning:</b>  {api_req.message}"
                )
                self._setup_key_banner.setVisible(True)
                self._btn_clear_key.setVisible(True)
                self._setup_key_edit.setPlaceholderText("Paste corrected key here  (sk-…)")
            else:
                self._setup_key_banner.setText(
                    "<b style='color:#c0392b'>No API key configured.</b>  "
                    "Vision AI will be skipped until you enter a key."
                )
                self._setup_key_banner.setVisible(True)
                self._btn_clear_key.setVisible(False)
                self._setup_key_edit.setPlaceholderText("Paste your key here:  sk-…")

        # Populate Tesseract path field with saved value (don't overwrite if user is typing)
        saved_tess = setup_checker.get_tesseract_cmd()
        if saved_tess and not self._setup_tess_edit.text().strip():
            self._setup_tess_edit.setText(saved_tess)

        return reqs

    def _setup_save_key(self):
        if setup_checker is None:
            return
        key = self._setup_key_edit.text().strip()
        if not key:
            self._setup_key_status.setText("Please paste your API key into the field above.")
            return
        if not key.startswith("sk-"):
            self._setup_key_status.setText(
                "Key does not look valid — OpenAI keys start with 'sk-'. Double-check and try again."
            )
            return
        ok, msg = setup_checker.save_api_key_to_env(key)
        self._setup_key_status.setText(msg)
        if ok:
            self._setup_key_edit.clear()
            self._btn_show_key.setChecked(False)
            self._refresh_setup_tab()

    def _setup_remove_key(self):
        if setup_checker is None:
            return
        reply = QMessageBox.question(
            self, "Remove API Key",
            "This will delete the saved OpenAI API key from the .env file.\n"
            "Vision AI will stop working until you enter a new key.\n\nContinue?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        ok, msg = setup_checker.save_api_key_to_env("")
        self._setup_key_status.setText("Key removed. " + msg if ok else msg)
        self._refresh_setup_tab()

    def _setup_install_deps(self):
        if setup_checker is None:
            return
        self._btn_install_deps.setEnabled(False)
        self._btn_install_deps.setText("Installing …")
        self._setup_log.clear()
        self._setup_log.append("Running pip install …\n")
        QTimer.singleShot(50, self._setup_run_install)

    def _setup_run_install(self):
        if setup_checker is None:
            return
        reqs = setup_checker.check_all()
        ok, output = setup_checker.install_missing(reqs)
        self._setup_log.setPlainText(output)
        cursor = self._setup_log.textCursor()
        cursor.movePosition(QTextCursor.End)
        self._setup_log.setTextCursor(cursor)
        if ok:
            self._setup_log.append("\n✔ Installation complete — refreshing status …")
        else:
            self._setup_log.append("\n✘ Installation failed — see output above.")
        self._refresh_setup_tab()

    def _setup_open_tesseract_url(self):
        from qgis.PyQt.QtGui import QDesktopServices
        from qgis.PyQt.QtCore import QUrl
        QDesktopServices.openUrl(QUrl("https://github.com/UB-Mannheim/tesseract/wiki"))

    def _setup_browse_tesseract(self):
        current = self._setup_tess_edit.text().strip()
        start_dir = str(Path(current).parent) if current and Path(current).parent.exists() else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Tesseract Executable",
            start_dir,
            "Executable (*.exe);;All files (*)",
        )
        if path:
            self._setup_tess_edit.setText(path)

    def _setup_save_tesseract(self):
        if setup_checker is None:
            return
        path = self._setup_tess_edit.text().strip()
        if path and not Path(path).is_file():
            self._setup_tess_status.setText(f"File not found: {path}")
            return
        ok, msg = setup_checker.save_tesseract_cmd(path)
        self._setup_tess_status.setText(msg)
        self._refresh_setup_tab()

    def _setup_clear_tesseract(self):
        if setup_checker is None:
            return
        self._setup_tess_edit.clear()
        ok, msg = setup_checker.save_tesseract_cmd("")
        self._setup_tess_status.setText(msg)
        self._refresh_setup_tab()

    def _setup_run_cleanup(self):
        if setup_checker is None:
            return

        anything = (
            self._chk_uninstall_pip.isChecked()
            or self._chk_remove_key.isChecked()
            or self._chk_remove_work.isChecked()
            or self._chk_remove_data.isChecked()
            or self._chk_remove_config.isChecked()
        )
        if not anything:
            self._setup_uninstall_log.setPlainText("Nothing selected.")
            return

        # Build confirmation message
        items = []
        if self._chk_uninstall_pip.isChecked():
            items.append("• Uninstall pip packages: " + ", ".join(setup_checker.PLUGIN_PACKAGES))
        if self._chk_remove_key.isChecked():
            items.append("• Delete saved API key (.env)")
        if self._chk_remove_work.isChecked():
            items.append("• Delete output/_work/ directory")
        if self._chk_remove_data.isChecked():
            items.append("• Delete plugin data files (log, training data, model, map sources)")
        if self._chk_remove_config.isChecked():
            items.append("• Delete user config files (manual_seed.json, project_address.json)")

        reply = QMessageBox.warning(
            self, "Confirm Cleanup",
            "The following will be permanently removed:\n\n" + "\n".join(items)
            + "\n\nThis cannot be undone. Continue?",
            QMessageBox.Yes | QMessageBox.Cancel,
        )
        if reply != QMessageBox.Yes:
            return

        self._btn_run_cleanup.setEnabled(False)
        self._btn_run_cleanup.setText("Running …")
        self._setup_uninstall_log.clear()

        opts = {
            "pip":        self._chk_uninstall_pip.isChecked(),
            "key":        self._chk_remove_key.isChecked(),
            "work":       self._chk_remove_work.isChecked(),
            "data":       self._chk_remove_data.isChecked(),
            "config":     self._chk_remove_config.isChecked(),
            "output_dir": Path(self._output_dir) if self._output_dir else None,
        }
        self._cleanup_thread = QThread()
        self._cleanup_worker = _CleanupWorker(setup_checker, opts)
        self._cleanup_worker.moveToThread(self._cleanup_thread)
        self._cleanup_thread.started.connect(self._cleanup_worker.run)
        self._cleanup_worker.log_line.connect(self._setup_uninstall_log.append)
        self._cleanup_worker.finished.connect(self._cleanup_thread.quit)
        self._cleanup_worker.finished.connect(self._on_cleanup_finished)
        self._cleanup_thread.start()

    def _on_cleanup_finished(self):
        self._btn_run_cleanup.setEnabled(True)
        self._btn_run_cleanup.setText("Run Cleanup")
        self._refresh_setup_tab()

    # ------------------------------------------------------------------
    def refresh_input_label(self):
        if self._ag is None:
            self._lbl_input.setText("Loading…")
            self._btn_run.setEnabled(False)
            return
        p = self._selected_input_path or self._ag.INPUT_PATH
        if p:
            self._lbl_input.setText(str(p))
            self._lbl_input.setStyleSheet("")
        else:
            self._lbl_input.setText(
                "No PDF or TIFF found in plan/  —  use Browse to select one")
            self._lbl_input.setStyleSheet("color: #ff6b6b;")
        self._btn_run.setEnabled(p is not None)

    # ------------------------------------------------------------------
    # Browse slots
    # ------------------------------------------------------------------
    def _browse_input(self):
        if self._ag is None:
            return
        start = (str(self._ag.INPUT_PATH.parent)
                 if self._ag.INPUT_PATH else str(self._ag.PLAN_FOLDER))
        path, _ = QFileDialog.getOpenFileName(
            self, "Select input plan file", start,
            "Plans (*.tif *.tiff *.pdf);;All files (*)"
        )
        if not path:
            return
        p = Path(path)
        self._selected_input_path = p
        self._selected_input_is_pdf = p.suffix.lower() == ".pdf"
        self._ag.INPUT_PATH   = p
        self._ag.INPUT_IS_PDF = self._selected_input_is_pdf
        self.refresh_input_label()

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select output folder", str(self._output_dir)
        )
        if not path:
            return
        self._output_dir = Path(path)
        self._orig_out_dir = self._output_dir
        self._lbl_output.setText(str(self._output_dir))
        # Reload map sources from the new output directory
        self._load_map_sources()
        self._refresh_wms_combo()

    # ------------------------------------------------------------------
    # Map source registry helpers
    # ------------------------------------------------------------------
    @property
    def _map_sources_path(self) -> Path:
        return self._orig_out_dir / "map_sources.json"

    def _load_map_sources(self) -> None:
        """Load (or reload) the map source registry from disk.
        On first run (no file yet) the preset catalog is used and immediately
        saved so the file reflects the starting state.
        """
        try:
            from georef_core.map_sources import load_map_sources, save_map_sources
            path = self._map_sources_path
            self._map_sources = load_map_sources(path)
            # If file was absent, persist the preset defaults now
            if not path.exists():
                try:
                    save_map_sources(self._map_sources, path)
                except Exception:
                    pass
        except Exception:
            self._map_sources = {}

    def _open_geopack_manager(self) -> None:
        """Open the geopack management dialog."""
        ag = self._ag
        if ag is None:
            QMessageBox.warning(self, "Not ready", "Plugin is still loading. Try again in a moment.")
            return
        try:
            from georef_core.geopack_manager import GeopackManager, default_store_dirs
            from pathlib import Path
            mgr = GeopackManager(default_store_dirs(Path(ag._ROOT), Path(ag.OUTPUT_DIR)))
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Could not load geopack manager:\n{exc}")
            return

        dlg = _GeopackManagerDialog(mgr, parent=self)
        dlg.exec_()
        # Reload geopacks and refresh CRS combo after user installs/removes
        if ag and hasattr(ag, "_load_geopacks"):
            ag._load_geopacks()
        self._refresh_crs_combo()

    def _refresh_crs_combo(self) -> None:
        """Populate the CRS dropdown from ag.CRS_PRESETS and restore saved selection."""
        ag = self._ag
        if ag is None or not hasattr(ag, "CRS_PRESETS"):
            return

        saved_key = self._settings.value("crs/preset_key", "epsg_25832", type=str)
        saved_custom = self._settings.value("crs/custom_epsg", "", type=str)

        self._combo_crs.blockSignals(True)
        self._combo_crs.clear()
        restore_idx = 0
        for i, (key, preset) in enumerate(ag.CRS_PRESETS.items()):
            self._combo_crs.addItem(preset.get("label", key), userData=key)
            if key == saved_key:
                restore_idx = i
        self._combo_crs.setCurrentIndex(restore_idx)
        self._combo_crs.blockSignals(False)

        # Restore custom EPSG field
        if saved_key == "custom" and saved_custom:
            self._edit_custom_epsg.setText(saved_custom)
            self._edit_custom_epsg.setVisible(True)

        # Apply saved preset to the backend immediately
        self._apply_crs_preset(saved_key, saved_custom)

    def _on_crs_changed(self, _index: int) -> None:
        """Called when the user selects a different CRS from the dropdown."""
        key = self._combo_crs.currentData()
        is_custom = (key == "custom")
        self._edit_custom_epsg.setVisible(is_custom)
        custom_epsg_str = self._edit_custom_epsg.text().strip() if is_custom else ""
        self._apply_crs_preset(key, custom_epsg_str)

    def _on_custom_epsg_changed(self) -> None:
        """Called when the user finishes editing the custom EPSG field."""
        if self._combo_crs.currentData() == "custom":
            self._apply_crs_preset("custom", self._edit_custom_epsg.text().strip())

    def _apply_crs_preset(self, key: str, custom_epsg_str: str = "") -> None:
        """Push the selected CRS preset into the ag module."""
        ag = self._ag
        if ag is None or not hasattr(ag, "set_active_crs_preset"):
            return
        try:
            if key == "custom":
                epsg = int(custom_epsg_str) if custom_epsg_str.isdigit() else None
                if epsg is None:
                    return  # wait for valid input
                ag.set_active_crs_preset("custom", custom_epsg=epsg)
            else:
                ag.set_active_crs_preset(key)
            # Refresh WMS combo since preset may have changed available map types
            self._refresh_wms_combo()
        except Exception as exc:
            ag.print(f"[!] CRS preset error: {exc}")

    def _refresh_wms_combo(self) -> None:
        """Repopulate the reference-map combo from the current _map_sources registry."""
        sources = self._map_sources

        # Remember currently selected key so we can restore it
        prev_key = self._combo_wms.currentData() if self._combo_wms.count() else None

        self._combo_wms.blockSignals(True)
        self._combo_wms.clear()
        # First entry is always "Auto-detect"
        self._combo_wms.addItem("Auto-detect", None)
        for sid, ms in sources.items():
            label = ms.name if hasattr(ms, "name") else str(sid)
            self._combo_wms.addItem(label, sid)
        self._combo_wms.blockSignals(False)

        # Restore previous selection if it still exists
        if prev_key is not None:
            for i in range(self._combo_wms.count()):
                if self._combo_wms.itemData(i) == prev_key:
                    self._combo_wms.setCurrentIndex(i)
                    break

    def _open_map_sources_manager(self) -> None:
        """Show the Map Sources manager dialog."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Map Sources")
        dlg.setMinimumSize(860, 500)
        layout = QVBoxLayout(dlg)

        lbl_info = QLabel(
            "All sources — including presets — can be removed and re-added.  "
            "Sources are saved to <i>map_sources.json</i> in your output folder."
        )
        lbl_info.setWordWrap(True)
        layout.addWidget(lbl_info)

        # ── table ──────────────────────────────────────────────────────
        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["Name", "WMS URL", "Layer", "Format", "Type"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setAlternatingRowColors(True)
        layout.addWidget(table)

        def _populate_table():
            table.setRowCount(0)
            for sid, ms in self._map_sources.items():
                row = table.rowCount()
                table.insertRow(row)
                name_item = QTableWidgetItem(ms.name)
                name_item.setData(Qt.UserRole, sid)
                table.setItem(row, 0, name_item)
                table.setItem(row, 1, QTableWidgetItem(ms.url))
                layer_display = ms.layer if len(ms.layer) <= 45 else ms.layer[:42] + "…"
                table.setItem(row, 2, QTableWidgetItem(layer_display))
                table.setItem(row, 3, QTableWidgetItem(ms.format))
                type_item = QTableWidgetItem("preset" if ms.builtin else "custom")
                if ms.builtin:
                    type_item.setForeground(QColor("#888888"))
                table.setItem(row, 4, type_item)

        _populate_table()

        # ── button row ─────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_add     = QPushButton("+ Add WMS…")
        btn_restore = QPushButton("↩ Restore Preset…")
        btn_test    = QPushButton("Test Connection")
        btn_remove  = QPushButton("Remove")
        btn_close   = QPushButton("Close")
        btn_row.addWidget(btn_add)
        btn_row.addWidget(btn_restore)
        btn_row.addWidget(btn_test)
        btn_row.addWidget(btn_remove)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        # ── Add WMS form (hidden until user clicks "+ Add WMS…") ───────
        add_group = QGroupBox("New WMS Source")
        add_group.hide()
        add_form = QFormLayout(add_group)
        edit_name    = QLineEdit()
        edit_url     = QLineEdit()
        edit_url.setPlaceholderText("https://example.com/wms")
        edit_layer   = QLineEdit()
        edit_layer.setPlaceholderText("layer_name  (comma-separate multiple layers)")
        combo_fmt    = QComboBox()
        combo_fmt.addItems(["image/jpeg", "image/png"])
        combo_ver    = QComboBox()
        combo_ver.addItems(["1.3.0", "1.1.1"])
        chk_bgcolor  = QCheckBox("White background  (use for PNG / vector maps)")
        edit_desc    = QLineEdit()
        add_form.addRow("Name *:", edit_name)
        add_form.addRow("WMS URL *:", edit_url)
        add_form.addRow("Layer *:", edit_layer)
        add_form.addRow("Format:", combo_fmt)
        add_form.addRow("WMS Version:", combo_ver)
        add_form.addRow("", chk_bgcolor)
        add_form.addRow("Description:", edit_desc)
        save_row = QHBoxLayout()
        btn_save_new  = QPushButton("Save")
        btn_cancel_add = QPushButton("Cancel")
        save_row.addWidget(btn_save_new)
        save_row.addWidget(btn_cancel_add)
        save_row.addStretch()
        add_form.addRow("", save_row)
        layout.addWidget(add_group)

        # ── Connections ────────────────────────────────────────────────
        def _show_add_form():
            edit_name.clear()
            edit_url.clear()
            edit_layer.clear()
            edit_desc.clear()
            chk_bgcolor.setChecked(False)
            add_group.show()
            edit_name.setFocus()

        def _hide_add_form():
            add_group.hide()

        def _save_new_source():
            from georef_core.map_sources import (
                MapSource, add_map_source, save_map_sources, new_source_id
            )
            name  = edit_name.text().strip()
            url   = edit_url.text().strip()
            layer = edit_layer.text().strip()
            if not name or not url or not layer:
                QMessageBox.warning(dlg, "Missing Fields", "Name, WMS URL and Layer are required.")
                return
            # Only allow http/https — reject file://, ftp://, etc.
            if not url.lower().startswith(("http://", "https://")):
                QMessageBox.warning(dlg, "Invalid URL",
                                    "WMS URL must start with http:// or https://")
                return
            ms = MapSource(
                id=          new_source_id(),
                name=        name,
                url=         url,
                layer=       layer,
                format=      combo_fmt.currentText(),
                wms_version= combo_ver.currentText(),
                bgcolor=     "0xFFFFFF" if chk_bgcolor.isChecked() else "",
                description= edit_desc.text().strip(),
                builtin=     False,
            )
            try:
                self._map_sources = add_map_source(self._map_sources, ms)
                save_map_sources(self._map_sources, self._map_sources_path)
                _populate_table()
                _hide_add_form()
                self._refresh_wms_combo()
            except Exception as exc:
                QMessageBox.warning(dlg, "Error", str(exc))

        def _remove_selected():
            from georef_core.map_sources import remove_map_source, save_map_sources
            row = table.currentRow()
            if row < 0:
                QMessageBox.information(dlg, "Remove", "Select a row first.")
                return
            sid = table.item(row, 0).data(Qt.UserRole)
            ms  = self._map_sources.get(sid)
            if ms is None:
                return
            hint = "  It can be re-added via '↩ Restore Preset'." if ms.builtin else ""
            reply = QMessageBox.question(
                dlg, "Remove Source",
                f"Remove '{ms.name}'?{hint}",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
            try:
                self._map_sources = remove_map_source(self._map_sources, sid)
                save_map_sources(self._map_sources, self._map_sources_path)
                _populate_table()
                self._refresh_wms_combo()
            except Exception as exc:
                QMessageBox.warning(dlg, "Error", str(exc))

        def _restore_preset():
            from georef_core.map_sources import (
                get_available_presets, add_map_source, save_map_sources
            )
            available = get_available_presets(self._map_sources)
            if not available:
                QMessageBox.information(dlg, "Restore Preset",
                                        "All presets are already in the registry.")
                return
            # Show a small picker dialog
            picker = QDialog(dlg)
            picker.setWindowTitle("Restore Preset")
            picker.setMinimumWidth(420)
            pl = QVBoxLayout(picker)
            pl.addWidget(QLabel("Select a preset to restore:"))
            combo_presets = QComboBox()
            for p in available:
                combo_presets.addItem(p.name, p.id)
            pl.addWidget(combo_presets)
            pr = QHBoxLayout()
            btn_ok = QPushButton("Add")
            btn_cancel_p = QPushButton("Cancel")
            pr.addWidget(btn_ok)
            pr.addWidget(btn_cancel_p)
            pr.addStretch()
            pl.addLayout(pr)
            btn_ok.clicked.connect(picker.accept)
            btn_cancel_p.clicked.connect(picker.reject)
            if picker.exec_() != QDialog.Accepted:
                return
            preset_id = combo_presets.currentData()
            preset = next((p for p in available if p.id == preset_id), None)
            if preset is None:
                return
            try:
                self._map_sources = add_map_source(self._map_sources, preset)
                save_map_sources(self._map_sources, self._map_sources_path)
                _populate_table()
                self._refresh_wms_combo()
            except Exception as exc:
                QMessageBox.warning(dlg, "Error", str(exc))

        def _test_selected():
            import urllib.request
            import urllib.parse
            row = table.currentRow()
            if row < 0:
                QMessageBox.information(dlg, "Test", "Select a row first.")
                return
            sid = table.item(row, 0).data(Qt.UserRole)
            ms  = self._map_sources.get(sid)
            if not ms:
                return
            # Validate scheme before making any network request
            if not ms.url.lower().startswith(("http://", "https://")):
                QMessageBox.warning(dlg, "Invalid URL",
                                    "Cannot test: URL must start with http:// or https://")
                return
            # Properly append GetCapabilities params even if URL already has a query string
            parsed = urllib.parse.urlparse(ms.url)
            existing_params = urllib.parse.parse_qs(parsed.query)
            caps_params = {**existing_params,
                           "SERVICE": ["WMS"], "REQUEST": ["GetCapabilities"]}
            caps_url = urllib.parse.urlunparse(parsed._replace(
                query=urllib.parse.urlencode(caps_params, doseq=True)
            ))
            try:
                req = urllib.request.Request(
                    caps_url, headers={"User-Agent": "AutoGeoreference/1.0"}
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    code = resp.getcode()
                    ct   = resp.headers.get("Content-Type", "")
                QMessageBox.information(
                    dlg, "Connection OK",
                    f"WMS responded successfully.\nHTTP {code}  —  {ct}",
                )
            except Exception as exc:
                QMessageBox.warning(dlg, "Connection Failed",
                                    f"Could not reach WMS:\n{exc}")

        btn_add.clicked.connect(_show_add_form)
        btn_restore.clicked.connect(_restore_preset)
        btn_cancel_add.clicked.connect(_hide_add_form)
        btn_save_new.clicked.connect(_save_new_source)
        btn_remove.clicked.connect(_remove_selected)
        btn_test.clicked.connect(_test_selected)
        btn_close.clicked.connect(dlg.accept)

        dlg.exec_()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def _run(self):
        if self._ag is None or (self._thread and self._thread.isRunning()):
            return

        try:
            self._ag = importlib.reload(self._ag)
        except Exception as exc:
            self._lbl_status.setText(f"Reload error: {exc}")
            return

        if self._selected_input_path is not None:
            self._ag.INPUT_PATH = self._selected_input_path
            self._ag.INPUT_IS_PDF = bool(self._selected_input_is_pdf)

        # Apply user WMS choice (currentData() stores the source id, None = auto)
        wms_key = self._combo_wms.currentData()
        self._ag.WMS_CONFIG_OVERRIDE = wms_key   # None = auto-detect
        # Inject user-added sources so select_wms_config() can resolve them
        try:
            from georef_core.map_sources import as_wms_configs
            user_sources = {k: v for k, v in self._map_sources.items() if not v.builtin}
            self._ag.WMS_CONFIGS_EXTRA = as_wms_configs(user_sources)
        except Exception:
            pass

        # Apply user scale override (blank = auto-detect)
        _scale_text = self._edit_scale.text().strip().replace(".", "").replace(",", "")
        try:
            _scale_val = int(_scale_text)
            self._ag.SCALE_OVERRIDE = _scale_val if 100 <= _scale_val <= 100_000 else None
        except (ValueError, AttributeError):
            self._ag.SCALE_OVERRIDE = None

        self._ag.USE_QGIS_CANVAS_VISION_CONTEXT = self._chk_canvas_context.isChecked()
        self._ag.STRICT_WMS_VALIDATION = self._chk_strict_wms.isChecked()
        self._ag.ENABLE_OSM_VECTOR_SNAPPING = self._chk_osm_snap.isChecked()
        self._ag.GEOCODE_DEBUG = self._chk_geocode_debug.isChecked()
        self._settings.setValue("vision/use_canvas_context", self._chk_canvas_context.isChecked())
        self._settings.setValue("wms/strict_validation", self._chk_strict_wms.isChecked())
        self._settings.setValue("seed/osm_snap", self._chk_osm_snap.isChecked())
        self._settings.setValue("seed/geocode_debug", self._chk_geocode_debug.isChecked())
        # Persist CRS selection
        _preset_key = self._combo_crs.currentData() or "epsg_25832"
        self._settings.setValue("crs/preset_key", _preset_key)
        if _preset_key == "custom":
            self._settings.setValue("crs/custom_epsg", self._edit_custom_epsg.text().strip())
        if self._chk_canvas_context.isChecked():
            QMessageBox.warning(
                self,
                "Canvas Context Enabled",
                "Canvas image context is enabled. This can bias location inference toward the current viewport instead of the plan. Disable it unless the plan has almost no readable location text.",
            )
        if not self._chk_strict_wms.isChecked():
            QMessageBox.warning(
                self,
                "Strict WMS Disabled",
                "Strict WMS validation is disabled. Weak matches are more likely to be accepted. This is not recommended for production runs.",
            )

        # Save project address to file before run (so derive_auto_seed can read it)
        _addr_text = self._edit_address.text().strip()
        _addr_file = self._orig_out_dir / "project_address.json"
        try:
            _addr_data = {
                "enabled": bool(_addr_text),
                "address": _addr_text,
                "notes": (
                    "Set enabled=true and enter any address, place name, or area description. "
                    "Geocoded via OpenStreetMap Nominatim. Used when QGIS canvas is unavailable."
                ),
            }
            _addr_file.write_text(
                json.dumps(_addr_data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass

        # Intermediates → _work/; only *_georef.tif goes to output_dir
        work_dir = self._output_dir / "_work"
        work_dir.mkdir(parents=True, exist_ok=True)
        self._ag.OUTPUT_DIR           = work_dir
        self._ag.LOG_FILE             = work_dir / "run.log"
        self._ag.MANUAL_SEED_FILE     = self._orig_out_dir / "manual_seed.json"
        self._ag.LAST_RESULT_FILE     = self._orig_out_dir / "last_result.json"
        self._ag.GEOREF_LIBRARY_FILE  = self._orig_out_dir / "georef_library.json"
        self._ag.PROJECT_ADDRESS_FILE = _addr_file

        self._ag.LOG_FILE.write_text("", encoding="utf-8")
        self._log_pos      = 0
        self._current_step = 0
        self._latest_case_bundle_path = None
        self._latest_result_path = None
        self._log.clear()
        self._set_review_buttons_enabled(False)

        self._progress.setRange(0, 0)   # indeterminate / pulse
        self._progress.setValue(0)
        self._lbl_status.setText("Starting…")
        self._btn_run.setEnabled(False)
        self._btn_run.setText("⏳ Running…")
        self._btn_stop.setEnabled(True)

        # Reset cancellation flag so a fresh run is never pre-cancelled
        self._ag._CANCEL_FLAG.clear()

        # Capture canvas on the main thread (canvas.grab() is not thread-safe)
        self._ag.CANVAS_INFO_OVERRIDE = self._ag.get_qgis_canvas_info()
        # Prevent load_in_qgis() from being called inside the worker thread —
        # this dialog handles layer loading on the main thread after the run.
        self._ag.SKIP_AUTO_LAYER_LOAD = True

        run_input_path = self._selected_input_path or self._ag.INPUT_PATH
        run_input_is_pdf = (
            self._selected_input_is_pdf
            if self._selected_input_path is not None
            else self._ag.INPUT_IS_PDF
        )
        self._current_run_input_path = run_input_path
        self._current_run_input_is_pdf = bool(run_input_is_pdf)
        if run_input_path is not None:
            if self._current_run_input_is_pdf:
                self._expected_result_name = f"{Path(run_input_path).stem}_rendered_georef.tif"
            else:
                self._expected_result_name = f"{Path(run_input_path).stem}_georef.tif"
        else:
            self._expected_result_name = None
        self._thread = QThread()
        self._worker = _Worker(
            self._ag,
            input_path=run_input_path,
            input_is_pdf=run_input_is_pdf,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._on_finished)
        self._thread.start()
        self._timer.start(400)

    # ------------------------------------------------------------------
    # Stop (cancel running job)
    # ------------------------------------------------------------------
    def _stop(self):
        self._shutdown_adjustment_tool()
        self._hide_adjustment_tab()
        if self._ag is not None:
            self._ag._CANCEL_FLAG.set()
        self._btn_stop.setEnabled(False)
        self._btn_stop.setText("Stopping…")
        self._lbl_status.setText("Cancelling — waiting for current tile…")

    # ------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------
    def _on_finished(self):
        self._timer.stop()
        self._poll_log()   # final flush

        # Restore OUTPUT_DIR to the original folder so that if the dialog is
        # re-opened (or the module re-imported), _deferred_import doesn't see
        # the _work subdir and create nested _work/_work folders.
        self._ag.OUTPUT_DIR = self._orig_out_dir

        work_dir  = self._output_dir / "_work"
        delivered = []
        expected_name = self._expected_result_name
        if expected_name:
            tif = work_dir / expected_name
            if tif.exists():
                dest = self._output_dir / tif.name
                delivered.append((tif, dest))
                meta_src = work_dir / f"{tif.stem}_meta.json"
                meta_dest = self._output_dir / meta_src.name if meta_src.exists() else None
                self._pending_result_work_path = tif
                self._pending_result_dest_path = dest
                self._pending_result_meta_src = meta_src if meta_src.exists() else None
                self._pending_result_meta_dest = meta_dest
        case_name = self._expected_case_bundle_name()
        if case_name:
            case_src = work_dir / case_name
            if case_src.exists():
                case_dest = self._output_dir / case_src.name
                shutil.copy2(str(case_src), str(case_dest))
                self._latest_case_bundle_path = case_dest
        if not delivered:
            candidates = sorted(
                work_dir.glob("*_georef.tif"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                tif = candidates[0]
                dest = self._output_dir / tif.name
                delivered.append((tif, dest))
                self._pending_result_work_path = tif
                self._pending_result_dest_path = dest
                meta_src = work_dir / f"{tif.stem}_meta.json"
                self._pending_result_meta_src = meta_src if meta_src.exists() else None
                self._pending_result_meta_dest = (
                    self._output_dir / meta_src.name if meta_src.exists() else None
                )
        if self._latest_case_bundle_path is None:
            case_candidates = sorted(
                work_dir.glob("*_case_bundle.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if case_candidates:
                case_dest = self._output_dir / case_candidates[0].name
                shutil.copy2(str(case_candidates[0]), str(case_dest))
                self._latest_case_bundle_path = case_dest

        if delivered:
            self._progress.setRange(0, _TOTAL_STEPS)
            self._progress.setValue(_TOTAL_STEPS)
            preview_path, output_path = delivered[0]
            self._lbl_status.setText(f"✅  Done — preview ready ({preview_path.name})")
            self._append_log(
                f"\n[i] Preview result ready: {preview_path}\n"
                f"[i] Final output will be written after adjustment: {output_path}\n"
            )
            if GeorefAdjustmentTool is None:
                finalized = self._publish_pending_result()
                if finalized is not None:
                    self._latest_result_path = finalized
                    self._load_result_layer(finalized)
                    self._append_log(f"\n✅  Result: {finalized}\n")
                    self._lbl_status.setText(f"✅  Done — {finalized.name}")
                else:
                    self._append_log("[~] Could not finalize result without adjustment tool.\n")
            else:
                # Auto-launch the adjustment tool so the user can verify / correct
                # the initial placement before the final output is written.
                def _zoom_then_adjust():
                    try:
                        from osgeo import gdal
                        from qgis.core import QgsRectangle
                        from qgis.utils import iface as _iface
                        # Derive the extent directly from GDAL — avoids creating a
                        # temporary QgsRasterLayer whose C++ object would be
                        # destroyed while the canvas renderer is still using it.
                        ds = gdal.Open(str(preview_path))
                        if ds is not None:
                            gt = ds.GetGeoTransform()
                            W2, H2 = ds.RasterXSize, ds.RasterYSize
                            ds = None  # close immediately
                            corners = [
                                (gt[0] + c * gt[1] + r * gt[2],
                                 gt[3] + c * gt[4] + r * gt[5])
                                for c, r in [(0, 0), (W2, 0), (W2, H2), (0, H2)]
                            ]
                            xs = [p[0] for p in corners]
                            ys = [p[1] for p in corners]
                            ext = QgsRectangle(min(xs), min(ys), max(xs), max(ys))
                            _iface.mapCanvas().setExtent(ext)
                            _iface.mapCanvas().refresh()
                    except Exception as exc:
                        self._append_log(f"[~] Could not zoom to result: {exc}\n")
                    QTimer.singleShot(100, lambda: self._launch_adjustment_tool())
                QTimer.singleShot(50, _zoom_then_adjust)
        else:
            self._progress.setRange(0, _TOTAL_STEPS)
            self._progress.setValue(0)
            self._lbl_status.setText("⚠️  No georef TIFF produced — check log")
            self._append_log("\n⚠️  Run finished but no *_georef.tif was produced.\n")

        self._btn_run.setEnabled(True)
        self._btn_run.setText("▶  Run")
        self._btn_stop.setEnabled(False)
        self._btn_stop.setText("⏹  Stop")
        self._thread = None
        self._worker = None

        self._set_review_buttons_enabled(self._latest_case_bundle_path is not None)
        self._log_library_accuracy()
        if self._close_when_finished:
            self._close_when_finished = False
            self.close()

    # ------------------------------------------------------------------
    # Library accuracy check
    # ------------------------------------------------------------------
    def _log_library_accuracy(self):
        """Compare AI pipeline result to library ground truth and log the delta."""
        if self._current_run_input_path is None:
            return
        try:
            from georef_core.library import load_library, find_library_match
        except Exception:
            return
        try:
            lib = load_library(self._library_path)
            stem = Path(self._current_run_input_path).stem
            entry = find_library_match(stem, lib, plan_path=Path(self._current_run_input_path))
            if entry is None:
                return
            # Read AI result centre from the pipeline TIFF
            result_path = self._pending_result_work_path or self._latest_result_path
            if result_path is None:
                return
            georef = self._read_tiff_georef(Path(result_path))
            if georef is None:
                return
            _gt, ai_ce, ai_cn, _W, _H = georef
            delta_m = ((ai_ce - entry.center_easting) ** 2
                       + (ai_cn - entry.center_northing) ** 2) ** 0.5
            self._append_log(
                f"\n[Library] Accuracy check vs verified entry ({entry.source}, {entry.reviewed_at[:10]}):\n"
                f"  AI result : E={ai_ce:.0f}  N={ai_cn:.0f}\n"
                f"  Library   : E={entry.center_easting:.0f}  N={entry.center_northing:.0f}\n"
                f"  Delta     : {delta_m:.1f} m\n"
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Log polling + status parsing
    # ------------------------------------------------------------------
    def _poll_log(self):
        try:
            text = self._ag.LOG_FILE.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return
        new = text[self._log_pos:]
        if not new:
            return
        self._log_pos = len(text)
        self._append_log(new)

        # Update progress bar from key log phrases
        for line in new.splitlines():
            for pattern, step, label in _STEPS:
                if re.search(pattern, line, re.IGNORECASE):
                    if step > self._current_step:
                        self._current_step = step
                        # Switch from indeterminate to determinate once we have a step
                        if self._progress.maximum() == 0:
                            self._progress.setRange(0, _TOTAL_STEPS)
                        self._progress.setValue(step)
                        self._lbl_status.setText(
                            f"Step {step}/{_TOTAL_STEPS}  –  {label}")
                    break

    def _append_log(self, text: str):
        cursor = self._log.textCursor()
        cursor.movePosition(QTextCursor.End)
        self._log.setTextCursor(cursor)
        self._log.insertPlainText(text)
        self._log.ensureCursorVisible()

    # ------------------------------------------------------------------
    def _open_output(self):
        import subprocess
        subprocess.Popen(["explorer", str(self._output_dir)])

    def _load_result_layer(self, path: Path):
        try:
            from qgis.core import QgsRasterLayer, QgsProject
            from qgis.utils import iface as _iface
            src_norm = str(path).replace("\\", "/")
            project = QgsProject.instance()
            # Remove any existing layer pointing at the same file so we don't
            # accumulate stale copies after repeated runs / adjustments.
            for layer in list(project.mapLayers().values()):
                try:
                    if getattr(layer, "source", lambda: "")().replace("\\", "/") == src_norm:
                        project.removeMapLayer(layer.id())
                except Exception:
                    pass
            lyr = QgsRasterLayer(src_norm, path.stem)
            if not lyr.isValid():
                self._append_log(
                    f"[!] Could not load result layer — QGIS reports the file as invalid:\n"
                    f"    {path}\n"
                    f"    (Check the file exists and the CRS is recognised.)\n"
                )
                return
            # addMapLayer(lyr) with default addToLegend=True is the simplest and
            # most robust way to get the layer into both the registry and the
            # Layers panel in one step.
            project.addMapLayer(lyr)
            ext = lyr.extent()
            QTimer.singleShot(200, lambda e=ext: (
                _iface.mapCanvas().setExtent(e),
                _iface.mapCanvas().refresh()
            ))
        except Exception as exc:
            self._append_log(f"[!] Layer load exception: {exc}\n")

    def _expected_case_bundle_name(self):
        if self._current_run_input_path is None:
            return None
        return f"{Path(self._current_run_input_path).stem}_case_bundle.json"

    def _set_review_buttons_enabled(self, enabled: bool):
        self._btn_accept.setEnabled(enabled)
        self._btn_review.setEnabled(enabled)
        self._btn_adjust.setEnabled(
            enabled and self._latest_result_path is not None
            and GeorefAdjustmentTool is not None
        )

    def _show_adjustment_tab(self):
        layout = getattr(self, "_adjust_content_layout", None)
        if layout is None:
            return None
        self._tabs.setTabEnabled(1, True)
        self._tabs.setCurrentIndex(1)
        if self._adj_panel_widget is None and _AdjustmentPanel is not None:
            self._adj_panel_widget = _AdjustmentPanel(self._adjust_content)
            self._adj_panel_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            layout.insertWidget(0, self._adj_panel_widget)
            layout.setStretchFactor(self._adj_panel_widget, 1)
        if self._adjust_placeholder is not None:
            self._adjust_placeholder.hide()
        if self._adj_panel_widget is not None:
            self._adj_panel_widget.show()
        return self._adj_panel_widget

    def _hide_adjustment_tab(self):
        self._tabs.setCurrentIndex(0)
        self._tabs.setTabEnabled(1, False)
        if self._adj_panel_widget is not None:
            self._adj_panel_widget.hide()
        if self._adjust_placeholder is not None:
            self._adjust_placeholder.show()

    def _case_bundle_path(self):
        if self._latest_case_bundle_path and Path(self._latest_case_bundle_path).exists():
            return Path(self._latest_case_bundle_path)
        expected = self._expected_case_bundle_name()
        if expected:
            candidate = self._output_dir / expected
            if candidate.exists():
                self._latest_case_bundle_path = candidate
                return candidate
        return None

    def _current_case_review_status(self):
        case_path = self._case_bundle_path()
        if case_path is None:
            return None
        try:
            payload = json.loads(case_path.read_text(encoding="utf-8"))
            review = payload.get("review") or {}
            return review.get("status")
        except Exception:
            return None

    def _save_review(self, review: dict):
        case_path = self._case_bundle_path()
        if case_path is None:
            QMessageBox.warning(self, "No Case Bundle", "No case bundle was found for the latest run.")
            return
        try:
            from georef_core.persistence import update_case_bundle_review
        except Exception as exc:
            QMessageBox.warning(self, "Review Error", f"Could not load review persistence: {exc}")
            return
        review.setdefault("reviewer", self._edit_reviewer.text().strip() or None)
        update_case_bundle_review(case_path, review)
        self._append_log(f"[i] Review saved -> {case_path.name}\n")

    # ------------------------------------------------------------------
    # Library helpers
    # ------------------------------------------------------------------
    @property
    def _library_path(self) -> Path:
        return self._orig_out_dir / "georef_library.json"

    def _read_tiff_georef(self, tiff_path) -> tuple | None:
        """Return (geotransform_list, center_e, center_n, W, H) or None."""
        try:
            from osgeo import gdal
            ds = gdal.Open(str(tiff_path))
            if ds is None:
                return None
            gt = list(ds.GetGeoTransform())
            W = ds.RasterXSize
            H = ds.RasterYSize
            ds = None
            ce = gt[0] + gt[1] * (W / 2.0) + gt[2] * (H / 2.0)
            cn = gt[3] + gt[4] * (W / 2.0) + gt[5] * (H / 2.0)
            return (gt, ce, cn, W, H)
        except Exception:
            return None

    def _read_bundle_context(self) -> dict:
        """Extract OCR hints, quality, epsg, scale, and validation confidence from the current case bundle."""
        result = {
            "ocr_hints": {}, "quality": {}, "epsg": (self._ag.TARGET_EPSG if self._ag else 25832),
            "scale_denominator": None, "validation_confidence": 0.85,
        }
        try:
            case_path = self._case_bundle_path()
            if case_path is None:
                return result
            payload = json.loads(case_path.read_text(encoding="utf-8"))
            arts = payload.get("artifacts") or {}
            hints = arts.get("structured_hints") or {}
            georef = arts.get("georef_result") or {}
            vision = arts.get("vision") or {}
            result["ocr_hints"] = {
                "site_city":   hints.get("site_city"),
                "site_street": hints.get("site_street"),
                "road_codes":  hints.get("road_codes") or [],
                "parcel_refs": hints.get("parcel_refs") or [],
            }
            result["quality"]  = georef.get("quality") or {}
            result["epsg"]     = int(georef.get("epsg") or 0) or (self._ag.TARGET_EPSG if self._ag else 25832)
            ov = vision.get("overview") or {}
            _sc = ov.get("scale")
            result["scale_denominator"] = int(_sc) if _sc else None
            # Best accepted validation confidence from decision_engine (range [0,1])
            for v in (arts.get("validations") or []):
                if v.get("accepted") and v.get("confidence") is not None:
                    result["validation_confidence"] = max(
                        result["validation_confidence"], float(v["confidence"])
                    )
        except Exception:
            pass
        return result

    def _write_library_entry(
        self,
        *,
        source: str,
        center_easting: float,
        center_northing: float,
        geotransform: list,
        confidence: float,
        pipeline_center_easting: float | None = None,
        pipeline_center_northing: float | None = None,
    ):
        """Write a verified placement into georef_library.json and log the result."""
        if self._current_run_input_path is None:
            return
        try:
            from georef_core.library import update_library
        except Exception as exc:
            self._append_log(f"[~] Library update skipped (import error): {exc}\n")
            return
        ctx = self._read_bundle_context()
        try:
            entry = update_library(
                library_path=self._library_path,
                plan_stem=Path(self._current_run_input_path).stem,
                plan_path=Path(self._current_run_input_path),
                center_easting=center_easting,
                center_northing=center_northing,
                geotransform=geotransform,
                epsg=ctx["epsg"],
                scale_denominator=ctx["scale_denominator"],
                source=source,
                confidence=confidence,
                reviewer=self._edit_reviewer.text().strip() or None,
                ocr_hints=ctx["ocr_hints"],
                quality=ctx["quality"],
                pipeline_center_easting=pipeline_center_easting,
                pipeline_center_northing=pipeline_center_northing,
            )
            delta_str = (
                f"  delta from pipeline: {entry.delta_from_pipeline_m:.1f} m"
                if entry.delta_from_pipeline_m is not None and entry.delta_from_pipeline_m > 0
                else ""
            )
            self._append_log(
                f"[✓] Library updated: {entry.plan_stem}"
                f"  E={entry.center_easting:.0f}  N={entry.center_northing:.0f}"
                f"  ({source}){delta_str}\n"
            )
        except Exception as exc:
            self._append_log(f"[~] Library update failed: {exc}\n")

    def _mark_accepted(self):
        self._save_review({
            "status": "accepted",
            "reason": "operator_accepted",
            "metadata": {},
        })
        # Write accepted (unmodified) pipeline result to library as ground truth.
        result_path = (
            Path(self._pending_result_work_path)
            if self._pending_result_work_path
            else (Path(self._latest_result_path) if self._latest_result_path else None)
        )
        if result_path and result_path.exists():
            georef = self._read_tiff_georef(result_path)
            if georef:
                gt, ce, cn, _W, _H = georef
                ctx = self._read_bundle_context()
                self._write_library_entry(
                    source="accepted",
                    center_easting=ce,
                    center_northing=cn,
                    geotransform=gt,
                    confidence=ctx.get("validation_confidence", 0.85),
                )

    def _mark_review_required(self):
        self._save_review({
            "status": "review_required",
            "reason": "operator_flagged",
            "metadata": {},
        })

    def _export_training_dataset(self):
        try:
            from georef_core.training_export import (
                export_review_label_datasets,
                export_training_dataset,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Export Error", f"Could not load exporter: {exc}")
            return
        out_path = self._output_dir / "training_dataset.jsonl"
        candidate_labels_path = self._output_dir / "candidate_labels.jsonl"
        transform_labels_path = self._output_dir / "transform_labels.jsonl"
        try:
            export_training_dataset(
                case_dir=self._output_dir,
                output_path=out_path,
                include_unreviewed=False,
            )
            export_review_label_datasets(
                case_dir=self._output_dir,
                candidate_output_path=candidate_labels_path,
                transform_output_path=transform_labels_path,
                include_unreviewed=False,
            )
            self._append_log(
                f"[i] Training dataset exported -> {out_path}\n"
                f"[i] Candidate labels exported -> {candidate_labels_path}\n"
                f"[i] Transform labels exported -> {transform_labels_path}\n"
            )
            QMessageBox.information(
                self,
                "Export Complete",
                "Review datasets written to:\n"
                f"{out_path}\n"
                f"{candidate_labels_path}\n"
                f"{transform_labels_path}",
            )
        except Exception as exc:
            QMessageBox.warning(self, "Export Error", f"Could not export training dataset: {exc}")

    def _retrain_ranker(self):
        """
        Retrain the candidate LinearRanker from the exported training dataset.

        Workflow:
          1. Load training_dataset.jsonl from the output directory.
          2. Check that at least 5 labelled examples exist.
          3. Report baseline accuracy (heuristic ranking, no model).
          4. Train a new LinearRanker via stochastic gradient descent.
          5. Report trained top-1 accuracy on the same dataset.
          6. Save the model to output/candidate_ranker_model.json.
        """
        try:
            from georef_core.ranker_training import (
                load_ranking_examples,
                train_linear_ranker,
                evaluate_ranker,
            )
            from georef_core.ranker import save_ranker_model, default_ranker_model_path
        except Exception as exc:
            QMessageBox.warning(self, "Import Error", f"Could not load ranker modules:\n{exc}")
            return

        dataset_path = self._output_dir / "training_dataset.jsonl"
        if not dataset_path.exists():
            QMessageBox.warning(
                self,
                "No Dataset",
                "training_dataset.jsonl not found.\n\n"
                "Click 'Export Dataset' first to generate it from reviewed case bundles.",
            )
            return

        try:
            examples = load_ranking_examples(dataset_path)
        except Exception as exc:
            QMessageBox.warning(self, "Load Error", f"Could not load training examples:\n{exc}")
            return

        if len(examples) < 5:
            QMessageBox.warning(
                self,
                "Too Few Examples",
                f"Only {len(examples)} labelled example(s) found.\n"
                "At least 5 are required to train a meaningful model.\n\n"
                "Review more case bundles (Accept / Adjust Placement) and export again.",
            )
            return

        self._append_log(
            f"[i] Retraining ranker on {len(examples)} examples "
            f"from {dataset_path.name}...\n"
        )

        # Split into train / held-out when we have enough cases.
        # Group by case_id first so we split whole cases, not individual
        # candidate rows (which would leak data across the split boundary).
        import random as _random
        _by_case: dict[str, list] = {}
        for ex in examples:
            _by_case.setdefault(ex.case_id, []).append(ex)
        _case_ids = list(_by_case.keys())
        _n_cases  = len(_case_ids)
        _held_out_examples: list = []
        _split_note = ""
        if _n_cases >= 10:
            _random.seed(0)
            _random.shuffle(_case_ids)
            _split = max(1, _n_cases // 5)   # 20 % held out
            _held_ids = set(_case_ids[:_split])
            _train_examples = [ex for ex in examples if ex.case_id not in _held_ids]
            _held_out_examples = [ex for ex in examples if ex.case_id in _held_ids]
            _split_note = f" (80/20 split — {_n_cases - _split} train / {_split} held-out)"
        else:
            _train_examples = examples
            _split_note = f" (train-only — accuracy is in-sample; need ≥10 cases for held-out eval)"

        try:
            baseline = evaluate_ranker(_train_examples, ranker=None)
            ranker   = train_linear_ranker(_train_examples)
            trained  = evaluate_ranker(_train_examples, ranker=ranker)
            held_out = evaluate_ranker(_held_out_examples, ranker=ranker) if _held_out_examples else None
        except Exception as exc:
            QMessageBox.warning(self, "Training Error", f"Training failed:\n{exc}")
            return

        model_path = default_ranker_model_path()
        try:
            save_ranker_model(ranker, model_path)
        except Exception as exc:
            QMessageBox.warning(self, "Save Error", f"Could not save model:\n{exc}")
            return

        n_cases     = int(trained.get("cases", 0))
        base_acc    = float(baseline.get("top1_accuracy", 0.0))
        trained_acc = float(trained.get("top1_accuracy", 0.0))
        delta       = trained_acc - base_acc
        held_line   = (
            f"   Held-out top-1: {held_out['top1_accuracy']:.1%} "
            f"({int(held_out['cases'])} cases)\n"
            if held_out else ""
        )

        self._append_log(
            f"[✓] Ranker retrained{_split_note}.\n"
            f"    Cases: {n_cases}   "
            f"Baseline top-1: {base_acc:.1%}   "
            f"Trained top-1: {trained_acc:.1%}   "
            f"Delta: {delta:+.1%}\n"
            + held_line +
            f"    Model saved -> {model_path}\n"
        )
        held_msg = (
            f"  Held-out top-1 accuracy : {held_out['top1_accuracy']:.1%} "
            f"({int(held_out['cases'])} cases)\n"
            if held_out else
            f"  (In-sample accuracy only — collect {max(0, 10 - _n_cases)} more reviewed cases for held-out eval)\n"
        )
        QMessageBox.information(
            self,
            "Retrain Complete",
            f"Ranker retrained on {n_cases} case(s).\n\n"
            f"  Baseline top-1 accuracy : {base_acc:.1%}\n"
            f"  Trained  top-1 accuracy : {trained_acc:.1%}  ({delta:+.1%})\n"
            + held_msg +
            f"\nModel saved to:\n{model_path}",
        )

    # ------------------------------------------------------------------
    # Interactive adjustment tool
    # ------------------------------------------------------------------
    def _publish_pending_result(self):
        work_path = self._pending_result_work_path
        dest_path = self._pending_result_dest_path
        if work_path is None or dest_path is None:
            self._set_review_buttons_enabled(self._latest_case_bundle_path is not None)
            return self._latest_result_path
        try:
            shutil.copy2(str(work_path), str(dest_path))
            if self._pending_result_meta_src and self._pending_result_meta_dest:
                shutil.copy2(str(self._pending_result_meta_src), str(self._pending_result_meta_dest))
        except Exception as exc:
            self._append_log(f"[~] Could not finalize preview result: {exc}\n")
            return None
        self._latest_result_path = dest_path
        self._pending_result_work_path = None
        self._pending_result_dest_path = None
        self._pending_result_meta_src = None
        self._pending_result_meta_dest = None
        self._set_review_buttons_enabled(self._latest_case_bundle_path is not None)
        return dest_path

    def _launch_adjustment_tool(self):
        if GeorefAdjustmentTool is None:
            QMessageBox.warning(
                self, "Not Available",
                "adjustment_tool.py could not be loaded.\nCheck for import errors.",
            )
            return
        source_path = self._pending_result_work_path or self._latest_result_path
        if source_path is None or not Path(source_path).exists():
            QMessageBox.warning(
                self, "No Result",
                "No georeferenced TIFF is available. Run the georeferencer first.",
            )
            return

        try:
            from osgeo import gdal
            ds = gdal.Open(str(source_path))
            if ds is None:
                raise FileNotFoundError(f"GDAL could not open: {source_path}")
            gt = ds.GetGeoTransform()
            W  = ds.RasterXSize
            H  = ds.RasterYSize
            ds = None   # close immediately — release Windows file lock
        except Exception as exc:
            QMessageBox.warning(self, "GDAL Error", f"Could not read TIFF metadata:\n{exc}")
            return

        # Stash for use in _apply_adjustment
        self._W_adj       = W
        self._H_adj       = H
        self._orig_adj_gt = tuple(gt)

        from qgis.utils import iface as _iface
        canvas = _iface.mapCanvas()
        self._shutdown_adjustment_tool()
        try:
            from qgis.core import QgsProject
            project = QgsProject.instance()
            hidden_sources = {
                str(Path(source_path)).replace("\\", "/"),
            }
            if self._latest_result_path:
                hidden_sources.add(str(Path(self._latest_result_path)).replace("\\", "/"))
            if self._pending_result_dest_path:
                hidden_sources.add(str(Path(self._pending_result_dest_path)).replace("\\", "/"))
            for layer in list(project.mapLayers().values()):
                try:
                    src = getattr(layer, "source", lambda: "")().replace("\\", "/")
                    if src in hidden_sources:
                        project.removeMapLayer(layer.id())
                except Exception:
                    pass
        except Exception:
            pass

        # Remember the previous map tool so we can restore it on cancel
        self._adj_prev_tool = canvas.mapTool()

        adj_panel = self._show_adjustment_tab()
        self._adj_tool = GeorefAdjustmentTool(canvas, source_path, gt, W, H, panel=adj_panel)
        self._adj_tool.accepted.connect(self._apply_adjustment)
        self._adj_tool.cancelled.connect(self._cancel_adjustment)
        self._adj_tool.path_changed.connect(self._on_adj_path_changed)
        canvas.setMapTool(self._adj_tool)

        self._append_log(
            "[i] Adjustment tool activated.\n"
            "    Left-drag: translate  |  green handle / right-drag: rotate  |  Shift+wheel: fine rotation\n"
            "    Use the Adjustment tab in this dialog for typed center, absolute rotation, scale, opacity, and Fast/Full TIFF display.\n"
            "    Click ✓ Accept in the panel to apply, or Cancel to discard.\n"
        )

    def _on_adj_path_changed(self, new_path: str):
        """Called when GCP warp inside the adjustment tool replaces the working file."""
        try:
            from osgeo import gdal
            ds = gdal.Open(new_path)
            if ds is None:
                return
            gt = ds.GetGeoTransform()
            W  = ds.RasterXSize
            H  = ds.RasterYSize
            ds = None
        except Exception:
            return
        # Point the pending work path at the warped file so _publish_pending_result
        # copies the correct (warped) pixels to the output destination.
        self._pending_result_work_path = Path(new_path)
        self._W_adj       = W
        self._H_adj       = H
        self._orig_adj_gt = tuple(gt)

    def _cancel_adjustment(self):
        self._shutdown_adjustment_tool()
        self._hide_adjustment_tab()
        finalized = self._publish_pending_result()
        if finalized is not None:
            self._load_result_layer(finalized)
            self._append_log(f"[i] Preview accepted without manual changes -> {finalized.name}\n")
        self._append_log("[i] Adjustment panel closed.\n")

    def _apply_adjustment(self, new_gt: tuple):
        import math
        from qgis.core import QgsProject

        self._shutdown_adjustment_tool()
        self._hide_adjustment_tab()

        tiff_path = Path(self._publish_pending_result() or self._latest_result_path)
        if not tiff_path.exists():
            QMessageBox.warning(self, "Write Error", f"Adjusted output path does not exist:\n{tiff_path}")
            return

        # Step 1: Remove QGIS layer to release Windows file lock
        project  = QgsProject.instance()
        src_norm = str(tiff_path).replace("\\", "/")
        for layer in list(project.mapLayers().values()):
            try:
                if getattr(layer, "source", lambda: "")().replace("\\", "/") == src_norm:
                    project.removeMapLayer(layer.id())
            except Exception:
                pass

        # Defer the actual write to the next event-loop tick so Qt fully
        # processes the layer-removal signal before we open the file.
        QTimer.singleShot(0, lambda: self._write_adjusted_gt(tiff_path, new_gt))

    def _write_adjusted_gt(self, tiff_path: Path, new_gt: tuple):
        import math
        from osgeo import gdal

        W = self._W_adj
        H = self._H_adj
        orig_gt = self._orig_adj_gt

        def _center(gt):
            return (
                gt[0] + (W / 2.0) * gt[1] + (H / 2.0) * gt[2],
                gt[3] + (W / 2.0) * gt[4] + (H / 2.0) * gt[5],
            )
        def _rot(gt):
            return math.degrees(math.atan2(-gt[2], gt[1]))

        # Write new geotransform in-place
        try:
            ds = gdal.Open(str(tiff_path), gdal.GA_Update)
            if ds is None:
                raise RuntimeError(f"gdal.Open(GA_Update) returned None for {tiff_path}")
            ds.SetGeoTransform(new_gt)
            ds.FlushCache()
            ds = None
        except Exception as exc:
            QMessageBox.warning(
                self, "Write Error",
                f"Could not write adjusted geotransform:\n{exc}\n\nThe file was not modified.",
            )
            self._load_result_layer(tiff_path)
            return

        # Compute deltas for logging and seed
        orig_ce, orig_cn = _center(orig_gt)
        new_ce,  new_cn  = _center(new_gt)
        delta_e   = new_ce  - orig_ce
        delta_n   = new_cn  - orig_cn
        delta_rot = _rot(new_gt) - _rot(orig_gt)

        self._append_log(
            f"[✓] Adjustment written to {tiff_path.name}\n"
            f"    ΔE={delta_e:+.1f} m  ΔN={delta_n:+.1f} m  Δrot={delta_rot:+.3f}°\n"
            f"    New centre: E={new_ce:.0f}  N={new_cn:.0f}\n"
        )

        # Save correction to case bundle (feeds training data)
        self._save_review({
            "status": "corrected",
            "reason": "interactive_adjustment",
            "corrected_center_easting":  round(new_ce,  2),
            "corrected_center_northing": round(new_cn,  2),
            "metadata": {
                "source":                  "interactive_adjustment_tool",
                "original_geotransform":   list(orig_gt),
                "corrected_geotransform":  list(new_gt),
                "delta_easting_m":         round(delta_e,   2),
                "delta_northing_m":        round(delta_n,   2),
                "delta_rotation_deg":      round(delta_rot, 4),
                "training_target": {
                    "center_easting":  round(new_ce, 2),
                    "center_northing": round(new_cn, 2),
                },
            },
        })

        # Record corrected placement as verified ground truth in the library.
        # orig_ce/orig_cn = uncorrected pipeline result; new_ce/new_cn = verified.
        self._write_library_entry(
            source="interactive_adjustment",
            center_easting=new_ce,
            center_northing=new_cn,
            geotransform=list(new_gt),
            confidence=0.97,
            pipeline_center_easting=orig_ce,
            pipeline_center_northing=orig_cn,
        )

        # Reload the corrected layer into QGIS
        self._load_result_layer(tiff_path)
        self._set_review_buttons_enabled(self._latest_case_bundle_path is not None)

    def _shutdown_adjustment_tool(self):
        from qgis.utils import iface as _iface

        canvas = _iface.mapCanvas()
        if self._adj_tool is not None:
            try:
                if canvas.mapTool() is self._adj_tool:
                    canvas.unsetMapTool(self._adj_tool)
                else:
                    self._adj_tool.deactivate()
            except Exception:
                pass
            self._adj_tool = None
        if self._adj_prev_tool is not None:
            try:
                if canvas.mapTool() is not self._adj_prev_tool:
                    canvas.setMapTool(self._adj_prev_tool)
            except Exception:
                pass
            self._adj_prev_tool = None

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        self._shutdown_adjustment_tool()
        self._hide_adjustment_tab()
        if self._thread and self._thread.isRunning():
            self._close_when_finished = True
            if self._ag is not None:
                self._ag._CANCEL_FLAG.set()
            event.ignore()
        else:
            event.accept()


class _GeopackManagerDialog(QDialog):
    """Simple dialog for browsing, installing and removing geopacks."""

    def __init__(self, mgr, parent=None):
        super().__init__(parent)
        self._mgr = mgr
        self.setWindowTitle("Geopack Manager")
        self.setMinimumWidth(640)
        self.setMinimumHeight(400)
        self._build_ui()
        self._refresh()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 12)

        desc = QLabel(
            "Geopacks are <b>.geopack.json</b> files that add regional CRS presets, "
            "WMS services and city lookup tables to the georeferencer."
        )
        desc.setWordWrap(True)
        root.addWidget(desc)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["Name", "Version", "ID", "Presets", "Source"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        root.addWidget(self._table)

        # Description area
        self._lbl_desc = QLabel("")
        self._lbl_desc.setWordWrap(True)
        self._lbl_desc.setStyleSheet("color: #555;")
        root.addWidget(self._lbl_desc)
        self._table.currentCellChanged.connect(self._on_row_changed)

        # Buttons
        btn_row = QHBoxLayout()
        btn_install = QPushButton("Install…")
        btn_install.setToolTip("Browse for a .geopack.json file and install it")
        btn_install.clicked.connect(self._install)
        btn_row.addWidget(btn_install)

        self._btn_remove = QPushButton("Remove")
        self._btn_remove.setToolTip("Remove the selected user-installed geopack")
        self._btn_remove.setEnabled(False)
        self._btn_remove.clicked.connect(self._remove)
        btn_row.addWidget(self._btn_remove)

        btn_row.addStretch()

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        root.addLayout(btn_row)

    def _refresh(self):
        self._packs = self._mgr.discover_all()
        self._table.setRowCount(len(self._packs))
        for row, pack in enumerate(self._packs):
            preset_count = len(pack.get("crs_presets") or {})
            source = "built-in" if pack.get("_builtin") else "user-installed"
            cells = [
                pack.get("name", ""),
                str(pack.get("version", "")),
                str(pack.get("id", "")),
                str(preset_count),
                source,
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                self._table.setItem(row, col, item)

    def _on_row_changed(self, row, _col, _prev_row, _prev_col):
        if 0 <= row < len(self._packs):
            pack = self._packs[row]
            self._lbl_desc.setText(pack.get("description", ""))
            self._btn_remove.setEnabled(not pack.get("_builtin", True))
        else:
            self._lbl_desc.setText("")
            self._btn_remove.setEnabled(False)

    def _install(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Install Geopack", "",
            "Geopack files (*.geopack.json);;JSON files (*.json);;All files (*)"
        )
        if not path:
            return
        try:
            pack = self._mgr.install(Path(path))
            QMessageBox.information(
                self, "Installed",
                f"Geopack '{pack.get('name', pack['id'])}' installed successfully.\n"
                "Close this dialog to refresh the Projection dropdown."
            )
            self._refresh()
        except Exception as exc:
            QMessageBox.critical(self, "Install failed", str(exc))

    def _remove(self):
        row = self._table.currentRow()
        if row < 0 or row >= len(self._packs):
            return
        pack = self._packs[row]
        if pack.get("_builtin"):
            QMessageBox.warning(self, "Cannot remove", "Built-in geopacks cannot be removed.")
            return
        name = pack.get("name", pack.get("id", "?"))
        if QMessageBox.question(
            self, "Remove geopack",
            f"Remove '{name}'?\nThe CRS presets it provides will no longer be available.",
            QMessageBox.Yes | QMessageBox.No
        ) == QMessageBox.Yes:
            ok = self._mgr.uninstall(str(pack["id"]))
            if ok:
                self._refresh()
            else:
                QMessageBox.warning(self, "Not found", "Could not find the geopack file to remove.")
