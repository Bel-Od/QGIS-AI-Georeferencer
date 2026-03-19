"""
plugin.py – QGIS plugin entry point for Auto Georeferencer.
Creates a dedicated "Auto Georef" toolbar with one button, and adds an
entry under Plugins menu so it is always findable.
"""
from pathlib import Path

from qgis.PyQt.QtWidgets import QAction
from qgis.PyQt.QtGui import QIcon


_PLUGIN_DIR = Path(__file__).resolve().parent   # resolves junction → real path
_SCRIPT_DIR = _PLUGIN_DIR.parent               # D:\TestDash\QGIS PY\
_ICON_PATH  = str(_PLUGIN_DIR / "icon.png")


class AutoGeorefPlugin:
    def __init__(self, iface):
        self.iface    = iface
        self._toolbar = None
        self._action  = None
        self._dialog  = None

    # ------------------------------------------------------------------
    # Plugin lifecycle
    # ------------------------------------------------------------------
    def initGui(self):
        icon = QIcon(_ICON_PATH) if Path(_ICON_PATH).exists() else QIcon()

        self._action = QAction(icon, "Auto Georeferencer", self.iface.mainWindow())
        self._action.setToolTip("Georeference the plan in the plan/ folder")
        self._action.triggered.connect(self.show_dialog)

        # Add to the built-in plugins toolbar (Erweiterungswerkzeugleiste)
        # and to the Plugins menu (Erweiterungen → Auto Georeferencer)
        self.iface.addToolBarIcon(self._action)
        self.iface.addPluginToMenu("Auto Georeferencer", self._action)

    def unload(self):
        self.iface.removeToolBarIcon(self._action)
        self.iface.removePluginMenu("Auto Georeferencer", self._action)
        if self._dialog:
            self._dialog.close()
            self._dialog = None

    # ------------------------------------------------------------------
    # Open / reuse the run dialog
    # ------------------------------------------------------------------
    def show_dialog(self):
        from .dialog import GeorefDialog
        if self._dialog is None:
            self._dialog = GeorefDialog(self.iface, _SCRIPT_DIR,
                                        parent=self.iface.mainWindow())
        self._dialog.refresh_input_label()
        self._dialog.show()
        self._dialog.raise_()
        self._dialog.activateWindow()
