"""
adjustment_tool.py – Interactive drag/rotate adjustment of a georeferenced
TIFF overlay within the QGIS map canvas.

Usage
-----
    tool = GeorefAdjustmentTool(canvas, tiff_path, gt_tuple, W, H)
    tool.accepted.connect(my_slot)   # slot receives new 6-tuple GT
    tool.cancelled.connect(...)
    canvas.setMapTool(tool)

The tool renders the plan image as a semi-transparent overlay (adjustable
opacity) on the map canvas and draws control handles on top:

    • Orange outline       – plan footprint
    • Green circle handle  – rotation, at top-edge midpoint
    • Magenta box handle   – pivot point (centre of rotation), draggable
    • Orange crosshair     – current image centre

Interactions
------------
    Left-drag body          → translate (pivot follows)
    Left-drag green handle  → rotate around pivot
    Left-drag magenta pivot → move rotation centre (image stays put)
    Right-drag anywhere     → rotate around pivot
    Mouse wheel             → fine rotation ±1° per notch (around pivot)
    Reset                   → restore original GT and pivot to centre
    Cancel                  → discard changes
    ✓ Accept                → emit accepted(new_gt)
"""
from __future__ import annotations

import math
from pathlib import Path

from qgis.PyQt import sip
from qgis.PyQt.QtCore    import Qt, pyqtSignal, QPoint, QRectF, QSize, QTimer, QThread, QObject
from qgis.PyQt.QtGui     import (
    QColor, QCursor, QTransform as QGuiTransform,
    QImage, QPixmap, QPainter, QPen,
)
from qgis.PyQt.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QSlider,
    QDoubleSpinBox, QRadioButton, QGroupBox, QGridLayout, QFrame,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QComboBox,
    QSizePolicy,
)

from qgis.core import (
    QgsWkbTypes,
    QgsPointXY,
    QgsGeometry,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsProject,
)


# ---------------------------------------------------------------------------
# Scroll-locked widgets — only respond to wheel when explicitly focused
# ---------------------------------------------------------------------------

class _ClickFocusSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that ignores the scroll wheel unless the user clicked it."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.ClickFocus)

    def wheelEvent(self, e):
        if self.hasFocus():
            super().wheelEvent(e)
        else:
            e.ignore()


class _ClickFocusSlider(QSlider):
    """QSlider that ignores the scroll wheel unless the user clicked it."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.ClickFocus)

    def wheelEvent(self, e):
        if self.hasFocus():
            super().wheelEvent(e)
        else:
            e.ignore()
from qgis.gui import QgsMapTool, QgsRubberBand, QgsMapCanvasItem


# ---------------------------------------------------------------------------
# Pure-math helpers  (no QGIS dependency)
# ---------------------------------------------------------------------------

def _gt_center(gt: tuple, W: int, H: int) -> tuple[float, float]:
    """Centre of the raster in plan CRS (e, n)."""
    return (
        gt[0] + (W / 2.0) * gt[1] + (H / 2.0) * gt[2],
        gt[3] + (W / 2.0) * gt[4] + (H / 2.0) * gt[5],
    )


def _gt_mpp(gt: tuple) -> float:
    """Isotropic pixel size in metres/pixel."""
    return math.sqrt(gt[1] ** 2 + gt[2] ** 2)


def _gt_axis_scales(gt: tuple) -> tuple[float, float]:
    """Column/row axis scales in map units per pixel."""
    return (
        math.sqrt(gt[1] ** 2 + gt[4] ** 2),
        math.sqrt(gt[2] ** 2 + gt[5] ** 2),
    )


def _gt_rotation_deg(gt: tuple) -> float:
    """θ = atan2(-GT[2], GT[1]) — matches auto_georeference.py convention."""
    return math.degrees(math.atan2(-gt[2], gt[1]))


def _gt_from_center_rot_scale(
    center_e: float, center_n: float,
    rot_deg: float, scale_x: float, scale_y: float,
    W: int, H: int,
) -> tuple:
    """
    Rebuild a GDAL 6-tuple from centre + rotation + axis scales.
    Convention matches auto_georeference.py (lines 3940-3942, 7318-7320):
        GT[1] =  sx*cos(θ),  GT[2] = -sy*sin(θ)
        GT[4] = -sx*sin(θ),  GT[5] = -sy*cos(θ)
    """
    theta = math.radians(rot_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    gt1, gt2 =  scale_x * cos_t, -scale_y * sin_t
    gt4, gt5 = -scale_x * sin_t, -scale_y * cos_t
    gt0 = center_e - (W / 2.0) * gt1 - (H / 2.0) * gt2
    gt3 = center_n - (W / 2.0) * gt4 - (H / 2.0) * gt5
    return (gt0, gt1, gt2, gt3, gt4, gt5)


def _gt_corners(gt: tuple, W: int, H: int) -> list[tuple[float, float]]:
    """Four corners in plan CRS: TL, TR, BR, BL."""
    return [
        (gt[0] + c * gt[1] + r * gt[2],
         gt[3] + c * gt[4] + r * gt[5])
        for c, r in [(0, 0), (W, 0), (W, H), (0, H)]
    ]


def _rotate_around(pe: float, pn: float,
                   ce: float, cn: float,
                   delta_deg: float) -> tuple[float, float]:
    """Rotate point (pe, pn) around centre (ce, cn) by delta_deg (CCW in plan CRS)."""
    rad = math.radians(delta_deg)
    cos_d, sin_d = math.cos(rad), math.sin(rad)
    re, rn = pe - ce, pn - cn
    return (ce + re * cos_d - rn * sin_d,
            cn + re * sin_d + rn * cos_d)


def _normalize_angle_deg(angle: float) -> float:
    """Normalize degrees to the [-180, 180) range for UI-facing rotations."""
    return ((float(angle) + 180.0) % 360.0) - 180.0


# ---------------------------------------------------------------------------
# Raster loader
# ---------------------------------------------------------------------------

def _load_raster_image(path: str, max_size: int | None = 1200):
    """
    Load the raster at *path* as a QImage, down-scaled so the largest
    dimension ≤ max_size pixels.

    Returns (QImage | None, src_width_px, src_height_px).
    """
    try:
        from osgeo import gdal
        import numpy as np
        from qgis.PyQt.QtGui import QImage as _QI

        ds = gdal.Open(str(path))
        if ds is None:
            return None, 1, 1
        src_w, src_h = ds.RasterXSize, ds.RasterYSize
        if max_size is None:
            scale = 1.0
        else:
            scale = min(max_size / src_w, max_size / src_h, 1.0)
        out_w   = max(1, int(src_w * scale))
        out_h   = max(1, int(src_h * scale))
        bc      = ds.RasterCount

        def _band(i):
            return ds.GetRasterBand(i).ReadAsArray(0, 0, src_w, src_h, out_w, out_h)

        if bc >= 3:
            r, g, b = _band(1), _band(2), _band(3)
            if bc >= 4:
                arr = np.stack([r, g, b, _band(4)], axis=2).astype(np.uint8)
                fmt = _QI.Format_RGBA8888
            else:
                arr = np.stack([r, g, b], axis=2).astype(np.uint8)
                fmt = _QI.Format_RGB888
        elif bc == 1:
            gray = _band(1)
            arr  = np.stack([gray, gray, gray], axis=2).astype(np.uint8)
            fmt  = _QI.Format_RGB888
        else:
            return None, src_w, src_h

        ds = None  # close
        h, w, c = arr.shape
        img = _QI(arr.tobytes(), w, h, w * c, fmt)
        return img.copy(), src_w, src_h   # .copy() owns the buffer

    except Exception:
        return None, 1, 1


# ---------------------------------------------------------------------------
# North-arrow picker widget  (zoomable, pannable thumbnail)
# ---------------------------------------------------------------------------

class ZoomableNorthArrowPickerWidget(QLabel):
    angle_measured = pyqtSignal(float)

    _THUMB_W = 420
    _THUMB_H = 300

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thumb: QPixmap | None = None
        self._tail: tuple[float, float] | None = None
        self._head: tuple[float, float] | None = None
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._panning = False
        self._pan_anchor: tuple[int, int] | None = None

        self.setMinimumSize(self._THUMB_W, self._THUMB_H)
        self.setMaximumHeight(self._THUMB_H)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#1a1a1a; border:1px solid #555; color:#888;")
        self.setText("Load image first")
        self.setCursor(QCursor(Qt.CrossCursor))
        self.setToolTip(
            "Left click tail, then head, to measure the north-arrow direction.\n"
            "Wheel zooms. Middle mouse drags the preview."
        )

    def sizeHint(self):
        return QSize(self._THUMB_W, self._THUMB_H)

    def minimumSizeHint(self):
        return QSize(self._THUMB_W, self._THUMB_H)

    def set_image(self, qimage: "QImage | None"):
        if qimage is None:
            self._thumb = None
            self._tail = None
            self._head = None
            self.setText("Load image first")
            return
        self._thumb = QPixmap.fromImage(qimage)
        self._tail = None
        self._head = None
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.setText("")
        self._redraw()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._redraw()

    def wheelEvent(self, event):
        if self._thumb is None:
            return
        step = event.angleDelta().y() / 120.0
        if step == 0:
            event.accept()
            return
        factor = 1.15 ** step
        old_zoom = self._zoom
        self._zoom = max(0.25, min(64.0, self._zoom * factor))
        ratio = self._zoom / old_zoom
        # Use pos() — compatible with both PyQt5 (Qt5) and PyQt6 (Qt6)
        pos = event.pos()
        cx = pos.x() - self.width() / 2.0
        cy = pos.y() - self.height() / 2.0
        self._pan_x = cx - ratio * (cx - self._pan_x)
        self._pan_y = cy - ratio * (cy - self._pan_y)
        self._redraw()
        event.accept()

    def mouseMoveEvent(self, event):
        if self._panning and self._pan_anchor is not None:
            ax, ay = self._pan_anchor
            self._pan_x += event.pos().x() - ax
            self._pan_y += event.pos().y() - ay
            self._pan_anchor = (event.pos().x(), event.pos().y())
            self._redraw()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = False
            self._pan_anchor = None
            self.setCursor(QCursor(Qt.CrossCursor))
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _display_rect(self):
        if self._thumb is None:
            return None
        iw = self._thumb.width()
        ih = self._thumb.height()
        if iw <= 0 or ih <= 0:
            return None
        fit = min(self.width() / iw, self.height() / ih)
        scale = fit * self._zoom
        draw_w = iw * scale
        draw_h = ih * scale
        ox = (self.width() - draw_w) / 2.0 + self._pan_x
        oy = (self.height() - draw_h) / 2.0 + self._pan_y
        return scale, ox, oy, draw_w, draw_h

    def _widget_to_image(self, x: float, y: float):
        rect = self._display_rect()
        if rect is None:
            return None
        scale, ox, oy, draw_w, draw_h = rect
        if x < ox or y < oy or x > ox + draw_w or y > oy + draw_h:
            return None
        return ((x - ox) / scale, (y - oy) / scale)

    def _image_to_widget(self, x: float, y: float):
        rect = self._display_rect()
        if rect is None:
            return None
        scale, ox, oy, _draw_w, _draw_h = rect
        return (ox + x * scale, oy + y * scale)

    def _redraw(self):
        if self._thumb is None:
            return
        canvas = QPixmap(self.size())
        canvas.fill(QColor(26, 26, 26))
        painter = QPainter(canvas)
        rect = self._display_rect()
        if rect is None:
            painter.end()
            self.setPixmap(canvas)
            return
        _scale, ox, oy, draw_w, draw_h = rect
        painter.drawPixmap(QRectF(ox, oy, draw_w, draw_h), self._thumb, QRectF(0, 0, self._thumb.width(), self._thumb.height()))
        if self._tail is not None:
            tail_pt = self._image_to_widget(*self._tail)
            if tail_pt is not None:
                tx, ty = tail_pt
                painter.setPen(QPen(QColor(255, 80, 80), 2))
                painter.drawEllipse(int(tx) - 6, int(ty) - 6, 12, 12)
        if self._head is not None:
            head_pt = self._image_to_widget(*self._head)
            if head_pt is not None:
                hx, hy = head_pt
                painter.setPen(QPen(QColor(80, 255, 80), 2))
                painter.drawEllipse(int(hx) - 5, int(hy) - 5, 10, 10)
        if self._tail is not None and self._head is not None:
            tail_pt = self._image_to_widget(*self._tail)
            head_pt = self._image_to_widget(*self._head)
            if tail_pt is not None and head_pt is not None:
                tx, ty = tail_pt
                hx, hy = head_pt
                painter.setPen(QPen(QColor(80, 255, 80), 2))
                painter.drawLine(int(tx), int(ty), int(hx), int(hy))
        painter.setPen(QPen(QColor(255, 220, 80), 1))
        if self._tail is None:
            painter.drawText(6, self.height() - 8, "Left click tail/head | Wheel zoom | Middle-drag pan")
        else:
            painter.drawText(6, self.height() - 8, "Click head")
        painter.end()
        self.setPixmap(canvas)

    def mousePressEvent(self, event):
        if self._thumb is None:
            return
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_anchor = (event.pos().x(), event.pos().y())
            self.setCursor(QCursor(Qt.SizeAllCursor))
            event.accept()
            return
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        img_pt = self._widget_to_image(event.pos().x(), event.pos().y())
        if img_pt is None:
            event.accept()
            return
        x, y = img_pt
        if self._tail is None:
            self._tail = (x, y)
            self._head = None
        else:
            tx, ty = self._tail
            self._head = (x, y)
            dx = x - tx
            dy = y - ty
            angle = math.degrees(math.atan2(dx, -dy))
            self.angle_measured.emit(angle)
            self._tail = None
        self._redraw()
        event.accept()

# ---------------------------------------------------------------------------
# GCP helpers
# ---------------------------------------------------------------------------

def _fit_affine_from_gcps(gcps: list) -> "tuple | None":
    """
    Solve the best-fit GDAL affine geotransform from >= 3 GCP pairs via
    ordinary least squares.

    gcps – list of (pixel_col, pixel_row, world_e, world_n)

    Returns (gt0, gt1, gt2, gt3, gt4, gt5) where
        world_E = gt0 + gt1*col + gt2*row
        world_N = gt3 + gt4*col + gt5*row
    or None if fewer than 3 GCPs are supplied.
    """
    if len(gcps) < 3:
        return None
    try:
        import numpy as np
    except ImportError:
        return None
    A  = np.array([[1.0, float(c), float(r)] for c, r, e, n in gcps])
    bE = np.array([float(e) for c, r, e, n in gcps])
    bN = np.array([float(n) for c, r, e, n in gcps])
    xE, _, _, _ = np.linalg.lstsq(A, bE, rcond=None)
    xN, _, _, _ = np.linalg.lstsq(A, bN, rcond=None)
    return (xE[0], xE[1], xE[2], xN[0], xN[1], xN[2])


def _apply_gcp_warp(
    src_path: "str | Path",
    gcps: list,            # list of (pixel_x, pixel_y, world_e, world_n)
    output_path: "str | Path",
    epsg: int = 25832,
    warp_type: str = "poly1",   # "poly1" | "poly2" | "tps"
    current_gt: "tuple | None" = None,   # current adjusted GT; used for output pixel size
    src_W: int = 0,
    src_H: int = 0,
) -> tuple:
    """
    Warp *src_path* using *gcps* and write the result to *output_path*.

    Returns ``(new_gt, width_px, height_px)`` for the warped file.

    gcps       – list of (pixel_x, pixel_y, world_e, world_n) in *epsg* coords
    warp_type  – "poly1"  → polynomial order 1 (affine, min 3 GCPs)
                 "poly2"  → polynomial order 2 (min 6 GCPs)
                 "tps"    → thin-plate spline   (min 3 GCPs recommended)
    current_gt – the live (possibly adjusted) geotransform; used only to derive
                 the output pixel size so resolution is preserved.  Output bounds
                 are left to GDAL to compute automatically from the polynomial
                 transform of the image corners (same as QGIS Georeferencer).
    """
    try:
        from osgeo import gdal, osr  # type: ignore
    except ImportError:
        raise RuntimeError("GDAL/osgeo is not available")

    src_path    = str(src_path)
    output_path = str(output_path)

    src_ds = gdal.Open(src_path, gdal.GA_ReadOnly)
    if src_ds is None:
        raise RuntimeError(f"Cannot open: {src_path}")

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    wkt = srs.ExportToWkt()

    gdal_gcps = [
        gdal.GCP(float(world_e), float(world_n), 0.0, float(px), float(py))
        for px, py, world_e, world_n in gcps
    ]

    # Set GCPs on the in-memory dataset object (GA_ReadOnly — the file on disk
    # is never modified).  When GCPs are present, gdal.Warp uses the GCP
    # polynomial transform for the pixel→world mapping and ignores the embedded
    # GT.  The output extent is derived by transforming the four source-image
    # corners through the polynomial — the embedded GT plays no part in that.
    src_ds.SetGCPs(gdal_gcps, wkt)

    # Derive output pixel size from the current (adjusted) geotransform so the
    # warp preserves the image resolution.  We do NOT set outputBounds — we let
    # GDAL compute the natural output extent by transforming the source image
    # corners through the polynomial, exactly as the QGIS Georeferencer does.
    # Setting an explicit AABB-based outputBounds for a rotated image results in
    # a large north-up rectangle mostly filled with black/NoData pixels.
    common_kwargs: dict = dict(
        dstSRS=wkt,
        resampleAlg=gdal.GRA_Bilinear,
        format="GTiff",
    )
    if current_gt is not None and src_W > 0 and src_H > 0:
        gt = current_gt
        xres = math.sqrt(gt[1] ** 2 + gt[4] ** 2)   # metres per pixel (x-axis)
        yres = math.sqrt(gt[2] ** 2 + gt[5] ** 2)   # metres per pixel (y-axis)
        if xres > 0 and yres > 0:
            common_kwargs["xRes"] = xres
            common_kwargs["yRes"] = yres

    if warp_type == "tps":
        warp_opts = gdal.WarpOptions(tps=True, **common_kwargs)
    else:
        poly_order = 2 if warp_type == "poly2" else 1
        warp_opts = gdal.WarpOptions(polynomialOrder=poly_order, **common_kwargs)

    out_ds = gdal.Warp(output_path, src_ds, options=warp_opts)
    src_ds = None
    if out_ds is None:
        raise RuntimeError("gdal.Warp returned None — check GCP count / warp type")

    new_gt = tuple(out_ds.GetGeoTransform())
    W = out_ds.RasterXSize
    H = out_ds.RasterYSize
    out_ds = None
    return new_gt, W, H


# ---------------------------------------------------------------------------
# Async raster loader  (QThread so GDAL I/O never blocks the UI thread)
# ---------------------------------------------------------------------------

class _RasterLoaderWorker(QObject):
    """Loads a raster image in a background thread and emits the result."""
    finished = pyqtSignal(object, int, int)   # (QImage | None, src_w, src_h)

    def __init__(self, path: str, max_size: "int | None"):
        super().__init__()
        self._path     = path
        self._max_size = max_size

    def run(self):
        qimage, src_w, src_h = _load_raster_image(self._path, self._max_size)
        self.finished.emit(qimage, src_w, src_h)


# ---------------------------------------------------------------------------
# Canvas overlay item  (renders the plan image with adjustable opacity)
# ---------------------------------------------------------------------------

class _PlanOverlayItem(QgsMapCanvasItem):
    """
    QgsMapCanvasItem that paints the plan raster at the current GT position
    using an affine QTransform so it stays aligned during drag/rotate.

    Position (0, 0) in scene coords = canvas pixel coords — no setPos() needed.
    """

    def __init__(self, canvas):
        super().__init__(canvas)
        self._canvas       = canvas
        self._qimage       = None
        self._opacity      = 0.5
        self._gt           = None
        self._W            = 1
        self._H            = 1
        self._xform_fwd    = None       # QgsCoordinateTransform: plan CRS → canvas CRS
        self._canvas_corners = None     # cached plan→canvas-CRS corners (list[QgsPointXY])
        self.setZValue(50)              # draw above base layers, below rubber bands

    def set_image(self, qimage):
        self._qimage = qimage
        self.update()

    def set_opacity(self, opacity: float):
        self._opacity = max(0.0, min(1.0, opacity))
        self.update()

    def set_rendering_info(self, gt: tuple, W: int, H: int, xform_fwd):
        """Call whenever the GT changes. Pre-computes plan→canvas CRS corners
        so paint() only needs the cheap mapToPixel step."""
        self._gt        = tuple(gt)
        self._W         = W
        self._H         = H
        self._xform_fwd = xform_fwd
        # Cache corners in canvas CRS (expensive PROJ transform — done once here,
        # not on every repaint call)
        corners_plan = _gt_corners(self._gt, self._W, self._H)
        try:
            if xform_fwd is not None and xform_fwd.isValid():
                self._canvas_corners = [
                    xform_fwd.transform(QgsPointXY(e, n))
                    for e, n in corners_plan
                ]
            else:
                self._canvas_corners = [QgsPointXY(e, n) for e, n in corners_plan]
        except Exception:
            self._canvas_corners = None
        self.update()

    def updatePosition(self):
        """Called by QGIS on map zoom/pan — just trigger repaint."""
        self.update()

    def boundingRect(self) -> QRectF:
        # Cover the entire canvas so nothing is clipped during rotation/translation.
        try:
            w = self._canvas.width()
            h = self._canvas.height()
        except Exception:
            w, h = 4096, 4096
        return QRectF(-20, -20, w + 40, h + 40)

    def paint(self, painter, _option, _widget):
        if self._qimage is None or self._gt is None or self._canvas_corners is None:
            return
        iw = self._qimage.width()
        ih = self._qimage.height()
        if iw == 0 or ih == 0:
            return

        # Canvas-CRS corners are pre-computed in set_rendering_info.
        # Only apply mapToPixel here (cheap linear transform, no PROJ math).
        mtp  = self._canvas.mapSettings().mapToPixel()
        ptpx = [mtp.transform(pt) for pt in self._canvas_corners]
        tl, tr, _br, bl = ptpx[0], ptpx[1], ptpx[2], ptpx[3]

        # Affine transform: image (col, row) → scene (x, y)
        m11 = (tr.x() - tl.x()) / iw
        m12 = (tr.y() - tl.y()) / iw
        m21 = (bl.x() - tl.x()) / ih
        m22 = (bl.y() - tl.y()) / ih

        painter.save()
        painter.setOpacity(self._opacity)
        painter.setTransform(QGuiTransform(m11, m12, m21, m22, tl.x(), tl.y()))
        painter.drawImage(0, 0, self._qimage)
        painter.restore()

    def cleanup(self):
        try:
            self._canvas.scene().removeItem(self)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Floating control panel
# ---------------------------------------------------------------------------

class _AdjustmentPanel(QWidget):
    """
    Dark floating panel anchored to the top-right of the canvas.
    Shows live ΔE/ΔN/Δrot, an opacity slider, and action buttons.
    """

    opacity_changed       = pyqtSignal(float)   # 0.0–1.0
    uniform_scale_changed = pyqtSignal(float)
    axis_scale_changed    = pyqtSignal(float, float)
    display_mode_changed  = pyqtSignal(str)
    center_changed        = pyqtSignal(float, float)
    rotation_changed      = pyqtSignal(float)
    # North arrow
    north_angle_measured  = pyqtSignal(float)
    # GCP correction
    gcp_pick_requested          = pyqtSignal()
    gcp_pick_stop_requested     = pyqtSignal()
    gcp_markers_clear_requested = pyqtSignal()
    gcp_warp_requested          = pyqtSignal(list, str)   # (gcps, warp_type)
    gcp_rows_removed            = pyqtSignal(list)        # list[int] of removed row indices

    def __init__(self, canvas: QWidget):
        super().__init__()
        self._syncing_scale = False
        self._syncing_pose = False
        self.setObjectName("autoGeorefAdjustPanel")
        self.setStyleSheet("")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        self._lbl_title = QLabel("Placement Adjustment")
        self._lbl_title.setObjectName("panelTitle")
        outer.addWidget(self._lbl_title)

        self._lbl_subtitle = QLabel(
            "Refine the overlay with direct numeric inputs for position, rotation, and scale."
        )
        self._lbl_subtitle.setObjectName("panelSubtitle")
        self._lbl_subtitle.setWordWrap(True)
        outer.addWidget(self._lbl_subtitle)

        # Mode hint
        self._lbl_mode = QLabel(
            "Drag: translate  |  ● rotate  |  ■ pivot  |  Shift+wheel: ±1°"
        )
        self._lbl_mode.setObjectName("modeHint")
        outer.addWidget(self._lbl_mode)
        self._lbl_mode.setText(
            "Drag: translate  |  rotate handle/right-drag  |  pivot handle  |  Shift+wheel: +/-1 deg"
        )

        # Delta readout
        self._lbl_delta = QLabel(
            "ΔE:   +0.0 m\nΔN:   +0.0 m\nΔrot: +0.000°\nSx:   1.0000\nSy:   1.0000"
        )
        self._lbl_delta.setObjectName("metricCard")
        outer.addWidget(self._lbl_delta)
        self._lbl_delta.setText(
            "Delta E:   +0.0 m\nDelta N:   +0.0 m\nDelta rot: +0.000 deg\nRotation:  +0.000 deg\nSx:        1.0000\nSy:        1.0000"
        )

        transform_box = QGroupBox("Transform")
        transform_layout = QGridLayout(transform_box)
        transform_layout.setHorizontalSpacing(8)
        transform_layout.setVerticalSpacing(8)
        center_lbl = QLabel("Center E")
        self._spin_center_e = _ClickFocusSpinBox()
        self._spin_center_e.setDecimals(2)
        self._spin_center_e.setRange(-10000000.0, 10000000.0)
        self._spin_center_e.setSingleStep(1.0)
        self._spin_center_e.setMaximumWidth(120)
        self._spin_center_e.setToolTip("Center Easting")
        self._spin_center_e.valueChanged.connect(self._on_center_changed)
        self._spin_center_n = _ClickFocusSpinBox()
        self._spin_center_n.setDecimals(2)
        self._spin_center_n.setRange(-10000000.0, 10000000.0)
        self._spin_center_n.setSingleStep(1.0)
        self._spin_center_n.setMaximumWidth(120)
        self._spin_center_n.setToolTip("Center Northing")
        self._spin_center_n.valueChanged.connect(self._on_center_changed)
        center_n_lbl = QLabel("Center N")
        transform_layout.addWidget(center_lbl, 0, 0)
        transform_layout.addWidget(self._spin_center_e, 0, 1)
        transform_layout.addWidget(center_n_lbl, 0, 2)
        transform_layout.addWidget(self._spin_center_n, 0, 3)

        rot_lbl = QLabel("Rotation")
        self._spin_rotation = _ClickFocusSpinBox()
        self._spin_rotation.setDecimals(3)
        self._spin_rotation.setRange(-360.0, 360.0)
        self._spin_rotation.setSingleStep(0.25)
        self._spin_rotation.setMaximumWidth(120)
        self._spin_rotation.setToolTip("Absolute rotation from the original image orientation")
        self._spin_rotation.valueChanged.connect(self._on_rotation_changed)
        transform_layout.addWidget(rot_lbl, 1, 0)
        transform_layout.addWidget(self._spin_rotation, 1, 1)

        scale_lbl = QLabel("Uniform scale")
        self._spin_scale = _ClickFocusSpinBox()
        self._spin_scale.setDecimals(4)
        self._spin_scale.setRange(0.0001, 1000000.0)
        self._spin_scale.setSingleStep(0.1)
        self._spin_scale.setMaximumWidth(90)
        self._spin_scale.setToolTip("Uniform map units per pixel")
        self._spin_scale.valueChanged.connect(self._on_uniform_scale_changed)
        transform_layout.addWidget(scale_lbl, 1, 2)
        transform_layout.addWidget(self._spin_scale, 1, 3)

        axis_x_lbl = QLabel("Scale X")
        self._spin_scale_x = _ClickFocusSpinBox()
        self._spin_scale_x.setDecimals(4)
        self._spin_scale_x.setRange(0.0001, 1000000.0)
        self._spin_scale_x.setSingleStep(0.1)
        self._spin_scale_x.setMaximumWidth(90)
        self._spin_scale_x.setToolTip("Column-axis map units per pixel")
        self._spin_scale_x.valueChanged.connect(self._on_axis_scale_changed)
        self._spin_scale_y = _ClickFocusSpinBox()
        self._spin_scale_y.setDecimals(4)
        self._spin_scale_y.setRange(0.0001, 1000000.0)
        self._spin_scale_y.setSingleStep(0.1)
        self._spin_scale_y.setMaximumWidth(90)
        self._spin_scale_y.setToolTip("Row-axis map units per pixel")
        self._spin_scale_y.valueChanged.connect(self._on_axis_scale_changed)
        axis_y_lbl = QLabel("Scale Y")
        transform_layout.addWidget(axis_x_lbl, 2, 0)
        transform_layout.addWidget(self._spin_scale_x, 2, 1)
        transform_layout.addWidget(axis_y_lbl, 2, 2)
        transform_layout.addWidget(self._spin_scale_y, 2, 3)
        outer.addWidget(transform_box)

        # ── North Arrow ────────────────────────────────────────────────
        north_box = QGroupBox("North Arrow")
        north_layout = QVBoxLayout(north_box)
        north_layout.setSpacing(6)

        north_info = QLabel(
            "Click TAIL then HEAD on the thumbnail to measure the bearing."
        )
        north_info.setWordWrap(True)
        north_layout.addWidget(north_info)

        self._north_picker = ZoomableNorthArrowPickerWidget()
        self._north_picker.angle_measured.connect(self._on_north_angle)
        north_layout.addWidget(self._north_picker, alignment=Qt.AlignHCenter)

        north_result_row = QHBoxLayout()
        self._lbl_north_angle = QLabel("Measured angle: —")
        self._btn_apply_north = QPushButton("Apply to Rotation")
        self._btn_apply_north.setEnabled(False)
        self._btn_apply_north.clicked.connect(self._on_apply_north)
        north_result_row.addWidget(self._lbl_north_angle)
        north_result_row.addStretch()
        north_result_row.addWidget(self._btn_apply_north)
        north_layout.addLayout(north_result_row)
        outer.addWidget(north_box)
        self._last_north_angle: float | None = None

        # ── GCP Correction ─────────────────────────────────────────────
        gcp_box = QGroupBox("GCP Correction")
        gcp_layout = QVBoxLayout(gcp_box)
        gcp_layout.setSpacing(6)

        gcp_info = QLabel(
            "Pick control points to correct position, rotation and distortion.\n"
            "Fit Affine GT: 3+ GCPs — solves rotation+offset directly, no resampling.\n"
            "Poly-1: 3+ GCPs, Poly-2: 6+ GCPs, TPS: 3+ GCPs (all resample pixels)."
        )
        gcp_info.setWordWrap(True)
        gcp_layout.addWidget(gcp_info)

        self._gcp_table = QTableWidget(0, 4)
        self._gcp_table.setHorizontalHeaderLabels(["Px X", "Px Y", "World E", "World N"])
        self._gcp_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._gcp_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._gcp_table.setMinimumHeight(100)
        self._gcp_table.setMaximumHeight(150)
        gcp_layout.addWidget(self._gcp_table)

        gcp_btn_row = QHBoxLayout()
        gcp_btn_row.setSpacing(4)
        self._btn_pick_gcp = QPushButton("Start Picking GCPs")
        self._btn_pick_gcp.setToolTip(
            "Starts two-phase GCP picking:\n"
            "  Phase 1 — click a feature on the plan (red cross marks it)\n"
            "  Phase 2 — click where that feature should be on the basemap (green cross)\n"
            "Repeats automatically. Press Stop or Escape to finish."
        )
        self._btn_pick_gcp.clicked.connect(self._on_pick_gcp_clicked)
        self._btn_stop_pick = QPushButton("Stop Picking")
        self._btn_stop_pick.setToolTip("Exit GCP pick mode (or press Escape on the canvas).")
        self._btn_stop_pick.setEnabled(False)
        self._btn_stop_pick.clicked.connect(self._on_stop_pick_clicked)
        self._btn_remove_gcp = QPushButton("Remove Row")
        self._btn_remove_gcp.clicked.connect(self._remove_gcp_row)
        self._btn_clear_markers = QPushButton("Clear Markers")
        self._btn_clear_markers.setToolTip("Remove all GCP marker overlays from the canvas.")
        self._btn_clear_markers.clicked.connect(self._on_clear_markers_clicked)
        gcp_btn_row.addWidget(self._btn_pick_gcp)
        gcp_btn_row.addWidget(self._btn_stop_pick)
        gcp_btn_row.addWidget(self._btn_remove_gcp)
        gcp_btn_row.addWidget(self._btn_clear_markers)
        gcp_btn_row.addStretch()
        gcp_layout.addLayout(gcp_btn_row)

        gcp_apply_row = QHBoxLayout()
        gcp_apply_row.setSpacing(6)
        warp_lbl = QLabel("Warp:")
        warp_lbl.setFixedWidth(40)
        self._combo_warp = QComboBox()
        self._combo_warp.addItem("Fit Affine GT (no resample)", "fit_gt")
        self._combo_warp.addItem("Polynomial 1 (affine)",       "poly1")
        self._combo_warp.addItem("Polynomial 2",                "poly2")
        self._combo_warp.addItem("Thin-Plate Spline (TPS)",     "tps")
        self._combo_warp.setCurrentIndex(0)   # default: Fit Affine GT
        self._btn_apply_gcp = QPushButton("Apply GCP Correction")
        self._btn_apply_gcp.clicked.connect(self._on_apply_gcp)
        gcp_apply_row.addWidget(warp_lbl)
        gcp_apply_row.addWidget(self._combo_warp)
        gcp_apply_row.addStretch()
        gcp_apply_row.addWidget(self._btn_apply_gcp)
        gcp_layout.addLayout(gcp_apply_row)

        self._lbl_gcp_status = QLabel("")
        self._lbl_gcp_status.setWordWrap(True)
        gcp_layout.addWidget(self._lbl_gcp_status)
        outer.addWidget(gcp_box)

        # ── Display ────────────────────────────────────────────────────
        view_box = QGroupBox("Display")
        view_layout = QVBoxLayout(view_box)
        view_layout.setSpacing(8)

        opac_row = QHBoxLayout()
        opac_row.setSpacing(4)
        opac_lbl = QLabel("Opacity:")
        opac_lbl.setFixedWidth(54)
        self._slider = _ClickFocusSlider(Qt.Horizontal)
        self._slider.setRange(0, 100)
        self._slider.setValue(50)
        self._slider.setMaximumWidth(110)
        self._slider.setToolTip("Overlay opacity (0 = transparent, 100 = opaque)")
        self._lbl_opac = QLabel("50%")
        self._lbl_opac.setFixedWidth(30)
        self._slider.valueChanged.connect(self._on_opacity_changed)
        opac_row.addWidget(opac_lbl)
        opac_row.addWidget(self._slider)
        opac_row.addWidget(self._lbl_opac)
        view_layout.addLayout(opac_row)

        quality_row = QHBoxLayout()
        quality_row.setSpacing(6)
        quality_lbl = QLabel("Display:")
        quality_lbl.setFixedWidth(54)
        self._rb_preview = QRadioButton("Fast")
        self._rb_full = QRadioButton("Full TIFF")
        self._rb_preview.setChecked(True)
        self._rb_preview.toggled.connect(self._on_display_mode_changed)
        self._rb_full.toggled.connect(self._on_display_mode_changed)
        quality_row.addWidget(quality_lbl)
        quality_row.addWidget(self._rb_preview)
        quality_row.addWidget(self._rb_full)
        quality_row.addStretch()
        view_layout.addLayout(quality_row)

        self._lbl_quality = QLabel("Using a down-scaled overlay for smoother interaction.")
        self._lbl_quality.setWordWrap(True)
        self._lbl_quality.setStyleSheet("color:#665a47;")
        view_layout.addWidget(self._lbl_quality)
        outer.addWidget(view_box)

        # ── Actions ────────────────────────────────────────────────────
        actions_box = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_box)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        self.btn_reset  = QPushButton("Reset")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_accept = QPushButton("✓ Accept")
        self.btn_accept.setObjectName("acceptButton")
        btn_row.addWidget(self.btn_reset)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_accept)
        actions_layout.addLayout(btn_row)
        outer.addWidget(actions_box)

        self.adjustSize()

    def _on_opacity_changed(self, val: int):
        self._lbl_opac.setText(f"{val}%")
        self.opacity_changed.emit(val / 100.0)

    def _on_display_mode_changed(self, checked: bool):
        if not checked:
            return
        self.display_mode_changed.emit("full" if self._rb_full.isChecked() else "preview")

    def _on_center_changed(self, _val: float):
        if self._syncing_pose:
            return
        self.center_changed.emit(self._spin_center_e.value(), self._spin_center_n.value())

    def _on_rotation_changed(self, val: float):
        if self._syncing_pose:
            return
        self.rotation_changed.emit(val)

    # ── North arrow helpers ────────────────────────────────────────────
    def set_thumbnail(self, qimage: "QImage | None"):
        """Feed the plan thumbnail to the north-arrow picker."""
        self._north_picker.set_image(qimage)

    def _on_north_angle(self, angle: float):
        self._last_north_angle = angle
        self._lbl_north_angle.setText(
            f"Measured arrow bearing: {angle:.2f}\u00b0 from north"
        )
        self._btn_apply_north.setEnabled(True)
        self.north_angle_measured.emit(angle)

    def _on_apply_north(self):
        if self._last_north_angle is not None:
            # The picker measures bearing on the raw image itself, so the
            # corrected absolute rotation is simply the opposite of that
            # bearing relative to the original raster orientation.
            self.rotation_changed.emit(_normalize_angle_deg(-self._last_north_angle))

    # ── GCP helpers ───────────────────────────────────────────────────
    def _on_pick_gcp_clicked(self):
        self._lbl_gcp_status.setText(
            "Phase 1 — click a recognisable feature ON THE PLAN (red cross will mark it)."
        )
        self._btn_pick_gcp.setEnabled(False)
        self._btn_stop_pick.setEnabled(True)
        self.gcp_pick_requested.emit()

    def _on_stop_pick_clicked(self):
        self._btn_pick_gcp.setEnabled(True)
        self._btn_stop_pick.setEnabled(False)
        self._lbl_gcp_status.setText("Pick mode stopped.")
        self.gcp_pick_stop_requested.emit()

    def _on_clear_markers_clicked(self):
        self.gcp_markers_clear_requested.emit()

    def set_gcp_phase_hint(self, phase: int):
        """Update status label to reflect current two-phase pick state."""
        if phase == 1:
            self._lbl_gcp_status.setText(
                "Phase 1 — click a recognisable feature ON THE PLAN (red cross will mark it)."
            )
        elif phase == 2:
            self._lbl_gcp_status.setText(
                "Phase 2 — click where that feature actually is on the BASEMAP (green cross)."
            )
        else:
            self._btn_pick_gcp.setEnabled(True)
            self._btn_stop_pick.setEnabled(False)

    def on_gcp_point_picked(self, px: float, py: float, world_e: float, world_n: float):
        """Called by the map tool when a full GCP pair has been picked."""
        row = self._gcp_table.rowCount()
        self._gcp_table.insertRow(row)
        for col, val in enumerate([px, py, world_e, world_n]):
            item = QTableWidgetItem(f"{val:.2f}")
            self._gcp_table.setItem(row, col, item)
        self._lbl_gcp_status.setText(
            f"GCP #{row + 1} added — pixel ({px:.0f}, {py:.0f}) → "
            f"world ({world_e:.1f}, {world_n:.1f}). "
            f"Click next feature on the plan (Phase 1)."
        )

    def _remove_gcp_row(self):
        rows = sorted(
            {idx.row() for idx in self._gcp_table.selectedIndexes()},
            reverse=True,
        )
        if not rows:
            return
        self.gcp_rows_removed.emit(rows)   # map tool removes rubber bands first
        for r in rows:
            self._gcp_table.removeRow(r)

    def clear_gcp_table(self):
        self._gcp_table.setRowCount(0)

    def update_gcp_row(self, row: int, px: float, py: float, world_e: float, world_n: float):
        """Update a GCP table row after a marker has been dragged."""
        if row < 0 or row >= self._gcp_table.rowCount():
            return
        for col, val in enumerate([px, py, world_e, world_n]):
            self._gcp_table.setItem(row, col, QTableWidgetItem(f"{val:.2f}"))

    def _get_gcps(self) -> list:
        """Return list of (pixel_x, pixel_y, world_e, world_n) from the table."""
        gcps = []
        for row in range(self._gcp_table.rowCount()):
            try:
                vals = [
                    float(self._gcp_table.item(row, col).text())
                    for col in range(4)
                ]
                gcps.append(tuple(vals))
            except (AttributeError, ValueError):
                pass
        return gcps

    def _on_apply_gcp(self):
        gcps = self._get_gcps()
        warp_type = self._combo_warp.currentData()
        min_gcps = {"fit_gt": 3, "poly1": 3, "poly2": 6, "tps": 3}.get(warp_type, 3)
        if len(gcps) < min_gcps:
            self._lbl_gcp_status.setText(
                f"{warp_type} requires at least {min_gcps} GCPs "
                f"({len(gcps)} provided)."
            )
            return
        self._lbl_gcp_status.setText(
            "Fitting affine GT…" if warp_type == "fit_gt" else "Applying GCP warp…"
        )
        self.gcp_warp_requested.emit(gcps, warp_type)

    def set_gcp_status(self, text: str):
        self._lbl_gcp_status.setText(text)

    def set_mode_hint(self, text: str):
        self._lbl_mode.setText(text)

    def update_delta(self, delta_e: float, delta_n: float, delta_rot: float, abs_rot: float):
        self._lbl_delta.setText(
            f"Delta E:   {delta_e:+.1f} m\n"
            f"Delta N:   {delta_n:+.1f} m\n"
            f"Delta rot: {delta_rot:+.3f} deg\n"
            f"Rotation:  {abs_rot:+.3f} deg\n"
            f"Sx:        {self._spin_scale_x.value():.4f}\n"
            f"Sy:        {self._spin_scale_y.value():.4f}"
        )

    def set_pose_values(self, center_e: float, center_n: float, rotation_deg: float):
        self._syncing_pose = True
        try:
            self._spin_center_e.setValue(center_e)
            self._spin_center_n.setValue(center_n)
            self._spin_rotation.setValue(rotation_deg)
        finally:
            self._syncing_pose = False

    def set_quality_text(self, text: str):
        self._lbl_quality.setText(text)

    def set_display_mode(self, mode: str):
        self._rb_preview.setChecked(mode != "full")
        self._rb_full.setChecked(mode == "full")

    def set_scale_values(self, scale_x: float, scale_y: float):
        self._syncing_scale = True
        try:
            self._spin_scale_x.setValue(scale_x)
            self._spin_scale_y.setValue(scale_y)
            if abs(scale_x - scale_y) <= max(scale_x, scale_y, 1.0) * 1e-9:
                self._spin_scale.setValue(scale_x)
            else:
                self._spin_scale.setValue((scale_x + scale_y) / 2.0)
        finally:
            self._syncing_scale = False

    def _on_uniform_scale_changed(self, val: float):
        if self._syncing_scale:
            return
        self.uniform_scale_changed.emit(val)

    def _on_axis_scale_changed(self, _val: float):
        if self._syncing_scale:
            return
        self.axis_scale_changed.emit(
            self._spin_scale_x.value(),
            self._spin_scale_y.value(),
        )

# ---------------------------------------------------------------------------
# Map tool
# ---------------------------------------------------------------------------

_HINT_IDLE = (
    "Drag: translate  |  rotate handle/right-drag  |  Shift+wheel: +/-1 deg  |  pivot handle: move pivot"
)

class GeorefAdjustmentTool(QgsMapTool):
    """
    Interactive drag/rotate adjustment with live image overlay and
    adjustable rotation pivot.

    Signals
    -------
    accepted(tuple)  – new 6-tuple GDAL geotransform
    cancelled()
    """

    accepted        = pyqtSignal(tuple)
    cancelled       = pyqtSignal()
    path_changed    = pyqtSignal(str)          # emitted when GCP warp replaces the file
    gcp_point_picked  = pyqtSignal(float, float, float, float)           # px, py, world_e, world_n
    gcp_pair_updated  = pyqtSignal(int, float, float, float, float)      # row, px, py, world_e, world_n

    def __init__(
        self,
        canvas,
        geotiff_path: "str | Path",
        gt_tuple: tuple,
        W: int,
        H: int,
        epsg: int = 25832,
        panel: QWidget | None = None,
    ):
        super().__init__(canvas)
        self._canvas   = canvas
        self._path     = Path(geotiff_path)
        self._orig_gt  = tuple(gt_tuple)
        self._gt       = list(gt_tuple)
        self._W        = W
        self._H        = H
        self._epsg     = epsg
        self._scale_x, self._scale_y = _gt_axis_scales(gt_tuple)
        self._orig_scale_x, self._orig_scale_y = self._scale_x, self._scale_y

        # Pivot (rotation centre) — starts at image centre, moves with translate
        orig_ce, orig_cn = _gt_center(gt_tuple, W, H)
        self._pivot_e  = orig_ce
        self._pivot_n  = orig_cn

        # Drag state
        self._drag_mode      = None   # None|"translate"|"rotate"|"move_pivot"
        self._press_plan     = None   # (e,n) plan-CRS at mouse press
        self._press_center   = None   # (ce,cn) at translate press
        self._press_pivot    = None   # (pe,pn) at translate press
        self._prev_rot_angle = None   # float, for incremental rotation

        # QGIS objects (built in activate())
        self._rb_outline  = None
        self._rb_center   = None
        self._rb_handle   = None   # green rotation handle (top-edge midpoint)
        self._rb_pivot    = None   # magenta pivot handle
        self._img_item    = None   # _PlanOverlayItem
        self._panel       = None
        self._external_panel = panel
        self._xform_fwd   = None
        self._xform_rev   = None
        self._preview_image = None
        self._full_image    = None
        self._src_image_size = (W, H)
        self._display_mode = "preview"

        # Cached screen-space positions for hit-tests
        self._handle_canvas_pt: QgsPointXY | None = None
        self._pivot_canvas_pt:  QgsPointXY | None = None

        # GCP pick mode — two-phase: 0=off, 1=waiting plan click, 2=waiting world click
        self._gcp_phase: int = 0
        self._gcp_pending_pixel: "tuple[float, float] | None" = None
        self._gcp_pending_map_pt: "QgsPointXY | None" = None
        self._gcp_pending_src_rb: "QgsRubberBand | None" = None  # Phase-1 marker before pair is complete
        self._gcp_saved_opacity: float = 0.5                     # saved when plan is hidden in Phase 2
        # Structured list of GCP pairs: each entry is a dict with keys:
        #   src_rb, tgt_rb, link_rb  – rubber bands
        #   src_map_pt, tgt_map_pt   – positions in canvas CRS (QgsPointXY)
        self._gcp_pairs: "list[dict]" = []
        # Drag state for moving existing GCP markers
        self._drag_gcp_idx:  "int | None" = None   # which pair is being dragged
        self._drag_gcp_type: str = ""               # "src" | "tgt"

        # Throttle: cap drag-update redraws to ~60 fps so canvasMoveEvent
        # doesn't flood the render pipeline during fast mouse moves.
        self._update_timer = QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(16)   # ≈60 fps
        self._update_timer.timeout.connect(self._update_all)
        self._update_dirty = False

        # Background loader thread handles
        self._loader_thread: QThread | None = None
        self._loader_worker: "_RasterLoaderWorker | None" = None

    # ----------------------------------------------------------------
    # Activation / deactivation
    # ----------------------------------------------------------------

    def activate(self):
        self._setup_transforms()
        self._build_rubber_bands()
        self._build_image_overlay()
        self._update_all()
        self._build_panel()
        self._canvas.setCursor(QCursor(Qt.SizeAllCursor))
        super().activate()

    def deactivate(self):
        self._update_timer.stop()
        self._cancel_async_load()
        self._disconnect_panel_signals()
        self._destroy_rubber_bands()
        if self._img_item is not None:
            self._img_item.cleanup()
            self._img_item = None
        if self._panel is not None:
            if self._external_panel is None:
                try:
                    self._panel.deleteLater()
                except Exception:
                    pass
            self._panel = None
        try:
            self._canvas.unsetCursor()
        except Exception:
            pass
        super().deactivate()

    # ----------------------------------------------------------------
    # CRS transforms
    # ----------------------------------------------------------------

    def _setup_transforms(self):
        plan_crs   = QgsCoordinateReferenceSystem(f"EPSG:{self._epsg}")
        canvas_crs = self._canvas.mapSettings().destinationCrs()
        proj       = QgsProject.instance()
        self._xform_fwd = QgsCoordinateTransform(plan_crs,   canvas_crs, proj)
        self._xform_rev = QgsCoordinateTransform(canvas_crs, plan_crs,   proj)

    def _to_canvas(self, e: float, n: float) -> QgsPointXY:
        pt = QgsPointXY(e, n)
        if self._xform_fwd is not None and self._xform_fwd.isValid():
            try:
                return self._xform_fwd.transform(pt)
            except Exception:
                pass
        return pt

    def _to_plan(self, canvas_pt: QgsPointXY) -> QgsPointXY:
        if self._xform_rev is not None and self._xform_rev.isValid():
            try:
                return self._xform_rev.transform(canvas_pt)
            except Exception:
                pass
        return canvas_pt

    def _canvas_to_image_pixel(self, plan_pt: QgsPointXY) -> tuple:
        """
        Convert a plan-CRS point to fractional image pixel (col, row)
        using the inverse of the current GDAL geotransform.
        """
        gt = self._gt
        e, n = plan_pt.x(), plan_pt.y()
        det = gt[1] * gt[5] - gt[2] * gt[4]
        if abs(det) < 1e-15:
            return 0.0, 0.0
        col = ((e - gt[0]) * gt[5] - (n - gt[3]) * gt[2]) / det
        row = ((n - gt[3]) * gt[1] - (e - gt[0]) * gt[4]) / det
        return col, row

    # ── GCP pick / warp handlers ──────────────────────────────────────

    def _on_gcp_pick_requested(self):
        self._gcp_phase = 1
        self._gcp_pending_pixel = None
        self._gcp_pending_map_pt = None
        self._canvas.setCursor(QCursor(Qt.CrossCursor))
        if self._panel:
            self._panel.set_mode_hint("GCP Phase 1 — click a feature on the plan")
            self._panel.set_gcp_phase_hint(1)

    def _on_gcp_pick_stop(self):
        # If stopped mid-Phase-2, restore plan visibility and remove dangling src marker
        if self._gcp_phase == 2:
            if self._img_item is not None:
                self._img_item.set_opacity(self._gcp_saved_opacity)
            if self._gcp_pending_src_rb is not None:
                try:
                    self._gcp_pending_src_rb.reset()
                    self._canvas.scene().removeItem(self._gcp_pending_src_rb)
                except Exception:
                    pass
                self._gcp_pending_src_rb = None
        self._gcp_phase = 0
        self._gcp_pending_pixel = None
        self._gcp_pending_map_pt = None
        self._canvas.setCursor(QCursor(Qt.SizeAllCursor))
        if self._panel:
            self._panel.set_mode_hint(_HINT_IDLE)
            self._panel.set_gcp_phase_hint(0)

    def _on_gcp_markers_clear(self):
        """Remove all GCP rubber bands and clear pending Phase-1 marker."""
        if self._gcp_pending_src_rb is not None:
            try:
                self._gcp_pending_src_rb.reset()
                self._canvas.scene().removeItem(self._gcp_pending_src_rb)
            except Exception:
                pass
            self._gcp_pending_src_rb = None
        for pair in self._gcp_pairs:
            for key in ("src_rb", "tgt_rb", "link_rb"):
                rb = pair.get(key)
                if rb is not None:
                    try:
                        rb.reset()
                        self._canvas.scene().removeItem(rb)
                    except Exception:
                        pass
        self._gcp_pairs.clear()

    def _on_gcp_rows_removed(self, rows: list):
        """Remove rubber bands for the given row indices (sorted descending)."""
        for r in sorted(rows, reverse=True):
            if 0 <= r < len(self._gcp_pairs):
                pair = self._gcp_pairs.pop(r)
                for key in ("src_rb", "tgt_rb", "link_rb"):
                    rb = pair.get(key)
                    if rb is not None:
                        try:
                            rb.reset()
                            self._canvas.scene().removeItem(rb)
                        except Exception:
                            pass

    def _make_src_rb(self, map_pt: "QgsPointXY") -> "QgsRubberBand":
        """Create a red cross rubber band at map_pt (canvas CRS)."""
        rb = QgsRubberBand(self._canvas, QgsWkbTypes.PointGeometry)
        rb.setColor(QColor(220, 30, 30, 255))
        rb.setWidth(3)
        rb.setIcon(QgsRubberBand.ICON_CROSS)
        rb.setIconSize(24)
        rb.setZValue(200)
        rb.addPoint(map_pt)
        return rb

    def _make_tgt_rb(self, map_pt: "QgsPointXY") -> "QgsRubberBand":
        """Create a green circle rubber band at map_pt (canvas CRS)."""
        rb = QgsRubberBand(self._canvas, QgsWkbTypes.PointGeometry)
        rb.setColor(QColor(30, 210, 30, 255))
        rb.setWidth(3)
        rb.setIcon(QgsRubberBand.ICON_CIRCLE)
        rb.setIconSize(24)
        rb.setZValue(200)
        rb.addPoint(map_pt)
        return rb

    def _make_link_rb(self, src_pt: "QgsPointXY", tgt_pt: "QgsPointXY") -> "QgsRubberBand":
        """Create a blue line rubber band connecting src to tgt."""
        rb = QgsRubberBand(self._canvas, QgsWkbTypes.LineGeometry)
        rb.setColor(QColor(60, 120, 220, 200))
        rb.setWidth(2)
        rb.setZValue(199)
        rb.addPoint(src_pt)
        rb.addPoint(tgt_pt)
        return rb

    def _gcp_pair_at(self, pos: "QPoint") -> "tuple[int, str] | tuple[None, None]":
        """
        Return (pair_index, "src"|"tgt") if pos is within 20 px of a GCP marker.
        Returns (None, None) if no marker is hit.
        Src markers take priority over tgt markers.
        """
        RADIUS2 = 400   # 20 px²
        for i, pair in enumerate(self._gcp_pairs):
            if self._pt_screen_dist2(pair.get("src_map_pt"), pos) <= RADIUS2:
                return i, "src"
        for i, pair in enumerate(self._gcp_pairs):
            if self._pt_screen_dist2(pair.get("tgt_map_pt"), pos) <= RADIUS2:
                return i, "tgt"
        return None, None

    def _on_gcp_warp_requested(self, gcps: list, warp_type: str):
        if warp_type == "fit_gt":
            self._on_gcp_fit_gt(gcps)
            return
        if self._panel:
            self._panel.set_gcp_status("Running GCP warp…")
        try:
            warped_path = self._path.parent / (self._path.stem + "_gcp_warped.tif")
            new_gt, new_W, new_H = _apply_gcp_warp(
                self._path, gcps, warped_path, self._epsg, warp_type,
                current_gt=tuple(self._gt), src_W=self._W, src_H=self._H,
            )
            # Update state
            self._path       = warped_path
            self._orig_gt    = new_gt
            self._gt         = list(new_gt)
            self._W          = new_W
            self._H          = new_H
            self._scale_x, self._scale_y = _gt_axis_scales(new_gt)
            self._orig_scale_x, self._orig_scale_y = self._scale_x, self._scale_y
            ce, cn = _gt_center(new_gt, new_W, new_H)
            self._pivot_e, self._pivot_n = ce, cn
            # Reload preview asynchronously — full load deferred to user request
            self._preview_image = None
            self._full_image    = None
            self._display_mode  = "preview"
            if self._panel:
                self._panel.set_display_mode("preview")
                self._panel.set_thumbnail(None)
                self._panel.set_scale_values(self._scale_x, self._scale_y)
                self._panel.set_gcp_status(
                    f"Warp applied. Reloading preview… ({warped_path.name})"
                )
            # Clear stale GCP markers/table — pixel positions are relative to pre-warp image
            self._on_gcp_markers_clear()
            if self._panel:
                self._panel.clear_gcp_table()
            self._start_async_load(str(warped_path), max_size=1200, is_full=False)
            self.path_changed.emit(str(warped_path))
            self._update_all()
        except Exception as exc:
            if self._panel:
                self._panel.set_gcp_status(f"Warp failed: {exc}")

    def _on_gcp_fit_gt(self, gcps: list):
        """
        Fit an affine geotransform from GCP pairs via least squares and apply it
        directly to the in-memory GT (no pixel resampling, no new file created).

        This solves for the best rotation + offset + scale that maps the current
        image pixels to the user-supplied world coordinates, then repositions the
        overlay instantly.  The fitted GT is committed to the TIFF on Accept.
        """
        new_gt = _fit_affine_from_gcps(gcps)
        if new_gt is None:
            if self._panel:
                self._panel.set_gcp_status("Affine fit needs >= 3 GCPs.")
            return

        # Compute residuals for status display
        max_res = 0.0
        for col, row, E, N in gcps:
            fit_E = new_gt[0] + new_gt[1] * col + new_gt[2] * row
            fit_N = new_gt[3] + new_gt[4] * col + new_gt[5] * row
            res = math.sqrt((fit_E - E) ** 2 + (fit_N - N) ** 2)
            max_res = max(max_res, res)

        self._gt         = list(new_gt)
        self._orig_gt    = new_gt          # treat fitted GT as the new baseline
        self._scale_x, self._scale_y = _gt_axis_scales(new_gt)
        self._orig_scale_x, self._orig_scale_y = self._scale_x, self._scale_y
        ce, cn = _gt_center(new_gt, self._W, self._H)
        self._pivot_e, self._pivot_n = ce, cn

        rot_deg = _gt_rotation_deg(new_gt)
        if self._panel:
            self._panel.set_scale_values(self._scale_x, self._scale_y)
            self._panel.set_gcp_status(
                f"Affine GT fitted from {len(gcps)} GCPs — "
                f"rotation {rot_deg:+.3f}°, max residual {max_res:.1f} m.  "
                f"GCPs cleared. Click ✓ Accept to save."
            )
        # GCPs remain valid (pixels haven't moved) but clear them so the user
        # starts fresh after the GT has been updated
        self._on_gcp_markers_clear()
        if self._panel:
            self._panel.clear_gcp_table()
        self._update_all()

    # ----------------------------------------------------------------
    # Visual elements — build
    # ----------------------------------------------------------------

    def _build_rubber_bands(self):
        # Footprint outline
        self._rb_outline = QgsRubberBand(self._canvas, QgsWkbTypes.PolygonGeometry)
        self._rb_outline.setColor(QColor(255, 165, 0, 200))
        self._rb_outline.setWidth(2)
        self._rb_outline.setFillColor(QColor(0, 0, 0, 0))  # no fill (image shows beneath)

        # Image centre crosshair
        self._rb_center = QgsRubberBand(self._canvas, QgsWkbTypes.PointGeometry)
        self._rb_center.setColor(QColor(255, 165, 0, 220))
        self._rb_center.setWidth(2)
        self._rb_center.setIcon(QgsRubberBand.ICON_CROSS)
        self._rb_center.setIconSize(16)

        # Green circle — rotation handle (top-edge midpoint)
        self._rb_handle = QgsRubberBand(self._canvas, QgsWkbTypes.PointGeometry)
        self._rb_handle.setColor(QColor(30, 210, 80, 230))
        self._rb_handle.setWidth(3)
        self._rb_handle.setIcon(QgsRubberBand.ICON_CIRCLE)
        self._rb_handle.setIconSize(18)

        # Magenta box — pivot / rotation centre (draggable)
        self._rb_pivot = QgsRubberBand(self._canvas, QgsWkbTypes.PointGeometry)
        self._rb_pivot.setColor(QColor(220, 50, 220, 230))
        self._rb_pivot.setWidth(3)
        self._rb_pivot.setIcon(QgsRubberBand.ICON_BOX)
        self._rb_pivot.setIconSize(16)

    def _destroy_rubber_bands(self):
        for rb in (self._rb_outline, self._rb_center, self._rb_handle, self._rb_pivot):
            if rb is not None:
                try:
                    rb.reset()
                    self._canvas.scene().removeItem(rb)
                except Exception:
                    pass
        self._rb_outline = self._rb_center = self._rb_handle = self._rb_pivot = None
        self._on_gcp_markers_clear()
        self._gcp_pending_src_rb = None

    def _build_image_overlay(self):
        """Create the canvas overlay item and kick off async preview load."""
        self._img_item    = _PlanOverlayItem(self._canvas)
        self._display_mode = "preview"
        self._start_async_load(str(self._path), max_size=1200, is_full=False)

    def _start_async_load(self, path: str, max_size: "int | None", is_full: bool):
        """Start a background thread to load the raster image without blocking the UI."""
        # Cancel any in-flight load first
        self._cancel_async_load()
        worker = _RasterLoaderWorker(path, max_size)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(lambda img, w, h: self._on_raster_loaded(img, w, h, is_full))
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_async_loader_refs)
        thread.start()
        self._loader_thread = thread
        self._loader_worker = worker

    def _clear_async_loader_refs(self):
        self._loader_thread = None
        self._loader_worker = None

    def _cancel_async_load(self):
        thread = self._loader_thread
        self._loader_thread = None
        self._loader_worker = None
        if thread is None:
            return
        try:
            if sip.isdeleted(thread):
                return
        except Exception:
            return
        try:
            if thread.isRunning():
                thread.quit()
                thread.wait(500)
        except RuntimeError:
            return
        self._loader_thread = None
        self._loader_worker = None

    def _on_raster_loaded(self, qimage, src_w: int, src_h: int, is_full: bool):
        """Called on the main thread when a background raster load finishes."""
        if is_full:
            self._full_image = qimage
            self._src_image_size = (src_w, src_h)
            if qimage is None:
                if self._panel:
                    self._panel.set_quality_text("Full TIFF could not be loaded.")
                    self._panel.set_display_mode("preview")
                return
            if self._display_mode == "full" and self._img_item:
                self._img_item.set_image(qimage)
            if self._panel:
                self._panel.set_thumbnail(qimage)
                self._panel.set_quality_text(
                    f"Showing full TIFF pixels ({src_w} x {src_h})."
                )
        else:
            self._preview_image = qimage
            self._src_image_size = (src_w, src_h)
            if self._img_item and self._display_mode == "preview":
                self._img_item.set_image(qimage)
            if self._panel:
                self._panel.set_thumbnail(qimage)
                self._panel.set_quality_text(
                    "Preview loaded. Use 'Full TIFF' for pixel-accurate display."
                )
            # Now that the overlay image is ready, wire it into the canvas
            self._update_all()

    def _set_overlay_display_mode(self, mode: str):
        mode = "full" if mode == "full" else "preview"
        if self._img_item is None:
            return
        self._display_mode = mode
        if mode == "full":
            if self._full_image is not None:
                # Already cached — switch immediately
                self._img_item.set_image(self._full_image)
                if self._panel:
                    self._panel.set_thumbnail(self._full_image)
                    self._panel.set_quality_text(
                        f"Showing full TIFF pixels ({self._src_image_size[0]} x {self._src_image_size[1]})."
                    )
            else:
                # Kick off async load; overlay stays on preview until it arrives
                if self._panel:
                    self._panel.set_quality_text("Loading full TIFF in background…")
                self._start_async_load(str(self._path), max_size=None, is_full=True)
        else:
            target = self._preview_image
            if target is not None:
                self._img_item.set_image(target)
            if self._panel:
                self._panel.set_thumbnail(target)
                self._panel.set_quality_text("Using down-scaled preview for smooth interaction.")

    # ----------------------------------------------------------------
    # Visual elements — update
    # ----------------------------------------------------------------

    def _update_all(self):
        self._update_dirty = False
        gt = self._gt
        W, H = self._W, self._H

        # Footprint
        corners_plan = _gt_corners(gt, W, H)
        canvas_pts   = [self._to_canvas(e, n) for e, n in corners_plan]
        self._rb_outline.setToGeometry(
            QgsGeometry.fromPolygonXY([canvas_pts + [canvas_pts[0]]]), None
        )

        # Image centre
        ce, cn = _gt_center(gt, W, H)
        cpt = self._to_canvas(ce, cn)
        self._rb_center.reset(QgsWkbTypes.PointGeometry)
        self._rb_center.addPoint(cpt)

        # Rotation handle — midpoint of top edge (TL→TR)
        he = (corners_plan[0][0] + corners_plan[1][0]) / 2.0
        hn = (corners_plan[0][1] + corners_plan[1][1]) / 2.0
        hpt = self._to_canvas(he, hn)
        self._rb_handle.reset(QgsWkbTypes.PointGeometry)
        self._rb_handle.addPoint(hpt)
        self._handle_canvas_pt = hpt

        # Pivot handle
        ppt = self._to_canvas(self._pivot_e, self._pivot_n)
        self._rb_pivot.reset(QgsWkbTypes.PointGeometry)
        self._rb_pivot.addPoint(ppt)
        self._pivot_canvas_pt = ppt

        # Image overlay
        if self._img_item is not None:
            self._img_item.set_rendering_info(gt, W, H, self._xform_fwd)

        self._update_panel_deltas()

    def _pt_screen_dist2(self, canvas_pt: QgsPointXY | None, pos: QPoint) -> float:
        """Squared screen-pixel distance from a canvas-CRS point to QPoint pos."""
        if canvas_pt is None:
            return float("inf")
        mtp = self._canvas.mapSettings().mapToPixel()
        px  = mtp.transform(canvas_pt)
        dx, dy = pos.x() - px.x(), pos.y() - px.y()
        return dx * dx + dy * dy

    def _is_on_handle(self, pos: QPoint) -> bool:
        return self._pt_screen_dist2(self._handle_canvas_pt, pos) <= 400   # 20 px

    def _is_on_pivot(self, pos: QPoint) -> bool:
        return self._pt_screen_dist2(self._pivot_canvas_pt, pos) <= 400    # 20 px

    # ----------------------------------------------------------------
    # Panel
    # ----------------------------------------------------------------

    def _disconnect_panel_signals(self):
        if self._panel is None:
            return
        signal_map = (
            (self._panel.btn_reset.clicked, self.reset_to_original),
            (self._panel.btn_cancel.clicked, self._on_cancel),
            (self._panel.btn_accept.clicked, self._on_accept),
            (self._panel.opacity_changed, self._on_opacity_changed),
            (self._panel.uniform_scale_changed, self._on_uniform_scale_changed),
            (self._panel.axis_scale_changed, self._on_axis_scale_changed),
            (self._panel.center_changed, self._on_center_changed),
            (self._panel.rotation_changed, self._on_rotation_changed),
            (self._panel.display_mode_changed, self._set_overlay_display_mode),
            (self._panel.gcp_pick_requested, self._on_gcp_pick_requested),
            (self._panel.gcp_pick_stop_requested, self._on_gcp_pick_stop),
            (self._panel.gcp_markers_clear_requested, self._on_gcp_markers_clear),
            (self._panel.gcp_rows_removed, self._on_gcp_rows_removed),
            (self._panel.gcp_warp_requested, self._on_gcp_warp_requested),
            (self.gcp_point_picked, self._panel.on_gcp_point_picked),
            (self.gcp_pair_updated, self._panel.update_gcp_row),
        )
        for signal, slot in signal_map:
            try:
                signal.disconnect(slot)
            except Exception:
                pass

    def _north_picker_image_for_mode(self, _mode: str):
        # Always use the already-loaded preview — the picker widget scales it
        # internally via its own zoom/pan so extra resolution gives no benefit,
        # and loading a separate full-resolution copy would double peak RAM usage.
        return self._preview_image

    def _build_panel(self):
        if self._external_panel is not None:
            self._panel = self._external_panel
        else:
            from qgis.utils import iface as _iface
            self._panel = _AdjustmentPanel(self._canvas)
        self._disconnect_panel_signals()
        self._panel.btn_reset.clicked.connect(self.reset_to_original)
        self._panel.btn_cancel.clicked.connect(self._on_cancel)
        self._panel.btn_accept.clicked.connect(self._on_accept)
        self._panel.opacity_changed.connect(self._on_opacity_changed)
        self._panel.uniform_scale_changed.connect(self._on_uniform_scale_changed)
        self._panel.axis_scale_changed.connect(self._on_axis_scale_changed)
        self._panel.center_changed.connect(self._on_center_changed)
        self._panel.rotation_changed.connect(self._on_rotation_changed)
        self._panel.display_mode_changed.connect(self._set_overlay_display_mode)
        self._panel.gcp_pick_requested.connect(self._on_gcp_pick_requested)
        self._panel.gcp_pick_stop_requested.connect(self._on_gcp_pick_stop)
        self._panel.gcp_markers_clear_requested.connect(self._on_gcp_markers_clear)
        self._panel.gcp_rows_removed.connect(self._on_gcp_rows_removed)
        self._panel.gcp_warp_requested.connect(self._on_gcp_warp_requested)
        self.gcp_point_picked.connect(self._panel.on_gcp_point_picked)
        self.gcp_pair_updated.connect(self._panel.update_gcp_row)
        self._panel.set_mode_hint(_HINT_IDLE)
        self._panel.set_scale_values(self._scale_x, self._scale_y)
        self._panel.set_display_mode(self._display_mode)
        self._set_overlay_display_mode(self._display_mode)
        self._panel.set_thumbnail(self._north_picker_image_for_mode(self._display_mode))
        self._update_panel_deltas()
        if self._external_panel is None:
            self._panel.show()

    def _update_panel_deltas(self):
        if self._panel is None:
            return
        orig_ce, orig_cn = _gt_center(self._orig_gt, self._W, self._H)
        cur_ce,  cur_cn  = _gt_center(self._gt,      self._W, self._H)
        abs_rot = _gt_rotation_deg(self._gt)
        self._panel.set_pose_values(cur_ce, cur_cn, abs_rot)
        self._panel.update_delta(
            cur_ce  - orig_ce,
            cur_cn  - orig_cn,
            abs_rot - _gt_rotation_deg(self._orig_gt),
            abs_rot,
        )

    def _on_opacity_changed(self, opacity: float):
        if self._img_item is not None:
            self._img_item.set_opacity(opacity)

    def _apply_scale(self, scale_x: float, scale_y: float):
        scale_x = max(0.0001, float(scale_x))
        scale_y = max(0.0001, float(scale_y))
        ce, cn = _gt_center(self._gt, self._W, self._H)
        rot = _gt_rotation_deg(self._gt)
        self._scale_x = scale_x
        self._scale_y = scale_y
        if self._panel is not None:
            self._panel.set_scale_values(self._scale_x, self._scale_y)
        self._gt = list(_gt_from_center_rot_scale(
            ce, cn, rot, self._scale_x, self._scale_y, self._W, self._H
        ))
        self._update_all()

    def _on_uniform_scale_changed(self, scale: float):
        self._apply_scale(scale, scale)

    def _on_axis_scale_changed(self, scale_x: float, scale_y: float):
        self._apply_scale(scale_x, scale_y)

    def _on_center_changed(self, center_e: float, center_n: float):
        cur_ce, cur_cn = _gt_center(self._gt, self._W, self._H)
        de = float(center_e) - cur_ce
        dn = float(center_n) - cur_cn
        if abs(de) <= 1e-12 and abs(dn) <= 1e-12:
            return
        self._pivot_e += de
        self._pivot_n += dn
        self._gt = list(_gt_from_center_rot_scale(
            float(center_e),
            float(center_n),
            _gt_rotation_deg(self._gt),
            self._scale_x,
            self._scale_y,
            self._W,
            self._H,
        ))
        self._update_all()

    def _on_rotation_changed(self, rotation_deg: float):
        target_rot = float(rotation_deg)
        cur_rot = _gt_rotation_deg(self._gt)
        delta_rot = target_rot - cur_rot
        if abs(delta_rot) <= 1e-12:
            return
        old_ce, old_cn = _gt_center(self._gt, self._W, self._H)
        new_ce, new_cn = _rotate_around(
            old_ce, old_cn, self._pivot_e, self._pivot_n, delta_rot
        )
        self._gt = list(_gt_from_center_rot_scale(
            new_ce,
            new_cn,
            target_rot,
            self._scale_x,
            self._scale_y,
            self._W,
            self._H,
        ))
        self._update_all()

    # ----------------------------------------------------------------
    # Mouse events
    # ----------------------------------------------------------------

    def canvasPressEvent(self, e):
        map_pt  = self.toMapCoordinates(e.pos())
        plan_pt = self._to_plan(map_pt)

        # ── Two-phase GCP pick ──────────────────────────────────────────
        if self._gcp_phase > 0 and e.button() == Qt.LeftButton:
            if self._gcp_phase == 1:
                # Phase 1: user clicks a recognisable feature ON THE PLAN.
                col, row_px = self._canvas_to_image_pixel(plan_pt)
                self._gcp_pending_pixel  = (col, row_px)
                self._gcp_pending_map_pt = map_pt
                self._gcp_pending_src_rb = self._make_src_rb(map_pt)
                # Hide plan so basemap is visible for Phase 2
                self._gcp_saved_opacity = (
                    self._img_item._opacity if self._img_item is not None else 0.5
                )
                if self._img_item is not None:
                    self._img_item.set_opacity(0.0)
                self._gcp_phase = 2
                if self._panel:
                    self._panel.set_mode_hint("GCP Phase 2 — click where it is on the basemap")
                    self._panel.set_gcp_phase_hint(2)
            else:
                # Phase 2: user clicks the TRUE world location on the basemap.
                plan_tgt = self._to_plan(map_pt)
                world_e, world_n = plan_tgt.x(), plan_tgt.y()
                col, row_px = self._gcp_pending_pixel
                tgt_rb  = self._make_tgt_rb(map_pt)
                link_rb = self._make_link_rb(self._gcp_pending_map_pt, map_pt)
                self._gcp_pairs.append({
                    "src_rb":     self._gcp_pending_src_rb,
                    "tgt_rb":     tgt_rb,
                    "link_rb":    link_rb,
                    "src_map_pt": self._gcp_pending_map_pt,
                    "tgt_map_pt": map_pt,
                })
                self._gcp_pending_pixel   = None
                self._gcp_pending_map_pt  = None
                self._gcp_pending_src_rb  = None
                # Restore plan visibility for the next Phase 1 click
                if self._img_item is not None:
                    self._img_item.set_opacity(self._gcp_saved_opacity)
                self._gcp_phase = 1
                self.gcp_point_picked.emit(col, row_px, world_e, world_n)
                if self._panel:
                    self._panel.set_mode_hint("GCP Phase 1 — click a feature on the plan")
            return

        if e.button() == Qt.LeftButton:
            # ── GCP marker drag — check before normal drag handles ──────
            pair_idx, drag_type = self._gcp_pair_at(e.pos())
            if pair_idx is not None:
                self._drag_gcp_idx  = pair_idx
                self._drag_gcp_type = drag_type
                self._drag_mode     = "drag_gcp"
                self._canvas.setCursor(QCursor(Qt.ClosedHandCursor))
                return
            if self._is_on_pivot(e.pos()):
                self._start_move_pivot(plan_pt)
            elif self._is_on_handle(e.pos()):
                self._start_rotate(plan_pt)
            else:
                self._start_translate(plan_pt)
        elif e.button() == Qt.RightButton:
            self._start_rotate(plan_pt)

    def _start_translate(self, plan_pt: QgsPointXY):
        self._drag_mode    = "translate"
        self._press_plan   = (plan_pt.x(), plan_pt.y())
        ce, cn             = _gt_center(self._gt, self._W, self._H)
        self._press_center = (ce, cn)
        self._press_pivot  = (self._pivot_e, self._pivot_n)
        if self._panel:
            self._panel.set_mode_hint("Translating…")

    def _start_rotate(self, plan_pt: QgsPointXY):
        self._drag_mode = "rotate"
        dx = plan_pt.x() - self._pivot_e
        dy = plan_pt.y() - self._pivot_n
        self._prev_rot_angle = math.degrees(math.atan2(dy, dx))
        if self._panel:
            self._panel.set_mode_hint("Rotating around pivot…")

    def _start_move_pivot(self, plan_pt: QgsPointXY):
        self._drag_mode = "move_pivot"
        if self._panel:
            self._panel.set_mode_hint("Moving rotation pivot…")

    def canvasMoveEvent(self, e):
        if self._drag_mode is None:
            # Cursor hints — check GCP markers first, then standard handles
            if self._gcp_pairs and self._gcp_pair_at(e.pos())[0] is not None:
                self._canvas.setCursor(QCursor(Qt.OpenHandCursor))
            elif self._is_on_pivot(e.pos()):
                self._canvas.setCursor(QCursor(Qt.PointingHandCursor))
            elif self._is_on_handle(e.pos()):
                self._canvas.setCursor(QCursor(Qt.CrossCursor))
            else:
                self._canvas.setCursor(QCursor(Qt.SizeAllCursor))
            return

        map_pt  = self.toMapCoordinates(e.pos())
        plan_pt = self._to_plan(map_pt)
        px, py  = plan_pt.x(), plan_pt.y()

        if self._drag_mode == "drag_gcp":
            pair = self._gcp_pairs[self._drag_gcp_idx]
            if self._drag_gcp_type == "src":
                pair["src_rb"].reset(QgsWkbTypes.PointGeometry)
                pair["src_rb"].addPoint(map_pt)
                pair["link_rb"].reset(QgsWkbTypes.LineGeometry)
                pair["link_rb"].addPoint(map_pt)
                pair["link_rb"].addPoint(pair["tgt_map_pt"])
                pair["src_map_pt"] = map_pt
            else:
                pair["tgt_rb"].reset(QgsWkbTypes.PointGeometry)
                pair["tgt_rb"].addPoint(map_pt)
                pair["link_rb"].reset(QgsWkbTypes.LineGeometry)
                pair["link_rb"].addPoint(pair["src_map_pt"])
                pair["link_rb"].addPoint(map_pt)
                pair["tgt_map_pt"] = map_pt
            return

        if self._drag_mode == "translate":
            de = px - self._press_plan[0]
            dn = py - self._press_plan[1]
            new_ce = self._press_center[0] + de
            new_cn = self._press_center[1] + dn
            self._pivot_e = self._press_pivot[0] + de   # pivot follows image
            self._pivot_n = self._press_pivot[1] + dn
            self._gt = list(_gt_from_center_rot_scale(
                new_ce, new_cn, _gt_rotation_deg(self._gt),
                self._scale_x, self._scale_y, self._W, self._H,
            ))

        elif self._drag_mode == "rotate":
            dx = px - self._pivot_e
            dy = py - self._pivot_n
            cur_angle   = math.degrees(math.atan2(dy, dx))
            delta_angle = cur_angle - self._prev_rot_angle
            self._prev_rot_angle = cur_angle
            old_ce, old_cn = _gt_center(self._gt, self._W, self._H)
            new_ce, new_cn = _rotate_around(
                old_ce, old_cn, self._pivot_e, self._pivot_n, delta_angle
            )
            new_rot = _gt_rotation_deg(self._gt) + delta_angle
            self._gt = list(_gt_from_center_rot_scale(
                new_ce, new_cn, new_rot, self._scale_x, self._scale_y, self._W, self._H,
            ))

        elif self._drag_mode == "move_pivot":
            self._pivot_e = px
            self._pivot_n = py

        self._schedule_update()

    def _schedule_update(self):
        """Throttle redraws to ~60 fps during fast drags."""
        self._update_dirty = True
        if not self._update_timer.isActive():
            self._update_timer.start()

    def canvasReleaseEvent(self, e):
        if self._drag_mode == "drag_gcp" and self._drag_gcp_idx is not None:
            # Recompute pixel / world coords from final marker positions and sync table
            pair = self._gcp_pairs[self._drag_gcp_idx]
            plan_src = self._to_plan(pair["src_map_pt"])
            col, row_px = self._canvas_to_image_pixel(plan_src)
            plan_tgt = self._to_plan(pair["tgt_map_pt"])
            world_e, world_n = plan_tgt.x(), plan_tgt.y()
            self.gcp_pair_updated.emit(self._drag_gcp_idx, col, row_px, world_e, world_n)
            self._drag_gcp_idx  = None
            self._drag_gcp_type = ""
            self._drag_mode     = None
            self._canvas.setCursor(QCursor(Qt.SizeAllCursor))
            return

        self._drag_mode      = None
        self._press_plan     = None
        self._press_center   = None
        self._press_pivot    = None
        self._prev_rot_angle = None
        # Flush any pending throttled update immediately on release
        if self._update_dirty:
            self._update_timer.stop()
            self._update_all()
        if self._panel and self._gcp_phase == 0:
            self._panel.set_mode_hint(_HINT_IDLE)

    def keyPressEvent(self, e):
        """Escape exits GCP pick mode; other keys pass through."""
        if self._gcp_phase > 0 and e.key() == Qt.Key_Escape:
            self._on_gcp_pick_stop()
            return
        super().keyPressEvent(e)

    def wheelEvent(self, e):
        """Shift+wheel rotates; plain wheel delegates to the canvas wheel zoom."""
        if not (e.modifiers() & Qt.ShiftModifier):
            try:
                self._canvas.wheelEvent(e)
            except Exception:
                e.ignore()
            return
        ticks = e.angleDelta().y() / 120.0
        if ticks == 0:
            e.accept()
            return
        delta_deg  = ticks * 1.0
        old_ce, old_cn = _gt_center(self._gt, self._W, self._H)
        new_ce, new_cn = _rotate_around(
            old_ce, old_cn, self._pivot_e, self._pivot_n, delta_deg
        )
        new_rot = _gt_rotation_deg(self._gt) + delta_deg
        self._gt = list(_gt_from_center_rot_scale(
            new_ce, new_cn, new_rot, self._scale_x, self._scale_y, self._W, self._H,
        ))
        self._update_all()
        e.accept()

    # ----------------------------------------------------------------
    # Public actions
    # ----------------------------------------------------------------

    def reset_to_original(self):
        self._gt    = list(self._orig_gt)
        self._scale_x, self._scale_y = self._orig_scale_x, self._orig_scale_y
        ce, cn      = _gt_center(self._orig_gt, self._W, self._H)
        self._pivot_e, self._pivot_n = ce, cn
        self._drag_mode = None
        self._update_all()
        if self._panel:
            self._panel.set_scale_values(self._scale_x, self._scale_y)
            self._panel.set_mode_hint(_HINT_IDLE)

    def _current_gt(self) -> tuple:
        return tuple(self._gt)

    def _on_accept(self):
        self.accepted.emit(self._current_gt())

    def _on_cancel(self):
        self.cancelled.emit()
