"""
fitsblink.py - Ein schlanker FITS-Viewer fuer Windows im Stil von PixInsight Blink / AstroBlink.

Bedienung:
  Ordner oeffnen        : Button oder Strg+O
  Naechstes / Vorheriges: Pfeil rechts / links
  Erstes / Letztes      : Pos1 / Ende
  In Papierkorb         : Entf
  Loeschen rueckgaengig : Strg+Z
  Zoom                  : Mausrad (zentriert auf Mauszeiger)
  Pan                   : Linke Maustaste ziehen
  Reset Ansicht         : Doppelklick
  Stretch sperren       : Lock-Button (oder L)

Installation:
  pip install pyqt6 astropy pyqtgraph send2trash colour-demosaicing numpy xisf
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from astropy.io import fits
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QAction, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from send2trash import send2trash

try:
    from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
    HAVE_DEMOSAIC = True
except ImportError:
    HAVE_DEMOSAIC = False

try:
    from xisf import XISF
    HAVE_XISF = True
except ImportError:
    HAVE_XISF = False


FITS_EXT = {".fit", ".fits", ".fts", ".fz"}
XISF_EXT = {".xisf"}
SUPPORTED_EXT = FITS_EXT | XISF_EXT


# ---------------------------------------------------------------------------
# FITS-Loader
# ---------------------------------------------------------------------------

@dataclass
class LoadedFits:
    data: np.ndarray          # float32, (H,W) mono oder (H,W,3) rgb, 0..1 normalisiert (roh)
    header: fits.Header
    is_color: bool
    path: Path


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    # Auf den vollen Wertebereich des Sensors normalisieren, ohne zu strecken.
    # Wir teilen einfach durch das Maximum des Datentyps falls vorhanden, sonst durch das Bildmaximum.
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)
    mn = float(finite.min())
    mx = float(finite.max())
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = np.zeros_like(arr)
    return np.clip(arr, 0.0, 1.0)


def _detect_bayer(header: fits.Header) -> Optional[str]:
    for key in ("BAYERPAT", "BAYRPAT", "COLORTYP", "MOSAIC"):
        if key in header:
            val = str(header[key]).strip().upper()
            if val in ("RGGB", "BGGR", "GRBG", "GBRG"):
                return val
    return None


def load_xisf(path: Path) -> LoadedFits:
    if not HAVE_XISF:
        raise RuntimeError("xisf-Bibliothek nicht installiert (pip install xisf)")
    x = XISF(str(path))
    images = x.get_images_metadata()
    if not images:
        raise ValueError(f"Keine Bilder in {path.name}")
    data = x.read_image(0)  # numpy array
    meta = images[0]

    # XISF-Daten kommen typischerweise als (H,W) oder (H,W,C)
    if data.ndim == 3 and data.shape[2] == 1:
        data = data[:, :, 0]

    # In ein FITS-aehnliches Header-Objekt fuer die Anzeige uebersetzen
    header = fits.Header()
    try:
        for k, v in (meta.get("FITSKeywords") or {}).items():
            # FITSKeywords ist {keyword: [{"value":..., "comment":...}, ...]}
            if isinstance(v, list) and v:
                val = v[0].get("value", "")
                com = v[0].get("comment", "")
                header[k[:8]] = (val, com)
    except Exception:
        pass
    # XISFProperties als Zusatz-Hinweis
    try:
        for k, v in (meta.get("XISFProperties") or {}).items():
            short = k.split(":")[-1][:8].upper()
            if short and short not in header:
                val = v.get("value") if isinstance(v, dict) else v
                header[short] = str(val)[:60]
    except Exception:
        pass

    is_color = data.ndim == 3 and data.shape[2] == 3
    if is_color:
        data = _normalize(data)
        return LoadedFits(data=data, header=header, is_color=True, path=path)

    data = _normalize(data)

    # Bayer-Pattern aus XISF-Metadaten extrahieren
    bayer = None
    try:
        cfa = (meta.get("XISFProperties") or {}).get("PCL:CFASourcePattern")
        if isinstance(cfa, dict):
            val = str(cfa.get("value", "")).strip().upper()
            if val in ("RGGB", "BGGR", "GRBG", "GBRG"):
                bayer = val
    except Exception:
        pass
    if bayer is None:
        bayer = _detect_bayer(header)

    if bayer and HAVE_DEMOSAIC:
        rgb = demosaicing_CFA_Bayer_bilinear(data, bayer)
        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
        return LoadedFits(data=rgb, header=header, is_color=True, path=path)

    return LoadedFits(data=data, header=header, is_color=False, path=path)


def load_image(path: Path) -> LoadedFits:
    """Dispatcher fuer FITS und XISF."""
    if path.suffix.lower() in XISF_EXT:
        return load_xisf(path)
    return load_fits(path)


def load_fits(path: Path) -> LoadedFits:
    with fits.open(path, memmap=False) as hdul:
        # Erste HDU mit Bilddaten suchen
        hdu = None
        for h in hdul:
            if h.data is not None and h.data.ndim >= 2:
                hdu = h
                break
        if hdu is None:
            raise ValueError(f"Keine Bilddaten in {path.name}")
        data = np.asarray(hdu.data)
        header = hdu.header.copy()

    # 3D: erste Ebene oder als RGB interpretieren
    if data.ndim == 3:
        # astropy liefert bei RGB FITS oft (3,H,W)
        if data.shape[0] == 3:
            data = np.transpose(data, (1, 2, 0))
            data = _normalize(data)
            return LoadedFits(data=data, header=header, is_color=True, path=path)
        else:
            data = data[0]

    data = _normalize(data)

    # Bayer-Debayering wenn erkannt und moeglich
    bayer = _detect_bayer(header)
    if bayer and HAVE_DEMOSAIC:
        rgb = demosaicing_CFA_Bayer_bilinear(data, bayer)
        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
        return LoadedFits(data=rgb, header=header, is_color=True, path=path)

    return LoadedFits(data=data, header=header, is_color=False, path=path)


# ---------------------------------------------------------------------------
# STF-Autostretch (PixInsight-Stil)
# ---------------------------------------------------------------------------

def _mtf(m: float, x: np.ndarray) -> np.ndarray:
    """Midtones Transfer Function von PixInsight."""
    if m <= 0.0:
        return np.ones_like(x)
    if m >= 1.0:
        return np.zeros_like(x)
    if m == 0.5:
        return x
    # (m-1)*x / ((2m-1)*x - m)
    num = (m - 1.0) * x
    den = (2.0 * m - 1.0) * x - m
    out = np.where(den != 0.0, num / den, 0.0)
    return out


def auto_stf_params(
    image: np.ndarray,
    shadows_clip: float = -2.8,
    target_bg: float = 0.25,
) -> tuple[float, float, float]:
    """
    Berechnet (shadows, midtones, highlights) im Bereich [0,1] aus dem Bild.
    Folgt der PixInsight ScreenTransferFunction-AutoStretch-Logik.
    """
    # Bei Farbbildern den Luminanz-Median nehmen, simpel ueber alle Kanaele
    sample = image
    if sample.ndim == 3:
        sample = sample.mean(axis=2)

    # Subsampling fuer Performance bei grossen Frames
    if sample.size > 1_000_000:
        step = int(np.ceil(np.sqrt(sample.size / 1_000_000)))
        sample = sample[::step, ::step]

    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return (0.0, 0.5, 1.0)

    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    # Normalisiertes MAD (Konsistenzfaktor mit sigma)
    nmad = mad * 1.4826 if mad > 0 else 1e-6

    # Shadows-Clipping: median + shadows_clip*nmad (shadows_clip ist negativ)
    c0 = median + shadows_clip * nmad
    c0 = float(np.clip(c0, 0.0, 1.0))

    # Midtone via MTF so dass Median nach Stretch auf target_bg landet
    if median > c0:
        m = _mtf(target_bg, np.array([median - c0]) / (1.0 - c0))[0]
        m = float(np.clip(m, 0.0, 1.0))
    else:
        m = 0.5

    return (c0, m, 1.0)


def apply_stf(image: np.ndarray, shadows: float, midtones: float, highlights: float) -> np.ndarray:
    """Wendet Shadows/Midtones/Highlights auf das Bild an. Ergebnis 0..1."""
    if highlights <= shadows:
        highlights = shadows + 1e-6
    x = (image - shadows) / (highlights - shadows)
    x = np.clip(x, 0.0, 1.0)
    x = _mtf(midtones, x)
    return x.astype(np.float32)


# ---------------------------------------------------------------------------
# Hintergrund-Loader (Caching der Nachbarn)
# ---------------------------------------------------------------------------

class LoaderWorker(QObject):
    loaded = pyqtSignal(str, object)  # path, LoadedFits oder None
    finished = pyqtSignal()

    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    def run(self):
        try:
            lf = load_image(self.path)
            self.loaded.emit(str(self.path), lf)
        except Exception as e:
            print(f"[loader] {self.path.name}: {e}", file=sys.stderr)
            self.loaded.emit(str(self.path), None)
        self.finished.emit()


class FitsCache:
    """Sehr einfacher LRU-Cache fuer geladene FITS."""

    def __init__(self, capacity: int = 5):
        self.capacity = capacity
        self._store: dict[str, LoadedFits] = {}
        self._order: list[str] = []

    def get(self, path: Path) -> Optional[LoadedFits]:
        key = str(path)
        if key in self._store:
            self._order.remove(key)
            self._order.append(key)
            return self._store[key]
        return None

    def put(self, path: Path, lf: LoadedFits) -> None:
        key = str(path)
        if key in self._store:
            self._order.remove(key)
        self._store[key] = lf
        self._order.append(key)
        while len(self._order) > self.capacity:
            old = self._order.pop(0)
            self._store.pop(old, None)

    def drop(self, path: Path) -> None:
        key = str(path)
        self._store.pop(key, None)
        if key in self._order:
            self._order.remove(key)


# ---------------------------------------------------------------------------
# Hauptfenster
# ---------------------------------------------------------------------------

class FitsBlink(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FitsBlink")
        self.resize(1400, 900)

        self.files: list[Path] = []
        self.index: int = -1
        self.cache = FitsCache(capacity=7)
        self.undo_stack: list[Path] = []  # Pfade im Papierkorb (fuer Anzeige; echtes Undo via os)
        self.deleted_originals: list[Path] = []

        # Stretch-State
        self.stretch_locked = False
        self.shadows = 0.0
        self.midtones = 0.5
        self.highlights = 1.0

        # Zoom/Pan-State (wird beim Blaettern beibehalten)
        self.saved_view_range: Optional[tuple] = None
        self.first_image_shown = False

        self._loader_threads: list[QThread] = []

        self._build_ui()
        self._build_shortcuts()

    # ---------- UI ----------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # Linke Seite: Bildanzeige
        left = QVBoxLayout()
        root.addLayout(left, stretch=4)

        topbar = QHBoxLayout()
        self.btn_open = QPushButton("Ordner oeffnen")
        self.btn_open.clicked.connect(self.open_directory)
        topbar.addWidget(self.btn_open)
        self.lbl_file = QLabel("Kein Ordner geladen")
        self.lbl_file.setStyleSheet("font-weight: bold;")
        topbar.addWidget(self.lbl_file, stretch=1)
        left.addLayout(topbar)

        self.image_view = pg.ImageView()
        self.image_view.ui.histogram.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        # Doppelklick = Reset
        self.image_view.getView().scene().sigMouseClicked.connect(self._on_scene_click)
        left.addWidget(self.image_view, stretch=1)

        self.lbl_status = QLabel("")
        left.addWidget(self.lbl_status)

        # Rechte Seite: Controls
        right = QVBoxLayout()
        root.addLayout(right, stretch=1)

        right.addWidget(QLabel("<b>Autostretch (STF)</b>"))

        self.cb_lock = QCheckBox("Stretch sperren (L)")
        self.cb_lock.stateChanged.connect(self._on_lock_toggled)
        right.addWidget(self.cb_lock)

        self.btn_reset_stretch = QPushButton("Auto neu berechnen")
        self.btn_reset_stretch.clicked.connect(self._recompute_auto_stretch)
        right.addWidget(self.btn_reset_stretch)

        self.sld_shadows, lbl_s = self._make_slider("Shadows", 0, 1000, 0, right)
        self.sld_midtones, lbl_m = self._make_slider("Midtones", 1, 999, 500, right)
        self.sld_highlights, lbl_h = self._make_slider("Highlights", 0, 1000, 1000, right)
        self.lbl_shadows_val = lbl_s
        self.lbl_midtones_val = lbl_m
        self.lbl_highlights_val = lbl_h
        for s in (self.sld_shadows, self.sld_midtones, self.sld_highlights):
            s.valueChanged.connect(self._on_slider_changed)

        right.addSpacing(20)
        right.addWidget(QLabel("<b>FITS-Header</b>"))
        self.txt_header = QTextEdit()
        self.txt_header.setReadOnly(True)
        self.txt_header.setStyleSheet("font-family: Consolas, monospace; font-size: 10pt;")
        right.addWidget(self.txt_header, stretch=1)

        right.addWidget(QLabel(
            "<small>← → blaettern &nbsp; Pos1/Ende erstes/letztes<br>"
            "Entf in Papierkorb &nbsp; Strg+Z rueckgaengig<br>"
            "Mausrad zoom &nbsp; Drag pan &nbsp; Doppelklick reset</small>"
        ))

    def _make_slider(self, name: str, lo: int, hi: int, val: int, parent_layout) -> tuple[QSlider, QLabel]:
        # Wichtig: wrap muss eine Python-Referenz behalten, sonst killt der GC den Slider mit.
        # Wir haengen es deshalb sofort ans parent_layout, damit Qt es als Kind haelt.
        wrap = QWidget()
        parent_layout.addWidget(wrap)
        v = QVBoxLayout(wrap)
        v.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel(name, parent=wrap))
        lbl_val = QLabel(f"{val/1000:.3f}", parent=wrap)
        lbl_val.setMinimumWidth(50)
        row.addWidget(lbl_val)
        v.addLayout(row)
        sld = QSlider(Qt.Orientation.Horizontal, parent=wrap)
        sld.setRange(lo, hi)
        sld.setValue(val)
        sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Pfeiltasten sollen NICHT vom Slider geschluckt werden
        v.addWidget(sld)
        return sld, lbl_val

    def _build_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self.open_directory)
        QShortcut(QKeySequence("Right"), self, activated=self.next_file)
        QShortcut(QKeySequence("Left"), self, activated=self.prev_file)
        QShortcut(QKeySequence("Home"), self, activated=lambda: self.goto(0))
        QShortcut(QKeySequence("End"), self, activated=lambda: self.goto(len(self.files) - 1))
        QShortcut(QKeySequence("Delete"), self, activated=self.delete_current)
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.undo_delete)
        QShortcut(QKeySequence("L"), self, activated=lambda: self.cb_lock.toggle())

    # ---------- Verzeichnis / Navigation ----------

    def open_directory(self):
        d = QFileDialog.getExistingDirectory(self, "FITS-Ordner waehlen")
        if not d:
            return
        path = Path(d)
        files = sorted(
            [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXT]
        )
        if not files:
            self.lbl_file.setText(f"{path.name} (keine FITS/XISF gefunden)")
            return
        self.files = files
        self.index = 0
        self.first_image_shown = False
        self.saved_view_range = None
        self.show_current()

    def goto(self, idx: int):
        if not self.files:
            return
        idx = max(0, min(len(self.files) - 1, idx))
        if idx == self.index:
            return
        self._capture_view_range()
        self.index = idx
        self.show_current()

    def next_file(self):
        if self.files and self.index < len(self.files) - 1:
            self._capture_view_range()
            self.index += 1
            self.show_current()

    def prev_file(self):
        if self.files and self.index > 0:
            self._capture_view_range()
            self.index -= 1
            self.show_current()

    # ---------- Anzeige ----------

    def show_current(self):
        if not self.files or self.index < 0:
            return
        path = self.files[self.index]
        self.lbl_file.setText(f"{path.name}   ({self.index + 1} / {len(self.files)})")
        self.lbl_status.setText("Lade...")
        QApplication.processEvents()

        lf = self.cache.get(path)
        if lf is None:
            try:
                lf = load_image(path)
                self.cache.put(path, lf)
            except Exception as e:
                self.lbl_status.setText(f"Fehler: {e}")
                return

        self._render(lf)
        self._update_header(lf.header)
        self.lbl_status.setText(
            f"{lf.data.shape[1]}x{lf.data.shape[0]}  "
            f"{'Farbe' if lf.is_color else 'Mono'}  "
            f"{'(Stretch gesperrt)' if self.stretch_locked else '(Auto-Stretch)'}"
        )
        self._prefetch_neighbors()

    def _render(self, lf: LoadedFits):
        if not self.stretch_locked:
            s, m, h = auto_stf_params(lf.data)
            self.shadows, self.midtones, self.highlights = s, m, h
            self._sync_sliders_from_state()

        stretched = apply_stf(lf.data, self.shadows, self.midtones, self.highlights)

        # pyqtgraph erwartet (W,H) bzw (W,H,3) - wir transponieren
        if stretched.ndim == 2:
            display = stretched.T
        else:
            display = np.transpose(stretched, (1, 0, 2))

        view = self.image_view.getView()
        if self.first_image_shown and self.saved_view_range is not None:
            self.image_view.setImage(display, autoRange=False, autoLevels=False, autoHistogramRange=False)
            (xr, yr) = self.saved_view_range
            view.setRange(xRange=xr, yRange=yr, padding=0)
        else:
            self.image_view.setImage(display, autoRange=True, autoLevels=False)
            self.first_image_shown = True

    def _update_header(self, header: fits.Header):
        lines = []
        for card in header.cards:
            try:
                k, v, c = card.keyword, card.value, card.comment
                if k:
                    lines.append(f"{k:<8} = {v}" + (f"  / {c}" if c else ""))
            except Exception:
                continue
        self.txt_header.setPlainText("\n".join(lines))

    # ---------- Stretch-Slider ----------

    def _on_lock_toggled(self, state):
        self.stretch_locked = bool(state)
        self.lbl_status.setText("(Stretch gesperrt)" if self.stretch_locked else "(Auto-Stretch)")

    def _recompute_auto_stretch(self):
        if not self.files:
            return
        lf = self.cache.get(self.files[self.index])
        if lf is None:
            return
        s, m, h = auto_stf_params(lf.data)
        self.shadows, self.midtones, self.highlights = s, m, h
        self._sync_sliders_from_state()
        self._render(lf)

    def _sync_sliders_from_state(self):
        for sld, val, lbl in (
            (self.sld_shadows, self.shadows, self.lbl_shadows_val),
            (self.sld_midtones, self.midtones, self.lbl_midtones_val),
            (self.sld_highlights, self.highlights, self.lbl_highlights_val),
        ):
            sld.blockSignals(True)
            sld.setValue(int(round(val * 1000)))
            sld.blockSignals(False)
            lbl.setText(f"{val:.3f}")

    def _on_slider_changed(self, _):
        self.shadows = self.sld_shadows.value() / 1000.0
        self.midtones = max(0.001, min(0.999, self.sld_midtones.value() / 1000.0))
        self.highlights = self.sld_highlights.value() / 1000.0
        self.lbl_shadows_val.setText(f"{self.shadows:.3f}")
        self.lbl_midtones_val.setText(f"{self.midtones:.3f}")
        self.lbl_highlights_val.setText(f"{self.highlights:.3f}")
        if not self.files:
            return
        lf = self.cache.get(self.files[self.index])
        if lf is None:
            return
        # Render mit gemerktem Zoom
        self._render_locked(lf)

    def _render_locked(self, lf: LoadedFits):
        # Wie _render, aber ohne Auto-Stretch zu ueberschreiben (Slider hat ihn ja gerade gesetzt)
        stretched = apply_stf(lf.data, self.shadows, self.midtones, self.highlights)
        if stretched.ndim == 2:
            display = stretched.T
        else:
            display = np.transpose(stretched, (1, 0, 2))
        view = self.image_view.getView()
        # aktuelle Range merken bevor wir setImage aufrufen
        xr, yr = view.viewRange()
        self.saved_view_range = (tuple(xr), tuple(yr))
        self.image_view.setImage(display, autoRange=False, autoLevels=False, autoHistogramRange=False)
        view.setRange(xRange=xr, yRange=yr, padding=0)

    # ---------- Zoom/Pan ----------

    def _on_scene_click(self, ev):
        if ev.double():
            self.image_view.getView().autoRange()
            self.saved_view_range = None
            ev.accept()

    def _capture_view_range(self):
        if not self.first_image_shown:
            return
        xr, yr = self.image_view.getView().viewRange()
        self.saved_view_range = (tuple(xr), tuple(yr))

    # ---------- Loeschen / Undo ----------

    def delete_current(self):
        if not self.files or self.index < 0:
            return
        # Aktuellen Zoom merken vor dem Wegnehmen
        self._capture_view_range()
        path = self.files[self.index]
        try:
            send2trash(str(path))
        except Exception as e:
            self.lbl_status.setText(f"Loeschen fehlgeschlagen: {e}")
            return
        self.cache.drop(path)
        self.deleted_originals.append(path)
        del self.files[self.index]
        if not self.files:
            self.image_view.clear()
            self.lbl_file.setText("Alle Dateien geloescht")
            self.index = -1
            return
        if self.index >= len(self.files):
            self.index = len(self.files) - 1
        self.show_current()

    def undo_delete(self):
        # send2trash bietet kein Undo. Wir koennen nur darauf hinweisen.
        if not self.deleted_originals:
            self.lbl_status.setText("Nichts rueckgaengig zu machen.")
            return
        last = self.deleted_originals.pop()
        self.lbl_status.setText(
            f"Letzte geloeschte Datei: {last.name} - bitte aus dem Papierkorb wiederherstellen "
            f"(Strg+Z im Explorer auf dem Desktop oder im Papierkorb)."
        )
        # Wir fuegen sie nicht automatisch wieder ein, weil wir nicht wissen ob der User sie wiederhergestellt hat.
        # Ein vollwertiges Undo wuerde ueber die Windows Shell API gehen - waere ein Folge-Feature.

    # ---------- Prefetch ----------

    def _prefetch_neighbors(self):
        for offset in (1, -1, 2):
            j = self.index + offset
            if 0 <= j < len(self.files):
                p = self.files[j]
                if self.cache.get(p) is None:
                    self._spawn_loader(p)

    def _spawn_loader(self, path: Path):
        thread = QThread()
        worker = LoaderWorker(path)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.loaded.connect(self._on_prefetch_done)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._loader_threads.append(thread)
        thread.start()

    def _on_prefetch_done(self, path_str: str, lf):
        if lf is not None:
            self.cache.put(Path(path_str), lf)

    # ---------- Resize: Zoom-Range aktuell halten ----------

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        # Nach Resize neu merken, falls der User gerade kein eigenes Zoom gesetzt hat
        if self.first_image_shown and self.saved_view_range is None:
            pass


def main():
    app = QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder="col-major")
    win = FitsBlink()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
