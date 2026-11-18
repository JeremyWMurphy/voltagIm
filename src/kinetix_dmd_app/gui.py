import time
from datetime import datetime
import numpy as np
import tifffile

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import config as cfg


class HistogramWidget(QWidget):
    """Lightweight real-time 8-bit histogram with Min/Max contrast window overlay."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(110)
        self.setMaximumHeight(130)
        self.hist = np.zeros(256, dtype=np.float32)
        self.min_val = 0
        self.max_val = 255

    def update_data(self, frame_8bit: np.ndarray, min_val: int, max_val: int):
        counts = np.bincount(frame_8bit.ravel(), minlength=256).astype(np.float32)
        counts = np.log1p(counts)
        peak = counts.max()
        self.hist = (counts / peak) if peak > 0 else counts
        self.min_val = min_val
        self.max_val = max_val
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor("#181818"))

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#4fc3f7"))
        bar_w = w / 256.0
        for i in range(256):
            bar_h = int(self.hist[i] * (h - 10))
            if bar_h > 0:
                painter.drawRect(int(i * bar_w), h - bar_h, max(1, int(bar_w) + 1), bar_h)

        x_min = int((self.min_val / 255.0) * (w - 1))
        x_max = int((self.max_val / 255.0) * (w - 1))

        painter.setPen(QPen(QColor("#66bb6a"), 2))
        painter.drawLine(x_min, 0, x_min, h)

        painter.setPen(QPen(QColor("#ffee58"), 2))
        painter.drawLine(x_max, 0, x_max, h)

        painter.setPen(QPen(QColor("#ffffff"), 1, Qt.DashLine))
        painter.drawLine(x_min, h - 2, x_max, 2)


class MicroscopeGUI(QMainWindow):
    def __init__(
        self,
        shm,
        write_idx,
        read_idx,
        preview_counter,
        preview_event,
        record_event,
        shutdown_event,
        exposure_us,
        roi_x,
        roi_y,
        roi_w,
        roi_h,
        run_folder_buf,
        dmd_ctrl,
    ):
        super().__init__()
        self.setWindowTitle("Kinetix22 (8-Bit Speed) + Static DMD Controller")
        self.resize(1060, 900)

        self.shm = shm
        self.ring_buffer = np.ndarray(
            (cfg.TOTAL_SLOTS, cfg.MAX_HEIGHT, cfg.MAX_WIDTH), dtype=cfg.DTYPE, buffer=shm.buf
        )
        self.write_idx = write_idx
        self.read_idx = read_idx
        self.preview_counter = preview_counter
        self.preview_event = preview_event
        self.record_event = record_event
        self.shutdown_event = shutdown_event
        self.exposure_us = exposure_us
        self.roi_x, self.roi_y, self.roi_w, self.roi_h = roi_x, roi_y, roi_w, roi_h
        self.run_folder_buf = run_folder_buf
        self.dmd_ctrl = dmd_ctrl

        self.last_t = time.perf_counter()
        self.last_w = 0
        self.latest_sub = None
        self.lut = np.arange(256, dtype=np.uint8)

        self._build_ui()

        default_mask = self.dmd_ctrl.create_center_aperture(radius_px=250)
        self.dmd_ctrl.display_static_mask(default_mask)

        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_ui)
        self.timer.start(66)

    def _build_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)

        # --- 1. DMD Controls ---
        dmd_box = QGroupBox("Static ViALUX DMD Mask")
        dmd_layout = QHBoxLayout(dmd_box)

        btn_load_mask = QPushButton("Load Mask (.npy / .tif / .png)")
        btn_load_mask.clicked.connect(self.load_mask_dialog)
        btn_aperture = QPushButton("Center Aperture (250 px)")
        btn_aperture.clicked.connect(
            lambda: self.dmd_ctrl.display_static_mask(self.dmd_ctrl.create_center_aperture(250))
        )
        btn_all_on = QPushButton("All Mirrors ON")
        btn_all_on.clicked.connect(
            lambda: self.dmd_ctrl.display_static_mask(
                np.full((self.dmd_ctrl.height, self.dmd_ctrl.width), 255, dtype=np.uint8)
            )
        )

        dmd_layout.addWidget(btn_load_mask)
        dmd_layout.addWidget(btn_aperture)
        dmd_layout.addWidget(btn_all_on)
        layout.addWidget(dmd_box)

        # --- 2. Exposure (us) & ROI Cropping Controls ---
        roi_box = QGroupBox("Timing (us) & Sensor ROI Cropping (Multiples of 8 px)")
        roi_layout = QHBoxLayout(roi_box)

        roi_layout.addWidget(QLabel("Exposure (us):"))
        self.spin_exp = QSpinBox()
        self.spin_exp.setRange(20, 1000000)
        self.spin_exp.setSingleStep(100)
        self.spin_exp.setValue(1000)
        self.spin_exp.valueChanged.connect(lambda v: setattr(self.exposure_us, "value", v))
        roi_layout.addWidget(self.spin_exp)

        self.spin_x = QSpinBox()
        self.spin_x.setRange(0, 2336)
        self.spin_x.setSingleStep(8)
        self.spin_x.setValue(0)

        self.spin_y = QSpinBox()
        self.spin_y.setRange(0, 2336)
        self.spin_y.setSingleStep(8)
        self.spin_y.setValue(0)

        self.spin_w = QSpinBox()
        self.spin_w.setRange(64, 2400)
        self.spin_w.setSingleStep(8)
        self.spin_w.setValue(2400)

        self.spin_h = QSpinBox()
        self.spin_h.setRange(64, 2400)
        self.spin_h.setSingleStep(8)
        self.spin_h.setValue(2400)

        for lbl, sp in [("X:", self.spin_x), ("Y:", self.spin_y), ("W:", self.spin_w), ("H:", self.spin_h)]:
            roi_layout.addWidget(QLabel(lbl))
            roi_layout.addWidget(sp)

        btn_apply_roi = QPushButton("Apply ROI")
        btn_apply_roi.clicked.connect(self.apply_roi_values)
        btn_center_strip = QPushButton("Center 2400x600 (Fast)")
        btn_center_strip.clicked.connect(lambda: self.set_roi_preset(0, 900, 2400, 600))
        btn_full_fov = QPushButton("Full 2400x2400")
        btn_full_fov.clicked.connect(lambda: self.set_roi_preset(0, 0, 2400, 2400))

        roi_layout.addWidget(btn_apply_roi)
        roi_layout.addWidget(btn_center_strip)
        roi_layout.addWidget(btn_full_fov)
        layout.addWidget(roi_box)

        # --- 3. Acquisition Controls ---
        cam_box = QGroupBox("Acquisition Control")
        cam_layout = QHBoxLayout(cam_box)

        self.chk_led_ttl = QCheckBox("LED Strobe Sync")
        self.chk_led_ttl.setChecked(True)
        cam_layout.addWidget(self.chk_led_ttl)

        self.btn_preview = QPushButton("Live Preview")
        self.btn_preview.setStyleSheet("background-color: #1565c0; color: white; font-weight: bold;")
        self.btn_preview.clicked.connect(self.on_preview)

        self.btn_start = QPushButton("Start Streaming to NVMe")
        self.btn_start.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")
        self.btn_start.clicked.connect(self.on_start)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setStyleSheet("background-color: #ef6c00; color: white; font-weight: bold;")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.on_stop)

        self.btn_quit = QPushButton("Quit")
        self.btn_quit.setStyleSheet("background-color: #b71c1c; color: white; font-weight: bold;")
        self.btn_quit.clicked.connect(self.on_quit)

        cam_layout.addWidget(self.btn_preview)
        cam_layout.addWidget(self.btn_start)
        cam_layout.addWidget(self.btn_stop)
        cam_layout.addWidget(self.btn_quit)
        layout.addWidget(cam_box)

        # --- 4. Telemetry ---
        self.lbl_status = QLabel(f"Status: Idle | Buffer: 0 / {cfg.BUFFER_SLOTS} frames")
        self.lbl_status.setStyleSheet("font-size: 13px; font-weight: bold;")
        layout.addWidget(self.lbl_status)

        # --- 5. Live Preview + Histogram Panel ---
        view_layout = QHBoxLayout()

        self.lbl_preview = QLabel("Live 8-Bit Preview")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setMinimumSize(560, 520)
        self.lbl_preview.setStyleSheet("background-color: #111; color: #888;")
        view_layout.addWidget(self.lbl_preview, stretch=3)

        hist_panel = QGroupBox("Display Contrast & Histogram (Raw Data Saved Unchanged)")
        hist_layout = QVBoxLayout(hist_panel)

        self.hist_widget = HistogramWidget()
        hist_layout.addWidget(self.hist_widget)

        self.lbl_min = QLabel("Black Point (Min): 0")
        self.slider_min = QSlider(Qt.Horizontal)
        self.slider_min.setRange(0, 254)
        self.slider_min.setValue(0)
        self.slider_min.valueChanged.connect(self.update_lut)

        self.lbl_max = QLabel("White Point (Max): 255")
        self.slider_max = QSlider(Qt.Horizontal)
        self.slider_max.setRange(1, 255)
        self.slider_max.setValue(255)
        self.slider_max.valueChanged.connect(self.update_lut)

        btn_auto_contrast = QPushButton("Auto-Stretch Contrast (0.5% - 99.5%)")
        btn_auto_contrast.clicked.connect(self.auto_stretch_contrast)
        btn_reset_contrast = QPushButton("Reset (0 - 255)")
        btn_reset_contrast.clicked.connect(
            lambda: (self.slider_min.setValue(0), self.slider_max.setValue(255))
        )

        hist_layout.addWidget(self.lbl_min)
        hist_layout.addWidget(self.slider_min)
        hist_layout.addWidget(self.lbl_max)
        hist_layout.addWidget(self.slider_max)
        hist_layout.addWidget(btn_auto_contrast)
        hist_layout.addWidget(btn_reset_contrast)
        hist_layout.addStretch()

        view_layout.addWidget(hist_panel, stretch=1)
        layout.addLayout(view_layout)

        self.setCentralWidget(central)

    def set_roi_preset(self, x, y, w, h):
        self.spin_x.setValue(x)
        self.spin_y.setValue(y)
        self.spin_w.setValue(w)
        self.spin_h.setValue(h)
        self.apply_roi_values()

    def apply_roi_values(self):
        w = max(64, (self.spin_w.value() // 8) * 8)
        h = max(64, (self.spin_h.value() // 8) * 8)
        x = max(0, min(cfg.MAX_WIDTH - w, (self.spin_x.value() // 8) * 8))
        y = max(0, min(cfg.MAX_HEIGHT - h, (self.spin_y.value() // 8) * 8))
        self.spin_x.setValue(x)
        self.spin_y.setValue(y)
        self.spin_w.setValue(w)
        self.spin_h.setValue(h)
        self.roi_x.value, self.roi_y.value, self.roi_w.value, self.roi_h.value = x, y, w, h

    def update_lut(self):
        lo = self.slider_min.value()
        hi = max(lo + 1, self.slider_max.value())

        if self.slider_max.value() <= lo:
            self.slider_max.setValue(hi)

        self.lbl_min.setText(f"Black Point (Min): {lo}")
        self.lbl_max.setText(f"White Point (Max): {hi}")
        x = np.arange(256, dtype=np.float32)
        scaled = np.clip((x - lo) * (255.0 / (hi - lo)), 0, 255)
        self.lut = scaled.astype(np.uint8)

    def auto_stretch_contrast(self):
        if self.latest_sub is None:
            return

        lo, hi = np.percentile(self.latest_sub, [0.5, 99.5])
        lo = int(max(0, min(253, lo)))
        hi = int(max(lo + 2, min(255, hi)))
        self.slider_min.setValue(lo)
        self.slider_max.setValue(hi)

    def load_mask_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select DMD Binary Mask", "", "Images/Arrays (*.npy *.tif *.tiff *.png *.bmp)"
        )
        if not path:
            return
        if path.endswith(".npy"):
            arr = np.load(path)
        elif path.lower().endswith((".tif", ".tiff")):
            arr = tifffile.imread(path)
        else:
            qimg = QImage(path).convertToFormat(QImage.Format_Grayscale8)
            arr = np.frombuffer(qimg.bits(), np.uint8).reshape((qimg.height(), qimg.width())).copy()

        self.dmd_ctrl.display_static_mask(arr)

    def on_preview(self):
        self.record_event.clear()
        self.apply_roi_values()
        self.exposure_us.value = self.spin_exp.value()
        self.preview_event.set()
        self.btn_preview.setEnabled(False)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(True)

    def on_start(self):
        self.preview_event.clear()
        time.sleep(0.05)
        self.apply_roi_values()
        with self.write_idx.get_lock():
            self.write_idx.value = 0
        with self.read_idx.get_lock():
            self.read_idx.value = 0
        self.last_w = 0
        timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_folder_buf.value = timestamp.encode("utf-8")
        self.exposure_us.value = self.spin_exp.value()
        self.record_event.set()
        self.btn_preview.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.spin_exp.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def on_stop(self):
        self.preview_event.clear()
        self.record_event.clear()
        self.btn_stop.setEnabled(False)
        self.btn_preview.setEnabled(True)
        self.btn_start.setEnabled(True)
        self.spin_exp.setEnabled(True)

    def on_quit(self):
        self.preview_event.clear()
        self.record_event.clear()
        self.shutdown_event.set()
        self.close()

    def closeEvent(self, event):
        self.preview_event.clear()
        self.record_event.clear()
        self.shutdown_event.set()
        super().closeEvent(event)

    def refresh_ui(self):
        now = time.perf_counter()
        w = self.write_idx.value
        r = self.read_idx.value

        dt = now - self.last_t
        live_fps = (w - self.last_w) / dt if dt > 0 else 0.0
        self.last_t = now
        self.last_w = w

        buffered = w - r
        roi_w, roi_h = self.roi_w.value, self.roi_h.value
        gb_used = (buffered * roi_w * roi_h) / (1024 ** 3)

        if self.record_event.is_set():
            mode_str = "RECORDING"
        elif self.preview_event.is_set():
            mode_str = "LIVE PREVIEW"
        elif buffered > 0:
            mode_str = "FLUSHING"
        else:
            mode_str = "IDLE"

        self.lbl_status.setText(
            f"[{mode_str} | {roi_w}x{roi_h}] Rate: {live_fps:.1f} FPS | "
            f"Buffer: {buffered} ({gb_used:.2f} GB) | Captured: {w} | Written: {r}"
        )

        display_slot = None
        if self.preview_event.is_set() and self.preview_counter.value > 0:
            display_slot = cfg.PREVIEW_SLOT
        elif w > 0:
            display_slot = (w - 1) % cfg.BUFFER_SLOTS

        if display_slot is not None:
            step = max(1, max(roi_w, roi_h) // 400)
            sub = np.ascontiguousarray(self.ring_buffer[display_slot, :roi_h:step, :roi_w:step])
            self.latest_sub = sub
            self.hist_widget.update_data(sub, self.slider_min.value(), self.slider_max.value())
            disp = np.ascontiguousarray(self.lut[sub])
            h, width = disp.shape
            qimg = QImage(disp.data, width, h, width, QImage.Format_Grayscale8)
            self.lbl_preview.setPixmap(
                QPixmap.fromImage(qimg).scaled(
                    self.lbl_preview.size(), Qt.KeepAspectRatio, Qt.FastTransformation
                )
            )
