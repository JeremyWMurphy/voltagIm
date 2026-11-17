import os
import time
import ctypes
from datetime import datetime
import numpy as np
import tifffile
from multiprocessing import Process, Value, Event, Array, shared_memory

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSpinBox, QGroupBox, QFileDialog, QCheckBox, QSlider
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen

from pyvcam import pvc
from pyvcam.camera import Camera

try:
    from ALP4 import ALP4
    HAS_ALP4 = True
except ImportError:
    HAS_ALP4 = False

# --- HARDWARE & BUFFER CONFIGURATION ---
MAX_HEIGHT = 2400
MAX_WIDTH = 2400
DTYPE = np.uint8
FRAME_BYTES = MAX_HEIGHT * MAX_WIDTH * np.dtype(DTYPE).itemsize  # 5.76 MB max slot stride

# ~30 GB Ring Buffer (5,250 full-frame slots; holds 21,000+ frames when cropped 4x!)
BUFFER_SLOTS = 5250
TOTAL_SLOTS = BUFFER_SLOTS + 1
PREVIEW_SLOT = BUFFER_SLOTS
SHM_NAME = "kinetix22_8bit_ring"

NVME_PATHS = [r"D:\kinetix_stream", r"D:\kinetix_stream"]
FRAMES_PER_TIFF = 1000

# ==============================================================================
# 1. STATIC DMD CONTROLLER
# ==============================================================================
class StaticDMDController:
    def __init__(self, alp_version="4.3"):
        self.dmd = None
        self.connected = False
        self.width = 1024
        self.height = 768

        if not HAS_ALP4:
            print("[DMD] ALP4lib not installed. Running in simulated DMD mode.")
            return

        try:
            self.dmd = ALP4(version=alp_version)
            self.dmd.Initialize()
            self.width = self.dmd.nSizeX
            self.height = self.dmd.nSizeY
            self.connected = True
            print(f"[DMD] Initialized ViALUX DMD ({self.width} x {self.height})")
        except Exception as e:
            print(f"[DMD] Hardware not detected ({e}). Running in simulated mode.")
            self.connected = False

    def display_static_mask(self, mask_2d: np.ndarray):
        if not self.connected:
            print(f"[DMD Simulated] Loaded static mask shape {mask_2d.shape}")
            return
        
        if mask_2d.shape != (self.height, self.width):
            raise ValueError(f"Mask shape {mask_2d.shape} does not match DMD ({self.height}, {self.width})")
        
        binary_mask = np.where(mask_2d > 0, 255, 0).astype(np.uint8)
        self.dmd.Halt()
        try:
            self.dmd.FreeSeq()
        except Exception:
            pass

        self.dmd.SeqAlloc(nbImg=1, bitDepth=1)
        self.dmd.SeqPut(imgData=binary_mask.ravel())
        self.dmd.Run(loop=True)

    def create_center_aperture(self, radius_px: int = 200) -> np.ndarray:
        y, x = np.ogrid[:self.height, :self.width]
        cy, cx = self.height // 2, self.width // 2
        return ((x - cx) ** 2 + (y - cy) ** 2 <= radius_px ** 2).astype(np.uint8) * 255

    def shutdown(self):
        if self.connected and self.dmd is not None:
            self.dmd.Halt()
            try:
                self.dmd.FreeSeq()
            except Exception:
                pass
            self.dmd.Free()


# ==============================================================================
# 2. KINETIX22 8-BIT SPEED ACQUISITION (MICROSECOND EXPOSURE + ROI CROPPING)
# ==============================================================================
def camera_worker(shm_name: str, write_idx: Value, read_idx: Value, preview_counter: Value,
                  preview_event: Event, record_event: Event, shutdown_event: Event,
                  exposure_us: Value, roi_x: Value, roi_y: Value, roi_w: Value, roi_h: Value,
                  gate_led_ttl: bool):
    shm = shared_memory.SharedMemory(name=shm_name)
    ring_buffer = np.ndarray((TOTAL_SLOTS, MAX_HEIGHT, MAX_WIDTH), dtype=DTYPE, buffer=shm.buf)

    pvc.init_pvcam()
    cam = None
    try:
        for attempt in range(6):
            try:
                cam = next(Camera.detect_camera())
                break
            except StopIteration:
                if attempt == 5:
                    raise RuntimeError("No PVCAM camera detected.")
                time.sleep(0.5)

        cam.open()
        cam.readout_port = 1
        cam.speed_table_index = 0
        cam.gain = 1
        cam.exp_mode = "Internal Trigger"

        # Switch exposure resolution to microseconds (1 = EXP_RES_ONE_MICROSEC)
        try:
            cam.exp_res = 1
        except Exception as e:
            print(f"[Camera] Could not set microsecond exp_res: {e}")

        if gate_led_ttl:
            try:
                cam.exp_out_mode = "First Row"
            except Exception as e:
                print(f"[Camera] Could not set exp_out_mode: {e}")

        print(f"[Camera] {cam.name} ready (8-bit Speed, µs exposure + ROI enabled).")

        def apply_roi_and_exp():
            w = max(64, (roi_w.value // 8) * 8)
            h = max(64, (roi_h.value // 8) * 8)
            x = max(0, min(MAX_WIDTH - w, (roi_x.value // 8) * 8))
            y = max(0, min(MAX_HEIGHT - h, (roi_y.value // 8) * 8))
            roi_x.value, roi_y.value, roi_w.value, roi_h.value = x, y, w, h
            cam.set_roi(x, y, w, h)
            cam.exp_time = max(10, int(exposure_us.value))
            return w, h

        while not shutdown_event.is_set():
            # --- MODE A: HIGH-SPEED STREAMING TO RAM/DISK ---
            if record_event.is_set():
                w, h = apply_roi_and_exp()
                cam.start_live(buffer_frame_count=256)
                print(f"[Camera] Recording {w}x{h} @ {cam.exp_time} µs exposure...")

                while record_event.is_set() and not shutdown_event.is_set():
                    if (write_idx.value - read_idx.value) >= BUFFER_SLOTS - 4:
                        print("[WARNING] RAM Ring Buffer Full! Dropping frame.")
                        time.sleep(0.0005)
                        continue

                    frame, fps, _ = cam.poll_frame()
                    slot = write_idx.value % BUFFER_SLOTS
                    ring_buffer[slot, :h, :w] = frame["pixel_data"]

                    with write_idx.get_lock():
                        write_idx.value += 1

                cam.finish()
                print(f"[Camera] Recording stopped. Captured: {write_idx.value}")

            # --- MODE B: LIVE PREVIEW ONLY ---
            elif preview_event.is_set():
                w, h = apply_roi_and_exp()
                active_params = (roi_x.value, roi_y.value, w, h, cam.exp_time)
                cam.start_live(buffer_frame_count=16)

                while preview_event.is_set() and not record_event.is_set() and not shutdown_event.is_set():
                    desired_params = (roi_x.value, roi_y.value, roi_w.value, roi_h.value, exposure_us.value)
                    if desired_params != active_params:
                        cam.finish()
                        w, h = apply_roi_and_exp()
                        active_params = (roi_x.value, roi_y.value, w, h, cam.exp_time)
                        cam.start_live(buffer_frame_count=16)

                    frame, fps, _ = cam.poll_frame()
                    ring_buffer[PREVIEW_SLOT, :h, :w] = frame["pixel_data"]
                    with preview_counter.get_lock():
                        preview_counter.value += 1

                    time.sleep(0.025)

                cam.finish()
            else:
                time.sleep(0.02)

    finally:
        if cam is not None and cam.is_open:
            cam.close()
        pvc.uninit_pvcam()
        shm.close()


# ==============================================================================
# 3. DUAL-NVME BIGTIFF WRITER PROCESSES (DYNAMIC ROI SHAPE)
# ==============================================================================
def disk_writer_worker(worker_id: int, base_dir: str, run_folder_buf: Array,
                       shm_name: str, write_idx: Value, read_idx: Value,
                       roi_w: Value, roi_h: Value,
                       record_event: Event, shutdown_event: Event):
    shm = shared_memory.SharedMemory(name=shm_name)
    ring_buffer = np.ndarray((TOTAL_SLOTS, MAX_HEIGHT, MAX_WIDTH), dtype=DTYPE, buffer=shm.buf)

    chunk_index = worker_id
    tif_writer = None
    frames_in_file = 0
    active_run_folder = ""

    while not shutdown_event.is_set() or (read_idx.value < write_idx.value):
        current_run_folder = run_folder_buf.value.decode("utf-8")
        if current_run_folder != active_run_folder and current_run_folder != "":
            active_run_folder = current_run_folder
            chunk_index = worker_id

        with read_idx.get_lock():
            if read_idx.value >= write_idx.value:
                claim_idx = None
            else:
                assigned_worker = (read_idx.value // FRAMES_PER_TIFF) % 2
                if assigned_worker == worker_id:
                    claim_idx = read_idx.value
                    read_idx.value += 1
                else:
                    claim_idx = None

        if claim_idx is None:
            if not record_event.is_set() and (read_idx.value >= write_idx.value) and tif_writer is not None:
                tif_writer.close()
                tif_writer = None
                frames_in_file = 0
                print(f"[Writer {worker_id}] Run flushed and file closed.")
            time.sleep(0.0005)
            continue

        if tif_writer is None:
            out_dir = os.path.join(base_dir, active_run_folder)
            os.makedirs(out_dir, exist_ok=True)
            filepath = os.path.join(out_dir, f"kinetix8bit_chunk_{chunk_index:04d}.tif")
            tif_writer = tifffile.TiffWriter(filepath, bigtiff=True, ome=False)
            frames_in_file = 0

        slot = claim_idx % BUFFER_SLOTS
        w, h = roi_w.value, roi_h.value
        # Slice exact ROI dimensions before writing to disk
        frame_view = np.ascontiguousarray(ring_buffer[slot, :h, :w])

        tif_writer.write(frame_view, contiguous=True)
        frames_in_file += 1

        if frames_in_file >= FRAMES_PER_TIFF:
            tif_writer.close()
            tif_writer = None
            chunk_index += 2

    if tif_writer is not None:
        tif_writer.close()
        shm.close()

# ==============================================================================
# 4. INTERACTIVE HISTOGRAM WIDGET + PYSIDE6 CONTROL GUI
# ==============================================================================
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
        # Log-scale heights so dim fluorescence peaks remain clearly visible
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

        # Draw histogram bars
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#4fc3f7"))
        bar_w = w / 256.0
        for i in range(256):
            bar_h = int(self.hist[i] * (h - 10))
            if bar_h > 0:
                painter.drawRect(int(i * bar_w), h - bar_h, max(1, int(bar_w) + 1), bar_h)

        # Draw Min (Green) and Max (Yellow) contrast cutoffs + ramp line
        x_min = int((self.min_val / 255.0) * (w - 1))
        x_max = int((self.max_val / 255.0) * (w - 1))

        painter.setPen(QPen(QColor("#66bb6a"), 2))
        painter.drawLine(x_min, 0, x_min, h)

        painter.setPen(QPen(QColor("#ffee58"), 2))
        painter.drawLine(x_max, 0, x_max, h)

        painter.setPen(QPen(QColor("#ffffff"), 1, Qt.DashLine))
        painter.drawLine(x_min, h - 2, x_max, 2)

class MicroscopeGUI(QMainWindow):
    def __init__(self, shm, write_idx, read_idx, preview_counter,
                 preview_event, record_event, shutdown_event,
                 exposure_us, roi_x, roi_y, roi_w, roi_h,
                 run_folder_buf, dmd_ctrl):
        super().__init__()
        self.setWindowTitle("Kinetix22 (8-Bit Speed) + Static DMD Controller")
        self.resize(1060, 900)

        self.shm = shm
        self.ring_buffer = np.ndarray((TOTAL_SLOTS, MAX_HEIGHT, MAX_WIDTH), dtype=DTYPE, buffer=shm.buf)
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
        btn_aperture.clicked.connect(lambda: self.dmd_ctrl.display_static_mask(
            self.dmd_ctrl.create_center_aperture(250)
        ))
        btn_all_on = QPushButton("All Mirrors ON")
        btn_all_on.clicked.connect(lambda: self.dmd_ctrl.display_static_mask(
            np.full((self.dmd_ctrl.height, self.dmd_ctrl.width), 255, dtype=np.uint8)
        ))

        dmd_layout.addWidget(btn_load_mask)
        dmd_layout.addWidget(btn_aperture)
        dmd_layout.addWidget(btn_all_on)
        layout.addWidget(dmd_box)

        # --- 2. Exposure (µs) & ROI Cropping Controls ---
        roi_box = QGroupBox("Timing (µs) & Sensor ROI Cropping (Multiples of 8 px)")
        roi_layout = QHBoxLayout(roi_box)

        roi_layout.addWidget(QLabel("Exposure (µs):"))
        self.spin_exp = QSpinBox()
        self.spin_exp.setRange(20, 1000000)
        self.spin_exp.setSingleStep(100)
        self.spin_exp.setValue(1000)  # 1000 µs = 1 ms default
        self.spin_exp.valueChanged.connect(lambda v: setattr(self.exposure_us, "value", v))
        roi_layout.addWidget(self.spin_exp)

        self.spin_x = QSpinBox(); self.spin_x.setRange(0, 2336); self.spin_x.setSingleStep(8); self.spin_x.setValue(0)
        self.spin_y = QSpinBox(); self.spin_y.setRange(0, 2336); self.spin_y.setSingleStep(8); self.spin_y.setValue(0)
        self.spin_w = QSpinBox(); self.spin_w.setRange(64, 2400); self.spin_w.setSingleStep(8); self.spin_w.setValue(2400)
        self.spin_h = QSpinBox(); self.spin_h.setRange(64, 2400); self.spin_h.setSingleStep(8); self.spin_h.setValue(2400)

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
        self.lbl_status = QLabel("Status: Idle | Buffer: 0 / 5250 frames")
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
        btn_reset_contrast.clicked.connect(lambda: (self.slider_min.setValue(0), self.slider_max.setValue(255)))

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
        x = max(0, min(MAX_WIDTH - w, (self.spin_x.value() // 8) * 8))
        y = max(0, min(MAX_HEIGHT - h, (self.spin_y.value() // 8) * 8))
        self.spin_x.setValue(x); self.spin_y.setValue(y)
        self.spin_w.setValue(w); self.spin_h.setValue(h)
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
        else:
            arr = tifffile.imread(path) if path.lower().endswith((".tif", ".tiff")) else None
        if arr is None:
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
            display_slot = PREVIEW_SLOT
        elif w > 0:
            display_slot = (w - 1) % BUFFER_SLOTS

        if display_slot is not None:
            step = max(1, max(roi_w, roi_h) // 400)
            sub = np.ascontiguousarray(self.ring_buffer[display_slot, :roi_h:step, :roi_w:step])
            self.latest_sub = sub
            # Update live histogram from raw 8-bit values
            self.hist_widget.update_data(sub, self.slider_min.value(), self.slider_max.value())
            # Apply contrast LUT for display only
            disp = np.ascontiguousarray(self.lut[sub])
            h, width = disp.shape
            qimg = QImage(disp.data, width, h, width, QImage.Format_Grayscale8)
            self.lbl_preview.setPixmap(QPixmap.fromImage(qimg).scaled(
                self.lbl_preview.size(), Qt.KeepAspectRatio, Qt.FastTransformation
            ))

#==============================================================================
#5. ENTRY POINT
#==============================================================================

if __name__ == "__main__":
    
    total_bytes = TOTAL_SLOTS * FRAME_BYTES
    try:
        shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=total_bytes)
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=SHM_NAME)

    write_idx = Value(ctypes.c_uint64, 0)
    read_idx = Value(ctypes.c_uint64, 0)
    preview_counter = Value(ctypes.c_uint64, 0)
    exposure_us = Value(ctypes.c_uint32, 1000)
    roi_x = Value(ctypes.c_uint32, 0)
    roi_y = Value(ctypes.c_uint32, 0)
    roi_w = Value(ctypes.c_uint32, 2400)
    roi_h = Value(ctypes.c_uint32, 2400)
    run_folder_buf = Array(ctypes.c_char, 64)
    preview_event = Event()
    record_event = Event()
    shutdown_event = Event()
    app = QApplication([])
    dmd_ctrl = StaticDMDController(alp_version="4.3")
    window = MicroscopeGUI(
        shm, write_idx, read_idx, preview_counter,
        preview_event, record_event, shutdown_event,
        exposure_us, roi_x, roi_y, roi_w, roi_h,
        run_folder_buf, dmd_ctrl
    )
    p_cam = Process(
        target=camera_worker,
            args=(SHM_NAME, write_idx, read_idx, preview_counter,
                preview_event, record_event, shutdown_event,
                exposure_us, roi_x, roi_y, roi_w, roi_h,
            window.chk_led_ttl.isChecked())
    )
    
    p_w0 = Process(
        target=disk_writer_worker,
        args=(0, NVME_PATHS[0], run_folder_buf, SHM_NAME, write_idx, read_idx,
        roi_w, roi_h, record_event, shutdown_event)
    )

    p_w1 = Process(
        target=disk_writer_worker,
        args=(1, NVME_PATHS[1], run_folder_buf, SHM_NAME, write_idx, read_idx,
        roi_w, roi_h, record_event, shutdown_event)
    )

    p_cam.start()
    p_w0.start()
    p_w1.start()
    window.show()
    app.exec()
    shutdown_event.set()
    preview_event.clear()
    record_event.clear()
    p_cam.join(timeout=5)
    p_w0.join()
    p_w1.join()
    dmd_ctrl.shutdown()
    shm.close()
    shm.unlink()