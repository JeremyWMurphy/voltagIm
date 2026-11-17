import os
import time
import ctypes
from datetime import datetime
import numpy as np
import tifffile
from multiprocessing import Process, Value, Event, Array, shared_memory

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QDoubleSpinBox, QGroupBox, QFileDialog, QCheckBox
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap

from pyvcam import pvc
from pyvcam.camera import Camera

try:
    from ALP4 import ALP4
    HAS_ALP4 = True
except ImportError:
    HAS_ALP4 = False

# --- HARDWARE & BUFFER CONFIGURATION ---
FRAME_HEIGHT = 2400
FRAME_WIDTH = 2400
DTYPE = np.uint8
FRAME_BYTES = FRAME_HEIGHT * FRAME_WIDTH * np.dtype(DTYPE).itemsize  # 5.76 MB

# Allocate ~60 GB Ring Buffer in RAM (10,500 recording slots + 1 dedicated preview slot)
BUFFER_SLOTS = 10500
TOTAL_SLOTS = BUFFER_SLOTS + 1
PREVIEW_SLOT = BUFFER_SLOTS
SHM_NAME = "kinetix22_8bit_ring"

# Dedicated NVMe Output Directories
NVME_PATHS = [r"D:\kinetix_stream", r"D:\kinetix_stream"]
FRAMES_PER_TIFF = 1000  # ~5.76 GB per BigTIFF chunk


# ==============================================================================
# 1. STATIC DMD CONTROLLER (MAIN THREAD / ON-DEMAND)
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
        """Uploads a 2D uint8 array (0 or 255) and displays it indefinitely."""
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
        print("[DMD] Static mask active on mirrors.")

    def create_center_aperture(self, radius_px: int = 200) -> np.ndarray:
        """Generates a circular illumination mask centered on the DMD."""
        y, x = np.ogrid[:self.height, :self.width]
        cy, cx = self.height // 2, self.width // 2
        mask = ((x - cx) ** 2 + (y - cy) ** 2 <= radius_px ** 2).astype(np.uint8) * 255
        return mask

    def shutdown(self):
        if self.connected and self.dmd is not None:
            self.dmd.Halt()
            try:
                self.dmd.FreeSeq()
            except Exception:
                pass
            self.dmd.Free()
            print("[DMD] Safely shut down.")


# ==============================================================================
# 2. KINETIX22 8-BIT SPEED ACQUISITION PROCESS (PREVIEW + RECORDING)
# ==============================================================================
def camera_worker(shm_name: str, write_idx: Value, read_idx: Value, preview_counter: Value,
                  preview_event: Event, record_event: Event, shutdown_event: Event,
                  exposure_ms: Value, gate_led_ttl: bool):
    shm = shared_memory.SharedMemory(name=shm_name)
    ring_buffer = np.ndarray((TOTAL_SLOTS, FRAME_HEIGHT, FRAME_WIDTH), dtype=DTYPE, buffer=shm.buf)

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

        if gate_led_ttl:
            try:
                cam.exp_out_mode = "First Row"
            except Exception as e:
                print(f"[Camera] Could not set exp_out_mode: {e}")

        print(f"[Camera] {cam.name} ready in 8-bit Speed Mode.")

        while not shutdown_event.is_set():
            # --- MODE A: HIGH-SPEED STREAMING TO RAM/DISK ---
            if record_event.is_set():
                cam.exp_time = max(1, int(round(exposure_ms.value)))
                cam.start_live(buffer_frame_count=128)
                print(f"[Camera] Recording started ({cam.exp_time} ms exposure)...")

                while record_event.is_set() and not shutdown_event.is_set():
                    if (write_idx.value - read_idx.value) >= BUFFER_SLOTS - 4:
                        print("[WARNING] RAM Ring Buffer Full! Dropping frame.")
                        time.sleep(0.001)
                        continue

                    frame, fps, _ = cam.poll_frame()
                    slot = write_idx.value % BUFFER_SLOTS
                    ring_buffer[slot][:] = frame["pixel_data"]

                    with write_idx.get_lock():
                        write_idx.value += 1

                cam.finish()
                print(f"[Camera] Recording stopped. Total frames captured: {write_idx.value}")

            # --- MODE B: LIVE PREVIEW ONLY (NO DISK WRITE) ---
            elif preview_event.is_set():
                cam.exp_time = max(1, int(round(exposure_ms.value)))
                cam.start_live(buffer_frame_count=16)
                print(f"[Camera] Live Preview started ({cam.exp_time} ms exposure)...")

                while preview_event.is_set() and not record_event.is_set() and not shutdown_event.is_set():
                    # Allow live exposure adjustment during preview
                    desired_exp = max(1, int(round(exposure_ms.value)))
                    if cam.exp_time != desired_exp:
                        cam.finish()
                        cam.exp_time = desired_exp
                        cam.start_live(buffer_frame_count=16)

                    frame, fps, _ = cam.poll_frame()
                    ring_buffer[PREVIEW_SLOT][:] = frame["pixel_data"]
                    with preview_counter.get_lock():
                        preview_counter.value += 1

                    # Throttle preview polling slightly (~30 Hz) to keep CPU/LED load light
                    time.sleep(0.025)

                cam.finish()
                print("[Camera] Live Preview stopped.")

            # --- MODE C: IDLE ---
            else:
                time.sleep(0.02)

    finally:
        if cam is not None and cam.is_open:
            cam.close()
        pvc.uninit_pvcam()
        shm.close()


# ==============================================================================
# 3. DUAL-NVME BIGTIFF WRITER PROCESSES (SUPPORTS MULTIPLE RUNS)
# ==============================================================================
def disk_writer_worker(worker_id: int, base_dir: str, run_folder_buf: Array,
                       shm_name: str, write_idx: Value, read_idx: Value,
                       record_event: Event, shutdown_event: Event):
    shm = shared_memory.SharedMemory(name=shm_name)
    ring_buffer = np.ndarray((TOTAL_SLOTS, FRAME_HEIGHT, FRAME_WIDTH), dtype=DTYPE, buffer=shm.buf)

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
            time.sleep(0.001)
            continue

        if tif_writer is None:
            out_dir = os.path.join(base_dir, active_run_folder)
            os.makedirs(out_dir, exist_ok=True)
            filepath = os.path.join(out_dir, f"kinetix8bit_chunk_{chunk_index:04d}.tif")
            tif_writer = tifffile.TiffWriter(filepath, bigtiff=True, ome=False)
            frames_in_file = 0

        slot = claim_idx % BUFFER_SLOTS
        tif_writer.write(ring_buffer[slot], contiguous=True)
        frames_in_file += 1

        if frames_in_file >= FRAMES_PER_TIFF:
            tif_writer.close()
            tif_writer = None
            chunk_index += 2

    if tif_writer is not None:
        tif_writer.close()
    shm.close()


# ==============================================================================
# 4. PYSIDE6 CONTROL & PREVIEW GUI
# ==============================================================================
class MicroscopeGUI(QMainWindow):
    def __init__(self, shm, write_idx, read_idx, preview_counter,
                 preview_event, record_event, shutdown_event,
                 exposure_val, run_folder_buf, dmd_ctrl):
        super().__init__()
        self.setWindowTitle("Kinetix22 (8-Bit Speed) + Static DMD Controller")
        self.resize(980, 820)

        self.shm = shm
        self.ring_buffer = np.ndarray((TOTAL_SLOTS, FRAME_HEIGHT, FRAME_WIDTH), dtype=DTYPE, buffer=shm.buf)
        self.write_idx = write_idx
        self.read_idx = read_idx
        self.preview_counter = preview_counter
        self.preview_event = preview_event
        self.record_event = record_event
        self.shutdown_event = shutdown_event
        self.exposure_val = exposure_val
        self.run_folder_buf = run_folder_buf
        self.dmd_ctrl = dmd_ctrl

        self.last_t = time.perf_counter()
        self.last_w = 0

        self._build_ui()

        default_mask = self.dmd_ctrl.create_center_aperture(radius_px=250)
        self.dmd_ctrl.display_static_mask(default_mask)

        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_ui)
        self.timer.start(66)

    def _build_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)

        # --- DMD Controls ---
        dmd_box = QGroupBox("Static ViALUX DMD Mask")
        dmd_layout = QHBoxLayout(dmd_box)

        btn_load_mask = QPushButton("Load Mask File (.npy / .tif / .png)")
        btn_load_mask.clicked.connect(self.load_mask_dialog)
        btn_aperture = QPushButton("Set Center Aperture (250 px)")
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

        # --- Camera & Acquisition Controls ---
        cam_box = QGroupBox("Kinetix22 Acquisition (2400x2400 @ 8-Bit Speed)")
        cam_layout = QHBoxLayout(cam_box)

        cam_layout.addWidget(QLabel("Exposure (ms):"))
        self.spin_exp = QDoubleSpinBox()
        self.spin_exp.setRange(1.0, 1000.0)
        self.spin_exp.setValue(2.0)
        self.spin_exp.valueChanged.connect(self.on_exposure_changed)
        cam_layout.addWidget(self.spin_exp)

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

        # --- Telemetry ---
        self.lbl_status = QLabel("Status: Idle | Buffer: 0 / 10500 frames (0.00 GB)")
        self.lbl_status.setStyleSheet("font-size: 13px; font-weight: bold;")
        layout.addWidget(self.lbl_status)

        # --- Live Preview Canvas ---
        self.lbl_preview = QLabel("Live 8-Bit Preview (Subsampled 6x)")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setMinimumSize(600, 600)
        self.lbl_preview.setStyleSheet("background-color: #111; color: #888;")
        layout.addWidget(self.lbl_preview)

        self.setCentralWidget(central)

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
                ptr = qimg.bits()
                arr = np.frombuffer(ptr, np.uint8).reshape((qimg.height(), qimg.width())).copy()
        self.dmd_ctrl.display_static_mask(arr)

    def on_exposure_changed(self, val):
        self.exposure_val.value = val

    def on_preview(self):
        self.record_event.clear()
        self.exposure_val.value = self.spin_exp.value()
        self.preview_event.set()

        self.btn_preview.setEnabled(False)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(True)

    def on_start(self):
        self.preview_event.clear()
        time.sleep(0.05)

        with self.write_idx.get_lock():
            self.write_idx.value = 0
        with self.read_idx.get_lock():
            self.read_idx.value = 0
        self.last_w = 0

        timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_folder_buf.value = timestamp.encode("utf-8")
        self.exposure_val.value = self.spin_exp.value()

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
        gb_used = (buffered * FRAME_BYTES) / (1024 ** 3)

        if self.record_event.is_set():
            mode_str = "RECORDING"
        elif self.preview_event.is_set():
            mode_str = "LIVE PREVIEW (No Save)"
        elif buffered > 0:
            mode_str = "FLUSHING TO NVMe"
        else:
            mode_str = "IDLE"

        self.lbl_status.setText(
            f"[{mode_str}] Rate: {live_fps:.1f} FPS | RAM Buffer: {buffered} / {BUFFER_SLOTS} "
            f"({gb_used:.2f} GB) | Captured: {w} | Written: {r}"
        )

        display_slot = None
        if self.preview_event.is_set() and self.preview_counter.value > 0:
            display_slot = PREVIEW_SLOT
        elif w > 0:
            display_slot = (w - 1) % BUFFER_SLOTS

        if display_slot is not None:
            sub = np.ascontiguousarray(self.ring_buffer[display_slot][::6, ::6])
            h, width = sub.shape
            qimg = QImage(sub.data, width, h, width, QImage.Format_Grayscale8)
            self.lbl_preview.setPixmap(QPixmap.fromImage(qimg).scaled(
                self.lbl_preview.size(), Qt.KeepAspectRatio, Qt.FastTransformation
            ))

# ==============================================================================
# 5. ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    for p in NVME_PATHS:
        drive = os.path.splitdrive(p)[0]
        if not os.path.exists(drive):
            print(f"[WARNING] Drive {drive} not found! Update NVME_PATHS to match your NVMe drive letters.")

    total_bytes = TOTAL_SLOTS * FRAME_BYTES
    try:
        shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=total_bytes)
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=SHM_NAME)

    write_idx = Value(ctypes.c_uint64, 0)
    read_idx = Value(ctypes.c_uint64, 0)
    preview_counter = Value(ctypes.c_uint64, 0)
    exposure_val = Value(ctypes.c_double, 2.0)
    run_folder_buf = Array(ctypes.c_char, 64)

    preview_event = Event()
    record_event = Event()
    shutdown_event = Event()

    app = QApplication([])
    dmd_ctrl = StaticDMDController(alp_version="4.3")
    window = MicroscopeGUI(
        shm, write_idx, read_idx, preview_counter,
        preview_event, record_event, shutdown_event,
        exposure_val, run_folder_buf, dmd_ctrl
    )

    p_cam = Process(
        target=camera_worker,
        args=(SHM_NAME, write_idx, read_idx, preview_counter,
              preview_event, record_event, shutdown_event,
              exposure_val, window.chk_led_ttl.isChecked())
    )
    p_w0 = Process(
        target=disk_writer_worker,
        args=(0, NVME_PATHS[0], run_folder_buf, SHM_NAME, write_idx, read_idx, record_event, shutdown_event)
    )

    p_w1 = Process(
        target=disk_writer_worker,
        args=(1, NVME_PATHS[1], run_folder_buf, SHM_NAME, write_idx, read_idx, record_event, shutdown_event)
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