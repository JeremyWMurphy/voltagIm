import ctypes
import os
import time
from multiprocessing import Array, Event, Process, Value, shared_memory
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

import config as cfg
from dmd import StaticDMDController
from gui import MicroscopeGUI
from workers import camera_worker, disk_writer_worker

# ... keep apply_dark_theme() and prefault_shared_memory() above ...

if __name__ == "__main__":
    # Unique name per launch prevents collisions with any stale mappings
    session_shm_name = f"{cfg.SHM_NAME}_{os.getpid()}"
    shm = None
    dmd_ctrl = None
    processes = []

    shutdown_event = Event()
    preview_event = Event()
    record_event = Event()

    try:
        shm = shared_memory.SharedMemory(
            name=session_shm_name, create=True, size=cfg.TOTAL_SHM_BYTES
        )
        prefault_shared_memory(shm)

        sdk_fps = Value(ctypes.c_double, 0.0)
        write_idx = Value(ctypes.c_uint64, 0)
        read_idx = Value(ctypes.c_uint64, 0)
        dropped_frames = Value(ctypes.c_uint64, 0)
        preview_counter = Value(ctypes.c_uint64, 0)
        exposure_us = Value(ctypes.c_uint32, 1000)

        roi_x = Value(ctypes.c_uint32, 0)
        roi_y = Value(ctypes.c_uint32, 0)
        roi_w = Value(ctypes.c_uint32, 2400)
        roi_h = Value(ctypes.c_uint32, 2400)

        active_slots = Value(
            ctypes.c_uint32, max(16, cfg.RING_BUFFER_BYTES // cfg.MAX_FRAME_BYTES)
        )
        frames_per_tiff = Value(
            ctypes.c_uint32, max(1, cfg.TARGET_TIFF_BYTES // cfg.MAX_FRAME_BYTES)
        )

        run_folder_buf = Array(ctypes.c_char, 64)

        app = QApplication([])
        apply_dark_theme(app)

        dmd_ctrl = StaticDMDController(alp_version="4.3")
        window = MicroscopeGUI(
            shm,
            write_idx,
            read_idx,
            preview_counter,
            sdk_fps,
            preview_event,
            record_event,
            shutdown_event,
            exposure_us,
            roi_x,
            roi_y,
            roi_w,
            roi_h,
            active_slots,
            frames_per_tiff,
            run_folder_buf,
            dmd_ctrl,
        )

        p_cam = Process(
            target=camera_worker,
            args=(
                session_shm_name,
                write_idx,
                read_idx,
                preview_counter,
                sdk_fps,
                preview_event,
                record_event,
                shutdown_event,
                exposure_us,
                roi_x,
                roi_y,
                roi_w,
                roi_h,
                active_slots,
                window.chk_led_ttl.isChecked(),
            ),
        )

        p_w0 = Process(
            target=disk_writer_worker,
            args=(
                0,
                cfg.NVME_PATHS[0],
                run_folder_buf,
                session_shm_name,
                write_idx,
                read_idx,
                roi_w,
                roi_h,
                active_slots,
                frames_per_tiff,
                record_event,
                shutdown_event,
            ),
        )

        p_w1 = Process(
            target=disk_writer_worker,
            args=(
                1,
                cfg.NVME_PATHS[1],
                run_folder_buf,
                session_shm_name,
                write_idx,
                read_idx,
                roi_w,
                roi_h,
                active_slots,
                frames_per_tiff,
                record_event,
                shutdown_event,
            ),
        )

        processes = [p_cam, p_w0, p_w1]
        for p in processes:
            p.start()

        window.show()
        app.exec()

    finally:
        # Always runs, even if an exception occurs during startup or GUI execution
        shutdown_event.set()
        preview_event.clear()
        record_event.clear()

        for p in processes:
            if p.is_alive():
                p.join(timeout=5)
                if p.is_alive():
                    print(f"[Cleanup] Force-terminating stuck worker {p.name}...")
                    p.terminate()
                    p.join()

        if dmd_ctrl is not None:
            dmd_ctrl.shutdown()

        if shm is not None:
            try:
                shm.close()
                shm.unlink()
            except Exception as e:
                print(f"[Cleanup] SharedMemory release warning: {e}")
