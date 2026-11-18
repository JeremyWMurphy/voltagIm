import ctypes
import time
from multiprocessing import Array, Event, Process, Value, shared_memory
import numpy as np
from PySide6.QtWidgets import QApplication

import config as cfg
from dmd import StaticDMDController
from gui import MicroscopeGUI
from workers import camera_worker, disk_writer_worker


def prefault_shared_memory(shm: shared_memory.SharedMemory, page_size: int = 4096):
    """Touch one byte per 4 KB OS page to commit physical RAM pages before acquisition."""
    t0 = time.perf_counter()
    gb = shm.size / (1024 ** 3)
    print(f"[Memory] Pre-faulting {gb:.1f} GB shared memory pool...")
    flat_view = np.frombuffer(shm.buf, dtype=np.uint8)
    flat_view[::page_size] = 0
    dt = time.perf_counter() - t0
    print(f"[Memory] Pre-fault complete in {dt:.2f} s.")


if __name__ == "__main__":
    try:
        shm = shared_memory.SharedMemory(
            name=cfg.SHM_NAME, create=True, size=cfg.TOTAL_SHM_BYTES
        )
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=cfg.SHM_NAME)

    prefault_shared_memory(shm)
    sdk_fps = Value(ctypes.c_double, 0.0)

    write_idx = Value(ctypes.c_uint64, 0)
    read_idx = Value(ctypes.c_uint64, 0)
    preview_counter = Value(ctypes.c_uint64, 0)
    exposure_us = Value(ctypes.c_uint32, 1000)

    roi_x = Value(ctypes.c_uint32, 0)
    roi_y = Value(ctypes.c_uint32, 0)
    roi_w = Value(ctypes.c_uint32, 2400)
    roi_h = Value(ctypes.c_uint32, 2400)

    # Initial defaults for full-frame 2400x2400 (updated dynamically on Start)
    active_slots = Value(
        ctypes.c_uint32, max(16, cfg.RING_BUFFER_BYTES // cfg.MAX_FRAME_BYTES)
    )
    frames_per_tiff = Value(
        ctypes.c_uint32, max(1, cfg.TARGET_TIFF_BYTES // cfg.MAX_FRAME_BYTES)
    )

    run_folder_buf = Array(ctypes.c_char, 64)
    preview_event = Event()
    record_event = Event()
    shutdown_event = Event()

    app = QApplication([])
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
            cfg.SHM_NAME,
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
            cfg.SHM_NAME,
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
            cfg.SHM_NAME,
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
