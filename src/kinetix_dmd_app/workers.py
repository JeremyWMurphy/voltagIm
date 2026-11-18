import os
import time
from multiprocessing import Array, Event, Value, shared_memory
import numpy as np
import tifffile
from pyvcam import pvc
from pyvcam.camera import Camera

import config as cfg

def camera_worker(
    shm_name: str,
    write_idx: Value,
    read_idx: Value,
    preview_counter: Value,
    preview_event: Event,
    record_event: Event,
    shutdown_event: Event,
    exposure_us: Value,
    roi_x: Value,
    roi_y: Value,
    roi_w: Value,
    roi_h: Value,
    gate_led_ttl: bool,
):
    shm = shared_memory.SharedMemory(name=shm_name)
    ring_buffer = np.ndarray(
        (cfg.TOTAL_SLOTS, cfg.MAX_HEIGHT, cfg.MAX_WIDTH), dtype=cfg.DTYPE, buffer=shm.buf
    )

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
            x = max(0, min(cfg.MAX_WIDTH - w, (roi_x.value // 8) * 8))
            y = max(0, min(cfg.MAX_HEIGHT - h, (roi_y.value // 8) * 8))
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
                    if (write_idx.value - read_idx.value) >= cfg.BUFFER_SLOTS - 4:
                        print("[WARNING] RAM Ring Buffer Full! Dropping frame.")
                        time.sleep(0.0005)
                        continue

                    frame, fps, _ = cam.poll_frame()
                    slot = write_idx.value % cfg.BUFFER_SLOTS
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

                while (
                    preview_event.is_set()
                    and not record_event.is_set()
                    and not shutdown_event.is_set()
                ):
                    desired_params = (
                        roi_x.value,
                        roi_y.value,
                        roi_w.value,
                        roi_h.value,
                        exposure_us.value,
                    )
                    if desired_params != active_params:
                        cam.finish()
                        w, h = apply_roi_and_exp()
                        active_params = (roi_x.value, roi_y.value, w, h, cam.exp_time)
                        cam.start_live(buffer_frame_count=16)

                    frame, fps, _ = cam.poll_frame()
                    ring_buffer[cfg.PREVIEW_SLOT, :h, :w] = frame["pixel_data"]
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


def disk_writer_worker(
    worker_id: int,
    base_dir: str,
    run_folder_buf: Array,
    shm_name: str,
    write_idx: Value,
    read_idx: Value,
    roi_w: Value,
    roi_h: Value,
    record_event: Event,
    shutdown_event: Event,
):
    shm = shared_memory.SharedMemory(name=shm_name)
    ring_buffer = np.ndarray(
        (cfg.TOTAL_SLOTS, cfg.MAX_HEIGHT, cfg.MAX_WIDTH), dtype=cfg.DTYPE, buffer=shm.buf
    )

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
                assigned_worker = (read_idx.value // cfg.FRAMES_PER_TIFF) % 2
                if assigned_worker == worker_id:
                    claim_idx = read_idx.value
                    read_idx.value += 1
                else:
                    claim_idx = None

        if claim_idx is None:
            if (
                not record_event.is_set()
                and (read_idx.value >= write_idx.value)
                and tif_writer is not None
            ):
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

        slot = claim_idx % cfg.BUFFER_SLOTS
        w, h = roi_w.value, roi_h.value
        frame_view = np.ascontiguousarray(ring_buffer[slot, :h, :w])

        tif_writer.write(frame_view, contiguous=True)
        frames_in_file += 1

        if frames_in_file >= cfg.FRAMES_PER_TIFF:
            tif_writer.close()
            tif_writer = None
            chunk_index += 2

    if tif_writer is not None:
        tif_writer.close()
    shm.close()
