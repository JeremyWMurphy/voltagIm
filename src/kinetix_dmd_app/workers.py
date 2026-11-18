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
    sdk_fps: Value,
    preview_event: Event,
    record_event: Event,
    shutdown_event: Event,
    exposure_us: Value,
    roi_x: Value,
    roi_y: Value,
    roi_w: Value,
    roi_h: Value,
    active_slots: Value,
    gate_led_ttl: bool,
):
    shm = shared_memory.SharedMemory(name=shm_name)

    # Dedicated preview frame anchored at the end of the shared memory block
    preview_buffer = np.ndarray(
        (cfg.MAX_HEIGHT, cfg.MAX_WIDTH),
        dtype=cfg.DTYPE,
        buffer=shm.buf,
        offset=cfg.RING_BUFFER_BYTES,
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

        print(f"[Camera] {cam.name} ready (8-bit Speed, us exposure + dynamic ROI ring enabled).")

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
            # --- MODE A: HIGH-SPEED STREAMING TO DYNAMIC RING ---
            if record_event.is_set():
                w, h = apply_roi_and_exp()
                n_slots = active_slots.value

                # Build zero-copy 3D view sized specifically for this run's ROI
                ring_buffer = np.ndarray(
                    (n_slots, h, w),
                    dtype=cfg.DTYPE,
                    buffer=shm.buf,
                    offset=0,
                )

                # 2048-frame SDK buffer provides ~1 s DMA cushion at 2 kHz
                sdk_frames = min(2048, n_slots)
                cam.start_live(buffer_frame_count=sdk_frames)
                print(
                    f"[Camera] Recording {w}x{h} @ {cam.exp_time} us "
                    f"({n_slots} slots in 96 GB ring)..."
                )

                while record_event.is_set() and not shutdown_event.is_set():
                    if (write_idx.value - read_idx.value) >= n_slots - 4:
                        print("[WARNING] RAM Ring Buffer Full! Dropping frame.")
                        time.sleep(0.0005)
                        continue

                    frame, fps, _ = cam.poll_frame()
                    sdk_fps.value = fps
                    slot = write_idx.value % n_slots
                    ring_buffer[slot] = frame["pixel_data"]

                    # Single-producer increment (no lock needed while recording)
                    write_idx.value += 1

                cam.finish()
                print(f"[Camera] Recording stopped. Captured: {write_idx.value}")

            # --- MODE B: LIVE PREVIEW ONLY ---
            elif preview_event.is_set():
                w, h = apply_roi_and_exp()
                active_params = (roi_x.value, roi_y.value, w, h, cam.exp_time)
                cam.start_live(buffer_frame_count=64)
                last_ui_copy = 0.0

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
                        cam.start_live(buffer_frame_count=64)

                    frame, fps, _ = cam.poll_frame()
                    sdk_fps.value = fps
                    preview_counter.value += 1

                    # Only copy pixels to preview_buffer at ~30 Hz to save CPU/RAM bandwidth
                    now = time.perf_counter()
                    if (now - last_ui_copy) >= 0.033:
                        preview_buffer[:h, :w] = frame["pixel_data"]
                        last_ui_copy = now

                cam.finish()


    finally:
        if cam is not None and cam.is_open:
            cam.close()
        pvc.uninit_pvcam()
        shm.close()
