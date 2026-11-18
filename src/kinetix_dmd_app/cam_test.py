import time
import numpy as np
from pyvcam import pvc
from pyvcam.camera import Camera

def verify_kinetix():
    pvc.init_pvcam()
    try:
        cam = next(Camera.detect_camera())
        cam.open()
        print(f"Connected to: {cam.name}")
        print(f"Sensor size: {cam.sensor_size}")
        print(f"Available readout ports/speeds: {cam.port_speed_gain_table}")

        # Configure for maximum speed free-run
        cam.exp_mode = "Internal Trigger"
        cam.exp_time = 2  # 2 ms exposure
        cam.start_live(buffer_frame_count=64)

        n_frames = 100
        t0 = time.perf_counter()
        for _ in range(n_frames):
            frame, fps, frame_count = cam.poll_frame()
        elapsed = time.perf_counter() - t0

        cam.finish()
        print(f"Captured {n_frames} frames in {elapsed:.3f}s ({n_frames / elapsed:.1f} FPS)")
        print(f"Frame shape: {frame['pixel_data'].shape}, dtype: {frame['pixel_data'].dtype}")
        cam.close()
    finally:
        pvc.uninit_pvcam()

if __name__ == "__main__":
    verify_kinetix()