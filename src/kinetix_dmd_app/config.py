import numpy as np

# --- HARDWARE & BUFFER CONFIGURATION ---
MAX_HEIGHT = 2400
MAX_WIDTH = 2400
DTYPE = np.uint8
FRAME_BYTES = MAX_HEIGHT * MAX_WIDTH * np.dtype(DTYPE).itemsize  # 5.76 MB max slot stride

# ~30 GB Ring Buffer (5,250 full-frame slots, holds more frames when cropped)
BUFFER_SLOTS = 5250
TOTAL_SLOTS = BUFFER_SLOTS + 1
PREVIEW_SLOT = BUFFER_SLOTS
SHM_NAME = "kinetix22_8bit_ring"

NVME_PATHS = [r"D:\kinetix_stream", r"D:\kinetix_stream"]
FRAMES_PER_TIFF = 1000
