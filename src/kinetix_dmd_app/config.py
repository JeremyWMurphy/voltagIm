import numpy as np

# --- HARDWARE & BUFFER CONFIGURATION ---
MAX_HEIGHT = 2400
MAX_WIDTH = 2400
DTYPE = np.uint8
MAX_FRAME_BYTES = MAX_HEIGHT * MAX_WIDTH * np.dtype(DTYPE).itemsize  # 5.76 MB

# 96 GB dynamic ring pool + 1 full-frame slot at the end for live preview
RING_BUFFER_BYTES = 96 * (1024 ** 3)
TOTAL_SHM_BYTES = RING_BUFFER_BYTES + MAX_FRAME_BYTES

# Target ~4 GB per BigTIFF file regardless of ROI dimensions
TARGET_TIFF_BYTES = 4 * (1024 ** 3)

# Point to two separate physical NVMe drives if available (e.g., D:\ and E:\)
NVME_PATHS = [r"D:\kinetix_stream", r"D:\kinetix_stream"]
