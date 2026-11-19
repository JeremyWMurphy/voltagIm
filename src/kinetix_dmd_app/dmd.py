import numpy as np

try:
    from ALP4 import ALP4
    HAS_ALP4 = True
except ImportError:
    HAS_ALP4 = False


class StaticDMDController:
    def __init__(self, alp_version="4.3"):
        self.dmd = None
        self.connected = False
        self.width = 1024
        self.height = 768
        self.last_mask_info = "none"

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

    def display_static_mask(self, mask_2d: np.ndarray, label: str = "custom_array"):
        self.last_mask_info = label

        if not self.connected:
            print(f"[DMD Simulated] Loaded static mask '{label}' with shape {mask_2d.shape}")
            return

        if mask_2d.shape != (self.height, self.width):
            raise ValueError(
                f"Mask shape {mask_2d.shape} does not match DMD ({self.height}, {self.width})"
            )

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
