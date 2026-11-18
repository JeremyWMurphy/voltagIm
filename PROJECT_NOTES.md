# Kinetix22 + ViALUX DMD High-Speed Acquisition App

## Hardware & Environment
- **Workstation:** Windows PC with 128 GB RAM and dual NVMe storage (`NVME_PATHS`).
- **Camera:** Teledyne Kinetix22 sCMOS (`pyvcam`), 8-bit speed mode, microsecond exposure (`exp_res = 1`), targeting ~2 kHz at `2400 x 512` ROI.
- **DMD:** ViALUX DMD (`ALP4` SDK v4.3) with automatic fallback to simulated mode if hardware is disconnected.
- **Environment Manager:** `uv` (`uv run main.py`).

## Modular File Structure
- **`config.py`**: Hardware constants (`MAX_WIDTH = 2400`, `MAX_HEIGHT = 2400`), `RING_BUFFER_BYTES = 96 GB`, `TARGET_TIFF_BYTES = 4 GB`, and `NVME_PATHS`.
- **`dmd.py`**: `StaticDMDController` class handling `ALP4` initialization, static binary mask uploads (`.npy`, `.tif`, `.png`), center apertures, and cleanup.
- **`workers.py`**: Background `multiprocessing` loops:
  - `camera_worker`: Polls frames via `pyvcam`, writes recording frames to the dynamic shared-memory ring (`offset=0`), and copies ~30 Hz frames to `preview_buffer` (`offset=RING_BUFFER_BYTES`) during unthrottled live preview.
  - `disk_writer_worker`: Two parallel consumers (`worker_id` 0 and 1) writing alternating ~4 GB BigTIFF chunks (`kinetix8bit_chunk_XXXX.tif`) via `tifffile`.
- **`gui.py`**: PySide6 `MicroscopeGUI` and `HistogramWidget`:
  - Computes `active_slots` and `frames_per_tiff` dynamically on `on_start()` based on active ROI dimensions.
  - Displays smoothed 0.5 s rolling FPS alongside PyVCAM's `sdk_fps`.
  - Applies non-destructive display LUT contrast stretching while saving raw 8-bit frames.
- **`main.py`**: Allocates `TOTAL_SHM_BYTES`, pre-faults physical RAM pages (`prefault_shared_memory`), applies global Fusion dark `QPalette` (`apply_dark_theme`), spawns worker processes, and handles shutdown/unlinking.

## Planned Next Features
- [ ] **Run Metadata Logging:** Save a JSON metadata file (ROI coordinates, exposure time, `sdk_fps`, frame counts, DMD mask info, timestamps) inside each run folder alongside BigTIFF chunks.
- [ ] **External Hardware Triggering:** Add trigger mode selector (`Internal Trigger` vs. `Edge Trigger` / `Trigger First`) to lock acquisition to a 2 kHz Teensy TTL clock.
- [ ] **Live LED Strobe Sync Toggle:** Allow `chk_led_ttl` (`exp_out_mode`) to update dynamically between runs rather than only at process startup.
- [ ] **ROI Presets:** Add quick-select preset buttons (e.g., `2400 x 512` for 2 kHz voltage imaging).

## Quick Resume Prompt for New AI Sessions
> "I am continuing work on my modular PySide6 + PyVCAM + ViALUX ALP4 DMD microscopy app (`config.py`, `dmd.py`, `workers.py`, `gui.py`, `main.py`). Please review the attached files and `PROJECT_NOTES.md`, then help me implement [INSERT NEXT FEATURE]. Output only the specific file or method that needs changing."
