from __future__ import annotations

import numpy as np


# window_preds: list of (L',) arrays ; starts: list of start frame indices ; total_frames: int -> (total_frames,)
def aggregate_windows(window_preds: list[np.ndarray], starts: list[int], total_frames: int) -> np.ndarray:
    accum = np.zeros(total_frames, dtype=np.float64)
    counts = np.zeros(total_frames, dtype=np.float64)

    for pred, start in zip(window_preds, starts):
        end = start + len(pred)
        accum[start:end] += pred
        counts[start:end] += 1

    # Frames covered by at least one window: average
    valid = counts > 0
    accum[valid] /= counts[valid]

    # Fill uncovered tail frames with last valid value
    if not valid.all() and valid.any():
        last_valid = np.where(valid)[0][-1]
        accum[last_valid + 1:] = accum[last_valid]

    return accum.astype(np.float32)
