from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class FaceCropConfig:
    margin: float = 0.35  
    min_size: int = 80    


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def expand_bbox(
    x1: int, y1: int, x2: int, y2: int, w: int, h: int, margin: float
) -> Tuple[int, int, int, int]:
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0

    scale = 1.0 + margin
    nw = bw * scale
    nh = bh * scale

    nx1 = int(round(cx - nw / 2.0))
    ny1 = int(round(cy - nh / 2.0))
    nx2 = int(round(cx + nw / 2.0))
    ny2 = int(round(cy + nh / 2.0))

    nx1 = clamp(nx1, 0, w - 1)
    ny1 = clamp(ny1, 0, h - 1)
    nx2 = clamp(nx2, 0, w - 1)
    ny2 = clamp(ny2, 0, h - 1)

    if nx2 <= nx1 + 1 or ny2 <= ny1 + 1:
        return x1, y1, x2, y2

    return nx1, ny1, nx2, ny2


def crop_face_from_bbox(
    frame_bgr: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
    cfg: FaceCropConfig,
) -> Optional[np.ndarray]:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy

    x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2, w, h, cfg.margin)

    bw = x2 - x1
    bh = y2 - y1
    if min(bw, bh) < cfg.min_size:
        return None

    face = frame_bgr[y1:y2, x1:x2].copy()
    return face


def bgr_to_rgb(face_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)