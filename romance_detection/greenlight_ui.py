from __future__ import annotations
import os
import sys
import time
import threading
from collections import deque
from typing import Optional, Tuple, List
import cv2
import numpy as np
import sounddevice as sd
import torch
import python_speech_features

from detector import DeepfakeDetector, DetectorConfig
from preprocess import FaceCropConfig, crop_face_from_bbox
from pathlib import Path

SYNCNET_REPO = (Path(__file__).resolve().parent / ".." / "syncnet_python").resolve()
sys.path.insert(0, str(SYNCNET_REPO))
from SyncNetInstance import SyncNetInstance, calc_pdist  

import mediapipe.python.solutions.face_mesh as face_mesh
import mediapipe.python.solutions.drawing_utils as drawing_utils

GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
DARK_GRAY = (30, 30, 30)

risk_history = deque(maxlen=100)

def draw_greenlight_ui(
    frame,
    final_risk,
    fps,
    blink_count,
    overlay_lines: Optional[List[str]] = None,
):
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.line(frame, (0, 80), (w, 80), (100, 100, 100), 2)

    if final_risk < 0.3:
        status_color = GREEN
        status_text = "SAFE: GREEN LIGHT"
        light_pos = (50, 40)
    elif final_risk < 0.7:
        status_color = YELLOW
        status_text = "CAUTION: CHECK SYNC"
        light_pos = (w // 2, 40)
    else:
        status_color = RED
        status_text = "DANGER: FAKE DETECTED"
        light_pos = (w - 50, 40)

    colors = [GREEN, YELLOW, RED]
    positions = [50, w // 2, w - 50]
    for i, pos in enumerate(positions):
        alpha = 1.0 if colors[i] == status_color else 0.2
        c = tuple(int(val * alpha) for val in colors[i])
        cv2.circle(frame, (pos, 40), 20, c, -1)
        if colors[i] == status_color:
            cv2.circle(frame, (pos, 40), 25, status_color, 2) 

    cv2.putText(frame, status_text, (max(10, w // 2 - 140), 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    cv2.putText(frame, "GREEN LIGHT SYSTEM v1.0", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    if overlay_lines:
        y0 = 110
        for i, line in enumerate(overlay_lines[:5]):
            y = y0 + i * 24
            cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    graph_h, graph_w = 100, 200
    graph_x, graph_y = 20, h - 120

    overlay = frame.copy()
    cv2.rectangle(overlay, (graph_x - 10, graph_y - 20), (graph_x + graph_w + 10, h - 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    risk_history.append(final_risk)
    if len(risk_history) > 1:
        for i in range(1, len(risk_history)):
            pt1 = (graph_x + (i-1) * 2, int(graph_y + graph_h - (risk_history[i-1] * graph_h)))
            pt2 = (graph_x + i * 2, int(graph_y + graph_h - (risk_history[i] * graph_h)))
            color = GREEN if risk_history[i] < 0.3 else (YELLOW if risk_history[i] < 0.7 else RED)
            cv2.line(frame, pt1, pt2, color, 2)

    cv2.putText(frame, "RISK HISTORY", (graph_x, graph_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    cv2.putText(frame, f"Blinks: {blink_count}", (w - 140, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 140, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    sync_score = max(0, (1.0 - final_risk) * 100)
    cv2.putText(frame, f"AV Sync: {sync_score:.1f}%", (w - 140, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
