from __future__ import annotations

import os
import sys
import time
import threading
from enum import Enum, auto
from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import mediapipe as mp
import python_speech_features

from greenlight_ui import draw_greenlight_ui
from detector import DeepfakeDetector, DetectorConfig
from preprocess import FaceCropConfig, crop_face_from_bbox


SYNCNET_REPO = (Path(__file__).resolve().parent / ".." / "syncnet_python").resolve()
sys.path.insert(0, str(SYNCNET_REPO))
from SyncNetInstance import SyncNetInstance, calc_pdist  

# MediaPipe FaceMesh 설정
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# 얼굴 랜드마크
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
EAR_THRESHOLD = 0.22

LIPS_OUTER = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78
]

# Deepfake detector 생성
det_cfg = DetectorConfig(
    model_id="prithivMLmods/Deep-Fake-Detector-v2-Model",
    device="cpu",     
    use_fp16=False,   
)
deepfake_detector = DeepfakeDetector(det_cfg)
crop_cfg = FaceCropConfig(margin=0.5, min_size=80)

# Audio buffer 설정
class AudioRingBuffer:
    def __init__(self, sr: int, max_sec: float = 12.0):
        self.sr = sr
        self.max_len = int(sr * max_sec)
        self.buf = deque()
        self.total = 0
        self.lock = threading.Lock()

    def push(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2:
            x = x[:, 0]
        with self.lock:
            self.buf.append(x)
            self.total += len(x)
            while self.total > self.max_len:
                y = self.buf.popleft()
                self.total -= len(y)

    def get_last(self, sec: float) -> np.ndarray:
        n = int(self.sr * sec)
        with self.lock:
            if self.total <= 0:
                return np.zeros((0,), np.float32)

            need = n
            chunks = []
            for c in reversed(self.buf):
                if need <= 0:
                    break
                take = min(len(c), need)
                chunks.append(c[-take:])
                need -= take

            if not chunks:
                return np.zeros((0,), np.float32)

            return np.concatenate(list(reversed(chunks)))


class MicStream:
    def __init__(self, sr: int = 16000, blocksize: int = 1024, device=None):
        self.sr = sr
        self.rb = AudioRingBuffer(sr=sr, max_sec=12.0)
        self.stream = sd.InputStream(
            samplerate=sr,
            channels=1,
            blocksize=blocksize,
            device=device,
            callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status):
        self.rb.push(indata.copy())

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()

    def last(self, sec: float) -> np.ndarray:
        return self.rb.get_last(sec)

# Utils
def calculate_ear(landmarks, indices, w, h):
    coords = np.array([[landmarks[idx].x * w, landmarks[idx].y * h] for idx in indices], dtype=np.float32)
    v1 = np.linalg.norm(coords[1] - coords[5])
    v2 = np.linalg.norm(coords[2] - coords[4])
    h_dist = np.linalg.norm(coords[0] - coords[3])
    return float((v1 + v2) / (2.0 * max(1e-6, h_dist)))


def get_mediapipe_bbox(landmarks, w, h):
    x_min, y_min = w, h
    x_max, y_max = 0, 0
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        x_min = min(x_min, x); x_max = max(x_max, x)
        y_min = min(y_min, y); y_max = max(y_max, y)
    return (x_min, y_min, x_max, y_max)


def clamp_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1: x2 = min(w - 1, x1 + 1)
    if y2 <= y1: y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2

'''
def mouth_crop_from_landmarks(frame_bgr: np.ndarray, landmarks, out_size: int = 224, margin: float = 0.35):
    h, w = frame_bgr.shape[:2]
    xs = []
    ys = []
    for idx in LIPS_OUTER:
        xs.append(landmarks[idx].x * w)
        ys.append(landmarks[idx].y * h)

    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = (x2 - x1)
    bh = (y2 - y1)
    size = max(bw, bh) * (1.0 + margin)

    nx1 = int(cx - size / 2)
    ny1 = int(cy - size / 2)
    nx2 = int(cx + size / 2)
    ny2 = int(cy + size / 2)

    nx1, ny1, nx2, ny2 = clamp_bbox(nx1, ny1, nx2, ny2, w, h)
    crop = frame_bgr[ny1:ny2, nx1:nx2]
    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return crop
'''

def mouth_crop_from_landmarks(
    frame_bgr: np.ndarray,
    landmarks,
    out_size: int = 224,
    margin: float = 0.35,
) -> Optional[np.ndarray]:
    h, w = frame_bgr.shape[:2]
    if h <= 1 or w <= 1:
        return None

    # LIPS_OUTER 인덱스가 유효한지 방어
    if landmarks is None or len(landmarks) <= max(LIPS_OUTER):
        return None

    xs, ys = [], []
    for idx in LIPS_OUTER:
        x = float(landmarks[idx].x) * w
        y = float(landmarks[idx].y) * h
        if not np.isfinite(x) or not np.isfinite(y):
            return None
        xs.append(x)
        ys.append(y)

    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)

    bw = (x2 - x1)
    bh = (y2 - y1)

    # 입 박스가 너무 작거나(추적 불안정), 역전되면 스킵
    if bw < 2 or bh < 2:
        return None

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    # margin으로 정사각 확대
    size = max(bw, bh) * (1.0 + margin)
    if not np.isfinite(size) or size < 2:
        return None

    nx1 = int(round(cx - size / 2))
    ny1 = int(round(cy - size / 2))
    nx2 = int(round(cx + size / 2))
    ny2 = int(round(cy + size / 2))

    # frame 범위로 clamp
    nx1 = max(0, min(w, nx1))
    ny1 = max(0, min(h, ny1))
    nx2 = max(0, min(w, nx2))
    ny2 = max(0, min(h, ny2))

    # 최종적으로 유효한 crop인지 체크
    if nx2 - nx1 < 2 or ny2 - ny1 < 2:
        return None

    crop = frame_bgr[ny1:ny2, nx1:nx2]
    if crop is None or crop.size == 0:
        return None

    # 혹시 단일 채널/이상 shape 방어
    if crop.ndim != 3 or crop.shape[0] < 2 or crop.shape[1] < 2:
        return None

    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return crop


def clamp01(x):
    return max(0.0, min(1.0, float(x)))


def sync_risk(av_offset_ms, av_conf):
    off = abs(av_offset_ms)
    r_off = clamp01((off - 150) / (800 - 150))

    g = clamp01(av_conf/ 3.0)
    return r_off * g


def resample_frames(frames, target_n: int):
    if len(frames) == 0:
        return frames
    if len(frames) == target_n:
        return frames
    idxs = np.linspace(0, len(frames) - 1, num=target_n)
    idxs = np.rint(idxs).astype(int)
    return [frames[i] for i in idxs]


# SyncNet 실시간 래퍼
@dataclass
class SyncCfg:
    initial_model: str = str((SYNCNET_REPO / "data" / "syncnet_v2.model").resolve())
    fps: int = 25
    audio_sr: int = 16000
    clip_sec: float = 2.0
    batch_size: int = 20
    vshift: int = 50
    device: str = "cpu"


class RealTimeSyncNet:
    def __init__(self, cfg: SyncCfg):
        self.cfg = cfg
        self.device = torch.device("cpu")  
        self.s = SyncNetInstance(dropout=0, num_layers_in_fc_layers=1024, device="cpu")
        self.s.loadParameters(cfg.initial_model)

        for name, obj in vars(self.s).items():
            if isinstance(obj, torch.nn.Module):
                obj.to(self.device)
        if hasattr(self.s, "__S__") and isinstance(getattr(self.s, "__S__"), torch.nn.Module):
            getattr(self.s, "__S__").to(self.device)

        self.s.eval()

    @torch.no_grad()
    def estimate(self, mouth_frames_224_bgr: List[np.ndarray], audio_16k: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        fps = self.cfg.fps
        clip_sec = self.cfg.clip_sec
        sr = self.cfg.audio_sr

        max_frames = int(round(fps * clip_sec))
        audio_16k = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
        samples_per_frame = int(round(sr / fps))

        avail_frames_by_video = len(mouth_frames_224_bgr)
        avail_frames_by_audio = len(audio_16k) // samples_per_frame

        min_length = min(max_frames, avail_frames_by_video, avail_frames_by_audio)

        if min_length < 5:
            return None, None

        mouth_frames_224_bgr = mouth_frames_224_bgr[-min_length:]

        need_samples = min_length * samples_per_frame
        audio_16k = audio_16k[-need_samples:]

        # video tensor: (1, T, H, W, C) -> (1, C, T, H, W)
        frames = np.stack(mouth_frames_224_bgr, axis=0).astype(np.float32)  # (T,H,W,C)
        frames = np.transpose(frames, (3, 0, 1, 2))  # (C,T,H,W)
        frames = np.expand_dims(frames, axis=0)  # (1,C,T,H,W)
        imtv = torch.from_numpy(frames).float()

        mfcc = python_speech_features.mfcc(audio_16k.astype(np.float32), samplerate=sr)
        mfcc = mfcc.T.astype(np.float32)  # (num_ceps, num_frames)
        cc = mfcc[np.newaxis, np.newaxis, :, :]      # (1,1,num_ceps,num_frames)
        cct = torch.from_numpy(cc).float()

        lastframe = min_length - 5
        if lastframe <= 0:
            return None, None

        need_mfcc_frames = (lastframe - 1) * 4 + 20

        if cct.shape[-1] < need_mfcc_frames:
            pad = need_mfcc_frames - cct.shape[-1]
            last = cct[..., -1:].repeat(1, 1, 1, pad)
            cct = torch.cat([cct, last], dim=-1)
        im_feat = []
        cc_feat = []

        net = getattr(self.s, "__S__", None)
        if net is None:
            return None, None

        for i in range(0, lastframe, self.cfg.batch_size):
            v_end = min(lastframe, i + self.cfg.batch_size)

            im_batch = []
            cc_batch = []
            for vframe in range(i, v_end):
                im_batch.append(imtv[:, :, vframe:vframe + 5, :, :])
                cc_batch.append(cct[:, :, :, vframe * 4:vframe * 4 + 20])

            im_in = torch.cat(im_batch, dim=0).to(self.device)
            cc_in = torch.cat(cc_batch, dim=0).to(self.device)

            im_out = net.forward_lip(im_in)
            cc_out = net.forward_aud(cc_in)

            im_feat.append(im_out.cpu())
            cc_feat.append(cc_out.cpu())

        im_feat = torch.cat(im_feat, dim=0)
        cc_feat = torch.cat(cc_feat, dim=0)

        # offset/conf 계산
        dists = calc_pdist(im_feat, cc_feat, vshift=self.cfg.vshift)
        mdist = torch.mean(torch.stack(dists, dim=1), dim=1)

        minval, minidx = torch.min(mdist, dim=0)
        offset_frames = int(self.cfg.vshift - int(minidx))
        conf = float(torch.median(mdist) - minval)

        offset_ms = float(offset_frames) * (1000.0 / float(fps))
        return offset_ms, conf


def generate_artifact_map(face_bgr):
    try:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap = np.absolute(lap)
        lap = lap * 5.0
        lap[lap > 255] = 255
        lap = np.uint8(lap)
        heatmap = cv2.applyColorMap(lap, cv2.COLORMAP_JET)
        out = cv2.addWeighted(face_bgr, 0.5, heatmap, 0.5, 0)
        return out
    except Exception:
        return face_bgr


def main():
    # Dev toggles (useful on macOS where PortAudio / ffmpeg setup may block progress)
    #   GREENLIGHT_NO_AUDIO=1  -> disable microphone capture (skips PortAudio)
    #   GREENLIGHT_NO_SYNC=1   -> disable SyncNet
    ENABLE_AUDIO = os.getenv("GREENLIGHT_NO_AUDIO", "0") != "1"
    ENABLE_SYNC = os.getenv("GREENLIGHT_NO_SYNC", "0") != "1"

    class RiskState(Enum):
        GREEN = auto()
        YELLOW = auto()
        RED = auto()

    def risk_to_state(r: float) -> RiskState:
        if r < 0.3:
            return RiskState.GREEN
        if r < 0.7:
            return RiskState.YELLOW
        return RiskState.RED

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없음")
        return

    # audio (optional)
    mic = None
    if ENABLE_AUDIO:
        mic = MicStream(sr=16000, blocksize=1024, device=None)
        mic.start()
    else:
        print("[DEV] Audio disabled (GREENLIGHT_NO_AUDIO=1)")

    # syncnet (optional)
    sync_cfg = SyncCfg(
        initial_model=str((SYNCNET_REPO / "data" / "syncnet_v2.model").resolve()),
        fps=25,
        audio_sr=16000,
        clip_sec=2.0,
        batch_size=20,
        vshift=50,
        device="cpu",
    )

    rt_sync = None
    if ENABLE_SYNC and ENABLE_AUDIO:
        rt_sync = RealTimeSyncNet(sync_cfg)
        print("SyncNet device: cpu")
    elif not ENABLE_SYNC:
        print("[DEV] SyncNet disabled (GREENLIGHT_NO_SYNC=1)")
    elif ENABLE_SYNC and not ENABLE_AUDIO:
        print("[DEV] SyncNet requires audio. SyncNet disabled because audio is disabled.")

    cap_frames = int(round(30.0 * sync_cfg.clip_sec))  # 60
    mouth_buf = deque(maxlen=cap_frames)
    frame_buf = deque(maxlen=cap_frames)

    # state
    blink_count = 0
    is_blinking = False
    last_blink_time = 0.0
    frame_count = 0
    fake_prob = 0.0
    unsync_prob = 0.0
    final_risk = 0.0
    w_df = 0.75
    w_sync = 0.25

    av_offset_ms: Optional[float] = None
    av_conf: Optional[float] = None

    SYNC_INTERVAL_SEC = 3.0
    DELAY_SEC = 0.8
    last_sync_time = 0.0
    sync_lock = threading.Lock()
    sync_running = False

    def sync_worker(frames_mouth_224: List[np.ndarray], audio_clip: np.ndarray):
        nonlocal av_offset_ms, av_conf, sync_running
        try:
            off, conf = rt_sync.estimate(frames_mouth_224, audio_clip)
            av_offset_ms, av_conf = off, conf
        except Exception as e:
            print("SyncNet error:", repr(e))
            av_offset_ms, av_conf = None, None
        finally:
            with sync_lock:
                sync_running = False

    prev_time = time.time()

    # UI/behavior state machine
    state: RiskState = RiskState.GREEN
    yellow_report_shown = False
    red_enter_time: Optional[float] = None
    red_continue_granted = False
    RED_COUNTDOWN_SEC = 5.0

    def render_report_image(
        risk_state: RiskState,
        final_risk: float,
        fake_prob: float,
        unsync_prob: float,
        av_offset_ms: Optional[float],
        av_conf: Optional[float],
    ) -> np.ndarray:
        """Simple CV report window (fast MVP)."""
        img = np.zeros((340, 640, 3), dtype=np.uint8)
        title = "RISK REPORT"
        cv2.putText(img, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        if risk_state == RiskState.GREEN:
            badge = "GREEN / SAFE"
            badge_color = (0, 255, 0)
        elif risk_state == RiskState.YELLOW:
            badge = "YELLOW / CAUTION"
            badge_color = (0, 255, 255)
        else:
            badge = "RED / DANGER"
            badge_color = (0, 0, 255)

        cv2.putText(img, badge, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, badge_color, 2)
        cv2.putText(img, f"Final risk: {final_risk:.3f}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
        cv2.putText(img, f"Deepfake prob: {fake_prob:.3f}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
        cv2.putText(img, f"Unsync prob: {unsync_prob:.3f}", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)

        if av_offset_ms is None or av_conf is None:
            cv2.putText(img, "AV Sync: n/a", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
        else:
            cv2.putText(img, f"AV offset (ms): {av_offset_ms:.1f}", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
            cv2.putText(img, f"AV confidence: {av_conf:.2f}", (20, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)

        return img

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            h, w = frame.shape[:2]

            frame_buf.append(frame.copy())

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            got_mouth = False

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                lm = face_landmarks.landmark

                # Blink
                left_ear = calculate_ear(lm, LEFT_EYE, w, h)
                right_ear = calculate_ear(lm, RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2.0

                if avg_ear < EAR_THRESHOLD:
                    if not is_blinking:
                        blink_count += 1
                        is_blinking = True
                        last_blink_time = time.time()
                    cv2.putText(frame, "BLINK!", (370, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                else:
                    is_blinking = False

                # Face bbox for deepfake model
                bbox = get_mediapipe_bbox(lm, w, h)

                # mouth crop for SyncNet
                mouth_crop = mouth_crop_from_landmarks(frame, lm, out_size=224, margin=0.35)
                mouth_buf.append(mouth_crop)
                got_mouth = True

                # Deepfake inference (3 frames)
                if frame_count % 3 == 0:
                    face_crop = crop_face_from_bbox(frame, bbox, crop_cfg)
                    if face_crop is not None:
                        raw_prob, conf, label = deepfake_detector.predict_face(face_crop, is_bgr=True, apply_ema=True)

                        final_prob = raw_prob
                        bonus_text = ""

                        if time.time() - last_blink_time < 2.0:
                            if raw_prob > 0.85:
                                final_prob = raw_prob
                                bonus_text = " (Fake Blink Ignored)"
                            else:
                                final_prob = max(0.0, raw_prob - 0.4)
                                bonus_text = " (Liveness Bonus)"

                        fake_prob = float(final_prob)
                        if av_offset_ms is not None and av_conf is not None:
                            unsync_prob = sync_risk(av_offset_ms, av_conf)
                            final_risk = clamp01(w_df * fake_prob + w_sync * unsync_prob)
                        else:
                            final_risk = fake_prob

                        # Artifact view
                        base_map = generate_artifact_map(face_crop)
                        if final_risk < 0.5:
                            display_factor = 0.0
                            status_text = "Clean (Real)"
                            text_color = (0, 255, 0)
                        else:
                            display_factor = max(0.6, final_risk)
                            status_text = "Noise Detected (Fake)"
                            text_color = (0, 0, 255)

                        final_viz = cv2.addWeighted(face_crop, 1.0 - display_factor, base_map, display_factor, 0)
                        artifact_map_big = cv2.resize(final_viz, (0, 0), fx=1.5, fy=1.5)
                        cv2.putText(artifact_map_big, f"{status_text}{bonus_text}", (10, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
                        cv2.imshow("Artifact Analysis (Smart View)", artifact_map_big)

                # bbox draw
                THRESHOLD = 0.8
                color = (0, 0, 255) if final_risk > THRESHOLD else (0, 255, 0)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            if not got_mouth:
                if len(mouth_buf) > 0:
                    mouth_buf.append(mouth_buf[-1].copy())
                else:
                    cx1 = int(w * 0.35); cx2 = int(w * 0.65)
                    cy1 = int(h * 0.55); cy2 = int(h * 0.85)
                    fallback = frame[cy1:cy2, cx1:cx2]
                    fallback = cv2.resize(fallback, (224, 224))
                    mouth_buf.append(fallback)

            now = time.time()
            with sync_lock:
                can_start = (not sync_running)

            if rt_sync is not None and mic is not None and (now - last_sync_time) >= SYNC_INTERVAL_SEC and len(mouth_buf) == mouth_buf.maxlen and can_start:
                last_sync_time = now

                audio3 = mic.last(sync_cfg.clip_sec + DELAY_SEC)
                audio_clip = audio3[: int(sync_cfg.audio_sr * sync_cfg.clip_sec)] 

                frames_30 = list(mouth_buf)  
                sync_frames = int(round(sync_cfg.fps * sync_cfg.clip_sec))  
                frames_25 = resample_frames(frames_30, sync_frames)
    
                with sync_lock:
                    sync_running = True
                th = threading.Thread(target=sync_worker, args=(frames_25, audio_clip), daemon=True)
                th.start()

            # FPS
            curr_time = time.time()
            fps = 1.0 / max(1e-6, (curr_time - prev_time))
            prev_time = curr_time

            # --- State machine (GREEN / YELLOW / RED) ---
            new_state = risk_to_state(final_risk)
            if new_state != state:
                state = new_state
                if state == RiskState.YELLOW:
                    yellow_report_shown = False
                if state == RiskState.RED:
                    red_enter_time = time.time()
                    red_continue_granted = False

            overlay_lines: List[str] = []

            if state == RiskState.GREEN:
                pass

            elif state == RiskState.YELLOW:
                overlay_lines.append("CAUTION: You are at a warning level.")
                overlay_lines.append("Report is shown. Decide whether to continue.")
                overlay_lines.append("(Press 'q' to quit)")
                report = render_report_image(state, final_risk, fake_prob, unsync_prob, av_offset_ms, av_conf)
                cv2.imshow("Risk Report", report)

            else:  # RED
                remain = None
                if red_enter_time is not None:
                    remain = RED_COUNTDOWN_SEC - (time.time() - red_enter_time)

                overlay_lines.append("DANGER: High risk detected.")
                if not red_continue_granted:
                    overlay_lines.append("This call will close automatically.")
                    overlay_lines.append("Press 'c' within 5 seconds to continue.")
                    if remain is not None:
                        overlay_lines.append(f"Auto-close in: {max(0.0, remain):.1f}s")
                else:
                    overlay_lines.append("You chose to continue (override).")
                    overlay_lines.append("Report explains why this is risky.")

                report = render_report_image(state, final_risk, fake_prob, unsync_prob, av_offset_ms, av_conf)
                cv2.imshow("Risk Report", report)

                if (not red_continue_granted) and (remain is not None) and (remain <= 0):
                    print("[RED] Auto-terminating webcam.")
                    break

            draw_greenlight_ui(frame, final_risk, fps, blink_count, overlay_lines=overlay_lines)
            cv2.imshow("DeepTrust Hybrid System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if state == RiskState.RED and key == ord("c"):
                red_continue_granted = True

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if mic is not None:
            mic.stop()


if __name__ == "__main__":
    main()
