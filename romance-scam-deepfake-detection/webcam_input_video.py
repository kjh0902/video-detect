import time
import threading
from collections import deque
from typing import Optional

import cv2
import numpy as np
import soundfile as sf

from detector import DeepfakeDetector, DetectorConfig
from preprocess import FaceCropConfig, crop_face_from_bbox
from syncnet_wrapper import RTConfig, RealTimeSyncNet

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
import mediapipe as mp

import os
import tempfile
import subprocess


# -----------------------------
# MediaPipe FaceLandmarker
# -----------------------------
MODEL_PATH = "models/face_landmarker.task"

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
EAR_THRESHOLD = 0.22


def make_face_landmarker(model_path: str):
    base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(options)


face_landmarker = make_face_landmarker(MODEL_PATH)


def detect_landmarks_bgr(frame_bgr: np.ndarray, timestamp_ms: int):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
    if not result.face_landmarks:
        return None
    return result.face_landmarks[0]


# -----------------------------
# Deepfake detector (image-based)
# -----------------------------
det_cfg = DetectorConfig(
    model_id="prithivMLmods/Deep-Fake-Detector-v2-Model",
    device="cuda",
    use_fp16=True,
)
deepfake_detector = DeepfakeDetector(det_cfg)
crop_cfg = FaceCropConfig(margin=0.5, min_size=80)


# -----------------------------
# Audio source from video (pre-extract once)
# -----------------------------
class VideoAudioReader:
    def __init__(self, video_path: str, sr: int = 16000):
        self.video_path = video_path
        self.sr = sr
        self.tmp_dir = tempfile.mkdtemp(prefix="video_audio_")
        self.wav_path = os.path.join(self.tmp_dir, "audio_16k.wav")

        cmd = (
            f'ffmpeg -y -i "{video_path}" '
            f'-vn -ac 1 -ar {sr} -acodec pcm_s16le "{self.wav_path}"'
        )
        r = subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r != 0:
            raise RuntimeError("ffmpeg audio extract failed")

        audio, read_sr = sf.read(self.wav_path, dtype="float32")
        if read_sr != sr:
            raise RuntimeError(f"audio sr mismatch: {read_sr} != {sr}")
        if audio.ndim == 2:
            audio = audio[:, 0]
        self.audio = np.asarray(audio, dtype=np.float32)

    def last_exact(self, sec: float, t_sec: float) -> np.ndarray:
        target = int(sec * self.sr)
        end = int(t_sec * self.sr)
        start = end - target

        if end <= 0:
            return np.zeros((target,), np.float32)

        start = max(0, start)
        end = min(len(self.audio), end)

        x = self.audio[start:end]
        if len(x) == target:
            return x
        if len(x) > target:
            return x[-target:]
        pad = target - len(x)
        return np.concatenate([np.zeros((pad,), np.float32), x], axis=0)

    def close(self):
        try:
            if os.path.exists(self.wav_path):
                os.remove(self.wav_path)
            os.rmdir(self.tmp_dir)
        except Exception:
            pass


# -----------------------------
# Score utils
# -----------------------------
def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def av_risk_score(
    av_offset_ms: Optional[float],
    av_conf: Optional[float],
    off_thr_ms: float = 200.0,
    conf_thr: float = 2.5,
) -> Optional[float]:
    if av_offset_ms is None or av_conf is None:
        return None
    offset_score = clamp01(abs(av_offset_ms) / off_thr_ms)
    conf_score = clamp01((conf_thr - av_conf) / conf_thr)
    return 0.6 * offset_score + 0.4 * conf_score


def fuse_scores(fake_prob: float, av_risk: Optional[float], w_df: float = 0.7) -> float:
    fake_prob = clamp01(fake_prob)
    if av_risk is None:
        return fake_prob
    return clamp01(w_df * fake_prob + (1.0 - w_df) * clamp01(av_risk))


# -----------------------------
# Face utils
# -----------------------------
def calculate_ear(landmarks, indices, w, h):
    coords = np.array([[landmarks[idx].x * w, landmarks[idx].y * h] for idx in indices])
    v1 = np.linalg.norm(coords[1] - coords[5])
    v2 = np.linalg.norm(coords[2] - coords[4])
    h_dist = np.linalg.norm(coords[0] - coords[3])
    return (v1 + v2) / max(1e-6, (2.0 * h_dist))


def get_mediapipe_bbox(landmarks, w, h):
    x_min, y_min = w, h
    x_max, y_max = 0, 0
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        x_min = min(x_min, x)
        x_max = max(x_max, x)
        y_min = min(y_min, y)
        y_max = max(y_max, y)
    return (x_min, y_min, x_max, y_max)


def draw_dashboard(frame, score, blink_count, fps, av_offset, av_conf, last_sync_age):
    cv2.rectangle(frame, (10, 10), (480, 210), (0, 0, 0), -1)

    color = (0, 255, 0) if score < 0.5 else (0, 0, 255)
    status = "REAL" if score < 0.5 else "FAKE WARNING"
    cv2.putText(frame, f"AI Analysis: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.rectangle(frame, (20, 50), (320, 70), (50, 50, 50), -1)
    fill_width = int(300 * clamp01(score))
    cv2.rectangle(frame, (20, 50), (20 + fill_width, 70), color, -1)
    cv2.putText(frame, f"{score * 100:.1f}%", (20 + min(fill_width, 280) + 5, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, f"Blinks: {blink_count}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    if av_offset is None or av_conf is None:
        cv2.putText(frame, f"AV Sync: ... (age={last_sync_age:.1f}s)", (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    else:
        sync_ok = (av_conf >= 2.5) and (abs(av_offset) <= 120.0)
        s_color = (0, 255, 0) if sync_ok else (0, 0, 255)
        cv2.putText(frame, f"AV offset(ms): {av_offset:.1f}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, s_color, 1)
        cv2.putText(frame, f"AV conf: {av_conf:.2f} | age={last_sync_age:.1f}s", (20, 195),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, s_color, 1)


# -----------------------------
# Main (Video input)
# -----------------------------
def main():
    VIDEO_PATH = "/home/junhyung/Documents/vscode/talking_man.mkv"

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps is None or src_fps <= 1e-3:
        src_fps = 25.0

    cfg = RTConfig(
        syncnet_repo="/home/junhyung/Documents/vscode/syncnet_python",
        data_dir="/home/junhyung/Documents/vscode/syncnet_python/data/rt_work",
        clip_sec=2.0,
        fps=25,
        audio_sr=16000,
    )

    rt_sync = RealTimeSyncNet(cfg)
    audio_src = VideoAudioReader(VIDEO_PATH, sr=cfg.audio_sr)

    frame_buf = deque(maxlen=int(cfg.fps * cfg.clip_sec))  # 50 frames
    blink_count = 0
    is_blinking = False
    frame_idx = 0
    prev_time = time.time()

    sync_lock = threading.Lock()
    sync_offset = None
    sync_conf = None
    sync_busy = False
    last_req_time = 0.0
    last_done_time = 0.0
    SYNC_INTERVAL_SEC = 3.0

    def _syncnet_worker(frames_clip, audio_clip):
        nonlocal sync_offset, sync_conf, sync_busy, last_done_time
        try:
            off, conf = rt_sync.estimate_from_recent_clip(frames_clip, audio_clip)
        except Exception:
            off, conf = None, None
        with sync_lock:
            sync_offset = off
            sync_conf = conf
            last_done_time = time.time()
            sync_busy = False

    WIN = "DeepTrust Hybrid System (Video Input)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            frame_buf.append(frame.copy())

            h, w, _ = frame.shape
            fake_prob = 0.0

            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_msec is None or pos_msec <= 0:
                t_sec = frame_idx / src_fps
                timestamp_ms = int(t_sec * 1000.0)
            else:
                t_sec = pos_msec / 1000.0
                timestamp_ms = int(pos_msec)

            landmarks = detect_landmarks_bgr(frame, timestamp_ms)
            if landmarks is not None:
                left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
                right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2.0

                if avg_ear < EAR_THRESHOLD:
                    if not is_blinking:
                        blink_count += 1
                        is_blinking = True
                else:
                    is_blinking = False

                bbox = get_mediapipe_bbox(landmarks, w, h)
                face_crop = crop_face_from_bbox(frame, bbox, crop_cfg)
                try:
                    fake_prob, _, _ = deepfake_detector.predict_face(face_crop, is_bgr=True, apply_ema=True)
                except Exception:
                    fake_prob = 0.0

                color = (0, 0, 255) if fake_prob > 0.5 else (0, 255, 0)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            now = time.time()
            with sync_lock:
                can_request = (
                    (now - last_req_time) >= SYNC_INTERVAL_SEC
                    and (len(frame_buf) == frame_buf.maxlen)
                    and (not sync_busy)
                )
                if can_request:
                    sync_busy = True
                    last_req_time = now

            if can_request:
                frames_clip = list(frame_buf)
                audio_clip = audio_src.last_exact(cfg.clip_sec, t_sec)
                threading.Thread(
                    target=_syncnet_worker,
                    args=(frames_clip, audio_clip),
                    daemon=True,
                ).start()

            with sync_lock:
                av_offset = sync_offset
                av_conf = sync_conf
                done_time = last_done_time

            last_sync_age = (time.time() - done_time) if done_time > 0 else 1e9

            curr_time = time.time()
            fps = 1.0 / max(1e-6, (curr_time - prev_time))
            prev_time = curr_time

            risk_av = av_risk_score(av_offset, av_conf)
            final_score = fuse_scores(fake_prob, risk_av, w_df=0.7)

            draw_dashboard(frame, final_score, blink_count, fps, av_offset, av_conf, last_sync_age)

            cv2.imshow(WIN, frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        audio_src.close()


if __name__ == "__main__":
    main()
