# demo_webcam.py
import time
import threading
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd

from detector import DeepfakeDetector, DetectorConfig
from preprocess import FaceCropConfig, crop_face_from_bbox

from syncnet_wrapper import RTConfig, RealTimeSyncNet

# MediaPipe Tasks (FaceLandmarker)
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision


# -----------------------------
# MediaPipe FaceLandmarker (Tasks API)
# -----------------------------
MODEL_PATH = "models/face_landmarker.task"

# FaceMesh indices (MediaPipe face landmark index는 동일하게 사용 가능)
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
    """
    return: list[NormalizedLandmark] or None
    """
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
    device="auto",
    use_fp16=True,
)
deepfake_detector = DeepfakeDetector(det_cfg)
crop_cfg = FaceCropConfig(margin=0.5, min_size=80)


# -----------------------------
# Audio ring buffer (non-blocking)
# -----------------------------
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


# -----------------------------
# Utils
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
        if x < x_min:
            x_min = x
        if x > x_max:
            x_max = x
        if y < y_min:
            y_min = y
        if y > y_max:
            y_max = y

    return (x_min, y_min, x_max, y_max)


def draw_dashboard(frame, fake_prob, blink_count, fps, av_offset, av_conf):
    cv2.rectangle(frame, (10, 10), (450, 210), (0, 0, 0), -1)

    color = (0, 255, 0) if fake_prob < 0.5 else (0, 0, 255)
    status = "REAL" if fake_prob < 0.5 else "FAKE WARNING"
    cv2.putText(frame, f"AI Analysis: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.rectangle(frame, (20, 50), (320, 70), (50, 50, 50), -1)
    fill_width = int(300 * max(0.0, min(1.0, fake_prob)))
    cv2.rectangle(frame, (20, 50), (20 + fill_width, 70), color, -1)
    cv2.putText(
        frame,
        f"{fake_prob * 100:.1f}%",
        (20 + min(fill_width, 280) + 5, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    cv2.putText(frame, f"Blinks: {blink_count}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    if av_offset is None or av_conf is None:
        cv2.putText(frame, "AV Sync: ...", (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    else:
        sync_ok = (av_conf >= 2.5) and (abs(av_offset) <= 120.0)
        s_color = (0, 255, 0) if sync_ok else (0, 0, 255)
        cv2.putText(frame, f"AV offset(ms): {av_offset:.1f}", (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, s_color, 1)
        cv2.putText(frame, f"AV conf: {av_conf:.2f}", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, s_color, 1)


# -----------------------------
# Main
# -----------------------------
def main():
    cap = cv2.VideoCapture(0)

    # SyncNet 설정
    cfg = RTConfig(
        syncnet_repo="/home/junhyung/Documents/vscode/syncnet_python",
        data_dir="/home/junhyung/Documents/vscode/syncnet_python/data/rt_work",
        clip_sec=2.0,
        fps=25,
        audio_sr=16000,
    )
    rt_sync = RealTimeSyncNet(cfg)

    # 오디오 스트림(논블로킹)
    mic = MicStream(sr=cfg.audio_sr, blocksize=1024, device=None)
    mic.start()

    # 최근 clip_sec 프레임 버퍼
    frame_buf = deque(maxlen=int(cfg.fps * cfg.clip_sec))

    # SyncNet 결과
    av_offset = None
    av_conf = None

    # SyncNet 호출 주기(무거워서 너무 짧게 하면 UI 버벅)
    SYNC_INTERVAL_SEC = 3.0
    last_sync_time = 0.0

    blink_count = 0
    is_blinking = False

    prev_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_buf.append(frame.copy())

            h, w, _ = frame.shape
            fake_prob = 0.0

            # Tasks API는 timestamp(ms)가 필요
            timestamp_ms = int(time.time() * 1000)
            landmarks = detect_landmarks_bgr(frame, timestamp_ms)

            if landmarks is not None:
                left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
                right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2.0

                if avg_ear < EAR_THRESHOLD:
                    if not is_blinking:
                        blink_count += 1
                        is_blinking = True
                    cv2.putText(frame, "BLINK!", (370, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                else:
                    is_blinking = False

                bbox = get_mediapipe_bbox(landmarks, w, h)
                face_crop = crop_face_from_bbox(frame, bbox, crop_cfg)

                fake_prob, conf, label = deepfake_detector.predict_face(face_crop, is_bgr=True, apply_ema=True)

                color = (0, 0, 255) if fake_prob > 0.5 else (0, 255, 0)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            # SyncNet 실행(주기적으로)
            now = time.time()
            if (now - last_sync_time) >= SYNC_INTERVAL_SEC and len(frame_buf) == frame_buf.maxlen:
                last_sync_time = now

                audio = mic.last(cfg.clip_sec)
                frames = list(frame_buf)

                try:
                    av_offset, av_conf = rt_sync.estimate_from_recent_clip(frames, audio)
                except Exception:
                    av_offset, av_conf = None, None

            # FPS
            curr_time = time.time()
            fps = 1.0 / max(1e-6, (curr_time - prev_time))
            prev_time = curr_time

            draw_dashboard(frame, fake_prob, blink_count, fps, av_offset, av_conf)
            cv2.imshow("DeepTrust Hybrid System", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        mic.stop()


if __name__ == "__main__":
    main()
