# syncnet_wrapper.py
from __future__ import annotations

import os, sys
SYNCNET_REPO = "/home/junhyung/Documents/vscode/syncnet_python"
sys.path.insert(0, SYNCNET_REPO)
from SyncNetInstance import SyncNetInstance

import time
import glob
import pickle
import shutil
import tempfile
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch


@dataclass
class RTConfig:
    syncnet_repo: str = "/home/junhyung/Documents/vscode/syncnet_python"              # syncnet_python 경로
    initial_model: str = "/home/junhyung/Documents/vscode/syncnet_python/data/syncnet_v2.model"
    data_dir: str = "/home/junhyung/Documents/vscode/syncnet_python/output"     # run_pipeline output 루트
    clip_sec: float = 2.0
    fps: int = 25
    audio_sr: int = 16000
    batch_size: int = 20
    vshift: int = 15
    device: str = "cuda" 


class AudioRecorder:
    def __init__(self, sr: int = 16000, channels: int = 1):
        self.sr = sr
        self.channels = channels

    def record(self, seconds: float) -> np.ndarray:
        x = sd.rec(int(seconds * self.sr), samplerate=self.sr, channels=self.channels, dtype="float32")
        sd.wait()
        if x.ndim == 2:
            x = x[:, 0]
        return x


class RealTimeSyncNet:
    def __init__(self, cfg: RTConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if (cfg.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
        # SyncNet 모델 로드
        self.s = SyncNetInstance()
        self.s.loadParameters(os.path.join(cfg.syncnet_repo, cfg.initial_model))
        self._move_syncnet_to_device()
        print("SyncNet device:", self.device)

        os.makedirs(cfg.data_dir, exist_ok=True)

    def _move_syncnet_to_device(self):
        # SyncNetInstance 안에 들어있는 torch.nn.Module을 찾아서 전부 to(device)
        moved = False
        for name, obj in vars(self.s).items():
            if isinstance(obj, torch.nn.Module):
                obj.to(self.device)
                moved = True

        # 어떤 구현은 내부 속성이 vars()에 바로 안 잡힐 수 있어서 한 번 더 보강
        for name in ["model", "net", "syncnet", "S", "__S__", "network"]:
            if hasattr(self.s, name):
                obj = getattr(self.s, name)
                if isinstance(obj, torch.nn.Module):
                    obj.to(self.device)
                    moved = True

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        if not moved:
            print("Warning: SyncNetInstance 내부에서 nn.Module을 못 찾음. (SyncNetInstance.py 수정이 필요할 수 있음)")


    def _write_clip_mp4(self, frames_bgr: List[np.ndarray], mp4_path: str):
        h, w = frames_bgr[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(mp4_path, fourcc, self.cfg.fps, (w, h))
        for fr in frames_bgr:
            vw.write(fr)
        vw.release()

    def _mux_av(self, video_mp4: str, audio_wav: str, out_mp4: str):
        dur = self.cfg.clip_sec
        cmd = (
            f'ffmpeg -y '
            f'-i "{video_mp4}" -i "{audio_wav}" '
            f'-c:v copy '
            f'-c:a aac -ac 1 -ar {self.cfg.audio_sr} '
            f'-filter:a "atrim=0:{dur},apad=pad_dur={dur}" '
            f'-t {dur} '
            f'"{out_mp4}"'
        )
        r = subprocess.call(cmd, shell=True)
        if r != 0:
            raise RuntimeError("ffmpeg mux 실패")


    def _run_pipeline(self, videofile: str, reference: str):
        # run_pipeline.py를 subprocess로 실행 (syncnet_repo 안에 있다고 가정)
        run_pipeline = os.path.join(self.cfg.syncnet_repo, "run_pipeline.py")
        min_track = max(15, int(self.cfg.fps * self.cfg.clip_sec * 0.6))  # 2초면 30프레임 정도
        cmd = (
            f'python "{run_pipeline}" '
            f'--videofile "{videofile}" '
            f'--reference "{reference}" '
            f'--data_dir "{self.cfg.data_dir}" '
            f'--frame_rate {self.cfg.fps} '
            f'--min_track {min_track} '
            f'--num_failed_det 10 '
            f'--min_face_size 60 '
        )
        r = subprocess.call(cmd, cwd=self.cfg.syncnet_repo, shell=True)
        if r != 0:
            raise RuntimeError("run_pipeline 실패")



    def estimate_from_recent_clip(self, frames_bgr, audio_16k):
        target_frames = int(self.cfg.fps * self.cfg.clip_sec)          # 50
        target_samples = int(self.cfg.audio_sr * self.cfg.clip_sec)    # 32000

        # frames: 정확히 50장
        if len(frames_bgr) >= target_frames:
            frames_bgr = frames_bgr[-target_frames:]
        else:
            if len(frames_bgr) == 0:
                return None, None
            pad = [frames_bgr[-1]] * (target_frames - len(frames_bgr))
            frames_bgr = frames_bgr + pad

        # audio: 정확히 32000샘플
        audio_16k = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
        if len(audio_16k) >= target_samples:
            audio_16k = audio_16k[-target_samples:]
        else:
            audio_16k = np.pad(audio_16k, (0, target_samples - len(audio_16k)), mode="constant")

        if len(frames_bgr) < int(self.cfg.fps * self.cfg.clip_sec * 0.8):
            return None, None
        if len(audio_16k) < int(self.cfg.audio_sr * self.cfg.clip_sec * 0.8):
            return None, None

        tmp = tempfile.mkdtemp(prefix="rt_syncnet_")
        try:
            video_only = os.path.join(tmp, "v.mp4")
            audio_wav = os.path.join(tmp, "a.wav")
            av_mp4 = os.path.join(tmp, "av.mp4")

            # 1) mp4 저장
            self._write_clip_mp4(frames_bgr, video_only)

            # 2) wav 저장(16k mono)
            sf.write(audio_wav, audio_16k, self.cfg.audio_sr)

            # 3) mux 해서 오디오 포함 mp4 만들기
            self._mux_av(video_only, audio_wav, av_mp4)

            # 4) run_pipeline으로 얼굴 트랙 crop 생성
            reference = f"seg_{int(time.time()*1000)}"
            self._run_pipeline(av_mp4, reference)

            # 5) 생성된 crop 영상들에 대해 SyncNet evaluate
            crop_dir = os.path.join(self.cfg.data_dir, "pycrop", reference)
            flist = sorted(glob.glob(os.path.join(crop_dir, "0*.avi")))
            if len(flist) == 0:
                return None, None

            # opt 비슷한 객체가 필요해서 간단히 dict-like로 구성
            class Opt:
                pass
            opt = Opt()
            opt.batch_size = self.cfg.batch_size
            opt.vshift = self.cfg.vshift
            opt.data_dir = self.cfg.data_dir
            opt.reference = reference
            opt.videofile = ""
            opt.tmp_dir = os.path.join(self.cfg.data_dir, "pytmp")   # 이 줄 추가
            os.makedirs(opt.tmp_dir, exist_ok=True)
            best_conf = -1e9
            best_offset = None

            for f in flist:
                offset, conf, dist = self.s.evaluate(opt, videofile=f)
                if conf > best_conf:
                    best_conf = float(conf)
                    best_offset = float(offset)

            return best_offset, best_conf

        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            # run_pipeline output 폴더도 계속 쌓이면 커져서 reference 단위로 지우는 게 좋음
            # 필요하면 여기서 self.cfg.data_dir 아래 reference 관련 폴더들 삭제하도록 추가 가능