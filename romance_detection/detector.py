from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

@dataclass
class DetectorConfig:
    model_id: str = "prithivMLmods/Deep-Fake-Detector-v2-Model"
    device: str = "auto"           
    ema_alpha: float = 0.15       
    unknown_hold_frames: int = 15  
    use_fp16: bool = True          


class DeepfakeDetector:
    """
    Face crop (BGR 또는 RGB 이미지) -> deepfake probability 예측기
    - label 매핑은 모델마다 다를 수 있어, 내부적으로 'fake_prob'를 robust하게 계산함
    """

    def __init__(self, cfg: DetectorConfig = DetectorConfig()):
        self.cfg = cfg

        if cfg.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = cfg.device

        self.processor = AutoImageProcessor.from_pretrained(cfg.model_id)
        self.model = AutoModelForImageClassification.from_pretrained(cfg.model_id)
        self.model.eval()
        self.model.to(self.device)

        self._ema_score: Optional[float] = None
        self._hold_left: int = 0

        # 라벨 이름들 확인용
        self.id2label: Dict[int, str] = getattr(self.model.config, "id2label", {})

    @torch.inference_mode()
    def predict_face(
        self,
        face_rgb_or_bgr: np.ndarray,
        is_bgr: bool = True,
        apply_ema: bool = True,
    ) -> Tuple[float, float, str]:
        """
        Returns:
          fake_prob: [0,1] (높을수록 fake)
          confidence: [0,1] (top1 - top2 softmax gap 기반)
          label: top1 label string
        """
        if face_rgb_or_bgr is None:
            return self._on_unknown()

        # PIL은 RGB 기준
        if is_bgr:
            img = Image.fromarray(face_rgb_or_bgr[:, :, ::-1])
        else:
            img = Image.fromarray(face_rgb_or_bgr)

        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if self.device == "cuda" and self.cfg.use_fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = self.model(**inputs).logits
        else:
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)[0]
        top2 = torch.topk(probs, k=min(2, probs.shape[-1]))
        top1_id = int(top2.indices[0].item())
        top1_p = float(top2.values[0].item())
        top2_p = float(top2.values[1].item()) if top2.values.numel() > 1 else 0.0

        label = self.id2label.get(top1_id, str(top1_id))
        confidence = float(max(0.0, min(1.0, top1_p - top2_p)))

        fake_prob = self._infer_fake_prob(probs)

        if apply_ema:
            fake_prob = self._ema(fake_prob)
            self._hold_left = self.cfg.unknown_hold_frames



        return fake_prob, confidence, label

    def _infer_fake_prob(self, probs: torch.Tensor) -> float:
        """
        모델마다 라벨이 "Deepfake"/"Realism" 또는 "fake"/"real" 등으로 달라서,
        label 문자열을 보고 fake 클래스 확률을 찾는다.
        못 찾으면 top1이 fake 라벨일 때 top1 확률을 fake_prob로 사용.
        """
        # 라벨 문자열 기반으로 fake 후보 찾기
        fake_ids = []
        real_ids = []

        for idx in range(probs.shape[-1]):
            name = str(self.id2label.get(idx, "")).lower()
            if any(k in name for k in ["fake", "deepfake", "manip", "synthetic"]):
                fake_ids.append(idx)
            if any(k in name for k in ["real", "authentic", "realism", "genuine"]):
                real_ids.append(idx)

        if fake_ids:
            p = float(probs[fake_ids].sum().item())
            return float(max(0.0, min(1.0, p)))

        # fake 라벨을 못 찾으면, real 라벨이 있으면 1-real로 처리
        if real_ids:
            p_real = float(probs[real_ids].sum().item())
            return float(max(0.0, min(1.0, 1.0 - p_real)))

        # 둘 다 없으면 top1이 fake라고 가정하기 어려우니 보수적으로 top1 확률 반환
        return float(max(0.0, min(1.0, float(probs.max().item()))))

    def _ema(self, x: float) -> float:
        if self._ema_score is None:
            self._ema_score = x
        else:
            a = self.cfg.ema_alpha
            self._ema_score = (1.0 - a) * self._ema_score + a * x
        return float(self._ema_score)

    def _on_unknown(self) -> Tuple[float, float, str]:
        """
        얼굴 미검출/입력 없음일 때의 처리:
        - hold 프레임이 남아있으면 직전 EMA 유지
        - 아니면 0.5(unknown)로 리셋
        """
        if self._ema_score is not None and self._hold_left > 0:
            self._hold_left -= 1
            return float(self._ema_score), 0.0, "unknown(hold)"
        self._ema_score = None
        self._hold_left = 0
        return 0.5, 0.0, "unknown"