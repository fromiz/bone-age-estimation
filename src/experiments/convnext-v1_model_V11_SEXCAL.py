# -*- coding: utf-8 -*-
r"""
수부 X-ray 뼈나이 예측 (ConvNeXt V1-Tiny + LDL + FiLM)
======================================================

파이프라인 (v11 학습 데이터 crop_yolo_seg 생성 조건과 동일)
--------------------------------------------------------
    원본 X-ray
      -> [1] YOLOX-S 손 검출 (최고 confidence 박스)
      -> [2] 분할용 확대 crop (좌우/상/하 10%)
      -> [3] 손 분할 마스크 생성 -> 최대 연결성분 + 구멍 메움
      -> [4] PCA 주축 + 손가락/손목 잔여각으로 회전
      -> [5] 회전된 마스크 경계로 재크롭 (좌우 4%, 상 3%, 하 2%)
      -> [6] 이미지별 1~99 퍼센타일 강도 정규화
      -> [7] 비율유지 리사이즈 + 0 패딩 -> 768x512 (HxW)
      -> [8] 그레이 3채널 복제 + ImageNet 정규화
      -> [9] ConvNeXt-Tiny + FiLM 성별조건화 -> 240-bin LDL
      -> [10] 회전 TTA 기대값 평균
      -> [11] 선택: 선형 캘리브레이션
      -> 골연령(개월)

이 스크립트는 추론 경로에서 ultralytics 를 사용하지 않습니다.

전처리 상수(입력 해상도 / 강도 정규화 / 패딩)와 출력 헤드 구성
(LDL 여부 / 빈 개수 / FiLM stage)은 models/best_model.pt 안의 arch
딕셔너리에서 읽습니다. 체크포인트를 바꿔 끼우면 설정이 자동으로 따라오므로
학습 설정과 추론 설정이 어긋날 수 없습니다.

실행:
    .\.venv\Scripts\python.exe convnext-v1_model.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF

import timm


# =============================================================================
# 기업 테스트 설정
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parent

IMAGES_DIR = PROJECT_DIR / "Images"
METADATA_CSV = PROJECT_DIR / "test.csv"

YOLOX_DIR = PROJECT_DIR / "YOLOX"
YOLOX_EXP = PROJECT_DIR / "yolox_s_hand.py"
YOLOX_MODEL = PROJECT_DIR / "models" / "yolox_s_hand_best.pth"
SEG_MODEL = PROJECT_DIR / "models" / "hand_seg_crop512_traced.pt"

BONEAGE_MODEL = PROJECT_DIR / "models" / "best_model.pt"
CALIBRATION_JSON = PROJECT_DIR / "models" / "calibration.json"

OUTPUT_CSV = PROJECT_DIR / "predictions.csv"
METRICS_CSV = PROJECT_DIR / "metrics.csv"
METRICS_JSON = PROJECT_DIR / "metrics.json"

DEVICE = "auto"
BATCH_SIZE = 8

# "raw"             : 원본 X-ray -> 전처리 전 과정 -> 뼈나이
# "already_cropped" : 이미 전처리된 손 영상 -> 검출/분할/정렬 생략
INPUT_MODE = "raw"

# 최종 모델 입력 캔버스를 crops_input 에 저장 (전처리 검증용)
SAVE_CROPS = False

# 각 단계 중간 결과를 qc 폴더에 저장 (마스크 / 정렬 결과)
SAVE_QC = False

# calibration.json 이 있고 "used": true 인 경우에만 적용
USE_CALIBRATION = True

# test.csv 에 정답 뼈나이 열이 있으면 성능 지표 자동 계산
EVALUATE_IF_LABELS = True
BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_SEED = 42
BOOTSTRAP_CI = 0.95
WORST_CASE_COUNT = 20


# =============================================================================
# 전처리 단계 구성  [v11 학습 데이터 생성 조건과 동일]
# =============================================================================

# 손목선 검출 기반 2차 정렬. v11 은 이 단계를 사용하지 않습니다.
USE_WRIST_ALIGN = False

# 마스크 바깥을 0 으로 채워 배경과 판독 마커를 제거. v11 학습 데이터(crop_yolo_seg)는 배경을 남긴 상태입니다.
MASK_OUT_BACKGROUND = False


# =============================================================================
# 고정 상수 - 학습 데이터 생성 스크립트와 동일해야 합니다
# =============================================================================

# ── YOLOX-S 손 검출 ──────────────────────────────────────────────────
YOLOX_CONF = 0.20
YOLOX_NMS = 0.70

# ── 분할 입력용 확대 crop (검출 박스 대비) ───────────────────────────
SEG_MARGIN_X = 0.10
SEG_MARGIN_TOP = 0.10
SEG_MARGIN_BOTTOM = 0.10

SEG_INPUT_SIZE = 512
SEG_THRESHOLD = 0.5
SEG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SEG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── 회전 후 마스크 기준 재크롭 margin ────────────────────────────────
MARGIN_LEFT = 0.04
MARGIN_RIGHT = 0.04
MARGIN_TOP = 0.03
MARGIN_BOTTOM = 0.02

# ── 1차 정렬 (PCA + 손가락/손목 잔여각) ──────────────────────────────
TOP_BAND_RATIO = 0.12
WRIST_BAND_LO = 0.82
WRIST_BAND_HI = 0.92
RESIDUAL_CLIP = 12.0

# ── 2차 정렬 (손목선 E1 / 전완축 E2) ─────────────────────────────────
SM_CLOSE = 0.06
SM_OPEN = 0.03
SM_SIGMA = 0.030

E1_BAND = 0.16
E1_EPS = 0.010
E1_MINLEN = 0.08
E1_MAXTILT = 60.0
E1_TOL = 4.0

E2_LO, E2_HI = 0.70, 0.92

AGREE_TOL = 8.0
ANGLE_MAX = 45.0
ROT_ITERS = 2

MASK_FEATHER = 2
WRIST_MARGIN_FRAC = 0.03
MIN_SIDE = 128

# ── 최종 캔버스 (기본값. 실제 값은 체크포인트 arch 에서 읽음) ────────
DEFAULT_IMG_H = 768
DEFAULT_IMG_W = 512
DEFAULT_NORM_MODE = "p1p99"
DEFAULT_RESIZE_MODE = "letterbox"
DEFAULT_PAD_VALUE = 0
DEFAULT_PAD_ANCHOR = "center"
DEFAULT_BACKBONE = "convnext_tiny.fb_in22k_ft_in1k_384"
DEFAULT_AGE_BINS = 240

# calibration.json 에 tta_angles 가 없을 때 사용
DEFAULT_TTA_ANGLES = (0, -6, -3, 3, 6)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PAD_NORM = [-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

EVAL_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# =============================================================================
# Unicode-safe image I/O
#   cv2.imread / imwrite 는 경로에 한글이 있으면 예외 없이 실패합니다.
# =============================================================================

def imread_unicode(path, flags=cv2.IMREAD_UNCHANGED):
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    except Exception:
        return None


def imwrite_unicode(path, image) -> bool:
    path = str(path)
    ext = os.path.splitext(path)[1] or ".png"
    try:
        ok, buf = cv2.imencode(ext, image)
        if ok:
            buf.tofile(path)
        return bool(ok)
    except Exception:
        return False


# =============================================================================
# [A] YOLOX-S 손 검출
# =============================================================================

def load_yolox_detector(
    *,
    yolox_dir: Path,
    exp_path: Path,
    checkpoint_path: Path,
    device: torch.device,
):
    if not yolox_dir.is_dir():
        raise FileNotFoundError(
            f"YOLOX source 폴더가 없습니다: {yolox_dir}"
        )

    if str(yolox_dir) not in sys.path:
        sys.path.insert(0, str(yolox_dir))

    try:
        from yolox.exp import get_exp
        from yolox.data.data_augment import ValTransform
        from yolox.utils import postprocess
    except ImportError as exc:
        raise ImportError(
            "YOLOX import 실패. YOLOX 폴더와 requirements 설치를 확인하세요."
        ) from exc

    exp = get_exp(str(exp_path), None)
    model = exp.get_model()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )

    model.load_state_dict(restore_float32(state), strict=True)
    model.to(device)
    model.eval()

    return {
        "exp": exp,
        "model": model,
        "preproc": ValTransform(legacy=False),
        "postprocess": postprocess,
    }


def to_detector_bgr8(image: np.ndarray) -> np.ndarray:
    """검출 입력만 임시 8-bit BGR 로 변환합니다.
       실제 crop 은 원본 비트깊이 배열에서 만듭니다."""
    if image is None:
        raise ValueError("image is None")

    x = image

    if x.ndim == 2:
        gray = x
    elif x.ndim == 3:
        if x.shape[2] == 1:
            gray = x[..., 0]
        elif x.shape[2] == 3:
            if x.dtype == np.uint8:
                return x
            gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        elif x.shape[2] == 4:
            if x.dtype == np.uint8:
                return cv2.cvtColor(x, cv2.COLOR_BGRA2BGR)
            gray = cv2.cvtColor(x, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(f"지원하지 않는 channel shape: {x.shape}")
    else:
        raise ValueError(f"지원하지 않는 image shape: {x.shape}")

    if gray.dtype == np.uint8:
        gray8 = gray
    else:
        arr = gray.astype(np.float32, copy=False)
        finite = np.isfinite(arr)

        if not finite.any():
            raise ValueError("finite pixel 이 없습니다.")

        values = arr[finite]
        lo, hi = float(values.min()), float(values.max())

        if hi <= lo:
            gray8 = np.zeros(gray.shape, dtype=np.uint8)
        else:
            scaled = (arr - lo) * (255.0 / (hi - lo))
            scaled[~finite] = 0.0
            gray8 = np.clip(scaled, 0, 255).astype(np.uint8)

    return cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)


def detect_hand_box(
    *,
    detector,
    detector_input: np.ndarray,
    image_height: int,
    image_width: int,
    device: torch.device,
):
    """최고 score 박스 (x0, y0, x1, y1) 를 원본 좌표로 반환. 없으면 None."""
    exp = detector["exp"]
    model = detector["model"]
    preproc = detector["preproc"]
    postprocess = detector["postprocess"]

    ratio = min(
        exp.test_size[0] / image_height,
        exp.test_size[1] / image_width,
    )

    tensor, _ = preproc(detector_input, None, exp.test_size)

    tensor = (
        torch.from_numpy(tensor)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    with torch.inference_mode():
        outputs = model(tensor)

        output = postprocess(
            outputs,
            num_classes=1,
            conf_thre=YOLOX_CONF,
            nms_thre=YOLOX_NMS,
            class_agnostic=True,
        )[0]

    if output is None or len(output) == 0:
        return None

    output = output.detach().cpu()

    boxes = output[:, :4] / ratio
    scores = output[:, 4] * output[:, 5]

    best = int(torch.argmax(scores).item())

    return tuple(
        float(v) for v in boxes[best].numpy().tolist()
    )


def expand_box(
    box_xyxy,
    *,
    image_width: int,
    image_height: int,
):
    x0, y0, x1, y1 = box_xyxy

    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)

    x0 = x0 - width * MARGIN_LEFT
    x1 = x1 + width * MARGIN_RIGHT
    y0 = y0 - height * MARGIN_TOP
    y1 = y1 + height * MARGIN_BOTTOM

    x0 = int(max(0, math.floor(x0)))
    y0 = int(max(0, math.floor(y0)))
    x1 = int(min(image_width, math.ceil(x1)))
    y1 = int(min(image_height, math.ceil(y1)))

    if x1 <= x0 or y1 <= y0:
        raise ValueError("유효하지 않은 crop box 입니다.")

    return x0, y0, x1, y1


# =============================================================================
# [B] 손 분할 및 마스크 정제
# =============================================================================

def load_segmentation_model(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"segmentation model 없음: {checkpoint_path}"
        )

    model = torch.jit.load(str(checkpoint_path), map_location=device)
    model.eval()
    return model


def predict_seg_mask(crop_bgr, seg_model, device: torch.device) -> np.ndarray:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    x = cv2.resize(
        rgb,
        (SEG_INPUT_SIZE, SEG_INPUT_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    x = x.astype(np.float32) / 255.0
    x = (x - SEG_MEAN) / SEG_STD

    x = (
        torch.from_numpy(x.transpose(2, 0, 1))
        .unsqueeze(0)
        .float()
        .to(device)
    )

    with torch.inference_mode():
        logits = seg_model(x)

        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        if logits.ndim != 4:
            raise RuntimeError(
                f"segmentation output shape 오류: {tuple(logits.shape)}"
            )

        if logits.shape[1] == 1:
            prob = torch.sigmoid(logits)[0, 0]
        else:
            prob = torch.softmax(logits, dim=1)[0, 1]

        mask = (
            prob.detach().cpu().numpy() >= SEG_THRESHOLD
        ).astype(np.uint8)

    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def clean_hand_mask(mask: np.ndarray) -> np.ndarray:
    """가장 큰 연결성분만 남기고 내부 구멍을 메웁니다."""
    mask = (mask > 0).astype(np.uint8)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if n_labels <= 1:
        return mask

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    clean = (labels == largest).astype(np.uint8)

    padded = np.pad(clean, 1, mode="constant", constant_values=0)

    flood = (1 - padded).astype(np.uint8)
    ff_mask = np.zeros(
        (flood.shape[0] + 2, flood.shape[1] + 2), dtype=np.uint8
    )

    cv2.floodFill(flood, ff_mask, seedPoint=(0, 0), newVal=2)

    holes = (flood == 1).astype(np.uint8)
    filled = np.clip(padded + holes, 0, 1).astype(np.uint8)

    return filled[1:-1, 1:-1]


# =============================================================================
# [C] 1차 정렬 - PCA 주축 + 손가락/손목 잔여각
# =============================================================================

def get_pca_axis(mask: np.ndarray):
    ys, xs = np.where(mask > 0)

    if len(xs) < 100:
        return None

    points = np.column_stack([xs, ys]).astype(np.float32)
    _, eigenvectors, _ = cv2.PCACompute2(points, mean=None)

    axis = eigenvectors[0]
    theta = np.degrees(np.arctan2(axis[1], axis[0])) % 180.0

    return float(theta - 90.0)


def rotate_mask_bound(mask: np.ndarray, angle: float) -> np.ndarray:
    h, w = mask.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(math.ceil(h * sin + w * cos))
    new_h = int(math.ceil(h * cos + w * sin))

    M[0, 2] += new_w / 2.0 - cx
    M[1, 2] += new_h / 2.0 - cy

    rotated = cv2.warpAffine(
        (mask * 255).astype(np.uint8),
        M,
        (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return (rotated > 127).astype(np.uint8)


def get_finger_wrist_residual(mask: np.ndarray):
    """손가락 끝 조각과 손목 중앙을 잇는 축의 기울기를 잔여각으로 씁니다."""
    mask = (mask > 0).astype(np.uint8)

    ys, xs = np.where(mask > 0)
    if len(xs) < 100:
        return None

    y_min, y_max = int(ys.min()), int(ys.max())
    height = y_max - y_min + 1

    top_limit = int(y_min + height * TOP_BAND_RATIO)

    top_mask = np.zeros_like(mask)
    top_mask[y_min:top_limit + 1] = mask[y_min:top_limit + 1]

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        top_mask, connectivity=8
    )

    candidates = []

    for idx in range(1, n):
        if int(stats[idx, cv2.CC_STAT_AREA]) < 10:
            continue

        comp_ys, comp_xs = np.where(labels == idx)

        candidates.append({
            "top_y": int(comp_ys.min()),
            "cx": float(comp_xs.mean()),
            "cy": float(comp_ys.mean()),
        })

    if not candidates:
        return None

    finger = min(candidates, key=lambda item: item["top_y"])

    wrist_y1 = int(y_min + height * WRIST_BAND_LO)
    wrist_y2 = int(y_min + height * WRIST_BAND_HI)

    local_ys, wrist_xs = np.where(mask[wrist_y1:wrist_y2 + 1] > 0)

    if len(wrist_xs) < 20:
        return None

    wrist_ys = local_ys + wrist_y1

    wrist_x = float(np.median(wrist_xs))
    wrist_y = float(np.median(wrist_ys))

    dx = finger["cx"] - wrist_x
    dy = finger["cy"] - wrist_y

    residual = float(np.degrees(np.arctan2(dx, -dy)))

    return float(np.clip(residual, -RESIDUAL_CLIP, RESIDUAL_CLIP))


def get_total_rotation(mask: np.ndarray):
    pca_angle = get_pca_axis(mask)

    if pca_angle is None:
        return None

    residual = get_finger_wrist_residual(rotate_mask_bound(mask, pca_angle))

    if residual is None:
        residual = 0.0

    return float(pca_angle + residual)


def estimate_background_value(image: np.ndarray, mask: np.ndarray):
    """회전 여백을 원본 배경값으로 채우기 위한 중앙값."""
    bg_pixels = image[mask == 0]

    if bg_pixels.size == 0:
        if image.ndim == 2:
            return 0
        return tuple(0 for _ in range(image.shape[2]))

    if image.ndim == 2:
        return float(np.median(bg_pixels))

    return tuple(float(v) for v in np.median(bg_pixels, axis=0))


def rotate_native_pair_once(image: np.ndarray, mask: np.ndarray, angle: float):
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(math.ceil(h * sin + w * cos))
    new_h = int(math.ceil(h * cos + w * sin))

    M[0, 2] += new_w / 2.0 - cx
    M[1, 2] += new_h / 2.0 - cy

    bg = estimate_background_value(image, mask)

    rotated_img = cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=bg,
    )

    rotated_mask = cv2.warpAffine(
        (mask * 255).astype(np.uint8), M, (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return rotated_img, (rotated_mask > 127).astype(np.uint8)


def crop_by_rotated_mask(image: np.ndarray, mask: np.ndarray):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1

    bw, bh = x2 - x1, y2 - y1

    ml = int(round(bw * MARGIN_LEFT))
    mr = int(round(bw * MARGIN_RIGHT))
    mt = int(round(bh * MARGIN_TOP))
    mb = int(round(bh * MARGIN_BOTTOM))

    H, W = image.shape[:2]

    fx1 = max(0, x1 - ml)
    fx2 = min(W, x2 + mr)
    fy1 = max(0, y1 - mt)
    fy2 = min(H, y2 + mb)

    if fx2 <= fx1 or fy2 <= fy1:
        return None

    return (
        image[fy1:fy2, fx1:fx2].copy(),
        mask[fy1:fy2, fx1:fx2].copy(),
    )


def to_gray_native(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image

    if image.shape[2] == 1:
        return image[..., 0]

    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    raise ValueError(f"지원하지 않는 channel shape: {image.shape}")


# =============================================================================
# [D] 2차 정렬 - 손목선 검출
#
#   서로 독립인 두 추정기로 각도를 재고, 둘이 일치할 때만 회전합니다.
#     E1  하단 윤곽의 최장 직선분   = 손목 절단면 자체
#     E2  전완 중심선 기울기의 수직 = 손목선은 전완축에 직교
#   불일치하면 회전을 포기합니다. 조용히 틀린 각도로 도는 것보다 낫습니다.
# =============================================================================

def smooth_mask(kp: np.ndarray) -> np.ndarray:
    """각도 측정 전용 실루엣. 저장되지 않으므로 크게 뭉개도 됩니다."""
    ys = np.nonzero(kp.sum(1))[0]

    if ys.size < 40:
        return kp

    h = max(1, int(ys.max() - ys.min()))

    def kernel(fraction):
        k = max(3, int(fraction * h) | 1)
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    m = cv2.morphologyEx(kp, cv2.MORPH_CLOSE, kernel(SM_CLOSE))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel(SM_OPEN))

    if m.sum() < 0.3 * kp.sum():
        m = kp.copy()

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return m

    c = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    sigma = SM_SIGMA * h

    if len(c) > 30 and sigma >= 1.0:
        r = int(3 * sigma)

        if r < len(c) // 2:
            g = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
            g /= g.sum()

            padded = np.vstack([c[-r:], c, c[:r]])
            smoothed = np.stack(
                [np.convolve(padded[:, i], g, mode="valid") for i in (0, 1)],
                axis=1,
            )

            out = np.zeros_like(m)
            cv2.fillPoly(out, [smoothed.astype(np.int32)], 1)

            if out.sum() > 0.3 * kp.sum():
                return out

    return m


def widest_run(row: np.ndarray):
    idx = np.nonzero(row)[0]

    if idx.size == 0:
        return None

    breaks = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    b = max(breaks, key=len)

    return int(b[0]), int(b[-1])


def est_bottom_edge(kp: np.ndarray):
    """E1: 하단 윤곽의 최장 직선분."""
    ys = np.nonzero(kp.sum(1))[0]

    if ys.size < 40:
        return None

    y0, y1 = int(ys.min()), int(ys.max())
    h = max(1, y1 - y0)

    contours, _ = cv2.findContours(kp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return None

    c = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)

    if len(c) < 20:
        return None

    cf = c.reshape(-1, 1, 2).astype(np.float32)
    ap = cv2.approxPolyDP(cf, E1_EPS * cv2.arcLength(cf, True), True)
    ap = ap.reshape(-1, 2).astype(np.float64)

    if len(ap) < 3:
        return None

    ylim = y1 - E1_BAND * h
    best = None

    for i in range(len(ap)):
        p, q = ap[i], ap[(i + 1) % len(ap)]

        if p[1] < ylim or q[1] < ylim:
            continue

        d = q - p
        length = float(np.hypot(d[0], d[1]))

        if length < E1_MINLEN * h:
            continue

        angle = float((np.degrees(np.arctan2(d[1], d[0])) + 90) % 180 - 90)

        if abs(angle) > E1_MAXTILT:
            continue

        if best is None or length > best[0]:
            best = (length, p.copy(), q.copy(), angle)

    if best is None:
        return None

    length, p, q, angle = best

    d = (q - p) / max(length, 1e-9)
    normal = np.array([-d[1], d[0]])

    t = (c - p) @ d
    selected = (
        (np.abs((c - p) @ normal) < E1_TOL)
        & (t > -0.15 * length)
        & (t < 1.15 * length)
    )

    P = c[selected]
    n_points = int(len(P))

    if n_points >= 20:
        Q = P - P.mean(0)
        _, v = np.linalg.eigh(np.cov(Q.T))
        w = v[:, -1]

        angle2 = float((np.degrees(np.arctan2(w[1], w[0])) + 90) % 180 - 90)

        if abs(angle2 - angle) < 15.0:
            angle = angle2
            d = np.array([
                np.cos(np.deg2rad(angle)),
                np.sin(np.deg2rad(angle)),
            ])

        tt = (P - p) @ d
        p = p + d * float(tt.min())
        q = p + d * float(tt.max() - tt.min())

    return {
        "p": (float(p[0]), float(p[1])),
        "q": (float(q[0]), float(q[1])),
        "angle": angle,
        "n": n_points,
    }


def est_forearm_axis(kp: np.ndarray):
    """E2: 전완 중심선의 수직. 손목선은 전완축에 직교합니다."""
    ys = np.nonzero(kp.sum(1))[0]

    if ys.size < 60:
        return None

    y0, y1 = int(ys.min()), int(ys.max())
    h = max(1, y1 - y0)
    W = kp.shape[1]

    points = []

    for y in range(int(y0 + E2_LO * h), min(y1, int(y0 + E2_HI * h)) + 1):
        run = widest_run(kp[y])

        # 화면 밖으로 잘린 행은 중심이 왜곡되므로 제외합니다.
        if run is None or run[0] <= 1 or run[1] >= W - 2:
            continue

        points.append((y, 0.5 * (run[0] + run[1])))

    if len(points) < 20:
        return None

    P = np.array(points, dtype=np.float64)

    k = max(1, len(P) // 60)
    S = P[::k]

    slopes = [
        (S[j, 1] - S[i, 1]) / (S[j, 0] - S[i, 0])
        for i in range(len(S))
        for j in range(i + 1, len(S))
        if S[j, 0] != S[i, 0]
    ]

    if not slopes:
        return None

    tilt = float(np.degrees(np.arctan(np.median(slopes))))
    angle = -tilt

    cx = float(np.median(P[:, 1]))
    cy = float(y1 - 0.03 * h)

    d = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
    L = 0.30 * h

    return {
        "p": (cx - d[0] * L, cy - d[1] * L),
        "q": (cx + d[0] * L, cy + d[1] * L),
        "angle": angle,
        "n": int(len(P)),
    }


def measure_wrist_angle(kp_raw: np.ndarray):
    """두 추정기의 합의 각도. 불일치하면 (None, info) 를 반환합니다."""
    sm = smooth_mask(kp_raw)

    e1 = est_bottom_edge(sm)
    e2 = est_forearm_axis(sm)

    info = {
        "e1": e1,
        "e2": e2,
        "a1": None if e1 is None else e1["angle"],
        "a2": None if e2 is None else e2["angle"],
        "diff": None,
        "conf": "NONE",
    }

    if e1 is None and e2 is None:
        info["conf"] = "NO_ESTIMATE"
        return None, info

    if e1 is None or e2 is None:
        info["conf"] = "SINGLE"
        return None, info

    diff = abs(e1["angle"] - e2["angle"])
    info["diff"] = float(diff)

    if diff > AGREE_TOL:
        info["conf"] = "LOW"
        return None, info

    info["conf"] = "HIGH"
    angle = 0.5 * (e1["angle"] + e2["angle"])

    if abs(angle) > ANGLE_MAX:
        info["conf"] = "ANGLE_REJECT"
        return None, info

    return float(angle), info


def finger_up(kp: np.ndarray):
    """손끝이 위쪽인지. 수직 단면의 연결 조각 수로 판정합니다."""
    ys, _ = np.nonzero(kp)

    if ys.size == 0:
        return None

    y0, y1 = int(ys.min()), int(ys.max())
    h = max(1, y1 - y0)

    def runs(band):
        if band.size == 0:
            return 0.0
        d = np.diff(
            np.pad((band > 0).astype(np.int8), ((0, 0), (1, 1))),
            axis=1,
        )
        return float((d == 1).sum(1).mean())

    top = runs(kp[y0:y0 + int(0.25 * h)])
    bottom = runs(kp[max(y0, y1 - int(0.25 * h)):y1])

    if abs(top - bottom) < 0.35:
        return None

    return bool(top > bottom)


def rot_pair(image: np.ndarray, kp: np.ndarray, degrees: float, expand=True):
    """원본 이미지와 마스크를 같은 각도로 회전. 양수 = 반시계."""
    h, w = kp.shape

    if expand:
        r = np.deg2rad(abs(degrees))
        new_w = int(h * np.sin(r) + w * np.cos(r))
        new_h = int(h * np.cos(r) + w * np.sin(r))
    else:
        new_w, new_h = w, h

    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degrees, 1.0)
    M[0, 2] += new_w / 2.0 - w / 2.0
    M[1, 2] += new_h / 2.0 - h / 2.0

    rotated_image = cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    rotated_mask = cv2.warpAffine(
        kp * 255, M, (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return rotated_image, (rotated_mask > 127).astype(np.uint8)


def wrist_align(image: np.ndarray, kp: np.ndarray):
    """180도 뒤집기 + 합의 각도로 회전. 회전은 항상 원본 화소에 적용합니다."""
    flipped = False

    if finger_up(kp) is False:
        image, kp = rot_pair(image, kp, 180.0, expand=False)
        flipped = True

    angle, info = measure_wrist_angle(kp)

    if angle is None:
        return image, kp, info, 0.0, flipped, info["conf"]

    total = 0.0

    for _ in range(max(1, ROT_ITERS)):
        if abs(angle) < 0.15 or abs(total + angle) > ANGLE_MAX:
            break

        image, kp = rot_pair(image, kp, angle, expand=True)
        total += angle

        angle, info = measure_wrist_angle(kp)

        if angle is None:
            break

    return image, kp, info, total, flipped, "OK"


def apply_mask_and_crop(image: np.ndarray, kp: np.ndarray):
    """마스크 바깥 제거 후 마스크 경계로 재크롭."""
    if MASK_OUT_BACKGROUND:
        m = kp

        if MASK_FEATHER > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * MASK_FEATHER + 1, 2 * MASK_FEATHER + 1),
            )
            eroded = cv2.erode(kp, k)

            if eroded.sum() >= 0.5 * kp.sum():
                m = eroded

        image = np.where(m > 0, image, 0).astype(np.uint8)
        kp = m

    ys, xs = np.nonzero(kp)

    if ys.size < 100:
        return None, None

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    pad = int(WRIST_MARGIN_FRAC * min(y1 - y0 + 1, x1 - x0 + 1))

    hh, ww = kp.shape
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    x1 = min(ww - 1, x1 + pad)
    y1 = min(hh - 1, y1 + pad)

    if y1 - y0 < MIN_SIDE or x1 - x0 < MIN_SIDE:
        return None, None

    return (
        image[y0:y1 + 1, x0:x1 + 1],
        kp[y0:y1 + 1, x0:x1 + 1],
    )


# =============================================================================
# [E] 최종 캔버스 - 학습 스크립트의 fit_canvas 와 동일
# =============================================================================

def to_gray8(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return array

    arr = array.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())

    if hi - lo < 1e-6:
        return np.zeros(arr.shape, np.uint8)

    return np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def normalize_intensity(array: np.ndarray, mode: str) -> np.ndarray:
    """이미지별 퍼센타일 스트레치.
       0 배경은 통계에서 제외하며 반드시 padding 이전에 적용합니다."""
    if mode == "none":
        return array

    values = array[array > 0]

    if values.size < 1000:
        return array

    if mode == "p1p99":
        lo, hi = np.percentile(values, (1, 99))
    elif mode == "p2p98":
        lo, hi = np.percentile(values, (2, 98))
    else:
        raise ValueError(f"지원하지 않는 norm mode: {mode}")

    if hi - lo < 1e-3:
        return array

    return np.clip(
        (array.astype(np.float32) - lo) / (hi - lo) * 255.0, 0, 255
    ).astype(np.uint8)


def fit_canvas(
    gray: np.ndarray,
    *,
    height_out: int,
    width_out: int,
    resize_mode: str,
    pad_value: int,
    pad_anchor: str,
    norm_mode: str,
) -> np.ndarray:
    arr = to_gray8(gray)
    arr = normalize_intensity(arr, norm_mode)

    h, w = arr.shape[:2]

    if h < 1 or w < 1:
        raise ValueError("빈 이미지입니다.")

    if resize_mode == "stretch":
        interp = (
            cv2.INTER_AREA
            if (height_out < h and width_out < w)
            else cv2.INTER_CUBIC
        )
        return cv2.resize(arr, (width_out, height_out), interpolation=interp)

    scale = min(height_out / float(h), width_out / float(w))

    new_h = max(1, min(height_out, int(round(h * scale))))
    new_w = max(1, min(width_out, int(round(w * scale))))

    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(arr, (new_w, new_h), interpolation=interp)

    top = 0 if pad_anchor == "topleft" else (height_out - new_h) // 2
    left = 0 if pad_anchor == "topleft" else (width_out - new_w) // 2

    return cv2.copyMakeBorder(
        resized,
        top, height_out - new_h - top,
        left, width_out - new_w - left,
        cv2.BORDER_CONSTANT,
        value=int(pad_value),
    )


# =============================================================================
# [F] 전처리 파이프라인 조립
# =============================================================================

def preprocess_image(
    *,
    detector,
    seg_model,
    image_path: Path,
    device: torch.device,
    canvas_config: Dict,
    qc_dir: Optional[Path],
    image_id: str,
) -> Tuple[np.ndarray, str]:
    """반환 (canvas, stage)

    stage:
        full        전 단계 정상 통과
        no_wrist    손목선 합의 실패 - 1차 정렬까지만 적용
        seg_failed  분할/정렬 실패 - 검출 박스 crop 으로 대체
        no_detect   손 미검출 - 원본 전체 사용
    """
    original = imread_unicode(image_path, cv2.IMREAD_UNCHANGED)

    if original is None:
        raise RuntimeError(f"이미지를 읽지 못했습니다: {image_path}")

    h, w = original.shape[:2]
    detector_input = to_detector_bgr8(original)

    box = detect_hand_box(
        detector=detector,
        detector_input=detector_input,
        image_height=h,
        image_width=w,
        device=device,
    )

    if box is None:
        return fit_canvas(to_gray_native(original), **canvas_config), "no_detect"

    x0, y0, x1, y1 = box

    # 검출 실패 시 되돌아갈 crop 을 미리 준비합니다.
    fx0, fy0, fx1, fy1 = expand_box(
        (x0, y0, x1, y1), image_width=w, image_height=h
    )
    fallback_crop = to_gray_native(original)[fy0:fy1, fx0:fx1].copy()

    try:
        # 분할 입력은 손끝이 잘리지 않도록 더 넓게 자릅니다.
        bw = max(1.0, x1 - x0)
        bh = max(1.0, y1 - y0)

        sx0 = int(max(0, math.floor(x0 - bw * SEG_MARGIN_X)))
        sx1 = int(min(w, math.ceil(x1 + bw * SEG_MARGIN_X)))
        sy0 = int(max(0, math.floor(y0 - bh * SEG_MARGIN_TOP)))
        sy1 = int(min(h, math.ceil(y1 + bh * SEG_MARGIN_BOTTOM)))

        if sx1 <= sx0 or sy1 <= sy0:
            raise ValueError("segmentation crop box 오류")

        seg_crop_bgr = detector_input[sy0:sy1, sx0:sx1].copy()
        native_crop = original[sy0:sy1, sx0:sx1].copy()

        mask = clean_hand_mask(
            predict_seg_mask(seg_crop_bgr, seg_model, device)
        )

        if int(mask.sum()) < 100:
            raise ValueError("segmentation mask too small")

        if qc_dir is not None:
            imwrite_unicode(qc_dir / f"{image_id}_mask.png", mask * 255)

        # ── 1차 정렬 (PCA + 손가락/손목 잔여각) ──────────────────────
        total_angle = get_total_rotation(mask)

        if total_angle is None:
            raise ValueError("orientation estimation failed")

        rotated_image, rotated_mask = rotate_native_pair_once(
            native_crop, mask, total_angle
        )

        cropped = crop_by_rotated_mask(rotated_image, rotated_mask)

        if cropped is None:
            raise ValueError("rotated-mask crop failed")

        stage_image, stage_mask = cropped
        gray = to_gray8(to_gray_native(stage_image))

        stage = "full"

        if USE_WRIST_ALIGN:
            # ── 2차 정렬 (손목선 검출) ──────────────────────────────
            aligned, aligned_mask, info, total, flipped, status = wrist_align(
                gray, stage_mask
            )

            if status != "OK":
                stage = "no_wrist"

            result = apply_mask_and_crop(aligned, aligned_mask)

            if result[0] is None:
                raise ValueError("wrist-align crop failed")

            gray = result[0]

            if qc_dir is not None:
                imwrite_unicode(qc_dir / f"{image_id}_aligned.png", gray)

        elif MASK_OUT_BACKGROUND:
            result = apply_mask_and_crop(gray, stage_mask)

            if result[0] is None:
                raise ValueError("mask crop failed")

            gray = result[0]

        return fit_canvas(gray, **canvas_config), stage

    except Exception:
        return fit_canvas(fallback_crop, **canvas_config), "seg_failed"


# =============================================================================
# [G] 뼈나이 모델 (LDL + FiLM)
# =============================================================================

def restore_float32(state: Dict) -> Dict:
    """용량 절감을 위해 fp16 으로 저장된 가중치를 fp32 로 되돌립니다.
       fp32 로 저장된 파일에는 아무 영향이 없습니다."""
    restored = {}

    for key, value in state.items():
        if torch.is_tensor(value) and value.dtype == torch.float16:
            restored[key] = value.float()
        else:
            restored[key] = value

    return restored


def create_backbone(model_name: str, drop_path: float = 0.0):
    if "convnextv2" in model_name:
        raise ValueError(
            f"{model_name}는 CC BY-NC 4.0 가중치입니다 (상업 사용 불가)."
        )

    return timm.create_model(
        model_name,
        pretrained=False,
        num_classes=0,
        global_pool="",
        drop_path_rate=drop_path,
    )


class GenderFiLM(nn.Module):
    """성별 스칼라 -> 채널별 (gamma, beta) 를 생성해 특징맵을 변조.

        feat <- feat * (1 + gamma) + beta
    """

    def __init__(self, n_ch: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_ch * 2),
        )

    def forward(self, feat, g):
        gamma, beta = self.net(g).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return feat * (1.0 + gamma) + beta


class ConvNeXtRegressor(nn.Module):
    """use_ldl=True  -> 240 로짓 (softmax 기대값으로 개월 환산)
       use_ldl=False -> 1 스칼라 (z-역정규화로 개월 환산)"""

    def __init__(
        self,
        *,
        backbone_name: str,
        img_hw: Tuple[int, int],
        head_type: str = "gap",
        head_dim: int = 512,
        gender_dim: int = 32,
        dropout: float = 0.10,
        drop_path: float = 0.15,
        use_ldl: bool = True,
        n_bins: int = DEFAULT_AGE_BINS,
        use_film: bool = True,
        film_stages: Iterable[int] = (2, 3),
        film_hidden: int = 64,
    ):
        super().__init__()

        self.backbone = create_backbone(backbone_name, drop_path)
        self.head_type = head_type

        # ── FiLM 성별 조건화 ────────────────────────────────────────
        self.use_film = bool(use_film) and hasattr(self.backbone, "stages")
        self.film = nn.ModuleDict()

        if self.use_film:
            with torch.no_grad():
                z = self.backbone.stem(torch.zeros(1, 3, 64, 64))
                channels = []

                for stage in self.backbone.stages:
                    z = stage(z)
                    channels.append(int(z.shape[1]))

            for i in film_stages:
                if 0 <= i < len(channels):
                    self.film[str(i)] = GenderFiLM(
                        channels[i], hidden=film_hidden
                    )

        height, width = int(img_hw[0]), int(img_hw[1])

        with torch.no_grad():
            feature = self._backbone_forward(
                torch.zeros(1, 3, height, width),
                torch.zeros(1, 1),
            )

        C = int(feature.shape[1])
        feat_h = int(feature.shape[2])
        feat_w = int(feature.shape[3])

        if head_type == "gap":
            self.norm = nn.LayerNorm(C)
            self.proj = nn.Sequential(
                nn.Linear(C, head_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            image_out = head_dim

        elif head_type == "paper":
            self.conv = nn.Conv2d(C, 256, 3, padding=1)
            self.pool = nn.MaxPool2d(3, 3)
            self.drop = nn.Dropout(dropout)
            image_out = 256 * (feat_h // 3) * (feat_w // 3)

        else:
            raise ValueError(f"지원하지 않는 head_type: {head_type}")

        self.use_ldl = bool(use_ldl)
        self.n_bins = int(n_bins)
        n_out = self.n_bins if self.use_ldl else 1

        self.gender = nn.Sequential(nn.Linear(1, gender_dim), nn.GELU())

        self.fc = nn.Sequential(
            nn.Linear(image_out + gender_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_out),
        )

    def _backbone_forward(self, x, g):
        if not self.use_film:
            return self.backbone(x)

        z = self.backbone.stem(x)

        for i, stage in enumerate(self.backbone.stages):
            z = stage(z)
            key = str(i)

            if key in self.film:
                z = self.film[key](z, g)

        z = self.backbone.norm_pre(z)

        # timm ConvNeXt 의 backbone(x) 는 forward_head 까지 포함합니다.
        # 이 단계를 빠뜨리면 FiLM ON/OFF 가 다른 특징 공간을 쓰게 됩니다.
        if hasattr(self.backbone, "forward_head"):
            z = self.backbone.forward_head(z)
        elif hasattr(self.backbone, "head"):
            z = self.backbone.head(z)

        return z

    def forward(self, x, g):
        f = self._backbone_forward(x, g)

        if self.head_type == "gap":
            z = F.adaptive_avg_pool2d(f, 1).flatten(1)
            z = self.proj(self.norm(z))
        else:
            z = self.drop(torch.flatten(self.pool(F.relu(self.conv(f))), 1))

        e = self.gender(g)
        out = self.fc(torch.cat([z, e], dim=1))

        return out if self.use_ldl else out.squeeze(1)


_KBINS_CACHE = {}


def kbins(n: int, device):
    """연령 인덱스 벡터 k = [1, 2, ..., n]."""
    key = (int(n), str(device))

    if key not in _KBINS_CACHE:
        _KBINS_CACHE[key] = torch.arange(
            1, int(n) + 1, dtype=torch.float32, device=device
        )

    return _KBINS_CACHE[key]


def out_to_months(out, age_mean: float, age_std: float):
    """모델 출력을 개월로 바꾸는 유일한 경로.

       LDL   : (B, K) 로짓 -> softmax -> 기대값 Σ k·p_k
       스칼라 : (B,) -> pred * std + mean

       분기 기준은 마지막 차원 크기입니다."""
    out = out.float()

    if out.ndim == 2 and out.size(1) > 1:
        p = torch.softmax(out, dim=1)
        return (p * kbins(out.size(1), out.device)).sum(1)

    if out.ndim == 2:
        out = out.squeeze(1)

    return out * age_std + age_mean


def torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_boneage_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch_load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise KeyError(
            f"bone-age checkpoint 에 model state 가 없습니다: {checkpoint_path}"
        )

    arch = dict(checkpoint.get("arch", {}))
    arch.setdefault("IMG_H", DEFAULT_IMG_H)
    arch.setdefault("IMG_W", DEFAULT_IMG_W)
    arch.setdefault("NORM_MODE", DEFAULT_NORM_MODE)
    arch.setdefault("RESIZE_MODE", DEFAULT_RESIZE_MODE)
    arch.setdefault("PAD_VALUE", DEFAULT_PAD_VALUE)
    arch.setdefault("PAD_ANCHOR", DEFAULT_PAD_ANCHOR)

    backbone_name = (
        arch.get("BACKBONE_RESOLVED")
        or arch.get("BACKBONE")
        or DEFAULT_BACKBONE
    )

    model = ConvNeXtRegressor(
        backbone_name=backbone_name,
        img_hw=(int(arch["IMG_H"]), int(arch["IMG_W"])),
        head_type=arch.get("HEAD_TYPE", "gap"),
        head_dim=int(arch.get("HEAD_DIM", 512)),
        gender_dim=int(arch.get("GENDER_EMB_DIM", 32)),
        dropout=float(arch.get("DROPOUT", 0.10)),
        drop_path=float(arch.get("DROP_PATH", 0.15)),
        use_ldl=bool(arch.get("USE_LDL", False)),
        n_bins=int(arch.get("AGE_BINS", DEFAULT_AGE_BINS)),
        use_film=bool(arch.get("USE_FILM", False)),
        film_stages=tuple(arch.get("FILM_STAGES", (2, 3))),
        film_hidden=int(arch.get("FILM_HIDDEN", 64)),
    )

    model.load_state_dict(restore_float32(checkpoint["model"]), strict=True)
    model.eval()
    model.to(device)
    model.to(memory_format=torch.channels_last)

    age_mean = float(checkpoint.get("age_mean", 0.0))
    age_std = float(checkpoint.get("age_std", 1.0))

    if age_std <= 0:
        raise ValueError("checkpoint 의 age_std 값이 올바르지 않습니다.")

    info = {
        "backbone": backbone_name,
        "epoch": checkpoint.get("epoch"),
        "val_mae": checkpoint.get("val_mae"),
        "best_from": checkpoint.get("best_from"),
        "use_ldl": bool(arch.get("USE_LDL", False)),
        "age_bins": int(arch.get("AGE_BINS", DEFAULT_AGE_BINS)),
        "use_film": bool(arch.get("USE_FILM", False)),
    }

    return model, arch, age_mean, age_std, info


def load_calibration(path: Path):
    """반환 (calibration_dict, tta_angles).

    지원:
    - 기존 global affine:
        {"used": true, "a": ..., "b": ...}
    - sex-specific affine:
        {"used": true, "type": "sex_affine",
         "female": {"a": ..., "b": ...},
         "male": {"a": ..., "b": ...}}
    - sex-specific median shift:
        {"used": true, "type": "sex_median_shift",
         "female": {"shift": ...},
         "male": {"shift": ...}}
    """
    if not path.is_file():
        return None, DEFAULT_TTA_ANGLES

    try:
        payload = json.load(open(path, encoding="utf-8"))
    except Exception:
        return None, DEFAULT_TTA_ANGLES

    angles = payload.get("tta_angles") or DEFAULT_TTA_ANGLES
    angles = tuple(int(a) for a in angles)

    if not bool(payload.get("used", False)):
        return None, angles

    cal_type = str(payload.get("type", "global_affine"))

    if cal_type == "sex_affine":
        for sex_name in ("female", "male"):
            cfg = payload.get(sex_name, {})
            if abs(float(cfg.get("a", 0.0))) < 1e-6:
                return None, angles
        return payload, angles

    if cal_type == "sex_median_shift":
        for sex_name in ("female", "male"):
            if "shift" not in payload.get(sex_name, {}):
                return None, angles
        return payload, angles

    slope = float(payload.get("a", 1.0))
    intercept = float(payload.get("b", 0.0))

    if abs(slope) < 1e-6:
        return None, angles

    return {
        "used": True,
        "type": "global_affine",
        "a": slope,
        "b": intercept,
    }, angles


def apply_calibration(months: float, male: float, calibration):
    if calibration is None:
        return float(months)

    cal_type = str(
        calibration.get("type", "global_affine")
    )

    if cal_type == "sex_affine":
        cfg = (
            calibration["male"]
            if float(male) >= 0.5
            else calibration["female"]
        )
        return (
            float(months) - float(cfg["b"])
        ) / float(cfg["a"])

    if cal_type == "sex_median_shift":
        cfg = (
            calibration["male"]
            if float(male) >= 0.5
            else calibration["female"]
        )
        return float(months) - float(cfg["shift"])

    return (
        float(months) - float(calibration["b"])
    ) / float(calibration["a"])


def calibration_description(calibration):
    if calibration is None:
        return "미적용"

    cal_type = str(
        calibration.get("type", "global_affine")
    )

    if cal_type == "sex_affine":
        f = calibration["female"]
        m = calibration["male"]
        return (
            "sex-affine "
            f"F(a={float(f['a']):.4f},b={float(f['b']):+.2f}) "
            f"M(a={float(m['a']):.4f},b={float(m['b']):+.2f})"
        )

    if cal_type == "sex_median_shift":
        f = calibration["female"]
        m = calibration["male"]
        return (
            "sex-shift "
            f"F={float(f['shift']):+.2f} "
            f"M={float(m['shift']):+.2f}"
        )

    return (
        f"a={float(calibration['a']):.4f} "
        f"b={float(calibration['b']):+.2f}"
    )


# =============================================================================
# [H] 메타데이터
# =============================================================================

def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def normalize_column_name(name: str) -> str:
    return (
        str(name).strip().lower()
        .replace(" ", "").replace("_", "").replace("-", "")
    )


def find_column(dataframe: pd.DataFrame, aliases: Iterable[str]):
    normalized = {
        normalize_column_name(c): c for c in dataframe.columns
    }

    for alias in aliases:
        key = normalize_column_name(alias)
        if key in normalized:
            return normalized[key]

    return None


def parse_male(value) -> float:
    if pd.isna(value):
        raise ValueError("sex 값이 비어 있습니다.")

    text = str(value).strip().lower()

    if text in {"m", "male", "man", "boy", "남", "남자", "1", "1.0", "true"}:
        return 1.0

    if text in {"f", "female", "woman", "girl", "여", "여자", "0", "0.0", "false"}:
        return 0.0

    raise ValueError(f"인식할 수 없는 sex 값: {value!r}")


def build_image_index(images_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}

    for path in sorted(images_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem in index:
                raise RuntimeError(
                    "동일한 파일 stem 이 여러 개 있습니다: "
                    f"{path.stem}\n{index[path.stem]}\n{path}"
                )
            index[path.stem] = path

    if not index:
        raise FileNotFoundError(f"지원되는 이미지가 없습니다: {images_dir}")

    return index


def prepare_metadata(metadata_csv: Path, images_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv, dtype=str)

    if len(df) == 0:
        raise ValueError("metadata CSV 가 비어 있습니다.")

    id_col = find_column(
        df,
        ["id", "image_id", "imageid", "patient_id", "patientid",
         "case_id", "caseid"],
    )

    sex_col = find_column(df, ["male", "sex", "gender"])

    filename_col = find_column(
        df,
        ["filename", "file", "image", "imagefile", "imagepath", "path"],
    )

    label_col = find_column(
        df,
        ["boneage", "bone_age", "boneagemonths", "bone_age_months",
         "label", "target", "gt", "groundtruth", "true_age", "actual_age",
         "정답", "실제뼈나이"],
    )

    if id_col is None:
        raise ValueError("metadata CSV 에 id 열이 필요합니다.")

    if sex_col is None:
        raise ValueError(
            "metadata CSV 에 sex/gender/male 열이 필요합니다. "
            "모델은 성별 입력을 사용합니다."
        )

    image_index = build_image_index(images_dir)

    rows = []

    for _, row in df.iterrows():
        image_id = str(row[id_col]).strip()

        if not image_id or image_id.lower() == "nan":
            raise ValueError("빈 id 가 있습니다.")

        male = parse_male(row[sex_col])

        image_path = None

        if filename_col is not None and not pd.isna(row[filename_col]):
            raw_name = str(row[filename_col]).strip()
            candidate = Path(raw_name)

            if not candidate.is_absolute():
                candidate = images_dir / candidate

            image_path = (
                candidate if candidate.is_file()
                else image_index.get(Path(raw_name).stem)
            )

        if image_path is None:
            image_path = image_index.get(Path(image_id).stem)

        if image_path is None:
            raise FileNotFoundError(
                f"id={image_id} 에 해당하는 이미지를 찾지 못했습니다."
            )

        true_age = np.nan

        if label_col is not None and not pd.isna(row[label_col]):
            raw_label = str(row[label_col]).strip()

            if raw_label and raw_label.lower() != "nan":
                try:
                    true_age = float(raw_label)
                except ValueError:
                    true_age = np.nan

        rows.append({
            "id": image_id,
            "filename": image_path.name,
            "image_path": str(image_path.resolve()),
            "male": male,
            "sex": "M" if male >= 0.5 else "F",
            "true_age": true_age,
        })

    return pd.DataFrame(rows)


# =============================================================================
# [I] 성능 지표
# =============================================================================

def bootstrap_mae_ci(errors, *, iterations, confidence, seed):
    if errors.size < 2 or iterations < 1:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, errors.size, size=(iterations, errors.size))
    means = errors[indices].mean(axis=1)

    lower_q = (1.0 - confidence) / 2.0 * 100.0
    upper_q = (1.0 + confidence) / 2.0 * 100.0

    lower, upper = np.percentile(means, (lower_q, upper_q))

    return float(lower), float(upper)


def compute_group_metrics(group: pd.DataFrame, label: str) -> Dict:
    errors = group["abs_error"].to_numpy(dtype=np.float64)
    signed = (
        group["predicted_age"].to_numpy(dtype=np.float64)
        - group["true_age"].to_numpy(dtype=np.float64)
    )

    if errors.size == 0:
        return {"group": label, "n": 0}

    ci_low, ci_high = bootstrap_mae_ci(
        errors,
        iterations=BOOTSTRAP_ITERATIONS,
        confidence=BOOTSTRAP_CI,
        seed=BOOTSTRAP_SEED,
    )

    return {
        "group": label,
        "n": int(errors.size),
        "mae": float(errors.mean()),
        "mae_ci_low": ci_low,
        "mae_ci_high": ci_high,
        "rmse": float(np.sqrt((signed ** 2).mean())),
        "bias": float(signed.mean()),
        "median_ae": float(np.median(errors)),
        "within_6m": float((errors <= 6.0).mean() * 100.0),
        "within_12m": float((errors <= 12.0).mean() * 100.0),
        "max_ae": float(errors.max()),
    }


def evaluate_predictions(result_df, *, metrics_csv, metrics_json, context):
    scored = result_df.dropna(subset=["true_age", "predicted_age"]).copy()

    if len(scored) == 0:
        return None

    scored["abs_error"] = (
        scored["predicted_age"] - scored["true_age"]
    ).abs()

    rows = [compute_group_metrics(scored, "overall")]

    for sex_label in ["M", "F"]:
        subset = scored[scored["sex"] == sex_label]

        if len(subset) > 0:
            rows.append(
                compute_group_metrics(
                    subset, "male" if sex_label == "M" else "female"
                )
            )

    # 전처리가 완주한 건만 따로 집계 - fallback 이 지표를 흐리지 않도록
    clean = scored[scored["stage"] == "full"]

    if 0 < len(clean) < len(scored):
        rows.append(compute_group_metrics(clean, "stage_full_only"))

    metrics_df = pd.DataFrame(rows)

    for column in metrics_df.columns:
        if column not in {"group", "n"}:
            metrics_df[column] = metrics_df[column].round(4)

    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")

    worst = (
        scored.sort_values("abs_error", ascending=False)
        .head(WORST_CASE_COUNT)
        [["id", "filename", "sex", "true_age", "predicted_age",
          "abs_error", "stage"]]
    )

    payload = dict(context)
    payload["metrics"] = rows
    payload["worst_cases"] = [
        {
            "id": r["id"],
            "filename": r["filename"],
            "sex": r["sex"],
            "true_age": round(float(r["true_age"]), 4),
            "predicted_age": round(float(r["predicted_age"]), 4),
            "abs_error": round(float(r["abs_error"]), 4),
            "stage": r["stage"],
        }
        for _, r in worst.iterrows()
    ]

    json.dump(
        payload,
        open(metrics_json, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )

    return rows[0]


# =============================================================================
# [J] 추론
# =============================================================================

def format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remain = divmod(seconds, 3600)
    minutes, seconds = divmod(remain, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@torch.no_grad()
def predict_batch(
    *,
    model,
    canvases,
    males,
    device,
    age_mean,
    age_std,
    tta_angles,
    use_amp,
):
    images = torch.stack(
        [EVAL_TRANSFORM(np.stack([c] * 3, axis=-1)) for c in canvases],
        dim=0,
    ).to(device, non_blocking=True).to(memory_format=torch.channels_last)

    male_tensor = torch.tensor(
        males, dtype=torch.float32, device=device
    ).reshape(-1, 1)

    angles = tta_angles if tta_angles else (0,)
    accumulated = 0.0

    with torch.amp.autocast("cuda", enabled=use_amp):
        for angle in angles:
            rotated = (
                images if angle == 0
                else TF.rotate(
                    images,
                    angle,
                    fill=PAD_NORM,
                    interpolation=TF.InterpolationMode.BILINEAR,
                )
            )
            # 개월 환산 후 평균 - LDL 은 로짓 평균이 아니라 기대값 평균이어야 합니다.
            accumulated = accumulated + out_to_months(
                model(rotated, male_tensor), age_mean, age_std
            )

    return (accumulated / len(angles)).detach().cpu().numpy()


def main():
    total_start = time.perf_counter()

    images_dir = IMAGES_DIR.expanduser().resolve()
    metadata_csv = METADATA_CSV.expanduser().resolve()
    yolox_dir = YOLOX_DIR.expanduser().resolve()
    yolox_exp = YOLOX_EXP.expanduser().resolve()
    yolox_model_path = YOLOX_MODEL.expanduser().resolve()
    seg_model_path = SEG_MODEL.expanduser().resolve()
    boneage_model_path = BONEAGE_MODEL.expanduser().resolve()
    calibration_path = CALIBRATION_JSON.expanduser().resolve()
    output_csv = OUTPUT_CSV.expanduser().resolve()

    input_mode = str(INPUT_MODE).strip().lower()

    if input_mode not in {"raw", "already_cropped"}:
        raise ValueError('INPUT_MODE 는 "raw" 또는 "already_cropped" 여야 합니다.')

    for path, label in [
        (images_dir, "Images 폴더"),
        (metadata_csv, "test.csv"),
        (boneage_model_path, "models/best_model.pt"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} 없음: {path}")

    if input_mode == "raw":
        for path, label in [
            (yolox_model_path, "models/yolox_s_hand_best.pth"),
            (seg_model_path, "models/hand_seg_crop512_traced.pt"),
            (yolox_exp, "yolox_s_hand.py"),
        ]:
            if not path.exists():
                raise FileNotFoundError(f"{label} 없음: {path}")

    if BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE 는 1 이상이어야 합니다.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    crop_dir = output_csv.parent / "crops_input"
    qc_dir = output_csv.parent / "qc"

    if SAVE_CROPS:
        crop_dir.mkdir(parents=True, exist_ok=True)

    if SAVE_QC:
        qc_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(DEVICE)
    use_amp = device.type == "cuda"

    metadata = prepare_metadata(metadata_csv, images_dir)
    total_cases = len(metadata)

    print(
        f"\r추론 준비 중... | 0/{total_cases} (0.0%)",
        end="",
        flush=True,
    )

    model, arch, age_mean, age_std, info = load_boneage_model(
        boneage_model_path, device
    )

    canvas_config = {
        "height_out": int(arch["IMG_H"]),
        "width_out": int(arch["IMG_W"]),
        "resize_mode": str(arch.get("RESIZE_MODE", DEFAULT_RESIZE_MODE)),
        "pad_value": int(arch.get("PAD_VALUE", DEFAULT_PAD_VALUE)),
        "pad_anchor": str(arch.get("PAD_ANCHOR", DEFAULT_PAD_ANCHOR)),
        "norm_mode": str(arch.get("NORM_MODE", DEFAULT_NORM_MODE)),
    }

    calibration, tta_angles = (None, DEFAULT_TTA_ANGLES)

    if USE_CALIBRATION:
        calibration, tta_angles = load_calibration(calibration_path)

    detector = None
    seg_model = None

    if input_mode == "raw":
        detector = load_yolox_detector(
            yolox_dir=yolox_dir,
            exp_path=yolox_exp,
            checkpoint_path=yolox_model_path,
            device=device,
        )
        seg_model = load_segmentation_model(seg_model_path, device)

    print("\r" + "=" * 70)
    print(f" 뼈나이 추론 | device {device} | batch {BATCH_SIZE}")
    print(f" 이미지   {images_dir}  ({total_cases:,}장)")
    print(f" 모드     {input_mode}"
          f" | 손목정렬 {'ON' if USE_WRIST_ALIGN else 'OFF'}"
          f" | 배경제거 {'ON' if MASK_OUT_BACKGROUND else 'OFF'}")
    print(f" 백본     {info['backbone']}")
    print(f" 헤드     "
          + (f"LDL {info['age_bins']}bin" if info["use_ldl"] else "스칼라 회귀")
          + f" | FiLM {'ON' if info['use_film'] else 'OFF'}")
    print(f" 입력     {arch['IMG_H']}x{arch['IMG_W']}(HxW)"
          f" | {canvas_config['norm_mode']}"
          f" | {canvas_config['resize_mode']}")
    print(f" TTA      {list(tta_angles)}")
    print(
        " 보정     "
        + calibration_description(calibration)
    )
    print("=" * 70, flush=True)

    results: List[dict] = []
    pending_canvases: List[np.ndarray] = []
    pending_males: List[float] = []
    pending_indices: List[int] = []

    stage_counts: Dict[str, int] = {}
    error_count = 0

    def flush_pending():
        if not pending_canvases:
            return

        predictions = predict_batch(
            model=model,
            canvases=pending_canvases,
            males=pending_males,
            device=device,
            age_mean=age_mean,
            age_std=age_std,
            tta_angles=tta_angles,
            use_amp=use_amp,
        )

        for index, prediction, male_value in zip(
            pending_indices,
            predictions,
            pending_males,
        ):
            months = apply_calibration(
                float(prediction),
                float(male_value),
                calibration,
            )

            results[index]["predicted_age"] = months

        pending_canvases.clear()
        pending_males.clear()
        pending_indices.clear()

    for row_index, row in metadata.iterrows():
        image_path = Path(row["image_path"])

        record = {
            "id": row["id"],
            "filename": row["filename"],
            "sex": row["sex"],
            "predicted_age": np.nan,
            "true_age": float(row["true_age"]),
            "stage": "",
        }

        results.append(record)

        try:
            if input_mode == "raw":
                canvas, stage = preprocess_image(
                    detector=detector,
                    seg_model=seg_model,
                    image_path=image_path,
                    device=device,
                    canvas_config=canvas_config,
                    qc_dir=qc_dir if SAVE_QC else None,
                    image_id=str(row["id"]),
                )
            else:
                gray = imread_unicode(image_path, cv2.IMREAD_UNCHANGED)

                if gray is None:
                    raise RuntimeError("이미지를 읽지 못했습니다.")

                canvas = fit_canvas(to_gray_native(gray), **canvas_config)
                stage = "skipped"

            record["stage"] = stage
            stage_counts[stage] = stage_counts.get(stage, 0) + 1

            if SAVE_CROPS:
                imwrite_unicode(crop_dir / f"{row['id']}.png", canvas)

            pending_canvases.append(canvas)
            pending_males.append(float(row["male"]))
            pending_indices.append(len(results) - 1)

            if len(pending_canvases) >= BATCH_SIZE:
                flush_pending()

        except Exception as exc:
            error_count += 1
            record["stage"] = "error"
            record["error"] = str(exc)[:200]

        done = row_index + 1

        print(
            f"\r추론 진행: {done}/{total_cases} "
            f"({done / total_cases * 100:.1f}%) "
            f"| 경과 {format_elapsed(time.perf_counter() - total_start)}",
            end="",
            flush=True,
        )

    flush_pending()

    result_df = pd.DataFrame(results)

    has_labels = bool(
        EVALUATE_IF_LABELS and result_df["true_age"].notna().any()
    )

    result_df["predicted_age"] = result_df["predicted_age"].round(6)

    if has_labels:
        result_df["abs_error"] = (
            result_df["predicted_age"] - result_df["true_age"]
        ).abs().round(6)

        columns = ["id", "filename", "sex", "predicted_age",
                   "true_age", "abs_error", "stage"]
    else:
        columns = ["id", "filename", "sex", "predicted_age", "stage"]

    if "error" in result_df.columns:
        columns.append("error")

    result_df[columns].to_csv(output_csv, index=False, encoding="utf-8-sig")

    total_elapsed = time.perf_counter() - total_start
    predicted_count = int(result_df["predicted_age"].notna().sum())

    print(
        f"\r추론 완료: {predicted_count}/{total_cases} "
        f"| 총 소요시간 {format_elapsed(total_elapsed)}" + " " * 20
    )
    print(f"결과 저장: {output_csv.name}")

    stage_labels = {
        "full": "전 단계 정상",
        "no_wrist": "손목선 합의 실패 (1차 정렬까지)",
        "seg_failed": "분할/정렬 실패 (검출 crop 사용)",
        "no_detect": "손 미검출 (원본 전체 사용)",
        "skipped": "전처리 생략",
    }

    print("\n[전처리 단계]")

    for stage, count in sorted(
        stage_counts.items(), key=lambda kv: -kv[1]
    ):
        label = stage_labels.get(stage, stage)
        ratio = count / total_cases * 100
        print(f"  {label:<30} {count:>5,}건 ({ratio:5.1f}%)")

    if error_count:
        print(f"  {'처리 오류':<30} {error_count:>5,}건")

    if SAVE_CROPS:
        print(f"\n모델 입력 캔버스: {crop_dir}")

    if SAVE_QC:
        print(f"중간 QC 이미지: {qc_dir}")

    if not has_labels:
        return

    context = {
        "boneage_model": boneage_model_path.name,
        "backbone": info["backbone"],
        "head": ("ldl_%d" % info["age_bins"]) if info["use_ldl"] else "scalar",
        "use_film": info["use_film"],
        "input_size": f"{arch['IMG_H']}x{arch['IMG_W']}",
        "norm_mode": canvas_config["norm_mode"],
        "wrist_align": USE_WRIST_ALIGN,
        "mask_out": MASK_OUT_BACKGROUND,
        "tta_angles": list(tta_angles),
        "calibration": calibration,
        "input_mode": input_mode,
        "total_cases": int(total_cases),
        "stage_counts": stage_counts,
        "error_count": int(error_count),
        "bootstrap_iterations": int(BOOTSTRAP_ITERATIONS),
        "bootstrap_ci": float(BOOTSTRAP_CI),
    }

    overall = evaluate_predictions(
        result_df,
        metrics_csv=METRICS_CSV.expanduser().resolve(),
        metrics_json=METRICS_JSON.expanduser().resolve(),
        context=context,
    )

    if overall is None:
        return

    ci_percent = int(round(BOOTSTRAP_CI * 100))

    print("\n" + "-" * 58)
    print(f"성능 지표 (정답 {overall['n']}건 기준)")
    print(f"  MAE        {overall['mae']:.3f} months  "
          f"[{ci_percent}% CI {overall['mae_ci_low']:.3f} "
          f"~ {overall['mae_ci_high']:.3f}]")
    print(f"  RMSE       {overall['rmse']:.3f} months")
    print(f"  Bias       {overall['bias']:+.3f} months")
    print(f"  Median AE  {overall['median_ae']:.3f} months")
    print(f"  ±6개월 이내  {overall['within_6m']:.1f}%")
    print(f"  ±12개월 이내 {overall['within_12m']:.1f}%")
    print("-" * 58)
    print(f"지표 저장: {METRICS_CSV.name}, {METRICS_JSON.name}")


if __name__ == "__main__":
    main()