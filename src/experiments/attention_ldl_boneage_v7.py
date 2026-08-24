# -*- coding: utf-8 -*-
# =========================================================================
# 손 X-ray 골연령 - Attention-Guided RoI Localization + Label Distribution Learning
#   논문: Chen et al., "Attention-Guided Discriminative Region Localization and
#         Label Distribution Learning for Bone Age Assessment" (arXiv:2006.00202)
#   공식 구현(Keras): https://github.com/chenchao666/Bone-Age-Assessment
#
#   ▶ v7 (현재 버전) = v6 + 성능 개선용 전처리/증강
#     [신규] ⓪ 마커 억제  : YOLO 크롭 대응 '보수적' 제거
#            - 마커가 없으면 아무것도 하지 않음(대부분의 경우)
#            - 잘린 마커도 가장자리 조건으로 인식
#            - 손 내부는 절대 건드리지 않음(겹침·면적·테두리 4중 가드)
#            - 후보가 너무 많으면 오검출로 보고 전부 취소
#     [신규] ⓪-2 회전 정렬 : 손 주축(PCA)을 세로로. 과한 각도면 취소,
#            손목이 위로 뒤집힌 경우 180° 교정(애매하면 유지)
#     [변경] ③ CLAHE 를 리사이즈 '후'(560)에 적용 -> 타일 격자가 실효
#            (패딩 앞에 적용해 여백 증폭 방지). CLAHE_STAGE 로 되돌릴 수 있음
#     [변경] 약한 기하 증강 ON (밝기/대비 증강은 의도적으로 제외)
#     [유지] ①정규화 ②Top-Hat, R1 밴드 방식, 좌표 역변환, ROI 비율유지+패딩
#     [유지] R2 = 논문 방식(E 재학습 CAM), 3채널 집계, Xception+성별+LDL
#     [분리] cache_attention_v7/<PRE_TAG> / checkpoints_attention_v7 / logs/*_v7_*
#
#   ▶ 실행 후 로그에서 반드시 확인:
#       [마커] 제거된 이미지 N/M  -> 마커 없는 데이터면 낮은 게 정상
#       [정렬] 회전 중앙값 / 180° 뒤집기 N장
#
#   ▶ 실행: python attention_ldl_boneage_v2.py
#       - 창(터미널/VSCode)을 닫거나 노트북을 꺼도 서버에서 학습은 계속됩니다.
#       - 다시 실행하면 진행 중인 로그에 자동으로 다시 붙습니다.
#       - 각 스테이지는 완료 표식을 남기고 자동 스킵 -> 중단돼도 이어서 진행.
#
#   ▶ 실행 옵션
#       --fg             백그라운드 분리 없이 바로 실행(디버그용)
#       --eval-only      Phase II 학습을 건너뛰고 best.pt 로 평가만
#       --rebuild-cache  손 크롭 -> 560 리사이즈 캐시를 강제로 재생성
#       --rebuild-roi    R1/R2/E 크롭 캐시를 강제로 재생성 (CAM 부터 재실행)
#       --qc-only        ROI/QC 시트만 만들고 종료 (밴드 튜닝용)
#
#   ▶ 표기 규칙
#       [논문] = 논문/공식코드 명시값      [추론] = 미명시 -> 합리적 추정
# =========================================================================

from pathlib import Path
import os, sys, time, json, subprocess
from datetime import datetime

PROJECT_DIR = Path(__file__).resolve().parent
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RUN_STATE = LOG_DIR / "boneage_attention_v7_running.json"
_WORKER_ENV = "BONEAGE_ATTENTION_V7_WORKER"

FOREGROUND    = "--fg" in sys.argv
EVAL_ONLY     = "--eval-only" in sys.argv
REBUILD_ROI   = "--rebuild-roi" in sys.argv
REBUILD_CACHE = "--rebuild-cache" in sys.argv
QC_ONLY       = "--qc-only" in sys.argv      # ROI/QC 만 만들고 종료 (밴드 튜닝용)


# -------------------------------------------------------------------------
# [A] 런처: 자기 자신을 '세션과 분리된' 백그라운드로 띄우고 로그만 흘려보낸다.
# -------------------------------------------------------------------------
def _pid_alive(pid):
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True).stdout
        return str(pid) in out
    except Exception:
        return False


def _follow(log_path, pid):
    """로그 파일을 실시간 표시. 창을 닫거나 Ctrl+C 해도 학습은 계속됨."""
    log_path = Path(log_path)
    for _ in range(200):
        if log_path.exists():
            break
        time.sleep(0.2)
    print("=" * 64)
    print(f" 학습이 백그라운드에서 실행 중입니다  (PID {pid})")
    print(f" 로그 파일: {log_path}")
    print(" 이 창을 닫거나 노트북을 꺼도 서버에서 학습은 계속됩니다.")
    print(f" 완전히 중지하려면:  taskkill /PID {pid} /F")
    print("=" * 64, flush=True)
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            idle = 0
            while True:
                line = f.readline()
                if line:
                    sys.stdout.write(line); sys.stdout.flush(); idle = 0
                else:
                    idle += 1
                    if idle % 6 == 0 and not _pid_alive(pid):
                        rest = f.read()
                        if rest:
                            sys.stdout.write(rest); sys.stdout.flush()
                        print("\n[프로세스가 종료되었습니다]", flush=True)
                        break
                    time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[로그 보기만 종료] - 학습은 백그라운드에서 계속됩니다.", flush=True)


def _spawn_detached():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"boneage_attention_v7_{ts}.log"
    logf = open(log_path, "a", buffering=1, encoding="utf-8")
    env = dict(os.environ); env[_WORKER_ENV] = "1"
    env["PYTHONIOENCODING"] = "utf-8"; env["PYTHONUTF8"] = "1"
    cmd = [sys.executable, "-u", "-X", "utf8", str(Path(__file__).resolve())] + sys.argv[1:]

    DETACHED_PROCESS = 0x00000008
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    CREATE_BREAKAWAY_FROM_JOB = 0x01000000
    last_err = None
    for flags in (DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB,
                  DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP):
        try:
            p = subprocess.Popen(cmd, stdout=logf, stderr=logf,
                                 stdin=subprocess.DEVNULL, cwd=str(PROJECT_DIR),
                                 env=env, creationflags=flags, close_fds=True)
            logf.close()
            RUN_STATE.write_text(json.dumps({"pid": p.pid, "log": str(log_path)}),
                                 encoding="utf-8")
            return p.pid, log_path
        except OSError as e:
            last_err = e
            continue
    logf.close()
    raise RuntimeError(f"백그라운드 실행 실패: {last_err}")


if os.name == "nt" and not FOREGROUND and os.environ.get(_WORKER_ENV) != "1":
    if RUN_STATE.exists():
        try:
            st = json.loads(RUN_STATE.read_text(encoding="utf-8"))
        except Exception:
            st = {}
        if st.get("pid") and _pid_alive(st["pid"]):
            print("이미 실행 중입니다 - 기존 로그에 다시 붙습니다.")
            _follow(st["log"], st["pid"])
            sys.exit(0)
    _pid, _logp = _spawn_detached()
    _follow(_logp, _pid)
    sys.exit(0)


# =========================================================================
# [B] 본체
# =========================================================================
os.environ["TORCH_HOME"] = str(PROJECT_DIR / ".torch_cache")
os.environ.setdefault("HF_HOME", str(PROJECT_DIR / ".hf_cache"))

import random, math
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

try:
    import timm
except ImportError:
    raise SystemExit("timm 이 없습니다.  설치:  pip install timm")


def log(*a):
    print(f"[{datetime.now():%H:%M:%S}]", *a, flush=True)


def torch_load(path, map_location=None):
    """PyTorch 2.6+ weights_only 기본값 변경 대응 로더."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


# -------------------------------------------------------------------------
# [B-1] 경로  (실제 폴더 구조에 맞춰 고정. 없으면 그 자리에서 바로 중단)
#   BASE_DIR       : 캐시 / 체크포인트가 놓이는 작업 폴더
#   HAND_CROP_DIR  : ★ 손 크롭 이미지 폴더 (아래 train / validation / test)
#   CSV_DIR        : ★ 라벨 CSV 폴더
# -------------------------------------------------------------------------
BASE_DIR = Path(os.environ.get("BONEAGE_BASE_DIR", PROJECT_DIR))

# ★ 손 크롭 이미지 폴더 (yolo_crop/train, yolo_crop/validation, yolo_crop/test)
HAND_CROP_DIR = Path(os.environ.get("BONEAGE_HAND_DIR", r"G:/Project/sinra_cho/crop_data"))

# ★ 라벨 CSV 폴더 (yolo_crop_csv/*.csv)
CSV_DIR = Path(os.environ.get("BONEAGE_CSV_DIR", r"G:/Project/sinra_cho/crop_data_csv"))

# split 하위 폴더명 (yolo_crop 아래)
SPLIT_SUBDIR = {"train": "training", "val": "validation", "test": "test"}


def _require(path, what):
    """경로가 없으면 즉시 중단. 잘못된 경로로 조용히 진행하지 않는다."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"[중단] {what} 경로가 없습니다:\n       {p}\n"
                         f"       실제 위치로 코드 상단 경로를 수정하세요.")
    return p


TRAIN_CSV = _require(CSV_DIR / "training.csv", "train.csv")
VAL_CSV   = _require(CSV_DIR / "validation.csv", "Validation Dataset.csv")
TEST_CSV  = _require(CSV_DIR / "test.csv", "Bone age ground truth.csv")


# =========================================================================
# [B-2] 재현 스위치
# =========================================================================
SEED = 42

CLS_SIZE_REGION = 560     # [논문] R1/R2 국소화용 분류모델 입력 (특징맵 17~18x18)
REG_SIZE        = 560     # [논문] Table IV - 560 이상은 개선 없음

# -- CAM 임계값 (0~255 스케일) ------------------------------------------------
#   낮출수록 마스크가 넓어져 박스가 커집니다. 논문 Fig.1/첨부 그림처럼 넓은
#   해부학적 박스를 얻으려면 tau 를 낮추고(아래) + 여백/병합을 크게 줍니다.
TAU_R1 = 30               # [조정] 50 -> 30 (박스 확대. 논문 τ∈{10..100} 범위 내)

SOFT_L   = 50             # [논문] 식(5) 삼각형 소프트 라벨 폭 l=50
AGE_BINS = 240            # [논문] 데이터셋 최대 연령(개월)

# Phase II 집계 채널 - [논문] Table V 최고 조합: H+R1+E (4.3) / H+R1+R2 (4.3)
#   사용 가능: "hand"(크롭된 손 이미지), "r1", "r2", "erased"
#   ★ R2 를 반드시 만들도록 r2 포함. (r1/r2/erased 캐시는 모두 생성되므로
#     나중에 채널만 바꿔 H+R1+E 와 H+R1+R2 를 비교할 수 있습니다.)
AGG_CHANNELS = ("hand", "r1", "erased")

PAPER_STRICT = False      # [v7] 증강 ON (논문은 미사용이나 RSNA 상위팀은 사용)
                          #      True 로 두면 증강 OFF + x/255 (논문 그대로)
USE_AUG   = (not PAPER_STRICT)
NORMALIZE = "div255"      # [v7] 전처리에서 이미 표준화했으므로 x/255 유지

# =========================================================================
# ★ [v5] 전처리 파이프라인  (골연령_Xception_TopHat_CLAHE 노트북에서 이식)
#
#   ① 밝기 정규화 (백분위 스트레치)
#      -> ② Top-Hat (연조직 배경 제거·골 구조 강조)
#         -> ③ CLAHE (국소 대비)
#            -> ④ 비율 유지 리사이즈 -> ⑤ 배경색 패딩 -> 560x560
#
#   [중요] 레터박스 패딩이 들어가므로 캔버스 좌표 != 원본 좌표 단순비례 입니다.
#          CAM 박스를 원본으로 되돌릴 때 scale 과 pad 오프셋을 함께 역산합니다
#          (canvas_box_to_src 함수).
# =========================================================================
# ① 밝기 정규화
NORM_MODE    = "percentile"   # "percentile" | "minmax" | "none"
NORM_LO, NORM_HI = 1.0, 99.0  # 백분위 하한/상한 (마커·핫픽셀 클리핑)

# ② Top-Hat  (White Top-Hat = 원본 - Opening(원본))
TOPHAT_MODE   = "add"     # "none" | "white" | "add"(권장) | "addsub"
TOPHAT_KFRAC  = 0.10      # 커널 = 긴 변 x KFRAC. ★ 커널은 뼈 굵기보다 커야 함
                          #   (실측 중수골 폭 84~88px, 긴 변 1252 -> 125px)
TOPHAT_KSIZE  = 0         # >0 이면 절대 픽셀로 고정 (KFRAC 무시)
TOPHAT_SHAPE  = "rect"    # "rect"(빠름, 권장) | "ellipse" | "cross"
TOPHAT_WEIGHT = 1.0       # add / addsub 가산 강도

# ③ CLAHE
CLAHE_CLIP   = 2.0
CLAHE_TILE   = 8

# ④ 배율 정책
SCALE_MODE   = "per_image"   # "per_image": 각 이미지 긴 변을 크기에 맞춤
                             # "global"   : 데이터셋 공통 배율(손 절대크기 보존)
GLOBAL_REF_PCTL = 99.0

# ⑤ 패딩
PAD_MODE     = "auto"     # "auto"(테두리 하위 백분위) | "zero" | "replicate" | 정수
PAD_BORDER_FRAC = 0.02
PAD_BORDER_PCTL = 25.0
PAD_ANCHOR   = "center"   # "center" | "topleft"

# =========================================================================
# ★ [v7 신규] ⓪ 마커 억제 / 회전 정렬 / CLAHE 단계 위치
#
#   입력이 YOLO 크롭이라 (a) 마커가 아예 없을 수도 (b) 잘려서 일부만 남을 수도
#   (c) 크롭이 타이트해 손이 이미지 경계에 닿을 수도 있습니다.
#   -> 마커 제거는 '확실한 것만' 지우고, 조금이라도 애매하면 원본을 그대로 둡니다.
#      (손을 잘못 지우는 조용한 실패가 마커를 남기는 것보다 훨씬 위험)
# =========================================================================
# ── ⓪ 마커 억제 ─────────────────────────────────────────────────────
MARKER_MODE = "bright_blob"   # "off"          : 사용 안 함
                              # "bright_blob"  : ★권장. 아래 조건을 '전부' 만족하는
                              #                  밝은 덩어리만 배경색으로 대체
                              # "outside_hand" : 손 마스크 밖을 전부 배경색으로
                              #                  (크롭이 여유로울 때만. 위험)
MARKER_BRIGHT_PCTL = 99.2   # 이 백분위 이상을 '밝은 후보'로 (마커는 거의 포화)
MARKER_BRIGHT_MIN  = 200    # 동시에 이 절대값 이상이어야 함 (이중 조건)
MARKER_MIN_AREA_FRAC = 0.0004  # 후보 최소 면적(전체 대비). 점 노이즈 제외
MARKER_MAX_AREA_FRAC = 0.030   # ★ 후보 최대 면적. 이보다 크면 마커가 아니라고 보고 건너뜀
MARKER_REQUIRE_BORDER = True   # ★ 테두리에 닿은 것만 마커로 인정
                               #   (마커는 보통 구석/가장자리. 손 한가운데 밝은 뼈를
                               #    지우는 사고를 막는 핵심 가드)
MARKER_BORDER_FRAC = 0.18   # '테두리'로 볼 띠 두께 (짧은 변 대비)
MARKER_HAND_OVERLAP_MAX = 0.12  # 손 마스크와 이만큼 넘게 겹치면 마커 아님 -> 건너뜀
MARKER_DILATE = 5           # 지울 때 여유 (경계 잔상 제거)
MARKER_MAX_COUNT = 4        # 한 장에서 지울 최대 개수. 넘으면 오검출로 보고 전부 취소
# 손 마스크가 아래 범위를 벗어나면(=Otsu 실패 의심) 마커 억제를 통째로 건너뜁니다
MARKER_HAND_MIN_FRAC = 0.10
MARKER_HAND_MAX_FRAC = 0.97

# ── ⓪-2 회전 정렬 ───────────────────────────────────────────────────
ALIGN_MODE = "pca"        # "off" | "pca" : 손 주축(PCA)을 세로로 세움
ALIGN_MAX_DEG = 35.0      # 이 각도보다 크게 돌려야 하면 오검출로 보고 회전 취소
ALIGN_FIX_FLIP = True     # ★ 손목이 위로 뒤집힌 경우 180° 추가 회전
                          #   (상/하 절반의 마스크 폭 비교 - 손가락 쪽이 좁음)
ALIGN_FLIP_MARGIN = 1.12  # 폭 비가 이 배수 이상 차이날 때만 뒤집기 (애매하면 유지)

# ── ③ CLAHE 적용 시점 ───────────────────────────────────────────────
CLAHE_STAGE = "post_resize"  # "pre_resize"  : 원본 해상도에서 (v5/v6 동작)
                             # "post_resize" : ★리사이즈 후 560에서 (권장)
                             #   네트워크가 실제로 보는 해상도에서 타일 격자가
                             #   의미를 갖습니다. 단, 패딩보다는 앞에 적용됩니다.

# ── 증강 ────────────────────────────────────────────────────────────
#   [주의] 밝기/대비 증강은 넣지 않습니다. 정규화+CLAHE 로 표준화한 것을
#          다시 흐트러뜨리기 때문입니다. 약한 기하 증강만 사용합니다.
AUG_ROT_DEG   = 12
AUG_TRANSLATE = 0.06
AUG_SCALE     = (0.92, 1.08)

# 전처리 태그 -> 캐시 폴더명에 포함되어 설정이 다르면 자동으로 분리됩니다
_NORM_TAG = "none" if NORM_MODE == "none" else (
    "mm" if NORM_MODE == "minmax" else f"p{NORM_LO:g}-{NORM_HI:g}")
_TH_TAG   = "off" if TOPHAT_MODE == "none" else (
    f"{TOPHAT_MODE}-{TOPHAT_SHAPE}-" +
    (f"k{TOPHAT_KSIZE}" if TOPHAT_KSIZE > 0 else f"f{TOPHAT_KFRAC:g}") +
    f"-w{TOPHAT_WEIGHT:g}")
_SC_TAG   = "img" if SCALE_MODE == "per_image" else f"glob{GLOBAL_REF_PCTL:g}"
_MK_TAG   = "off" if MARKER_MODE == "off" else f"{MARKER_MODE[:2]}p{MARKER_BRIGHT_PCTL:g}"
_AL_TAG   = "off" if ALIGN_MODE == "off" else f"{ALIGN_MODE}{'F' if ALIGN_FIX_FLIP else ''}"
PRE_TAG   = (f"{REG_SIZE}_n{_NORM_TAG}_mk{_MK_TAG}_al{_AL_TAG}_th{_TH_TAG}"
             f"_c{CLAHE_CLIP:g}t{CLAHE_TILE}{'@post' if CLAHE_STAGE=='post_resize' else ''}"
             f"_{_SC_TAG}_pad{PAD_MODE}")

# =========================================================================
# [v6] R1 = 밴드 방식 유지 (v5 그대로) / R2 = ★ 논문 방식으로 복원
#
#   R1 (유지): 이미지 비율 기준 고정 밴드 안에서 CAM 최대 영역 -> 박스 확대
#              -> QC 에서 확인한 크기가 그대로 나옵니다.
#   R2 (복원): 논문 §III-A "Localization of Region-2"
#              R1 을 랜덤값으로 지운 이미지(E)로 분류모델을 '다시 학습' 시키고,
#              그 모델의 CAM 으로 R2 를 찾습니다. 밴드 제약·박스 확대 없음.
#              -> 네트워크가 R1 외 영역에 근거해 예측하도록 강제되어
#                 그 다음으로 판별력 있는 영역(중수골)이 자동으로 드러납니다.
# =========================================================================
R1_BAND = {"x0": 0.12, "y0": 0.52, "x1": 0.88, "y1": 0.98}   # 수근골 밴드 (유지)
BAND_CFG = {"r1": R1_BAND}
CLAMP_TO_BAND = True        # R1 박스가 밴드 밖으로 못 나가게 고정

# -- R1 박스 확대 파라미터 (v5 와 동일 - 변경 없음) ---------------------------
MERGE_COMPONENTS   = True   # tau 이상 화소 전체의 외접 박스 (조각 병합)

PAD_FRAC_R1_X      = 0.10
PAD_FRAC_R1_Y      = 0.12
MIN_W_FRAC_R1      = 0.48   # 이미지 폭 대비 최소 폭 (QC 빨강 박스 기준)
MIN_H_FRAC_R1      = 0.34

BOX_CFG = {
    "r1": {"pad_x": PAD_FRAC_R1_X, "pad_y": PAD_FRAC_R1_Y,
           "min_w": MIN_W_FRAC_R1, "min_h": MIN_H_FRAC_R1},
    # r2 는 논문 방식(박스 확대 없음) -> BOX_CFG 에 항목을 두지 않습니다
}

# -- ★ R2 : 논문 방식 -------------------------------------------------------
#   [논문] "we generate input images by replacing the pixels in Region-1 with
#           random values ... we can localize Region-2 in the same way"
#   [논문] 임계값 tau = 50 (Table I: 중수골 mIoU 0.565 / AP50 0.735 최대)
TAU_R2_PAPER   = 50       # [논문] R2 임계값
R2_MERGE       = False    # 논문은 '가장 판별력 있는 영역' -> 최대 연결성분 1개
                          #   True 로 두면 tau 이상 화소 전체를 감싸 커집니다
R2_PAD_FRAC    = 0.0      # [논문] 박스 여백 없음 (0 = 논문 그대로)
R2_MIN_W_FRAC  = 0.0      # [논문] 최소 크기 강제 없음
R2_MIN_H_FRAC  = 0.0
BOX_CFG_R2 = (None if (R2_PAD_FRAC == 0 and R2_MIN_W_FRAC == 0 and R2_MIN_H_FRAC == 0)
              else {"pad_x": R2_PAD_FRAC, "pad_y": R2_PAD_FRAC,
                    "min_w": R2_MIN_W_FRAC, "min_h": R2_MIN_H_FRAC})

# -- 크기 맞추기 -------------------------------------------------------------
#   전처리와 동일하게 '비율 유지 리사이즈 + 배경색 패딩'(fit_canvas)
PAD_TO_SQUARE = True

ERASE_FILL    = "noise"   # [논문] R1 픽셀을 랜덤값으로 대체
USE_EXCLUDE   = True
N_QC          = 6

# 전처리 설정이 다르면 캐시가 자동으로 분리됩니다 (PRE_TAG)
CACHE_DIR = BASE_DIR / "cache_attention_v7" / PRE_TAG
CKPT_DIR  = BASE_DIR / "checkpoints_attention_v7"
SPLITS    = ("train", "val", "test")
SUBDIRS   = ("hand560", "r1", "r2", "erased")

for d in [CACHE_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)
for sub in SUBDIRS:
    for sp in SPLITS:
        (CACHE_DIR / sub / sp).mkdir(parents=True, exist_ok=True)

BASE_DONE = CACHE_DIR / "_DONE_hand560.json"
REF_JSON  = CACHE_DIR / "_REF_LONG.json"   # SCALE_MODE="global" 기준 긴 변 캐시
ROI_DONE  = CACHE_DIR / "_DONE_roi.json"
BBOX_CSV  = CKPT_DIR / "roi_bboxes.csv"

CLS_R1_BEST = CKPT_DIR / "cls_r1_best.pt"
CLS_R1_LAST = CKPT_DIR / "cls_r1_last.pt"
CLS_R2_BEST = CKPT_DIR / "cls_r2_best.pt"
CLS_R2_LAST = CKPT_DIR / "cls_r2_last.pt"

BEST_CKPT    = CKPT_DIR / "best.pt"
LAST_CKPT    = CKPT_DIR / "last.pt"
HISTORY_JSON = CKPT_DIR / "history.json"
RESULTS_TXT  = CKPT_DIR / "results.txt"
RESULTS_JSON = CKPT_DIR / "results.json"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

EXCLUDE_IDS = {"1405", "1430", "1431", "1521", "1545", "1599", "1607", "1779", "1799", "1826", 
               "1840", "1863", "1926", "1973", "2133", "2193", "2213", "2256", "2296", "2372", 
                "2414", "2505", "2662", "2687", "2823", "2848", "2934", "3079", "3100", "3110", 
                "3156", "3157", "3387", "3475", "3528", "3655", "3722", "3736", "3752", "3823", 
                "3880", "3883", "3884", "3885", "3899", "3219", "3905", "3931", "3964", "3998", 
                "3999", "4004", "4067", "4071", "4128", "4193", "4210", "4217", "4230", "4243", 
                "4284", "4792", "6232", "6293", "6484", "6573", "6784", "6886", "7048", "7179", 
                "7235", "7358", "7491", "7507", "7555", "7758", "7784", "7822", "7826", "7840", 
                "7884", "7893", "7963", "7979", "8124", "8142", "8451", "8566", "8599", "8607", 
                "8623", "8680", "8821", "8836", "9024", "9194", "9401", "9728", "10059", "10087", 
                "10278", "10573", "10715", "10720", "11043", "11079", "11152", "11367", "11863", 
                "11910", "11917", "11971", "11987", "12036", "12074", "12192", "12296", "12335", 
                "12351", "12684", "13130", "14011", "14086", "14152", "14179", "14234", "14235", 
                "14281", "14343", "14552", "14595", "14742", "14770", "15035", "15114", "15234", 
                "15398", "15413",

                "1397", "1450", "1537", "1583", "1687", "1740", "2190", "2945", "3131", "3319", 
                "3326", "3404", "3625", "3853", "3868", "4022", "4119", "4274", "4800", "5618", 
                "6165", "6393", "7629", "7790", "8549", "9607", "10389", "10543", "11146", "11183", 
                "11312", "11559", "11774", "11839", "12110", "13308", "13806", "14325", "14746",

                "4389", "4423", "4432", "4455", "4483"
                } if USE_EXCLUDE else set()

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

print("=" * 64)
log("Attention(R1/R2) + LDL 골연령 - v2 (손 크롭 폴더 입력) 시작")
log(f"PyTorch {torch.__version__} | torchvision {torchvision.__version__} | timm {timm.__version__}")
log(f"device {device} | CUDA {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log(f"GPU {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda}")
log(f"BASE_DIR       {BASE_DIR}")
log(f"HAND_CROP_DIR  {HAND_CROP_DIR}")
log(f"채널 {AGG_CHANNELS} | tau(R1/R2) {TAU_R1}/{TAU_R2_PAPER} | 정규화 {NORMALIZE} | 증강 {USE_AUG}")
print("=" * 64, flush=True)
# [주의] RTX 5060(Blackwell, sm_120)은 최신 PyTorch 필요.
#   CUDA=False 또는 'no kernel image' 오류 시:
#   pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128


def imread_kr(path, flags=cv2.IMREAD_GRAYSCALE):
    """한글/유니코드 경로에서도 동작하는 이미지 로드. 실패 시 None."""
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    except Exception:
        return None


def imwrite_kr(path, img):
    """한글/유니코드 경로에도 저장 가능한 이미지 쓰기."""
    path = str(path); ext = os.path.splitext(path)[1] or ".png"
    try:
        ok, buf = cv2.imencode(ext, img)
        if ok:
            buf.tofile(path)
        return ok
    except Exception:
        return False


# =========================================================================
# [B-3] 전처리 함수  ① 정규화 → ② Top-Hat → ③ CLAHE → ④ 배율 → ⑤ 리사이즈+패딩
#       (골연령_Xception_TopHat_CLAHE 노트북에서 이식)
# =========================================================================
def _to_uint8(g):
    """16bit·float 입력도 안전하게 8bit로."""
    if g.dtype == np.uint8:
        return g
    g = g.astype(np.float32)
    lo, hi = float(g.min()), float(g.max())
    if hi - lo < 1e-6:
        return np.zeros(g.shape, np.uint8)
    return np.clip((g - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


# ── ① 밝기 정규화 ──────────────────────────────────────────────────
def normalize_intensity(g, mode=None, lo=None, hi=None):
    """이미지 간 밝기·계조 편차 제거. CLAHE보다 반드시 앞에 옵니다."""
    mode = NORM_MODE if mode is None else mode
    lo   = NORM_LO   if lo   is None else lo
    hi   = NORM_HI   if hi   is None else hi
    g = _to_uint8(g)
    if mode == "none":
        return g
    if mode == "minmax":
        a, b = float(g.min()), float(g.max())
    else:                                        # "percentile"
        a, b = np.percentile(g, [float(lo), float(hi)])
    if b - a < 1e-6:
        return g
    return np.clip((g.astype(np.float32) - a) / (b - a) * 255.0, 0, 255).astype(np.uint8)


# ── ⓪ 손 마스크 (마커 억제·정렬 공용) ──────────────────────────────
def hand_mask(g):
    """Otsu + 최대 연결성분 = 손. (mask, ok) 반환.
       ok=False 면 마스크를 신뢰할 수 없다는 뜻 -> 호출부는 아무것도 하지 않습니다."""
    g8 = _to_uint8(g)
    blur = cv2.GaussianBlur(g8, (5, 5), 0)
    _, m = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None, False
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    hm = (lab == i).astype(np.uint8)
    frac = hm.sum() / float(hm.size)
    # 손이 너무 작거나(배경을 손으로 오인) 거의 전체(임계 실패)면 신뢰 불가
    if not (MARKER_HAND_MIN_FRAC <= frac <= MARKER_HAND_MAX_FRAC):
        return hm, False
    return hm, True


def suppress_markers(g, debug=False):
    """★ YOLO 크롭 대응 보수적 마커 제거.
       '확실한 마커'만 지우고 조금이라도 애매하면 원본을 그대로 둡니다.
       마커가 없으면(대부분) 아무 일도 일어나지 않습니다.

       마커로 인정하는 조건 (전부 만족해야 함):
         1) 매우 밝음        : 백분위 MARKER_BRIGHT_PCTL 이상 AND 절대값 MIN 이상
         2) 면적이 적당      : MIN_AREA_FRAC ~ MAX_AREA_FRAC (너무 크면 뼈/손일 수 있음)
         3) 테두리에 닿음    : MARKER_REQUIRE_BORDER (마커는 보통 구석. 잘려도 가장자리)
         4) 손과 거의 안 겹침: 겹침 비율 MARKER_HAND_OVERLAP_MAX 이하
       그리고 후보가 MARKER_MAX_COUNT 를 넘으면 오검출로 보고 '전부 취소'합니다."""
    info = {"found": 0, "skipped": False, "reason": ""}
    if MARKER_MODE == "off":
        info["skipped"] = True; info["reason"] = "off"
        return (g, info) if debug else g

    g8 = _to_uint8(g)
    h, w = g8.shape[:2]
    hm, ok = hand_mask(g8)
    if not ok:
        info["skipped"] = True; info["reason"] = "hand_mask_unreliable"
        return (g8, info) if debug else g8      # ★ 애매하면 손대지 않음

    if MARKER_MODE == "outside_hand":
        # 공격적 모드: 손 밖 전부 배경색. 크롭이 여유로울 때만 사용하세요.
        hd = cv2.dilate(hm, np.ones((MARKER_DILATE * 3, MARKER_DILATE * 3), np.uint8))
        bg_pix = g8[hd == 0]
        if bg_pix.size < 50:
            info["skipped"] = True; info["reason"] = "no_background"
            return (g8, info) if debug else g8
        out = g8.copy()
        out[hd == 0] = int(np.percentile(bg_pix, 25))
        info["found"] = 1
        return (out, info) if debug else out

    # ---- bright_blob 모드 -------------------------------------------------
    thr = max(float(np.percentile(g8, MARKER_BRIGHT_PCTL)), float(MARKER_BRIGHT_MIN))
    bright = (g8 >= thr).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    if n <= 1:
        info["reason"] = "no_bright_blob"
        return (g8, info) if debug else g8

    bt = max(1, int(round(min(h, w) * MARKER_BORDER_FRAC)))
    area_img = float(h * w)
    cands = []
    for i in range(1, n):
        x, y, bw, bh, area = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                              stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT],
                              stats[i, cv2.CC_STAT_AREA])
        af = area / area_img
        if af < MARKER_MIN_AREA_FRAC or af > MARKER_MAX_AREA_FRAC:
            continue                                    # 조건 2
        if MARKER_REQUIRE_BORDER:
            touches = (x <= bt or y <= bt or (x + bw) >= (w - bt) or (y + bh) >= (h - bt))
            if not touches:
                continue                                # 조건 3
        comp = (lab == i)
        ov = float((comp & (hm > 0)).sum()) / float(comp.sum())
        if ov > MARKER_HAND_OVERLAP_MAX:
            continue                                    # 조건 4
        cands.append(i)

    if not cands:
        info["reason"] = "no_candidate"
        return (g8, info) if debug else g8
    if len(cands) > MARKER_MAX_COUNT:
        info["skipped"] = True; info["reason"] = f"too_many({len(cands)})"
        return (g8, info) if debug else g8              # ★ 오검출 의심 -> 전부 취소

    rm = np.isin(lab, cands).astype(np.uint8)
    if MARKER_DILATE > 0:
        rm = cv2.dilate(rm, np.ones((MARKER_DILATE, MARKER_DILATE), np.uint8))
    rm[hm > 0] = 0                                      # 손 영역은 절대 건드리지 않음
    bg_pix = g8[(hm == 0) & (rm == 0)]
    fill = int(np.percentile(bg_pix, 50)) if bg_pix.size >= 50 else int(np.percentile(g8, 10))
    out = g8.copy()
    out[rm > 0] = fill
    info["found"] = len(cands)
    return (out, info) if debug else out


# ── ⓪-2 회전 정렬 ──────────────────────────────────────────────────
def align_hand(g, debug=False):
    """손 주축(PCA)을 세로로 세웁니다. 각도가 과하면(오검출) 회전을 취소합니다.
       ALIGN_FIX_FLIP=True 면 손목이 위로 뒤집힌 경우 180° 추가 회전합니다."""
    info = {"rot": 0.0, "flip": False, "skipped": False}
    if ALIGN_MODE == "off":
        info["skipped"] = True
        return (g, info) if debug else g
    g8 = _to_uint8(g)
    hm, ok = hand_mask(g8)
    if not ok:
        info["skipped"] = True
        return (g8, info) if debug else g8

    ys, xs = np.where(hm > 0)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    mean, ev = cv2.PCACompute(pts, mean=None, maxComponents=1)
    ang = float(np.degrees(np.arctan2(ev[0, 1], ev[0, 0])))
    rot = -(ang - 90.0)
    while rot > 90:  rot -= 180
    while rot < -90: rot += 180
    if abs(rot) > ALIGN_MAX_DEG:          # ★ 과한 각도 = 마스크 오검출 의심
        info["skipped"] = True
        return (g8, info) if debug else g8

    cx, cy = float(mean[0][0]), float(mean[0][1])
    h, w = g8.shape[:2]
    fill = int(np.percentile(g8[hm == 0], 25)) if (hm == 0).sum() > 50 else 0
    M = cv2.getRotationMatrix2D((cx, cy), rot, 1.0)
    out = cv2.warpAffine(g8, M, (w, h), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    hmr = cv2.warpAffine(hm, M, (w, h), flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    if ALIGN_FIX_FLIP:
        # 손가락 쪽이 손목 쪽보다 '폭이 좁다' -> 위쪽이 넓으면 뒤집힌 것
        half = h // 2
        top_w = float(hmr[:half].sum(axis=1).mean() + 1e-6)
        bot_w = float(hmr[half:].sum(axis=1).mean() + 1e-6)
        if top_w > bot_w * ALIGN_FLIP_MARGIN:      # 애매하면(마진 미만) 유지
            out = cv2.rotate(out, cv2.ROTATE_180)
            info["flip"] = True
    info["rot"] = rot
    return (out, info) if debug else out
_TH_SHAPES = {"rect": cv2.MORPH_RECT, "ellipse": cv2.MORPH_ELLIPSE, "cross": cv2.MORPH_CROSS}


def tophat_ksize(h, w, kfrac=None, ksize=None):
    """커널 한 변(홀수). 원본 해상도가 제각각이라 긴 변 대비 비율로 정합니다."""
    kfrac = TOPHAT_KFRAC if kfrac is None else kfrac
    ksize = TOPHAT_KSIZE if ksize is None else ksize
    k = int(ksize) if ksize and ksize > 0 else int(round(max(h, w) * float(kfrac)))
    k = max(3, min(k, min(h, w) - 1 if min(h, w) > 4 else 3))
    return k if k % 2 == 1 else k + 1


def apply_tophat(g, mode=None, kfrac=None, ksize=None, shape=None, weight=None,
                 return_k=False):
    """White Top-Hat = g - Opening(g). 커널보다 작은 밝은 구조(=골)만 남깁니다.
       ★ 커널이 뼈 굵기보다 작으면 굵은 뼈 내부가 파입니다(실측 중수골 폭 84~88px)."""
    mode   = TOPHAT_MODE   if mode   is None else mode
    shape  = TOPHAT_SHAPE  if shape  is None else shape
    weight = TOPHAT_WEIGHT if weight is None else weight
    g = _to_uint8(g)
    if mode == "none":
        return (g, 0) if return_k else g
    h, w = g.shape[:2]
    k = tophat_ksize(h, w, kfrac, ksize)
    kern = cv2.getStructuringElement(_TH_SHAPES.get(shape, cv2.MORPH_RECT), (k, k))
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, kern)
    if mode == "white":
        out = th
    elif mode == "addsub":
        bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, kern)
        out = np.clip(g.astype(np.int16) + float(weight) * th.astype(np.int16)
                      - float(weight) * bh.astype(np.int16), 0, 255).astype(np.uint8)
    else:                                                  # "add"
        out = np.clip(g.astype(np.int16) + float(weight) * th.astype(np.int16),
                      0, 255).astype(np.uint8)
    return (out, k) if return_k else out


# ── ③ CLAHE ───────────────────────────────────────────────────────
def apply_clahe(g, clip=None, tile=None):
    """대비제한 적응 히스토그램 평활화(국소 대비 강화)."""
    clip = CLAHE_CLIP if clip is None else clip
    tile = CLAHE_TILE if tile is None else tile
    c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tile), int(tile)))
    return c.apply(_to_uint8(g))


# ── ④ 배율 결정 ────────────────────────────────────────────────────
def compute_scale(h, w, size=None, scale_mode=None, ref_long=None):
    """리사이즈 배율. global 모드에서도 캔버스를 넘치지 않도록 상한을 겁니다."""
    size       = REG_SIZE   if size       is None else size
    scale_mode = SCALE_MODE if scale_mode is None else scale_mode
    if scale_mode == "global":
        ref = ref_long if ref_long else globals().get("REF_LONG")
        if not ref:
            raise RuntimeError("SCALE_MODE='global'인데 REF_LONG 이 없습니다.")
        s = min(size / float(ref), size / float(max(h, w)))
    else:                                        # "per_image"
        s = size / float(max(h, w))
    return s


# ── ⑤ 배경색 추정 + 리사이즈 + 패딩 ────────────────────────────────
def estimate_bg(g, frac=None, pctl=None):
    """테두리 링의 하위 백분위 = 배경색 추정치."""
    frac = PAD_BORDER_FRAC if frac is None else frac
    pctl = PAD_BORDER_PCTL if pctl is None else pctl
    h, w = g.shape[:2]
    t = max(1, int(round(min(h, w) * float(frac))))
    ring = np.concatenate([g[:t, :].ravel(), g[-t:, :].ravel(),
                           g[:, :t].ravel(), g[:, -t:].ravel()])
    return int(np.percentile(ring, float(pctl)))


def fit_canvas(g, size=None, scale=None, pad_mode=None, anchor=None,
               frac=None, pctl=None):
    """배율만큼 축소한 뒤 남는 여백을 채워 size x size 완성. 반환 (canvas, info)"""
    size     = REG_SIZE   if size     is None else size
    pad_mode = PAD_MODE   if pad_mode is None else pad_mode
    anchor   = PAD_ANCHOR if anchor   is None else anchor
    h, w = g.shape[:2]
    s = compute_scale(h, w, size) if scale is None else scale
    nh, nw = max(1, min(size, int(round(h * s)))), max(1, min(size, int(round(w * s))))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    r = cv2.resize(g, (nw, nh), interpolation=interp)
    if CLAHE_STAGE == "post_resize":
        # ★ 네트워크가 실제로 보는 해상도에서 타일 격자가 의미를 갖도록.
        #   패딩 '앞'에 적용해야 균일한 여백이 자기 타일에서 증폭되지 않습니다.
        r = apply_clahe(r)
    top  = 0 if anchor == "topleft" else (size - nh) // 2
    left = 0 if anchor == "topleft" else (size - nw) // 2
    bottom, right = size - nh - top, size - nw - left
    if pad_mode == "replicate":
        out, pad_val = cv2.copyMakeBorder(r, top, bottom, left, right,
                                          cv2.BORDER_REPLICATE), -1
    else:
        if   pad_mode == "zero": pad_val = 0
        elif pad_mode == "auto": pad_val = estimate_bg(r, frac, pctl)
        else:                    pad_val = int(pad_mode)
        out = cv2.copyMakeBorder(r, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=int(pad_val))
    info = {"scale": float(s), "src": (h, w), "resized": (nh, nw),
            "pad": (top, bottom, left, right), "pad_val": pad_val,
            "pad_frac": 1.0 - (nh * nw) / float(size * size)}
    return out, info


def enhance(gray, return_k=False, return_info=False):
    """[v7] ⓪마커억제 → ①정규화 → ⓪-2회전정렬 → ②Top-Hat → (③CLAHE)
       ★ 해상도 변경 없음. 원본 해상도에서 1회만 수행하고 캔버스/ROI 크롭에 재사용.
       ★ CLAHE_STAGE='post_resize' 면 여기서 CLAHE 를 걸지 않고,
         fit_canvas 의 리사이즈 직후·패딩 직전에 적용합니다."""
    inf = {}
    g, mi = suppress_markers(gray, debug=True)          # ⓪ 마커 (보수적)
    inf["marker"] = mi
    g = normalize_intensity(g)                          # ① 정규화
    g, ai = align_hand(g, debug=True)                   # ⓪-2 회전 정렬
    inf["align"] = ai
    g, k = apply_tophat(g, return_k=True)               # ② Top-Hat (원본 해상도)
    inf["tophat_k"] = k
    if CLAHE_STAGE == "pre_resize":
        g = apply_clahe(g)                              # ③ CLAHE (구 동작)
    if return_info:
        return g, inf
    return (g, k) if return_k else g


def preprocess(gray, size=None, return_info=False):
    """전체 파이프라인: enhance -> 배율 -> 리사이즈 -> (CLAHE) -> 패딩."""
    size = REG_SIZE if size is None else size
    g, inf = enhance(gray, return_info=True)
    out, info = fit_canvas(g, size)
    info.update(inf)
    return (out, info) if return_info else out


def canvas_box_to_src(box, info):
    """★ 캔버스(560, 레터박스 패딩 포함) 좌표 -> 원본 좌표 역변환.
       패딩 오프셋을 빼고 배율로 나눕니다. 이 역변환이 없으면 박스가 어긋납니다."""
    top, _bottom, left, _right = info["pad"]
    s = info["scale"]
    h, w = info["src"]
    x0, y0, x1, y1 = box
    X0 = int(round((x0 - left) / s)); X1 = int(round((x1 - left) / s))
    Y0 = int(round((y0 - top) / s));  Y1 = int(round((y1 - top) / s))
    X0, Y0 = max(0, X0), max(0, Y0)
    X1, Y1 = min(w, max(X0 + 1, X1)), min(h, max(Y0 + 1, Y1))
    return X0, Y0, X1, Y1


def scan_ref_long(dfs, pctl=None, cache_path=None, force=False):
    """SCALE_MODE='global' 용: 헤더만 읽어 긴 변 백분위를 계산하고 캐시."""
    from PIL import Image
    pctl = GLOBAL_REF_PCTL if pctl is None else pctl
    cache_path = REF_JSON if cache_path is None else cache_path
    if (not force) and Path(cache_path).exists():
        info = json.load(open(cache_path, encoding="utf-8"))
        if info.get("pctl") == pctl:
            return int(info["ref_long"]), info
    longs = []
    for df in dfs:
        for p in df["hand_path"]:
            try:
                with Image.open(p) as im:
                    longs.append(max(im.size))
            except Exception:
                pass
    if not longs:
        raise RuntimeError("이미지 크기를 하나도 읽지 못했습니다 - 경로를 확인하세요.")
    longs = np.array(longs)
    ref = int(np.percentile(longs, pctl))
    info = {"ref_long": ref, "pctl": pctl, "n": int(longs.size),
            "min": int(longs.min()), "median": int(np.median(longs)),
            "max": int(longs.max())}
    json.dump(info, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return ref, info


REF_LONG = None    # SCALE_MODE='global' 이면 캐시 생성 전에 채워집니다


# =========================================================================
# [C] 라벨 로드 + 손 크롭 파일 인덱싱
# =========================================================================
def load_labels(csv_path):
    """id/boneage/male 컬럼을 유연하게 탐지해 표준 DataFrame 으로 반환."""
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(keys):
        for k in keys:
            for lc, orig in cols.items():
                if k in lc:
                    return orig
        return None

    id_col  = pick(["id", "case", "image"]) or df.columns[0]
    age_col = pick(["boneage", "bone age", "age"])
    sex_col = pick(["male", "sex", "gender"])
    assert age_col and sex_col, f"컬럼 탐지 실패: {list(df.columns)}"
    out = pd.DataFrame()
    out["id"]      = df[id_col].astype(str).str.replace(".png", "", regex=False).str.strip()
    out["boneage"] = pd.to_numeric(df[age_col], errors="coerce")
    s = df[sex_col]
    if s.dtype == bool:
        male = s.astype(int)
    else:
        sv = s.astype(str).str.lower().str.strip()
        male = sv.map({"true": 1, "false": 0, "m": 1, "f": 0,
                       "male": 1, "female": 0, "1": 1, "0": 0})
        if male.isna().any():
            male = pd.to_numeric(s, errors="coerce")
    out["male"] = male.astype(float)
    return out.dropna(subset=["boneage", "male"]).reset_index(drop=True)


def find_split_dir(split):
    """HAND_CROP_DIR/<고정 폴더명> 반환. 없으면 즉시 중단."""
    p = HAND_CROP_DIR / SPLIT_SUBDIR[split]
    if not p.exists():
        # test 는 없을 수도 있으니 test 만 None 허용, train/val 은 필수
        if split == "test":
            return None
        raise SystemExit(f"[중단] {split} 이미지 폴더가 없습니다: {p}")
    return p


def index_hand_files(split):
    """{id(stem): 파일경로} 인덱스. 하위 폴더까지 재귀 탐색."""
    d = find_split_dir(split)
    if d is None:
        return {}, None
    idx = {}
    for ext in ("*.png", "*.PNG", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
        for p in d.rglob(ext):
            idx.setdefault(p.stem.strip(), p)
    return idx, d


if not HAND_CROP_DIR.exists():
    raise SystemExit(f"[중단] 손 크롭 폴더가 없습니다: {HAND_CROP_DIR}\n"
                     f"       HAND_CROP_DIR 값을 확인하거나 환경변수 BONEAGE_HAND_DIR 로 지정하세요.")

for _n, _p in [("TRAIN_CSV", TRAIN_CSV), ("VAL_CSV", VAL_CSV), ("TEST_CSV", TEST_CSV)]:
    log(f"  {_n:<9} {'OK ' if Path(_p).exists() else '없음'} {_p}")

train_df = load_labels(TRAIN_CSV)
val_df   = load_labels(VAL_CSV)
test_df  = load_labels(TEST_CSV)

SPLIT_DFS = {"train": train_df, "val": val_df, "test": test_df}
HAND_INDEX = {}

for sp in SPLITS:
    idx, d = index_hand_files(sp)
    HAND_INDEX[sp] = idx
    log(f"  손크롭 {sp:<5} {'OK ' if d else '없음'} {d}  (파일 {len(idx):,}장)")

for sp in SPLITS:
    df = SPLIT_DFS[sp]
    if not len(df):
        continue
    before = len(df)
    if EXCLUDE_IDS:
        df = df[~df["id"].isin(EXCLUDE_IDS)]
    # 손 크롭이 실제로 존재하는 행만 사용 (크롭 실패분 자동 제외)
    idx = HAND_INDEX[sp]
    df = df[df["id"].isin(idx.keys())].reset_index(drop=True)
    df["hand_path"] = df["id"].map(lambda i: str(idx[i]))
    SPLIT_DFS[sp] = df
    log(f"  · {sp}: 라벨 {before:,} -> 사용 {len(df):,} "
        f"(제외/미크롭 {before - len(df):,})")

train_df, val_df, test_df = SPLIT_DFS["train"], SPLIT_DFS["val"], SPLIT_DFS["test"]
if not len(train_df):
    raise SystemExit("[중단] train 에서 사용 가능한 손 크롭 이미지가 0장입니다. 폴더/파일명을 확인하세요.")
log(f"최종 | train {len(train_df):,} | val {len(val_df):,} | test {len(test_df):,} "
    f"| 나이 {train_df.boneage.mean():.1f}±{train_df.boneage.std():.1f}개월 "
    f"| 남 {train_df.male.mean():.1%}")


# =========================================================================
# [D] hand560 캐시 - 손 크롭을 560x560 으로 리사이즈해 1회만 저장
#     * CAM 입력이자 Phase II 의 'hand' 채널로 동시에 쓰입니다.
#     * 고해상도 R1/R2 크롭은 원본(손 크롭 파일)에서 직접 뜹니다.
# =========================================================================
PRE_PARAMS = {"IMG_SIZE": REG_SIZE, "NORM_MODE": NORM_MODE, "NORM_LO": NORM_LO,
              "NORM_HI": NORM_HI, "TOPHAT_MODE": TOPHAT_MODE, "TOPHAT_KFRAC": TOPHAT_KFRAC,
              "TOPHAT_KSIZE": TOPHAT_KSIZE, "TOPHAT_SHAPE": TOPHAT_SHAPE,
              "TOPHAT_WEIGHT": TOPHAT_WEIGHT, "CLAHE_CLIP": CLAHE_CLIP,
              "CLAHE_TILE": CLAHE_TILE, "SCALE_MODE": SCALE_MODE,
              "PAD_MODE": PAD_MODE, "PAD_BORDER_FRAC": PAD_BORDER_FRAC,
              "PAD_BORDER_PCTL": PAD_BORDER_PCTL, "PAD_ANCHOR": PAD_ANCHOR,
              "MARKER_MODE": MARKER_MODE, "MARKER_BRIGHT_PCTL": MARKER_BRIGHT_PCTL,
              "MARKER_BRIGHT_MIN": MARKER_BRIGHT_MIN,
              "MARKER_MAX_AREA_FRAC": MARKER_MAX_AREA_FRAC,
              "MARKER_REQUIRE_BORDER": MARKER_REQUIRE_BORDER,
              "MARKER_BORDER_FRAC": MARKER_BORDER_FRAC,
              "MARKER_MAX_COUNT": MARKER_MAX_COUNT,
              "ALIGN_MODE": ALIGN_MODE, "ALIGN_MAX_DEG": ALIGN_MAX_DEG,
              "ALIGN_FIX_FLIP": ALIGN_FIX_FLIP, "CLAHE_STAGE": CLAHE_STAGE}


def hand_cache_valid():
    if REBUILD_CACHE or not BASE_DONE.exists():
        return False
    try:
        info = json.load(open(BASE_DONE, encoding="utf-8"))
    except Exception:
        return False
    return (info.get("pre") == PRE_PARAMS
            and info.get("train") == len(train_df)
            and info.get("val") == len(val_df)
            and info.get("test") == len(test_df))


def build_hand_cache():
    """손 크롭 -> 전처리 5단계 -> 560 캔버스 저장.
       ★ Top-Hat 은 원본 해상도에서 수행해야 커널 크기(뼈 굵기 대비)가 의미를 가집니다."""
    ks, pfs = [], []
    mstat = {"found": 0, "n_blob": 0, "skipped": 0, "reasons": {}}
    astat = {"skipped": 0, "flip": 0, "rots": []}
    for sp in SPLITS:
        df = SPLIT_DFS[sp]
        if not len(df):
            continue
        made = skipped = failed = 0
        for i, (_, r) in enumerate(df.iterrows(), 1):
            dst = CACHE_DIR / "hand560" / sp / f"{r['id']}.png"
            if dst.exists() and not REBUILD_CACHE:
                skipped += 1
            else:
                g = imread_kr(r["hand_path"], cv2.IMREAD_GRAYSCALE)
                if g is None:
                    failed += 1
                else:
                    canvas, info = preprocess(g, REG_SIZE, return_info=True)
                    imwrite_kr(dst, canvas)
                    ks.append(info["tophat_k"]); pfs.append(info["pad_frac"])
                    mk = info.get("marker", {}); al = info.get("align", {})
                    mstat["found"] += int(mk.get("found", 0) > 0)
                    mstat["n_blob"] += int(mk.get("found", 0))
                    if mk.get("skipped"):
                        mstat["skipped"] += 1
                        mstat["reasons"][mk.get("reason", "?")] = \
                            mstat["reasons"].get(mk.get("reason", "?"), 0) + 1
                    if al.get("skipped"):
                        astat["skipped"] += 1
                    else:
                        astat["rots"].append(abs(al.get("rot", 0.0)))
                        astat["flip"] += int(al.get("flip", False))
                    made += 1
            if i % 2000 == 0:
                log(f"  hand560 {sp} {i}/{len(df)}")
        log(f"[hand560:{sp}] 생성 {made} | 스킵 {skipped} | 실패 {failed}")
    if ks:
        n_tot = len(ks)
        log(f"  Top-Hat 커널 중앙값 {int(np.median(ks))}px "
            f"(뼈 굵기 84~88px 보다 커야 정상) | 패딩 비율 중앙값 {np.median(pfs):.1%}")
        log(f"  [마커] 제거된 이미지 {mstat['found']}/{n_tot} "
            f"({mstat['found']/max(n_tot,1):.1%}) | 제거 덩어리 총 {mstat['n_blob']}개 "
            f"| 안전상 건너뜀 {mstat['skipped']}장 {mstat['reasons']}")
        log(f"       ↳ 마커가 없는 크롭은 '제거 0'이 정상입니다. 비율이 비정상적으로"
            f" 높으면 MARKER_* 조건을 조이세요.")
        if astat["rots"]:
            log(f"  [정렬] 회전 중앙값 {np.median(astat['rots']):.1f}° "
                f"| 최대 {np.max(astat['rots']):.1f}° | 180° 뒤집기 {astat['flip']}장 "
                f"| 건너뜀 {astat['skipped']}장")
    json.dump({"pre": PRE_PARAMS, "train": len(train_df),
               "val": len(val_df), "test": len(test_df)},
              open(BASE_DONE, "w", encoding="utf-8"))


if SCALE_MODE == "global":
    REF_LONG, _ri = scan_ref_long([SPLIT_DFS[s] for s in SPLITS if len(SPLIT_DFS[s])])
    log(f"SCALE_MODE=global | 기준 긴 변 {REF_LONG}px ({_ri})")

if hand_cache_valid():
    log("hand560 캐시 유효 - 스킵")
else:
    log(f"hand560 캐시 생성: 정규화({NORM_MODE}) -> TopHat({TOPHAT_MODE},{TOPHAT_SHAPE},"
        f"{'k'+str(TOPHAT_KSIZE) if TOPHAT_KSIZE>0 else 'frac'+str(TOPHAT_KFRAC)}) -> "
        f"CLAHE({CLAHE_CLIP},{CLAHE_TILE}) -> scale({SCALE_MODE}) -> pad({PAD_MODE}) -> {REG_SIZE}px")
    build_hand_cache()
    log("hand560 캐시 완료")


def filter_cached(df, sub, sp):
    if not len(df):
        return df
    ok = df["id"].apply(lambda i: (CACHE_DIR / sub / sp / f"{i}.png").exists())
    return df[ok].reset_index(drop=True)


for _sp in SPLITS:
    SPLIT_DFS[_sp] = filter_cached(SPLIT_DFS[_sp], "hand560", _sp)
train_df, val_df, test_df = SPLIT_DFS["train"], SPLIT_DFS["val"], SPLIT_DFS["test"]
log(f"캐시 확인 후 | train {len(train_df):,} | val {len(val_df):,} | test {len(test_df):,}")


# =========================================================================
# [E] Phase I : 분류 모델 (InceptionV3 + soft label) - 논문 §III-A
#     식(1)~(3): Y_t = (1/HW) sum_ij sum_k W_kt F_ijk
#       -> 마지막 FC 는 반드시 '선형'이어야 CAM 수식이 성립 (활성함수 없음)
#     식(5)   : Y_i = max(0, 1 - |i-t|/l), l=50
#               원-핫으로는 수렴하지 않는다고 논문이 명시
# =========================================================================
def make_inception_backbone(pretrained=True):
    last_err = None
    for name in ("inception_v3", "tf_inception_v3"):
        try:
            m = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="")
            log(f"[backbone] timm '{name}' 로드 완료 (pretrained={pretrained})")
            return m
        except Exception as e:
            last_err = e
    raise RuntimeError(f"InceptionV3 백본 생성 실패: {last_err}")


class CAMClassifier(nn.Module):
    """CAM 추출용 분류 헤드.
       pool='gmp' -> 작고 뾰족한 영역(R1/R2)에 적합 (논문 사용)
       pool='gap' -> 넓은 영역용 (v2 에서는 손을 이미 크롭했으므로 미사용)"""

    def __init__(self, pool="gmp", n_bins=AGE_BINS, size=CLS_SIZE_REGION, pretrained=True):
        super().__init__()
        self.backbone = make_inception_backbone(pretrained)
        self.pool_kind = pool
        with torch.no_grad():
            f = self.backbone(torch.zeros(1, 3, size, size))
        self.c = f.shape[1]
        self.fc = nn.Linear(self.c, n_bins)          # [논문] 활성함수 없음
        log(f"[cls-{pool}] feat {tuple(f.shape[1:])} -> FC {self.c}->{n_bins}")

    def features(self, x):
        return self.backbone(x)

    def forward(self, x):
        f = self.features(x)
        p = (F.adaptive_avg_pool2d(f, 1) if self.pool_kind == "gap"
             else F.adaptive_max_pool2d(f, 1))
        return self.fc(torch.flatten(p, 1))          # (B, 240) 선형 출력


def soft_labels(y_month, n_bins=AGE_BINS, l=SOFT_L):
    """식(5) 삼각형 소프트 라벨. (B,) -> (B, n_bins)"""
    k = torch.arange(1, n_bins + 1, device=y_month.device, dtype=torch.float32)
    d = (k[None, :] - y_month[:, None]).abs()
    return torch.clamp(1.0 - d / float(l), min=0.0)


class ClsDataset(Dataset):
    """Phase I 학습용. sub 로 hand560 / erased 중 어느 캐시를 쓸지 지정."""

    def __init__(self, df, split, sub, size=CLS_SIZE_REGION):
        self.df = df.reset_index(drop=True)
        self.split, self.sub, self.size = split, sub, size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        g = imread_kr(CACHE_DIR / self.sub / self.split / f"{r['id']}.png", cv2.IMREAD_GRAYSCALE)
        if g is None:
            g = np.zeros((self.size, self.size), np.uint8)
        if g.shape[0] != self.size or g.shape[1] != self.size:
            g = cv2.resize(g, (self.size, self.size), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(np.stack([g, g, g], 0)).float() / 255.0     # [논문] x/255
        return x, torch.tensor(float(r["boneage"]), dtype=torch.float32)


# -- Phase I 하이퍼파라미터 (학습 루프 바로 앞) --------------------------------
CLS_BATCH   = 8       # [논문] 32. 8GB GPU + 560 입력이면 8 권장 (OOM 시 4)
CLS_EPOCHS  = 70      # [논문] 70 (3e-4 x 50 epoch -> 1e-4 x 20 epoch)
CLS_LR1, CLS_LR2 = 3e-4, 1e-4
CLS_SWITCH  = 50      # [논문] lr 전환 시점
CLS_WORKERS = 0       # 윈도우 안전값 (리눅스면 4 이상)
CLS_LOG_EVERY = 100

# ★ 조기 종료 (논문에는 없음 - 시간 절약용 엔지니어링 추가) --------------------
CLS_EARLY_STOP_PATIENCE = 8    # val argmax-MAE 가 이 에폭 수만큼 개선 없으면 중단 (<0 이면 끔)
CLS_MIN_DELTA           = 0.05 # 개선으로 인정할 최소 감소량(개월)
CLS_MIN_EPOCHS          = 20   # 최소 이 에폭까지는 조기종료하지 않음(초반 요동 방지)

# 빠른 점검용 프리셋 - 파이프라인이 끝까지 도는지만 확인할 때
CLS_QUICK = False
if CLS_QUICK:
    CLS_EPOCHS, CLS_SWITCH, CLS_MIN_EPOCHS, CLS_EARLY_STOP_PATIENCE = 12, 8, 3, 3
    log("[주의] CLS_QUICK=True - 논문 스케줄이 아닙니다. RoI 품질 저하 가능")


def train_cam_classifier(tag, best_path, last_path, pool, sub, epochs=CLS_EPOCHS):
    """Phase I 분류 모델 학습.
       - best_path 가 있고 학습이 이미 끝났으면(_DONE 플래그) 로드만 하고 스킵
       - last_path 가 있으면 자동으로 이어서 학습
       - 조기 종료 지원
    """
    done_flag = best_path.with_suffix(".done")
    if best_path.exists() and done_flag.exists():
        log(f"[{tag}] 학습 완료 상태 - 로드만 수행 ({best_path.name})")
        ck = torch_load(best_path, map_location=device)
        m = CAMClassifier(pool=pool, pretrained=False).to(device)
        m.load_state_dict(ck["model"]); m.eval()
        return m

    log(f"[{tag}] 분류 모델 학습 | pool={pool} src={sub} size={CLS_SIZE_REGION} epochs={epochs}")
    model = CAMClassifier(pool=pool, pretrained=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=CLS_LR1, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    crit = nn.L1Loss()                          # [논문] soft label 에 대한 MAE

    start_ep, best, no_improve, best_ep = 1, float("inf"), 0, 1
    if last_path.exists():
        try:
            ck = torch_load(last_path, map_location=device)
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
            scaler.load_state_dict(ck["scaler"])
            start_ep = ck["epoch"] + 1; best = ck["best"]
            no_improve = ck.get("no_improve", 0); best_ep = ck.get("best_epoch", 1)
            log(f"[{tag}] 이어서 학습: epoch {start_ep} 부터 (best {best:.2f}, 정체 {no_improve})")
        except Exception as e:
            log(f"[{tag}] [경고] last 로드 실패({e}) - 새로 학습")

    tr_loader = DataLoader(ClsDataset(train_df, "train", sub), batch_size=CLS_BATCH,
                           shuffle=True, num_workers=CLS_WORKERS, pin_memory=True, drop_last=True)
    va_loader = DataLoader(ClsDataset(val_df, "val", sub), batch_size=CLS_BATCH,
                           shuffle=False, num_workers=CLS_WORKERS, pin_memory=True)
    n_b = len(tr_loader)
    ep = start_ep - 1
    try:
        for ep in range(start_ep, epochs + 1):
            lr = CLS_LR1 if ep <= CLS_SWITCH else CLS_LR2      # [논문] 2단 스텝
            for pg in opt.param_groups:
                pg["lr"] = lr
            model.train(); run, seen, t0 = 0.0, 0, time.time()
            for step, (x, y) in enumerate(tr_loader, 1):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                tgt = soft_labels(y)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    out = model(x)
                loss = crit(out.float(), tgt)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                run += float(loss) * x.size(0); seen += x.size(0)
                if step % CLS_LOG_EVERY == 0 or step == n_b:
                    log(f"  [{tag}] ep{ep:02d} {step}/{n_b} loss {run/seen:.4f} "
                        f"lr {lr:.0e} ({time.time()-t0:.0f}s)")

            # 검증: argmax 를 나이로 본 대략적 MAE (모니터링/조기종료 기준)
            model.eval(); errs = []
            with torch.no_grad():
                for x, y in va_loader:
                    x = x.to(device)
                    with torch.amp.autocast("cuda", enabled=USE_AMP):
                        out = model(x)
                    errs.append(np.abs(out.float().argmax(1).cpu().numpy() + 1 - y.numpy()))
            vmae = float(np.concatenate(errs).mean()) if errs else float("nan")

            if vmae < best - CLS_MIN_DELTA:
                best, no_improve, best_ep = vmae, 0, ep
                torch.save({"model": model.state_dict(), "pool": pool,
                            "size": CLS_SIZE_REGION, "val_argmax_mae": best, "epoch": ep},
                           best_path)
                flag = f"* best 저장 ({best:.2f})"
            else:
                no_improve += 1
                flag = f"개선없음 {no_improve}/{CLS_EARLY_STOP_PATIENCE}"
            log(f"  [{tag}] Epoch {ep:02d} 완료 | train L1 {run/seen:.4f} "
                f"| val argmax-MAE {vmae:.2f} | {flag}")

            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "scaler": scaler.state_dict(), "epoch": ep, "best": best,
                        "no_improve": no_improve, "best_epoch": best_ep}, last_path)

            if (CLS_EARLY_STOP_PATIENCE >= 0 and ep >= CLS_MIN_EPOCHS
                    and no_improve >= CLS_EARLY_STOP_PATIENCE):
                log(f"  [{tag}] 조기종료 | best argmax-MAE {best:.2f} @ epoch {best_ep}")
                break
        else:
            log(f"[{tag}] 전체 {epochs}에폭 완료 | best {best:.2f} @ epoch {best_ep}")
    except KeyboardInterrupt:
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "best": best,
                    "no_improve": no_improve, "best_epoch": best_ep}, last_path)
        log(f"[{tag}] 중단됨 - last 저장(epoch {ep}). 다시 실행하면 이어서 학습합니다.")
        raise

    done_flag.write_text(json.dumps({"best": best, "best_epoch": best_ep}), encoding="utf-8")
    ck = torch_load(best_path, map_location=device)
    model.load_state_dict(ck["model"]); model.eval()
    return model


# =========================================================================
# [F] CAM -> 마스크 -> bbox -> 손 크롭 원본에서 고해상도 재크롭  (식 2 / 4)
#
#  [공식코드 참고] func_utils.GAPAttention()
#     · 클래스 인덱스 t = argmax(prediction)
#     · 가중치는 W[:, t-5:t+5] 의 '평균' -> 인접 나이 채널 노이즈 완화
#     · heatmap = heatmap/max ; uint8(255*heatmap)
#       -> 논문의 tau(10~100)는 0~255 축 위의 값
#  [원코드 결함 -> 수정] 음수 값에서 uint8 언더플로 래핑 발생 -> clip(0,None) 후 정규화
#  [논문 미명시 -> 추론] 마스크가 여러 조각일 때 -> '최대 연결성분의 bbox'
# =========================================================================
@torch.no_grad()
def compute_cam(model, x, span=5):
    """x: (1,3,H,W) -> (cam(h,w) float>=0, 예측 나이)"""
    feat = model.features(x).float()
    p = (F.adaptive_avg_pool2d(feat, 1) if model.pool_kind == "gap"
         else F.adaptive_max_pool2d(feat, 1))
    logits = model.fc(torch.flatten(p, 1))
    t = int(logits.argmax(1).item())
    W = model.fc.weight                                     # (240, C)
    lo, hi = max(0, t - span), min(W.size(0), t + span)
    w = W[lo:hi].mean(0)
    cam = (feat[0] * w[:, None, None]).sum(0)               # 식(2)
    return torch.clamp(cam, min=0).cpu().numpy(), t + 1


def _enlarge_box(x0, y0, x1, y1, out_w, out_h, cfg):
    """박스를 (1) 방향별 여백 확장 -> (2) 최소 크기 보장(중심 유지) 순으로 키운다.
       cfg = {"pad_x","pad_y","min_w","min_h"} (min_* 는 이미지 크기 대비 비율)."""
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    # (1) 방향별 여백
    px, py = w * cfg["pad_x"], h * cfg["pad_y"]
    x0, x1 = x0 - px, x1 + px
    y0, y1 = y0 - py, y1 + py
    w, h = x1 - x0, y1 - y0
    # (2) 최소 크기 (중심 유지하며 필요한 만큼만 확장)
    min_w, min_h = cfg["min_w"] * out_w, cfg["min_h"] * out_h
    if w < min_w:
        x0, x1 = cx - min_w / 2.0, cx + min_w / 2.0
    if h < min_h:
        y0, y1 = cy - min_h / 2.0, cy + min_h / 2.0
    # 경계 클램프
    x0, y0 = int(max(0, round(x0))), int(max(0, round(y0)))
    x1, y1 = int(min(out_w, round(x1))), int(min(out_h, round(y1)))
    return x0, y0, x1, y1


def cam_to_bbox(cam, tau, out_w, out_h, box_cfg=None, merge=None,
                region=None, clamp_region=None, fallback_box=None):
    """CAM -> (x0,y0,x1,y1). 실패 시 전체 영역 반환.
       merge=True 면 tau 이상 화소 전체의 외접 박스(조각 병합).
       box_cfg 가 주어지면 방향별 여백 + 최소 크기로 확대(첨부 그림처럼 넓게)."""
    if merge is None:
        merge = MERGE_COMPONENTS
    m = cv2.resize(cam.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    # region(밴드)이 주어지면 그 밖은 0 으로 만들고 밴드 안 최대값으로 정규화
    if region is not None:
        rx0, ry0, rx1, ry1 = region
        masked = np.zeros_like(m)
        masked[ry0:ry1, rx0:rx1] = m[ry0:ry1, rx0:rx1]
        m = masked
    _fb = fallback_box or (0, 0, out_w, out_h)
    mx = float(m.max())
    if mx <= 1e-8:
        return _fb, False
    heat = np.uint8(255.0 * m / mx)
    mask = (heat >= tau).astype(np.uint8)                   # 식(4)
    if mask.sum() == 0:
        return _fb, False

    if merge:
        # tau 이상 화소 '전체'의 외접 박스 -> 조각나도 넓게 감쌈
        ys, xs = np.where(mask > 0)
        x, y = int(xs.min()), int(ys.min())
        w, h = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    else:
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n <= 1:
            return _fb, False
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                      int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))

    x0, y0, x1, y1 = x, y, x + w, y + h
    if box_cfg is not None:
        x0, y0, x1, y1 = _enlarge_box(x0, y0, x1, y1, out_w, out_h, box_cfg)
    else:
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(out_w, x1), min(out_h, y1)
    if clamp_region is not None:
        # 밴드 밖으로 나가면 '잘라내지 말고' 밴드 안으로 밀어넣는다.
        # (그냥 자르면 경계에 걸린 박스가 납작해짐)
        cx0, cy0, cx1, cy1 = clamp_region
        bw, bh = cx1 - cx0, cy1 - cy0
        w, h = min(x1 - x0, bw), min(y1 - y0, bh)
        if x0 < cx0: x0 = cx0
        if y0 < cy0: y0 = cy0
        x1, y1 = x0 + w, y0 + h
        if x1 > cx1: x0, x1 = cx1 - w, cx1
        if y1 > cy1: y0, y1 = cy1 - h, cy1
        x0, y0 = max(cx0, x0), max(cy0, y0)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return _fb, False
    return (x0, y0, x1, y1), True


def band_abs(kind, W, H):
    """kind('r1'|'r2') 의 고정 밴드를 절대 좌표로. 서로 독립(겹쳐도 무방)."""
    b = BAND_CFG[kind]
    return (int(b["x0"] * W), int(b["y0"] * H),
            int(b["x1"] * W), int(b["y1"] * H))


def crop_roi(enh_full, box_canvas, info, size=REG_SIZE):
    """★ v5: 캔버스 박스를 '원본 좌표'로 역변환한 뒤, 전처리된 원본 해상도
       이미지에서 잘라 fit_canvas 로 정사각화(비율유지 리사이즈 + 배경색 패딩).
         enh_full   : enhance() 를 거친 원본 해상도 이미지
         box_canvas : 560 캔버스 좌표계의 박스
         info       : preprocess/fit_canvas 가 준 {scale, pad, src}
       레터박스 패딩 때문에 단순 비례로는 좌표가 어긋나므로 반드시 역변환합니다."""
    X0, Y0, X1, Y1 = canvas_box_to_src(box_canvas, info)
    patch = enh_full[Y0:Y1, X0:X1]
    if patch.size == 0:
        patch = enh_full
    if PAD_TO_SQUARE:
        out, _ = fit_canvas(patch, size)          # 비율 유지 + 배경색 패딩
        return out
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)


def roi_cache_valid():
    if REBUILD_ROI or not ROI_DONE.exists():
        return False
    try:
        info = json.load(open(ROI_DONE, encoding="utf-8"))
    except Exception:
        return False
    return (info.get("tau_r1") == TAU_R1 and info.get("tau_r2") == TAU_R2_PAPER
            and info.get("r2_method") == "paper_erase"
            and info.get("box_cfg_r2") == BOX_CFG_R2 and info.get("r2_merge") == R2_MERGE
            and info.get("reg_size") == REG_SIZE and info.get("has_r2", False)
            and info.get("box_cfg") == BOX_CFG and info.get("merge") == MERGE_COMPONENTS
            and info.get("bands") == BAND_CFG
            and info.get("clamp") == CLAMP_TO_BAND
            and info.get("pad_square") == PAD_TO_SQUARE
            and info.get("train") == len(train_df) and info.get("val") == len(val_df)
            and info.get("test") == len(test_df))


@torch.no_grad()
def localize_r1_and_erase(model, records):
    """[Phase I-a] R1(수근골) + E(R1 지운 이미지) 생성.
       R1 은 v5 와 동일하게 '밴드 안 CAM' 방식 -> QC 에서 확인한 박스 크기 유지.
       E 는 논문 Fig.3(f): R1 픽셀을 랜덤값으로 대체한 '전체 손 이미지'."""
    model.eval()
    ok_all = fb_all = 0
    for sp in SPLITS:
        df = SPLIT_DFS[sp]
        if not len(df):
            continue
        ok = fb = miss = 0
        for i, (_, r) in enumerate(df.iterrows(), 1):
            dst_r1 = CACHE_DIR / "r1" / sp / f"{r['id']}.png"
            dst_e  = CACHE_DIR / "erased" / sp / f"{r['id']}.png"
            if dst_r1.exists() and dst_e.exists() and not REBUILD_ROI:
                continue
            raw = imread_kr(r["hand_path"], cv2.IMREAD_GRAYSCALE)
            if raw is None:
                miss += 1
                continue
            enh_full = enhance(raw)                       # ①정규화 ②TopHat ③CLAHE
            ref, info = fit_canvas(enh_full, REG_SIZE)    # ④배율 ⑤패딩

            x = torch.from_numpy(np.stack([ref] * 3, 0)).float().div(255.).unsqueeze(0).to(device)
            cam, _ = compute_cam(model, x)
            band = band_abs("r1", REG_SIZE, REG_SIZE)
            box, good = cam_to_bbox(cam, TAU_R1, REG_SIZE, REG_SIZE,
                                    box_cfg=BOX_CFG.get("r1"), region=band,
                                    clamp_region=band if CLAMP_TO_BAND else None,
                                    fallback_box=band)
            ok += int(good); fb += int(not good)
            imwrite_kr(dst_r1, crop_roi(enh_full, box, info, REG_SIZE))

            # E: R1 을 랜덤값으로 덮음 -> Phase I-b 의 학습 입력
            e = ref.copy()
            x0, y0, x1, y1 = box
            if ERASE_FILL == "noise":
                e[y0:y1, x0:x1] = np.random.randint(0, 256, (y1 - y0, x1 - x0), dtype=np.uint8)
            else:
                e[y0:y1, x0:x1] = int(ref.mean())
            imwrite_kr(dst_e, e)

            sx0, sy0, sx1, sy1 = canvas_box_to_src(box, info)
            records.append({"split": sp, "id": r["id"], "kind": "r1",
                            "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3],
                            "w": box[2] - box[0], "h": box[3] - box[1],
                            "src_x0": sx0, "src_y0": sy0, "src_x1": sx1, "src_y1": sy1,
                            "ref_size": REG_SIZE, "found": int(good)})
            if i % 1000 == 0:
                log(f"  [r1] {sp} {i}/{len(df)}")
        log(f"[r1:{sp}] CAM {ok} | 밴드폴백 {fb} | 파일없음 {miss}")
        ok_all += ok; fb_all += fb
    return ok_all, fb_all


@torch.no_grad()
def localize_r2_paper(model, records):
    """[Phase I-b] ★ 논문 방식 R2(중수골).
       E(R1 지운 이미지)로 재학습한 분류모델의 CAM 을 그대로 사용합니다.
       밴드 제약 없음 / 박스 확대 없음 / tau=50 / 최대 연결성분 (논문 Eq.4)."""
    model.eval()
    for sp in SPLITS:
        df = SPLIT_DFS[sp]
        if not len(df):
            continue
        ok = fb = miss = 0
        for i, (_, r) in enumerate(df.iterrows(), 1):
            dst = CACHE_DIR / "r2" / sp / f"{r['id']}.png"
            if dst.exists() and not REBUILD_ROI:
                continue
            e = imread_kr(CACHE_DIR / "erased" / sp / f"{r['id']}.png", cv2.IMREAD_GRAYSCALE)
            raw = imread_kr(r["hand_path"], cv2.IMREAD_GRAYSCALE)
            if e is None or raw is None:
                miss += 1
                continue
            enh_full = enhance(raw)
            _ref, info = fit_canvas(enh_full, REG_SIZE)

            # CAM 은 'R1 이 지워진 이미지'에서 계산 -> 그 외 영역에 근거하게 됨
            x = torch.from_numpy(np.stack([e] * 3, 0)).float().div(255.).unsqueeze(0).to(device)
            cam, _ = compute_cam(model, x)
            box, good = cam_to_bbox(cam, TAU_R2_PAPER, REG_SIZE, REG_SIZE,
                                    box_cfg=BOX_CFG_R2, merge=R2_MERGE)
            ok += int(good); fb += int(not good)
            # 크롭은 '전처리된 원본'에서 (지워지지 않은 실제 조직)
            imwrite_kr(dst, crop_roi(enh_full, box, info, REG_SIZE))

            sx0, sy0, sx1, sy1 = canvas_box_to_src(box, info)
            records.append({"split": sp, "id": r["id"], "kind": "r2",
                            "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3],
                            "w": box[2] - box[0], "h": box[3] - box[1],
                            "src_x0": sx0, "src_y0": sy0, "src_x1": sx1, "src_y1": sy1,
                            "ref_size": REG_SIZE, "found": int(good)})
            if i % 1000 == 0:
                log(f"  [r2] {sp} {i}/{len(df)}")
        log(f"[r2:{sp}] CAM {ok} | 마스크실패 {fb} | 파일없음 {miss}")


if roi_cache_valid():
    log("R1/R2/E 캐시 유효 - Phase I 전체 스킵 (다시 만들려면 --rebuild-roi)")
else:
    bbox_records = []

    # === Phase I-a : R1(수근골) + E ==========================================
    #     R1 은 v5 와 동일 (밴드 + 박스 확대) -> QC 에서 맞춘 크기 유지
    log("### Phase I-a : R1(수근골) 국소화 + Erased 생성 ###")
    log(f"  R1 밴드 {R1_BAND} | clamp {CLAMP_TO_BAND} | tau {TAU_R1}")
    log(f"  크기 맞추기: {'비율유지 + 배경색 패딩(왜곡 0)' if PAD_TO_SQUARE else '정사각 강제 리사이즈'}")
    m_r1 = train_cam_classifier("cls-r1", CLS_R1_BEST, CLS_R1_LAST, "gmp", "hand560")
    localize_r1_and_erase(m_r1, bbox_records)
    del m_r1; torch.cuda.empty_cache()

    # === Phase I-b : R2(중수골) - ★ 논문 방식 ================================
    #     [논문] R1 이 지워진 이미지(E)로 분류모델을 '다시 학습' -> 그 CAM 이 R2
    log("### Phase I-b : R2(중수골) 국소화 - 논문 방식(erase 재학습) ###")
    log(f"  E 이미지로 cls-r2 재학습 -> CAM tau={TAU_R2_PAPER} "
        f"| 밴드 제약 없음 | 박스 확대 {'없음' if BOX_CFG_R2 is None else BOX_CFG_R2} "
        f"| 최대연결성분 {not R2_MERGE}")
    m_r2 = train_cam_classifier("cls-r2", CLS_R2_BEST, CLS_R2_LAST, "gmp", "erased")
    localize_r2_paper(m_r2, bbox_records)
    del m_r2; torch.cuda.empty_cache()

    if bbox_records:
        pd.DataFrame(bbox_records).to_csv(BBOX_CSV, index=False, encoding="utf-8-sig")
        log(f"bbox 로그 저장: {BBOX_CSV}")
        try:
            bdf = pd.DataFrame(bbox_records)
            for k in ("r1", "r2"):
                sub = bdf[bdf["kind"] == k]
                if len(sub):
                    log(f"  [{k}] 박스 폭 중앙값 {sub['w'].median():.0f}px "
                        f"({sub['w'].median()/REG_SIZE:.0%}) | "
                        f"높이 중앙값 {sub['h'].median():.0f}px "
                        f"({sub['h'].median()/REG_SIZE:.0%})")
        except Exception:
            pass
    json.dump({"tau_r1": TAU_R1, "tau_r2": TAU_R2_PAPER, "reg_size": REG_SIZE,
               "has_r2": True, "channels": list(AGG_CHANNELS),
               "box_cfg": BOX_CFG, "box_cfg_r2": BOX_CFG_R2,
               "merge": MERGE_COMPONENTS, "r2_merge": R2_MERGE,
               "bands": BAND_CFG, "clamp": CLAMP_TO_BAND,
               "pad_square": PAD_TO_SQUARE, "r2_method": "paper_erase",
               "train": len(train_df), "val": len(val_df), "test": len(test_df)},
              open(ROI_DONE, "w", encoding="utf-8"))
    log("R1/R2/E 캐시 완료")


# -- QC 시트: 밴드/박스 오버레이 + 크롭 결과 (preview 대체) --------------------
def _overlay_bands(ref_gray, cam_norm, boxes):
    """손 이미지 + CAM 히트맵 + 밴드(점선색) + 박스(진한색) 오버레이."""
    base = cv2.cvtColor(ref_gray, cv2.COLOR_GRAY2BGR)
    if cam_norm is not None:
        heat = cv2.applyColorMap(np.uint8(255 * cam_norm), cv2.COLORMAP_JET)
        base = cv2.addWeighted(base, 0.65, heat, 0.35, 0)
    H, W = ref_gray.shape[:2]
    colors = {"r1": (60, 60, 255), "r2": (255, 120, 0)}     # R1=빨강 R2=파랑(BGR)
    for kind in ("r1", "r2"):
        bx = band_abs(kind, W, H)
        cv2.rectangle(base, (bx[0], bx[1]), (bx[2], bx[3]), (150, 150, 150), 1)   # 밴드
        if kind in boxes:
            b = boxes[kind]
            cv2.rectangle(base, (b[0], b[1]), (b[2], b[3]), colors[kind], 3)      # 박스
    return cv2.cvtColor(base, cv2.COLOR_BGR2RGB)


def build_qc_sheet(n=N_QC, tag="qc_roi"):
    """QC 시트. 열: [원본] [전처리+R1밴드/박스] [r1] [E+R2박스] [r2]
       R1 은 밴드 방식, R2 는 논문 방식(E 로 학습한 cls-r2 의 CAM)."""
    if not len(val_df):
        return
    samp = val_df.sample(min(n, len(val_df)), random_state=SEED)

    def _load(pth):
        if not Path(pth).exists():
            return None
        try:
            ck = torch_load(pth, map_location=device)
            mm = CAMClassifier(pool="gmp", pretrained=False).to(device)
            mm.load_state_dict(ck["model"]); mm.eval()
            return mm
        except Exception as ex:
            log(f"[QC] 모델 로드 실패 {Path(pth).name}: {ex}")
            return None

    m1, m2 = _load(CLS_R1_BEST), _load(CLS_R2_BEST)
    cols = ["원본 | 마커·정렬 후", "전처리+R1밴드+박스", "r1 crop", "E + R2박스", "r2 crop"]
    fig, axes = plt.subplots(len(samp), len(cols),
                             figsize=(2.7 * len(cols), 2.7 * len(samp)))
    axes = np.atleast_2d(axes)
    for rr, (_, r) in enumerate(samp.iterrows()):
        raw = imread_kr(r["hand_path"], cv2.IMREAD_GRAYSCALE)
        ref = imread_kr(CACHE_DIR / "hand560" / "val" / f"{r['id']}.png", cv2.IMREAD_GRAYSCALE)
        e   = imread_kr(CACHE_DIR / "erased" / "val" / f"{r['id']}.png", cv2.IMREAD_GRAYSCALE)
        if raw is not None:
            # 마커/정렬 효과를 위아래로 붙여 한 칸에 표시
            try:
                mk_out, mk_i = suppress_markers(raw, debug=True)
                al_out, al_i = align_hand(normalize_intensity(mk_out), debug=True)
                th = max(raw.shape[0], 1)
                a = cv2.resize(_to_uint8(raw), (200, 260))
                b = cv2.resize(_to_uint8(al_out), (200, 260))
                axes[rr, 0].imshow(np.hstack([a, np.full((260, 6), 255, np.uint8), b]),
                                   cmap="gray")
                axes[rr, 0].set_title(f"{r['id']} | 원본 | 마커{mk_i['found']}"
                                      f"·회전{al_i['rot']:+.0f}°"
                                      f"{'·flip' if al_i['flip'] else ''}", fontsize=6)
            except Exception:
                axes[rr, 0].imshow(raw, cmap="gray")
                axes[rr, 0].set_title(f"{r['id']}", fontsize=7)

        # R1 (밴드 방식)
        b1, cn1 = None, None
        if ref is not None and m1 is not None:
            with torch.no_grad():
                x = torch.from_numpy(np.stack([ref] * 3, 0)).float().div(255.).unsqueeze(0).to(device)
                cam, _ = compute_cam(m1, x)
            cr = cv2.resize(cam.astype(np.float32), (REG_SIZE, REG_SIZE))
            cn1 = cr / max(float(cr.max()), 1e-8)
            band = band_abs("r1", REG_SIZE, REG_SIZE)
            b1, _ok = cam_to_bbox(cam, TAU_R1, REG_SIZE, REG_SIZE,
                                  box_cfg=BOX_CFG.get("r1"), region=band,
                                  clamp_region=band if CLAMP_TO_BAND else None,
                                  fallback_box=band)
        if ref is not None:
            axes[rr, 1].imshow(_overlay_bands(ref, cn1, {"r1": b1} if b1 else {}))
            axes[rr, 1].set_title(f"R1 {b1[2]-b1[0]}x{b1[3]-b1[1]}" if b1 else "전처리 후",
                                  fontsize=7)
        im = imread_kr(CACHE_DIR / "r1" / "val" / f"{r['id']}.png", cv2.IMREAD_GRAYSCALE)
        if im is not None:
            axes[rr, 2].imshow(im, cmap="gray")
        axes[rr, 2].set_title(cols[2], fontsize=7)

        # R2 (논문 방식: E 로 학습한 cls-r2 의 CAM, 밴드 없음)
        b2, cn2 = None, None
        if e is not None and m2 is not None:
            with torch.no_grad():
                x = torch.from_numpy(np.stack([e] * 3, 0)).float().div(255.).unsqueeze(0).to(device)
                cam2, _ = compute_cam(m2, x)
            cr2 = cv2.resize(cam2.astype(np.float32), (REG_SIZE, REG_SIZE))
            cn2 = cr2 / max(float(cr2.max()), 1e-8)
            b2, _ok = cam_to_bbox(cam2, TAU_R2_PAPER, REG_SIZE, REG_SIZE,
                                  box_cfg=BOX_CFG_R2, merge=R2_MERGE)
        if e is not None:
            base = cv2.cvtColor(e, cv2.COLOR_GRAY2BGR)
            if cn2 is not None:
                hm = cv2.applyColorMap(np.uint8(255 * cn2), cv2.COLORMAP_JET)
                base = cv2.addWeighted(base, 0.65, hm, 0.35, 0)
            if b2:
                cv2.rectangle(base, (b2[0], b2[1]), (b2[2], b2[3]), (255, 120, 0), 3)
            axes[rr, 3].imshow(cv2.cvtColor(base, cv2.COLOR_BGR2RGB))
            axes[rr, 3].set_title(f"R2 {b2[2]-b2[0]}x{b2[3]-b2[1]}" if b2 else cols[3], fontsize=7)
        im = imread_kr(CACHE_DIR / "r2" / "val" / f"{r['id']}.png", cv2.IMREAD_GRAYSCALE)
        if im is not None:
            axes[rr, 4].imshow(im, cmap="gray")
        axes[rr, 4].set_title(cols[4], fontsize=7)

    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    out = CKPT_DIR / f"{tag}.png"
    plt.savefig(out, dpi=110); plt.close()
    log(f"QC 시트 저장: {out}")
    log("  1열=원본|마커·정렬 후 / 2열=전처리+R1밴드(회색)+R1박스(빨강) / 4열=E+R2박스(파랑)")
    log("  ※ 1열 오른쪽에서 마커가 사라졌는지, 손이 세로로 섰는지 확인하세요.")
    for mm in (m1, m2):
        if mm is not None:
            del mm
    torch.cuda.empty_cache()


if N_QC > 0:
    try:
        build_qc_sheet()
    except Exception as e:
        log(f"[경고] QC 시트 생성 실패: {e}")

if QC_ONLY:
    log("--qc-only : QC 시트만 생성하고 종료합니다.")
    log("  밴드/박스가 마음에 들면 옵션 없이 다시 실행해 Phase II 로 진행하세요.")
    sys.exit(0)


# =========================================================================
# [G] Phase II : Xception + 성별 + LDL 기대값 회귀 - 논문 §III-B
#     Xception(no top) -> Conv2d(256,3x3) -> MaxPool(3x3) -> Flatten
#     gender(±1) -> Linear(32) -> concat -> Linear(240) -> softmax -> 기대값
# =========================================================================
# [v7] 약한 기하 증강만. 밝기/대비 증강은 넣지 않습니다(전처리 표준화를 흐트러뜨림).
_aug = [transforms.RandomRotation(AUG_ROT_DEG),
        transforms.RandomAffine(0, translate=(AUG_TRANSLATE, AUG_TRANSLATE),
                                scale=AUG_SCALE)]
_norm = ([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
         if NORMALIZE == "imagenet" else [transforms.ToTensor()])   # ToTensor 가 /255 수행

train_tf = transforms.Compose([transforms.ToPILImage()] + (_aug if USE_AUG else []) + _norm)
eval_tf  = transforms.Compose([transforms.ToPILImage()] + _norm)
log(f"Phase II 정규화 {NORMALIZE} | 증강 {'ON' if USE_AUG else 'OFF (논문 기준선)'}")

KIND2SUB = {"hand": "hand560", "r1": "r1", "r2": "r2", "erased": "erased"}


class AggDataset(Dataset):
    """서로 다른 국소 패치를 '입력 채널'에 하나씩 꽂는다 (공식 main_aggregation.py 방식)."""

    def __init__(self, df, split, tf, channels=AGG_CHANNELS):
        self.df = df.reset_index(drop=True)
        self.split, self.tf, self.ch = split, tf, list(channels)

    def __len__(self):
        return len(self.df)

    def _load(self, kind, iid):
        g = imread_kr(CACHE_DIR / KIND2SUB.get(kind, kind) / self.split / f"{iid}.png",
                      cv2.IMREAD_GRAYSCALE)
        if g is None:
            g = np.zeros((REG_SIZE, REG_SIZE), np.uint8)
        if g.shape[0] != REG_SIZE or g.shape[1] != REG_SIZE:
            g = cv2.resize(g, (REG_SIZE, REG_SIZE), interpolation=cv2.INTER_AREA)
        return g

    def __getitem__(self, i):
        r = self.df.iloc[i]
        chans = [self._load(k, r["id"]) for k in self.ch]
        while len(chans) < 3:
            chans.append(chans[0])
        x = self.tf(np.stack(chans[:3], -1))
        gender = torch.tensor([1.0 if r["male"] >= 0.5 else -1.0],   # [논문] 남 +1 / 여 -1
                              dtype=torch.float32)
        y_month = torch.tensor([float(r["boneage"])], dtype=torch.float32)
        return x, gender, y_month


def make_xception_backbone(pretrained=True):
    last_err = None
    for name in ("legacy_xception", "xception"):
        try:
            m = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="")
            log(f"[backbone] timm '{name}' 로드 완료 (pretrained={pretrained})")
            return m
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Xception 백본 생성 실패: {last_err}")


class AttentionLDLRegressor(nn.Module):
    def __init__(self, img_size=REG_SIZE, gender_dim=32, n_bins=AGE_BINS,
                 pretrained=True, head_relu=True, gender_relu=True):
        super().__init__()
        self.backbone = make_xception_backbone(pretrained)
        with torch.no_grad():
            f = self.backbone(torch.zeros(1, 3, img_size, img_size))
        c = f.shape[1]
        self.conv = nn.Conv2d(c, 256, kernel_size=3)        # [논문] Conv 256 3x3 (valid)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)   # [논문] MaxPool 3x3
        self.head_relu, self.gender_relu = head_relu, gender_relu
        with torch.no_grad():
            z = self.pool(self.conv(f))
        self.feat_dim = int(z.numel())
        self.gender = nn.Linear(1, gender_dim)              # [논문] Table III: 1->32
        self.fc = nn.Linear(self.feat_dim + gender_dim, n_bins)   # 식(6) 출력 240
        self.n_bins = n_bins
        log(f"[reg] backbone out {tuple(f.shape[1:])} -> z {self.feat_dim:,} "
            f"| params {sum(p.numel() for p in self.parameters())/1e6:.1f}M")

    def forward(self, x, g):
        z = self.conv(self.backbone(x))
        if self.head_relu:
            z = F.relu(z)
        z = torch.flatten(self.pool(z), 1)
        e = self.gender(g)
        if self.gender_relu:
            e = F.relu(e)
        return self.fc(torch.cat([z, e], 1))                # (B,240) 로짓


def build_reg_model(arch, pretrained=False):
    m = AttentionLDLRegressor(img_size=arch["IMG_SIZE"],
                              gender_dim=arch.get("GENDER_EMB_DIM", 32),
                              n_bins=arch.get("AGE_BINS", 240),
                              pretrained=pretrained,
                              head_relu=arch.get("HEAD_RELU", True),
                              gender_relu=arch.get("GENDER_RELU", True))
    return m.to(device)


# -------------------------------------------------------------------------
#  LDL + 기대값 회귀 - 식(7)~(12)
#    식(7)  p_k = softmax(z_k)
#    식(8)  yhat = sum_k k * p_k
#    식(9)  l_MAE = |y - yhat|
#    식(11) G_k = N(k ; y, delta^2), delta = 15 [논문]
#    식(10) l_reg = sum_k G_k (ln G_k - ln p_k) = D_KL(G||p)
#    식(12) l = l_MAE + lambda * l_reg, lambda in [0.1, 1] [논문 최적 구간]
#
#  [원문 표기 주의] 식(10) 좌변은 D_KL(p||G) 로 적혀 있으나 우변 전개식
#      -sum G ln(p/G) = sum G ln(G/p) 는 D_KL(G||p). 전개식을 그대로 구현.
#  [논문 미명시 -> 추론] G 는 240구간 합이 1이 아니므로(경계 나이에서 절반 잘림)
#      KL 성립을 위해 합=1 로 재정규화.
# -------------------------------------------------------------------------
def logits_to_months(logits):
    logits = logits.float()
    p = torch.softmax(logits, dim=1)                                  # 식(7)
    k = torch.arange(1, logits.size(1) + 1, device=logits.device, dtype=torch.float32)
    return (p * k).sum(dim=1), p                                      # 식(8)


class LDLExpectationLoss(nn.Module):
    def __init__(self, n_bins=AGE_BINS, delta=15.0, lam=0.5):
        super().__init__()
        self.delta, self.lam = float(delta), float(lam)
        self.register_buffer("k", torch.arange(1, n_bins + 1, dtype=torch.float32))

    def forward(self, logits, y_month):
        logits = logits.float()
        logp = F.log_softmax(logits, dim=1)
        yhat = (logp.exp() * self.k).sum(dim=1)                       # 식(8)
        l_mae = (yhat - y_month).abs().mean()                         # 식(9)
        G = torch.exp(-((self.k[None, :] - y_month[:, None]) ** 2) / (2.0 * self.delta ** 2))
        G = G / G.sum(dim=1, keepdim=True).clamp_min(1e-12)           # 재정규화
        l_reg = (G * (G.clamp_min(1e-12).log() - logp)).sum(dim=1).mean()   # 식(10)
        return l_mae + self.lam * l_reg, l_mae.detach(), l_reg.detach()


@torch.no_grad()
def predict_months(model, loader, use_amp=True):
    model.eval(); preds, trues = [], []
    for x, g, ym in loader:
        x, g = x.to(device), g.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(x, g)
        preds.append(logits_to_months(out)[0].cpu()); trues.append(ym.squeeze(1))
    if not preds:
        return np.array([]), np.array([])
    return torch.cat(preds).numpy(), torch.cat(trues).numpy()


def mae_rmse(preds, trues):
    err = preds - trues
    return float(np.abs(err).mean()), float(np.sqrt((err ** 2).mean()))


def save_checkpoint(path, model, arch, optimizer=None, scaler=None, epoch=None,
                    best_val=None, history=None, no_improve=None, best_epoch=None):
    ck = {"model": model.state_dict(), "arch": arch}
    if optimizer is not None:  ck["optimizer"] = optimizer.state_dict()
    if scaler is not None:     ck["scaler"] = scaler.state_dict()
    if epoch is not None:      ck["epoch"] = epoch
    if best_val is not None:   ck["best_val"] = best_val
    if history is not None:    ck["history"] = history
    if no_improve is not None: ck["no_improve"] = no_improve
    if best_epoch is not None: ck["best_epoch"] = best_epoch
    torch.save(ck, path)


# =========================================================================
# [H] Phase II 학습 설정 (하이퍼파라미터는 학습 루프 '바로 앞')
# =========================================================================
# -- 하이퍼파라미터 -----------------------------------------------------------
BATCH_SIZE   = 8       # [논문] 16. 8GB GPU + 560 입력이면 8 권장 (OOM 시 4)
EPOCHS       = 120     # [논문] 120 epoch
LR_STEPS     = [(60, 3e-4), (90, 1e-4), (120, 1e-5)]   # [논문] (누적epoch, lr)
WEIGHT_DECAY = 0.0     # [논문] 미언급 -> 0
NUM_WORKERS  = 0       # 윈도우 안전값 (리눅스면 4 이상)
AUTO_RESUME  = True
LOG_EVERY    = 100

# ★ 조기 종료 (논문에는 없음 - 엔지니어링 추가) --------------------------------
EARLY_STOP_PATIENCE = 15   # val MAE 가 이 에폭 수만큼 개선 없으면 중단 (<0 이면 끔)
MIN_DELTA           = 0.01 # 개선으로 인정할 최소 MAE 감소(개월)
MIN_EPOCHS          = 65   # [권장] lr 1차 강하(60ep) 이후부터 조기종료 판단
                           #        그 전에 멈추면 논문 스케줄의 이득을 못 봅니다.

# -- 모델 구조 ----------------------------------------------------------------
GENDER_EMB_DIM = 32    # [논문] Table III 최적
HEAD_RELU      = True  # [추론] Conv3x3 뒤 활성함수
GENDER_RELU    = True  # [추론] 성별 Dense 뒤 활성함수

# -- LDL ----------------------------------------------------------------------
LDL_DELTA  = 15.0      # [논문] 식(11) delta=15
LDL_LAMBDA = 0.5       # [논문] 최적 구간 lambda in [0.1, 1] 의 중앙값

ARCH = {"BACKBONE": "xception", "IMG_SIZE": REG_SIZE, "AGE_BINS": AGE_BINS,
        "GENDER_EMB_DIM": GENDER_EMB_DIM, "HEAD_RELU": HEAD_RELU,
        "GENDER_RELU": GENDER_RELU, "LDL_DELTA": LDL_DELTA, "LDL_LAMBDA": LDL_LAMBDA,
        "AGG_CHANNELS": list(AGG_CHANNELS), "NORMALIZE": NORMALIZE, "USE_AUG": USE_AUG,
        "TAU_R1": TAU_R1, "TAU_R2": TAU_R2_PAPER, "R2_METHOD": "paper_erase",
        "CLS_SIZE_REGION": CLS_SIZE_REGION,
        "PRE": PRE_PARAMS, "PRE_TAG": PRE_TAG, "HAND_SOURCE": str(HAND_CROP_DIR),
        "BOX_CFG": BOX_CFG, "MERGE_COMPONENTS": MERGE_COMPONENTS,
        "BANDS": BAND_CFG, "CLAMP_TO_BAND": CLAMP_TO_BAND,
        "PAD_TO_SQUARE": PAD_TO_SQUARE}
log(f"Phase II 설정: {ARCH}")
log(f"조기종료 | Phase I patience {CLS_EARLY_STOP_PATIENCE} "
    f"| Phase II patience {EARLY_STOP_PATIENCE} (최소 {MIN_EPOCHS}에폭 이후)")

train_loader = DataLoader(AggDataset(train_df, "train", train_tf), batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
train_eval_loader = DataLoader(AggDataset(train_df, "train", eval_tf), batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(AggDataset(val_df, "val", eval_tf), batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = (DataLoader(AggDataset(test_df, "test", eval_tf), batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
               if len(test_df) else None)
log(f"배치 수 | train {len(train_loader)} | val {len(val_loader)} "
    f"| test {len(test_loader) if test_loader else 0}")


def lr_at(epoch):
    """[논문] 3단 스텝: 3e-4(1~60) -> 1e-4(61~90) -> 1e-5(91~120)"""
    for upto, lr in LR_STEPS:
        if epoch <= upto:
            return lr
    return LR_STEPS[-1][1]


if EVAL_ONLY:
    log("--eval-only : Phase II 학습을 건너뛰고 best.pt 로 평가만 진행합니다.")
else:
    model = build_reg_model(ARCH, pretrained=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_at(1),
                                 betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    criterion = LDLExpectationLoss(AGE_BINS, LDL_DELTA, LDL_LAMBDA).to(device)
    log(f"[LDL] loss = l_MAE + {LDL_LAMBDA} * D_KL(G||p) | bins {AGE_BINS} | delta {LDL_DELTA}")

    start_epoch, best_val, epochs_no_improve, best_epoch = 1, float("inf"), 0, 1
    history = {"train_mae": [], "val_mae": [], "val_rmse": [], "kl": []}
    if AUTO_RESUME and LAST_CKPT.exists():
        ck = torch_load(LAST_CKPT, map_location=device)
        prev = ck.get("arch", {})
        bad = [k for k in ("IMG_SIZE", "AGE_BINS", "GENDER_EMB_DIM", "AGG_CHANNELS")
               if prev.get(k) != ARCH.get(k)]
        if bad:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            LAST_CKPT.rename(LAST_CKPT.with_name(f"last_stale_{ts}.pt"))
            log(f"[경고] 체크포인트 구조 불일치 {bad} -> last.pt 보관 후 새로 학습")
        else:
            model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
            scaler.load_state_dict(ck["scaler"])
            start_epoch = ck["epoch"] + 1; best_val = ck["best_val"]
            history = ck["history"]; epochs_no_improve = ck.get("no_improve", 0)
            best_epoch = ck.get("best_epoch", 1); history.setdefault("kl", [])
            log(f"이어서 학습: epoch {start_epoch} 부터 (best val MAE {best_val:.2f}, "
                f"정체 {epochs_no_improve})")
            log("  * 처음부터 다시 하려면 checkpoints_attention_v7 의 last.pt 삭제 후 실행")
    else:
        log("Phase II 새로 학습 시작")

    epoch = start_epoch - 1
    n_batches = len(train_loader)
    try:
        for epoch in range(start_epoch, EPOCHS + 1):
            lr = lr_at(epoch)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            model.train()
            run_abs, seen, run_kl, run_nb, t0 = 0.0, 0, 0.0, 0, time.time()
            for step, (x, g, ym) in enumerate(train_loader, 1):
                x, g = x.to(device, non_blocking=True), g.to(device, non_blocking=True)
                y = ym.to(device).squeeze(1)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    out = model(x, g)
                # 손실은 autocast 밖 fp32 (softmax/KL 수치 안정성)
                loss, l_mae, l_reg = criterion(out, y)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                run_abs += float(l_mae) * x.size(0); seen += x.size(0)
                run_kl += float(l_reg); run_nb += 1
                if step % LOG_EVERY == 0 or step == n_batches:
                    log(f"  Epoch {epoch:03d} {step}/{n_batches} train_mae {run_abs/seen:.2f} "
                        f"kl {run_kl/max(run_nb,1):.3f} lr {lr:.0e} ({time.time()-t0:.0f}s)")

            tr_mae = run_abs / max(seen, 1)
            vp, vt = predict_months(model, val_loader, USE_AMP)
            va_mae, va_rmse = mae_rmse(vp, vt)
            history["train_mae"].append(tr_mae)
            history["val_mae"].append(va_mae)
            history["val_rmse"].append(va_rmse)
            history["kl"].append(run_kl / max(run_nb, 1))

            if va_mae < best_val - MIN_DELTA:
                best_val, epochs_no_improve, best_epoch = va_mae, 0, epoch
                save_checkpoint(BEST_CKPT, model, ARCH, best_val=best_val, best_epoch=epoch)
                flag = f"* best 저장 (val MAE {best_val:.2f})"
            else:
                epochs_no_improve += 1
                flag = f"개선없음 {epochs_no_improve}/{EARLY_STOP_PATIENCE}"
            log(f"[Epoch {epoch:03d} 완료] train MAE {tr_mae:.2f} | val MAE {va_mae:.2f} "
                f"| val RMSE {va_rmse:.2f} | lr {lr:.0e} | {flag}")

            save_checkpoint(LAST_CKPT, model, ARCH, optimizer, scaler, epoch,
                            best_val, history, epochs_no_improve, best_epoch)
            json.dump(history, open(HISTORY_JSON, "w"))

            if (EARLY_STOP_PATIENCE >= 0 and epoch >= MIN_EPOCHS
                    and epochs_no_improve >= EARLY_STOP_PATIENCE):
                log(f"조기종료: {EARLY_STOP_PATIENCE}에폭 연속 개선 없음 "
                    f"| 최고 val MAE {best_val:.2f} @ epoch {best_epoch}")
                break
        else:
            log(f"학습 완료(전체 {EPOCHS}에폭) | 최고 val MAE {best_val:.2f} @ epoch {best_epoch}")
    except KeyboardInterrupt:
        save_checkpoint(LAST_CKPT, model, ARCH, optimizer, scaler, epoch,
                        best_val, history, epochs_no_improve, best_epoch)
        json.dump(history, open(HISTORY_JSON, "w"))
        log(f"중단됨 - last.pt 저장(epoch {epoch}). 다시 실행하면 이어서 학습합니다.")


# =========================================================================
# [I] 최종 평가 (best.pt 만 로드 - 학습 없이 이 부분만 재실행 가능)
# =========================================================================
print("=" * 64)
log("최종 평가 (best.pt 로드)")
if not BEST_CKPT.exists():
    log(f"[중단] best.pt 가 없습니다: {BEST_CKPT}")
    sys.exit(0)

ck = torch_load(BEST_CKPT, map_location=device)
eval_model = build_reg_model(ck["arch"], pretrained=False)
eval_model.load_state_dict(ck["model"]); eval_model.eval()
log(f"best.pt 로드 | 채널 {ck['arch'].get('AGG_CHANNELS')} "
    f"| lambda {ck['arch'].get('LDL_LAMBDA')} | delta {ck['arch'].get('LDL_DELTA')}")

results = {"method": "attention(R1/R2) + LDL, hand crop given",
           "when": datetime.now().isoformat(timespec="seconds"),
           "arch": ck["arch"], "splits": {}}
lines = ["=" * 58,
         f"Attention + LDL 골연령 | 채널 {'+'.join(ck['arch'].get('AGG_CHANNELS', []))}",
         f"{datetime.now():%Y-%m-%d %H:%M}", "=" * 58,
         f"{'split':>6} | {'N':>6} | {'MAE(mo)':>8} | {'RMSE(mo)':>9} | {'bias':>7}"]
log(lines[-1])

split_loaders = [("train", train_eval_loader), ("val", val_loader)]
if test_loader is not None:
    split_loaders.append(("test", test_loader))

for name, loader in split_loaders:
    preds, trues = predict_months(eval_model, loader, USE_AMP)
    if not len(trues):
        continue
    mae, rmse = mae_rmse(preds, trues); bias = float(np.mean(preds - trues))
    results["splits"][name] = {"N": int(len(trues)), "mae": mae, "rmse": rmse, "bias": bias}
    row = f"{name:>6} | {len(trues):>6,} | {mae:>8.2f} | {rmse:>9.2f} | {bias:>+7.2f}"
    lines.append(row); log(row)
    plt.figure(figsize=(6, 6)); plt.scatter(trues, preds, s=8, alpha=.4)
    lim = [0, max(trues.max(), preds.max()) + 5]; plt.plot(lim, lim, "r--")
    plt.xlabel("True (months)"); plt.ylabel("Pred (months)")
    plt.title(f"{name} | MAE={mae:.2f} · RMSE={rmse:.2f} mo"); plt.tight_layout()
    plt.savefig(CKPT_DIR / f"scatter_{name}.png", dpi=120); plt.close()

grp_name, grp_loader = ("test", test_loader) if test_loader is not None else ("val", val_loader)
gp, gt = predict_months(eval_model, grp_loader, USE_AMP)
if len(gt):
    lines += ["-" * 58, f"[{grp_name} 연령대별]",
              f"{'group':>7} | {'N':>5} | {'MAE':>6} | {'RMSE':>6} | {'bias':>6}"]
    grp = {}
    for lo, hi, lab in zip([0, 48, 96, 144, 192], [48, 96, 144, 192, 10 ** 5],
                           ["0-4y", "4-8y", "8-12y", "12-16y", ">16y"]):
        m = (gt >= lo) & (gt < hi)
        if m.sum():
            gm, gr = mae_rmse(gp[m], gt[m]); gb = float(np.mean(gp[m] - gt[m]))
            grp[lab] = {"N": int(m.sum()), "mae": gm, "rmse": gr, "bias": gb}
            lines.append(f"{lab:>7} | {m.sum():>5} | {gm:>6.2f} | {gr:>6.2f} | {gb:>+6.2f}")
    results[f"{grp_name}_by_age"] = grp
lines.append("=" * 58)

# -- 학습된 나이 분포 (논문 Fig.4) --------------------------------------------
try:
    samp = val_df.sample(min(4, len(val_df)), random_state=7)
    ds = AggDataset(samp, "val", eval_tf, ck["arch"].get("AGG_CHANNELS", AGG_CHANNELS))
    plt.figure(figsize=(7, 4.5))
    for i in range(len(ds)):
        x, g, ym = ds[i]
        with torch.no_grad():
            yh, pp = logits_to_months(eval_model(x.unsqueeze(0).to(device),
                                                 g.unsqueeze(0).to(device)))
        plt.plot(np.arange(1, pp.size(1) + 1), pp[0].cpu().numpy(),
                 label=f"y={int(ym.item())}, yhat={yh.item():.1f}")
    plt.xlabel("Age (months)"); plt.ylabel("Probability")
    plt.title(f"Learned age distribution (lambda={ck['arch'].get('LDL_LAMBDA')}, "
              f"delta={ck['arch'].get('LDL_DELTA')})")
    plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(CKPT_DIR / "age_distribution.png", dpi=120); plt.close()
    log(f"나이 분포 저장: {CKPT_DIR/'age_distribution.png'}")
except Exception as e:
    log(f"[경고] 나이 분포 그림 생략: {e}")

# -- 학습 곡선 ----------------------------------------------------------------
if HISTORY_JSON.exists():
    try:
        h = json.load(open(HISTORY_JSON)); ep = range(1, len(h["train_mae"]) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(ep, h["train_mae"], "-o", ms=3, label="train MAE")
        plt.plot(ep, h["val_mae"], "-o", ms=3, label="val MAE")
        if len(h.get("val_rmse", [])) == len(h["train_mae"]):
            plt.plot(ep, h["val_rmse"], "-s", ms=3, label="val RMSE", alpha=.7)
        plt.axhline(4.3, ls="--", c="green", label="paper 4.3 (H+R1+E, LDL)")
        plt.xlabel("Epoch"); plt.ylabel("months"); plt.title("Learning curve (Attention+LDL)")
        plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(CKPT_DIR / "learning_curve.png", dpi=120); plt.close()
    except Exception as e:
        log(f"[경고] 학습곡선 저장 실패: {e}")

json.dump(results, open(RESULTS_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
open(RESULTS_TXT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)
log(f"결과 저장: {RESULTS_TXT}")


# =========================================================================
# [J] 단일 이미지 추론
#     입력은 '손이 이미 크롭된 이미지' 입니다 (학습 파이프라인과 동일 조건).
#     내부에서 R1 / E (필요 시 R2) 를 CAM 으로 만들어 3채널을 구성합니다.
# =========================================================================
_CLS_CACHE = {}


def _load_cls(ckpt_path):
    key = str(ckpt_path)
    if key not in _CLS_CACHE:
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"분류 체크포인트가 없습니다: {ckpt_path}")
        c = torch_load(ckpt_path, map_location=device)
        m = CAMClassifier(pool=c.get("pool", "gmp"), pretrained=False).to(device)
        m.load_state_dict(c["model"]); m.eval()
        _CLS_CACHE[key] = m
    return _CLS_CACHE[key]


def make_rois_from_hand(hand_gray, channels=AGG_CHANNELS):
    """손 크롭 -> {kind: 560x560}. 학습과 동일:
       전처리 -> 캔버스 -> (R1: 밴드 CAM) -> E -> (R2: E 로 학습한 cls-r2 의 CAM)."""
    out = {}
    enh_full = enhance(hand_gray)
    ref, info = fit_canvas(enh_full, REG_SIZE)
    out["hand"] = ref
    need_r1 = ("r1" in channels) or ("erased" in channels) or ("r2" in channels)
    e = None
    if need_r1:
        m1 = _load_cls(CLS_R1_BEST)
        x = torch.from_numpy(np.stack([ref] * 3, 0)).float().div(255.).unsqueeze(0).to(device)
        cam, _ = compute_cam(m1, x)
        band = band_abs("r1", REG_SIZE, REG_SIZE)
        b1, _ok = cam_to_bbox(cam, TAU_R1, REG_SIZE, REG_SIZE,
                              box_cfg=BOX_CFG.get("r1"), region=band,
                              clamp_region=band if CLAMP_TO_BAND else None,
                              fallback_box=band)
        if "r1" in channels:
            out["r1"] = crop_roi(enh_full, b1, info, REG_SIZE)
        e = ref.copy()
        x0, y0, x1, y1 = b1
        e[y0:y1, x0:x1] = np.random.randint(0, 256, (y1 - y0, x1 - x0), dtype=np.uint8)
        if "erased" in channels:
            out["erased"] = e
    if "r2" in channels and e is not None:
        m2 = _load_cls(CLS_R2_BEST)          # ★ 논문 방식: 별도 학습된 R2 모델
        x = torch.from_numpy(np.stack([e] * 3, 0)).float().div(255.).unsqueeze(0).to(device)
        cam2, _ = compute_cam(m2, x)
        b2, _ok = cam_to_bbox(cam2, TAU_R2_PAPER, REG_SIZE, REG_SIZE,
                              box_cfg=BOX_CFG_R2, merge=R2_MERGE)
        out["r2"] = crop_roi(enh_full, b2, info, REG_SIZE)
    return out


def predict_bone_age(hand_image_path, is_male, ckpt_path=BEST_CKPT, return_dist=False):
    """손이 크롭된 이미지 경로 + 성별(True=남) -> 골연령(개월)."""
    c = torch_load(ckpt_path, map_location=device)
    m = build_reg_model(c["arch"], pretrained=False)
    m.load_state_dict(c["model"]); m.eval()
    g = imread_kr(hand_image_path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise FileNotFoundError(hand_image_path)
    kinds = c["arch"].get("AGG_CHANNELS", list(AGG_CHANNELS))
    rois = make_rois_from_hand(g, kinds)
    chans = []
    for k in kinds:
        im = rois.get(k)
        if im is None:
            im, _ = fit_canvas(enhance(g), REG_SIZE)
        if im.shape[0] != REG_SIZE:
            im = cv2.resize(im, (REG_SIZE, REG_SIZE), interpolation=cv2.INTER_AREA)
        chans.append(im)
    while len(chans) < 3:
        chans.append(chans[0])
    x = eval_tf(np.stack(chans[:3], -1)).unsqueeze(0).to(device)
    gd = torch.tensor([[1.0 if is_male else -1.0]], dtype=torch.float32, device=device)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
        out = m(x, gd)
    months, dist = logits_to_months(out)
    if not return_dist:
        return float(months.item())
    return float(months.item()), dist[0].cpu().numpy()


# 예시:
#   months = predict_bone_age(HAND_CROP_DIR / "validation" / "1377.png", is_male=True)
#   print(f"예측 골연령: {months:.1f} 개월")
log(f"추론 함수 준비 완료 · best.pt: {BEST_CKPT}")
log("=== 전체 완료 ===")
