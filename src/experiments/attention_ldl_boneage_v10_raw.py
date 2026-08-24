# -*- coding: utf-8 -*-
# =========================================================================
# 손 X-ray 골연령 - Attention-Guided RoI Localization + Label Distribution Learning
#   논문: Chen et al., "Attention-Guided Discriminative Region Localization and
#         Label Distribution Learning for Bone Age Assessment" (arXiv:2006.00202)
#
#   ▶ v10 = ★ 전처리 완전 제거 베이스라인 (crop_data 원본 그대로)
#     [삭제] 마커 억제       suppress_markers()      - 함수 자체를 없앰
#     [삭제] 회전 정렬       align_hand()            - 함수 자체를 없앰
#     [삭제] 밝기 정규화     normalize_intensity()   - 함수 자체를 없앰
#     [삭제] White Top-Hat   apply_tophat()          - 함수 자체를 없앰
#     [삭제] CLAHE           apply_clahe()           - 함수 자체를 없앰
#     [삭제] 배경색 추정     estimate_bg()           - 패딩은 상수 0 으로 고정
#     [삭제] global 배율     scan_ref_long()/REF_LONG- per-image 만 남김
#     [유지] 비율유지 리사이즈 + 패딩 (네트워크 입력 크기를 맞추기 위한 기하 변환
#            일 뿐, 화소값을 건드리지 않으므로 '전처리'로 보지 않습니다)
#     [유지] R1 밴드 CAM / R2 논문 erase 방식 / 성별 임베딩 / LDL 손실
#     [유지] Phase II 백본 = ConvNeXt-Tiny (v8 과 동일 - 비교 가능하게)
#     [분리] cache_attention_v10/<PRE_TAG> / checkpoints_attention_v10 / logs/*_v10_*
#
#   ▶ 이미지 체인 (최종)
#       crop_data 원본 --> [uint8 변환] --> 비율유지 리사이즈 --> 0 패딩 --> 512x512
#       * uint8 변환은 16bit/float PNG 를 안전하게 다루기 위한 것이며,
#         입력이 이미 8bit 면 아무 일도 하지 않습니다 (완전 무손실 통과).
#
#   ▶ v8 대비 이 파일이 답하려는 질문
#       "전처리(정규화/TopHat/CLAHE)가 실제로 MAE 를 개선했는가?"
#       -> v8 결과와 이 파일의 결과를 test MAE 로 직접 비교하면 됩니다.
#          (같은 split / 같은 EXCLUDE_IDS / 같은 백본 / 같은 LDL 설정)
#
#   ▶ 실행: python attention_ldl_boneage_v10_raw.py
#       - 창을 닫아도 서버에서 학습은 계속됩니다. 다시 실행하면 로그에 재부착.
#       - 각 스테이지는 완료 표식을 남기고 자동 스킵 -> 중단돼도 이어서 진행.
#
#   ▶ 실행 옵션
#       --fg             백그라운드 분리 없이 바로 실행(디버그용)
#       --eval-only      Phase II 학습을 건너뛰고 best.pt 로 평가만
#       --rebuild-cache  손 크롭 -> 512 캔버스 캐시를 강제로 재생성
#       --rebuild-roi    R1/R2/E 크롭 캐시를 강제로 재생성 (CAM 부터 재실행)
#       --qc-only        ROI/QC 시트만 만들고 종료 (밴드 튜닝용)
#
#   ▶ 표기 규칙
#       [논문] = 논문/공식코드 명시값      [추론] = 미명시 -> 합리적 추정
#       [v10] = 이 버전에서 바뀐 항목
# =========================================================================
from pathlib import Path
import os, sys, time, json, subprocess
from datetime import datetime

PROJECT_DIR = Path(__file__).resolve().parent
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RUN_STATE = LOG_DIR / "boneage_attention_v10_running.json"
_WORKER_ENV = "BONEAGE_ATTENTION_V10_WORKER"

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
    log_path = LOG_DIR / f"boneage_attention_v10_{ts}.log"
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
#   HAND_CROP_DIR  : ★ 손 크롭 이미지 폴더 (아래 training / validation / test)
#   CSV_DIR        : ★ 라벨 CSV 폴더
# -------------------------------------------------------------------------
BASE_DIR = Path(r"G:/Project/sinra_cho")

# ★ 손 크롭 이미지 폴더 (crop_data/training, crop_data/validation, crop_data/test)
HAND_CROP_DIR = Path(r"G:/Project/sinra_cho/crop_data")

# ★ 라벨 CSV 폴더 (crop_data_csv/*.csv)
CSV_DIR = Path(r"G:/Project/sinra_cho/crop_data_csv")

# split 하위 폴더명 (HAND_CROP_DIR 아래)
SPLIT_SUBDIR = {"train": "training", "val": "validation", "test": "test"}


def _require(path, what):
    """경로가 없으면 즉시 중단. 잘못된 경로로 조용히 진행하지 않는다."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"[중단] {what} 경로가 없습니다:\n       {p}\n"
                         f"       실제 위치로 코드 상단 경로를 수정하세요.")
    return p


TRAIN_CSV = _require(CSV_DIR / "training.csv", "training.csv")
VAL_CSV   = _require(CSV_DIR / "validation.csv", "validation.csv")
TEST_CSV  = _require(CSV_DIR / "test.csv", "test.csv")


# =========================================================================
# [B-2] 재현 스위치
# =========================================================================
SEED = 42

CLS_SIZE_REGION = 512     # Phase I CAM 입력 (InceptionV3 특징맵 약 14x14)
REG_SIZE        = 512     # Phase II 입력

# -- CAM 임계값 (0~255 스케일) ------------------------------------------------
TAU_R1 = 30               # [조정] 논문 τ∈{10..100} 범위 내. 낮출수록 박스가 커짐

SOFT_L   = 50             # [논문] 식(5) 삼각형 소프트 라벨 폭 l=50
AGE_BINS = 240            # [논문] 데이터셋 최대 연령(개월)

# Phase II 집계 채널 - [논문] Table V 최고 조합: H+R1+E (4.3) / H+R1+R2 (4.3)
#   사용 가능: "hand"(크롭된 손 이미지), "r1", "r2", "erased"
AGG_CHANNELS = ("hand", "r1", "r2")

# -- Phase II 백본 (v8 과 동일하게 유지 -> 전처리 유무만 단독 비교) ---------
REG_BACKBONE = "convnext_tiny.fb_in22k_ft_in1k"

PAPER_STRICT = False      # True 로 두면 증강 OFF (논문 그대로)
USE_AUG   = (not PAPER_STRICT)
NORMALIZE = "div255"      # v8 과 동일. "imagenet" 으로 바꾸면 ImageNet 통계 사용

# =========================================================================
# ★ [v10] 이미지 준비 = '기하 변환만'
#
#   전처리(화소값 조작)는 전부 제거했습니다. 남은 것은 네트워크 입력 크기를
#   맞추기 위한 두 가지뿐입니다.
#
#   RESIZE_MODE = "letterbox" : 비율 유지 리사이즈 + 0 패딩  (왜곡 0, 권장)
#                 "stretch"   : 정사각으로 강제 리사이즈     (패딩 0, 왜곡 있음)
#
#   [중요] letterbox 를 쓰면 캔버스 좌표 != 원본 좌표 단순비례 입니다.
#          CAM 박스를 원본으로 되돌릴 때 scale 과 pad 오프셋을 함께 역산합니다
#          (canvas_box_to_src 함수). stretch 는 x/y 배율이 달라 역시 역산 필요.
# =========================================================================
RESIZE_MODE = "letterbox"   # "letterbox" | "stretch"
PAD_VALUE   = 0             # [v10] 배경색 추정 제거 -> 상수 0 고정
PAD_ANCHOR  = "center"      # "center" | "topleft"

# ── 증강 (약한 기하 증강만) ─────────────────────────────────────────
AUG_ROT_DEG   = 12
AUG_TRANSLATE = 0.06
AUG_SCALE     = (0.92, 1.08)

# 이미지 준비 태그 -> 캐시 폴더명에 포함되어 설정이 다르면 자동으로 분리됩니다
PRE_TAG = f"raw{REG_SIZE}_{RESIZE_MODE}_pad{PAD_VALUE}_{PAD_ANCHOR}"

# =========================================================================
# R1 = 밴드 방식 / R2 = 논문 erase 방식 (v6~v8 과 동일)
#
#   R1: 이미지 비율 기준 고정 밴드 안에서 CAM 최대 영역 -> 박스 확대
#   R2: R1 을 랜덤값으로 지운 이미지(E)로 분류모델을 '다시 학습' 시키고,
#       그 모델의 CAM 으로 R2 를 찾습니다. 밴드 제약·박스 확대 없음.
# =========================================================================
R1_BAND = {"x0": 0.12, "y0": 0.52, "x1": 0.88, "y1": 0.98}   # 수근골 밴드
BAND_CFG = {"r1": R1_BAND}
CLAMP_TO_BAND = True        # R1 박스가 밴드 밖으로 못 나가게 고정

# -- R1 박스 확대 파라미터 ----------------------------------------------------
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
R2_PAD_FRAC    = 0.0      # [논문] 박스 여백 없음
R2_MIN_W_FRAC  = 0.0      # [논문] 최소 크기 강제 없음
R2_MIN_H_FRAC  = 0.0
BOX_CFG_R2 = (None if (R2_PAD_FRAC == 0 and R2_MIN_W_FRAC == 0 and R2_MIN_H_FRAC == 0)
              else {"pad_x": R2_PAD_FRAC, "pad_y": R2_PAD_FRAC,
                    "min_w": R2_MIN_W_FRAC, "min_h": R2_MIN_H_FRAC})

ERASE_FILL    = "noise"   # [논문] R1 픽셀을 랜덤값으로 대체
USE_EXCLUDE   = True
N_QC          = 6

# 설정이 다르면 캐시가 자동으로 분리됩니다 (PRE_TAG)
CACHE_DIR = BASE_DIR / "cache_attention_v10" / PRE_TAG
CKPT_DIR  = BASE_DIR / "checkpoints_attention_v10"
SPLITS    = ("train", "val", "test")
SUBDIRS   = ("hand", "r1", "r2", "erased")

for d in [CACHE_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)
for sub in SUBDIRS:
    for sp in SPLITS:
        (CACHE_DIR / sub / sp).mkdir(parents=True, exist_ok=True)

BASE_DONE = CACHE_DIR / "_DONE_hand.json"
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

EXCLUDE_IDS = {""} if USE_EXCLUDE else set()

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

print("=" * 64)
log("Attention(R1/R2) + LDL 골연령 - v10 (전처리 없음 / raw crop_data) 시작")
log(f"PyTorch {torch.__version__} | torchvision {torchvision.__version__} | timm {timm.__version__}")
log(f"device {device} | CUDA {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log(f"GPU {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda}")
log(f"BASE_DIR       {BASE_DIR}")
log(f"HAND_CROP_DIR  {HAND_CROP_DIR}")
log(f"이미지 준비    전처리 없음 | {RESIZE_MODE} -> {REG_SIZE}px | pad {PAD_VALUE}")
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
# [B-3] ★ v10 이미지 준비 - 화소값을 바꾸는 코드는 여기에 '없습니다'
#       to_gray8()  : 8bit 보장 (이미 8bit 면 그대로 통과)
#       fit_canvas(): 비율유지 리사이즈 + 0 패딩 (또는 stretch)
#       canvas_box_to_src(): 캔버스 박스 -> 원본 좌표 역변환
# =========================================================================
def to_gray8(g):
    """16bit·float 입력만 8bit 로 낮춥니다.
       ★ 입력이 이미 uint8 이면 아무 연산도 하지 않고 그대로 반환합니다
         (= crop_data PNG 는 손대지 않음)."""
    if g.dtype == np.uint8:
        return g
    g = g.astype(np.float32)
    lo, hi = float(g.min()), float(g.max())
    if hi - lo < 1e-6:
        return np.zeros(g.shape, np.uint8)
    return np.clip((g - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def fit_canvas(g, size=None, anchor=None):
    """size x size 캔버스로 맞춥니다. 화소값 보정 없음. 반환 (canvas, info)
         info["sx"], info["sy"] : x/y 배율 (letterbox 면 동일)
         info["pad"]            : (top, bottom, left, right)
       ★ 이 두 값이 있어야 CAM 박스를 원본 좌표로 정확히 되돌릴 수 있습니다."""
    size   = REG_SIZE   if size   is None else size
    anchor = PAD_ANCHOR if anchor is None else anchor
    g = to_gray8(g)
    h, w = g.shape[:2]

    if RESIZE_MODE == "stretch":
        interp = cv2.INTER_AREA if (size < h and size < w) else cv2.INTER_CUBIC
        out = cv2.resize(g, (size, size), interpolation=interp)
        info = {"sx": size / float(w), "sy": size / float(h),
                "src": (h, w), "resized": (size, size),
                "pad": (0, 0, 0, 0), "pad_val": 0, "pad_frac": 0.0}
        return out, info

    # letterbox: 긴 변을 size 에 맞춤 -> 비율 왜곡 0
    s = size / float(max(h, w))
    nh = max(1, min(size, int(round(h * s))))
    nw = max(1, min(size, int(round(w * s))))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    r = cv2.resize(g, (nw, nh), interpolation=interp)

    top  = 0 if anchor == "topleft" else (size - nh) // 2
    left = 0 if anchor == "topleft" else (size - nw) // 2
    bottom, right = size - nh - top, size - nw - left
    out = cv2.copyMakeBorder(r, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=int(PAD_VALUE))
    info = {"sx": float(s), "sy": float(s), "src": (h, w), "resized": (nh, nw),
            "pad": (top, bottom, left, right), "pad_val": int(PAD_VALUE),
            "pad_frac": 1.0 - (nh * nw) / float(size * size)}
    return out, info


def canvas_box_to_src(box, info):
    """★ 캔버스(512, 패딩 포함) 좌표 -> 원본 좌표 역변환.
       패딩 오프셋을 빼고 배율로 나눕니다. 이 역변환이 없으면 박스가 어긋납니다."""
    top, _bottom, left, _right = info["pad"]
    sx, sy = info["sx"], info["sy"]
    h, w = info["src"]
    x0, y0, x1, y1 = box
    X0 = int(round((x0 - left) / sx)); X1 = int(round((x1 - left) / sx))
    Y0 = int(round((y0 - top) / sy));  Y1 = int(round((y1 - top) / sy))
    X0, Y0 = max(0, X0), max(0, Y0)
    X1, Y1 = min(w, max(X0 + 1, X1)), min(h, max(Y0 + 1, Y1))
    return X0, Y0, X1, Y1


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
    raise SystemExit(f"[중단] 손 크롭 폴더가 없습니다: {HAND_CROP_DIR}")

for _n, _p in [("TRAIN_CSV", TRAIN_CSV), ("VAL_CSV", VAL_CSV), ("TEST_CSV", TEST_CSV)]:
    log(f"  {_n:<9} OK  {_p}")

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
# [D] hand 캐시 - 손 크롭을 512x512 캔버스로 1회만 저장 (화소값 보정 없음)
#     * CAM 입력이자 Phase II 의 'hand' 채널로 동시에 쓰입니다.
#     * 고해상도 R1/R2 크롭은 원본(손 크롭 파일)에서 직접 뜹니다.
# =========================================================================
PRE_PARAMS = {"PREPROCESS": "none", "IMG_SIZE": REG_SIZE,
              "RESIZE_MODE": RESIZE_MODE, "PAD_VALUE": PAD_VALUE,
              "PAD_ANCHOR": PAD_ANCHOR}


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
    """손 크롭 -> (리사이즈 + 패딩) -> 512 캔버스 저장. 화소값 보정 없음."""
    pfs = []
    for sp in SPLITS:
        df = SPLIT_DFS[sp]
        if not len(df):
            continue
        made = skipped = failed = 0
        for i, (_, r) in enumerate(df.iterrows(), 1):
            dst = CACHE_DIR / "hand" / sp / f"{r['id']}.png"
            if dst.exists() and not REBUILD_CACHE:
                skipped += 1
            else:
                g = imread_kr(r["hand_path"], cv2.IMREAD_GRAYSCALE)
                if g is None:
                    failed += 1
                else:
                    canvas, info = fit_canvas(g, REG_SIZE)
                    imwrite_kr(dst, canvas)
                    pfs.append(info["pad_frac"])
                    made += 1
            if i % 2000 == 0:
                log(f"  hand {sp} {i}/{len(df)}")
        log(f"[hand:{sp}] 생성 {made} | 스킵 {skipped} | 실패 {failed}")
    if pfs:
        log(f"  패딩 비율 중앙값 {np.median(pfs):.1%} "
            f"(YOLO 크롭이 타이트하면 10~30% 가 정상)")
    json.dump({"pre": PRE_PARAMS, "train": len(train_df),
               "val": len(val_df), "test": len(test_df)},
              open(BASE_DONE, "w", encoding="utf-8"))


if hand_cache_valid():
    log("hand 캐시 유효 - 스킵")
else:
    log(f"hand 캐시 생성: 전처리 없음 -> {RESIZE_MODE}({REG_SIZE}px) -> pad {PAD_VALUE}")
    build_hand_cache()
    log("hand 캐시 완료")


def filter_cached(df, sub, sp):
    if not len(df):
        return df
    ok = df["id"].apply(lambda i: (CACHE_DIR / sub / sp / f"{i}.png").exists())
    return df[ok].reset_index(drop=True)


for _sp in SPLITS:
    SPLIT_DFS[_sp] = filter_cached(SPLIT_DFS[_sp], "hand", _sp)
train_df, val_df, test_df = SPLIT_DFS["train"], SPLIT_DFS["val"], SPLIT_DFS["test"]
log(f"캐시 확인 후 | train {len(train_df):,} | val {len(val_df):,} | test {len(test_df):,}")


# =========================================================================
# [E] Phase I : 분류 모델 (InceptionV3 + soft label) - 논문 §III-A
#     식(1)~(3): Y_t = (1/HW) sum_ij sum_k W_kt F_ijk
#       -> 마지막 FC 는 반드시 '선형'이어야 CAM 수식이 성립 (활성함수 없음)
#     식(5)   : Y_i = max(0, 1 - |i-t|/l), l=50
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
    """CAM 추출용 분류 헤드. pool='gmp' -> 작고 뾰족한 영역(R1/R2)에 적합 (논문 사용)."""

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
    """Phase I 학습용. sub 로 hand / erased 중 어느 캐시를 쓸지 지정."""

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
CLS_BATCH   = 8       # [논문] 32. 8GB GPU + 512 입력이면 8 권장 (OOM 시 4)
CLS_EPOCHS  = 100      # [논문] 70 (3e-4 x 50 epoch -> 1e-4 x 20 epoch)
CLS_LR1, CLS_LR2 = 3e-4, 1e-4
CLS_SWITCH  = 50      # [논문] lr 전환 시점
CLS_WORKERS = 0       # 윈도우 안전값
CLS_LOG_EVERY = 100

# ★ 조기 종료 (논문에는 없음 - 시간 절약용 엔지니어링 추가) --------------------
CLS_EARLY_STOP_PATIENCE = 10    # val argmax-MAE 가 이 에폭 수만큼 개선 없으면 중단 (<0 이면 끔)
CLS_MIN_DELTA           = 0.05 # 개선으로 인정할 최소 감소량(개월)
CLS_MIN_EPOCHS          = 20   # 최소 이 에폭까지는 조기종료하지 않음

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
#  [원코드 결함 -> 수정] 음수에서 uint8 언더플로 래핑 -> clip(0,None) 후 정규화
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
    """박스를 (1) 방향별 여백 확장 -> (2) 최소 크기 보장(중심 유지) 순으로 키운다."""
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    px, py = w * cfg["pad_x"], h * cfg["pad_y"]
    x0, x1 = x0 - px, x1 + px
    y0, y1 = y0 - py, y1 + py
    w, h = x1 - x0, y1 - y0
    min_w, min_h = cfg["min_w"] * out_w, cfg["min_h"] * out_h
    if w < min_w:
        x0, x1 = cx - min_w / 2.0, cx + min_w / 2.0
    if h < min_h:
        y0, y1 = cy - min_h / 2.0, cy + min_h / 2.0
    x0, y0 = int(max(0, round(x0))), int(max(0, round(y0)))
    x1, y1 = int(min(out_w, round(x1))), int(min(out_h, round(y1)))
    return x0, y0, x1, y1


def cam_to_bbox(cam, tau, out_w, out_h, box_cfg=None, merge=None,
                region=None, clamp_region=None, fallback_box=None):
    """CAM -> (x0,y0,x1,y1). 실패 시 fallback 반환."""
    if merge is None:
        merge = MERGE_COMPONENTS
    m = cv2.resize(cam.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_LINEAR)
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
    """kind('r1') 의 고정 밴드를 절대 좌표로."""
    b = BAND_CFG[kind]
    return (int(b["x0"] * W), int(b["y0"] * H),
            int(b["x1"] * W), int(b["y1"] * H))


def crop_roi(src_full, box_canvas, info, size=REG_SIZE):
    """캔버스 박스를 '원본 좌표'로 역변환한 뒤, 원본 해상도 이미지에서 잘라
       fit_canvas 로 정사각화. 화소값 보정은 어디에도 없습니다.
         src_full   : 원본 해상도 손 크롭 (uint8)
         box_canvas : 512 캔버스 좌표계의 박스
         info       : fit_canvas 가 준 {sx, sy, pad, src}"""
    X0, Y0, X1, Y1 = canvas_box_to_src(box_canvas, info)
    patch = src_full[Y0:Y1, X0:X1]
    if patch.size == 0:
        patch = src_full
    out, _ = fit_canvas(patch, size)
    return out


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
            and info.get("resize_mode") == RESIZE_MODE
            and info.get("train") == len(train_df) and info.get("val") == len(val_df)
            and info.get("test") == len(test_df))


@torch.no_grad()
def localize_r1_and_erase(model, records):
    """[Phase I-a] R1(수근골) + E(R1 지운 이미지) 생성.
       R1 = '밴드 안 CAM' 방식. E 는 논문 Fig.3(f): R1 픽셀을 랜덤값으로 대체."""
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
            src_full = to_gray8(raw)                      # ★ 보정 없음
            ref, info = fit_canvas(src_full, REG_SIZE)

            x = torch.from_numpy(np.stack([ref] * 3, 0)).float().div(255.).unsqueeze(0).to(device)
            cam, _ = compute_cam(model, x)
            band = band_abs("r1", REG_SIZE, REG_SIZE)
            box, good = cam_to_bbox(cam, TAU_R1, REG_SIZE, REG_SIZE,
                                    box_cfg=BOX_CFG.get("r1"), region=band,
                                    clamp_region=band if CLAMP_TO_BAND else None,
                                    fallback_box=band)
            ok += int(good); fb += int(not good)
            imwrite_kr(dst_r1, crop_roi(src_full, box, info, REG_SIZE))

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
            src_full = to_gray8(raw)
            _ref, info = fit_canvas(src_full, REG_SIZE)

            # CAM 은 'R1 이 지워진 이미지'에서 계산 -> 그 외 영역에 근거하게 됨
            x = torch.from_numpy(np.stack([e] * 3, 0)).float().div(255.).unsqueeze(0).to(device)
            cam, _ = compute_cam(model, x)
            box, good = cam_to_bbox(cam, TAU_R2_PAPER, REG_SIZE, REG_SIZE,
                                    box_cfg=BOX_CFG_R2, merge=R2_MERGE)
            ok += int(good); fb += int(not good)
            # 크롭은 '원본'에서 (지워지지 않은 실제 조직)
            imwrite_kr(dst, crop_roi(src_full, box, info, REG_SIZE))

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
    log("### Phase I-a : R1(수근골) 국소화 + Erased 생성 ###")
    log(f"  R1 밴드 {R1_BAND} | clamp {CLAMP_TO_BAND} | tau {TAU_R1}")
    m_r1 = train_cam_classifier("cls-r1", CLS_R1_BEST, CLS_R1_LAST, "gmp", "hand")
    localize_r1_and_erase(m_r1, bbox_records)
    del m_r1; torch.cuda.empty_cache()

    # === Phase I-b : R2(중수골) - ★ 논문 방식 ================================
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
               "resize_mode": RESIZE_MODE, "r2_method": "paper_erase",
               "train": len(train_df), "val": len(val_df), "test": len(test_df)},
              open(ROI_DONE, "w", encoding="utf-8"))
    log("R1/R2/E 캐시 완료")


# -- QC 시트: 밴드/박스 오버레이 + 크롭 결과 ----------------------------------
def _overlay_bands(ref_gray, cam_norm, boxes):
    """손 이미지 + CAM 히트맵 + 밴드(회색) + 박스(진한색) 오버레이."""
    base = cv2.cvtColor(ref_gray, cv2.COLOR_GRAY2BGR)
    if cam_norm is not None:
        heat = cv2.applyColorMap(np.uint8(255 * cam_norm), cv2.COLORMAP_JET)
        base = cv2.addWeighted(base, 0.65, heat, 0.35, 0)
    H, W = ref_gray.shape[:2]
    colors = {"r1": (60, 60, 255), "r2": (255, 120, 0)}     # R1=빨강 R2=파랑(BGR)
    for kind in ("r1",):
        bx = band_abs(kind, W, H)
        cv2.rectangle(base, (bx[0], bx[1]), (bx[2], bx[3]), (150, 150, 150), 1)   # 밴드
        if kind in boxes:
            b = boxes[kind]
            cv2.rectangle(base, (b[0], b[1]), (b[2], b[3]), colors[kind], 3)      # 박스
    return cv2.cvtColor(base, cv2.COLOR_BGR2RGB)


def build_qc_sheet(n=N_QC, tag="qc_roi"):
    """QC 시트. 열: [raw crop] [canvas+R1 band/box] [r1] [E+R2 box] [r2]
       ★ v10 은 전처리가 없으므로 1열(raw)과 2열(canvas)의 '밝기·대비가 동일'해야
         정상입니다. 다르면 어딘가에 보정이 남아 있다는 뜻입니다."""
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
    cols = ["raw crop", "canvas + R1 band/box", "r1 crop", "E + R2 box", "r2 crop"]
    fig, axes = plt.subplots(len(samp), len(cols),
                             figsize=(2.7 * len(cols), 2.7 * len(samp)))
    axes = np.atleast_2d(axes)
    for rr, (_, r) in enumerate(samp.iterrows()):
        raw = imread_kr(r["hand_path"], cv2.IMREAD_GRAYSCALE)
        ref = imread_kr(CACHE_DIR / "hand" / "val" / f"{r['id']}.png", cv2.IMREAD_GRAYSCALE)
        e   = imread_kr(CACHE_DIR / "erased" / "val" / f"{r['id']}.png", cv2.IMREAD_GRAYSCALE)
        if raw is not None:
            axes[rr, 0].imshow(to_gray8(raw), cmap="gray", vmin=0, vmax=255)
            axes[rr, 0].set_title(f"{r['id']} | {raw.shape[1]}x{raw.shape[0]}", fontsize=7)

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
            axes[rr, 1].set_title(f"R1 {b1[2]-b1[0]}x{b1[3]-b1[1]}" if b1 else cols[1],
                                  fontsize=7)
        im = imread_kr(CACHE_DIR / "r1" / "val" / f"{r['id']}.png", cv2.IMREAD_GRAYSCALE)
        if im is not None:
            axes[rr, 2].imshow(im, cmap="gray", vmin=0, vmax=255)
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
            axes[rr, 4].imshow(im, cmap="gray", vmin=0, vmax=255)
        axes[rr, 4].set_title(cols[4], fontsize=7)

    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    out = CKPT_DIR / f"{tag}.png"
    plt.savefig(out, dpi=110); plt.close()
    log(f"QC 시트 저장: {out}")
    log("  1열=원본 크롭 / 2열=캔버스+R1밴드(회색)+R1박스(빨강) / 4열=E+R2박스(파랑)")
    log("  ※ 1열과 2열의 밝기·대비가 같아야 정상입니다(전처리 없음 확인).")
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
# [G] Phase II : 백본 + 성별 + LDL 기대값 회귀 - 논문 §III-B
#     Backbone(no top) -> Conv2d(256,3x3) -> MaxPool(3x3) -> Flatten
#     gender(±1) -> Linear(32) -> concat -> Linear(240) -> softmax -> 기대값
# =========================================================================
_aug = [transforms.RandomRotation(AUG_ROT_DEG),
        transforms.RandomAffine(0, translate=(AUG_TRANSLATE, AUG_TRANSLATE),
                                scale=AUG_SCALE)]
_norm = ([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
         if NORMALIZE == "imagenet" else [transforms.ToTensor()])   # ToTensor 가 /255 수행

train_tf = transforms.Compose([transforms.ToPILImage()] + (_aug if USE_AUG else []) + _norm)
eval_tf  = transforms.Compose([transforms.ToPILImage()] + _norm)
log(f"Phase II 정규화 {NORMALIZE} | 증강 {'ON' if USE_AUG else 'OFF (논문 기준선)'}")

KIND2SUB = {"hand": "hand", "r1": "r1", "r2": "r2", "erased": "erased"}


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


def make_reg_backbone(pretrained=True):
    """Phase II 백본. REG_BACKBONE 한 줄로 교체됩니다.
       global_pool="" 로 특징'맵'을 받아야 뒤의 Conv3x3+MaxPool 구조가 논문 그대로
       유지됩니다 (풀링된 벡터가 아니라 (C,H,W) 가 나와야 함)."""
    try:
        m = timm.create_model(REG_BACKBONE, pretrained=pretrained,
                              num_classes=0, global_pool="")
    except Exception as e:
        raise SystemExit(f"[중단] 백본 '{REG_BACKBONE}' 생성 실패: {e}\n"
                         f"       timm 버전을 확인하거나 모델명을 수정하세요.")
    log(f"[backbone] timm '{REG_BACKBONE}' 로드 완료 (pretrained={pretrained})")
    return m


class AttentionLDLRegressor(nn.Module):
    def __init__(self, img_size=REG_SIZE, gender_dim=32, n_bins=AGE_BINS,
                 pretrained=True, head_relu=True, gender_relu=True):
        super().__init__()
        self.backbone = make_reg_backbone(pretrained)
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
#  [논문 미명시 -> 추론] G 는 240구간 합이 1이 아니므로 KL 성립을 위해 합=1 재정규화.
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
BATCH_SIZE   = 8       # 512 입력 + ConvNeXt-Tiny. OOM 이면 낮추세요
EPOCHS       = 120     # [논문] 120 epoch
LR_STEPS     = [(60, 1e-4), (90, 3e-5), (120, 1e-5)]   # 논문 LR 의 1/3
WEIGHT_DECAY = 0.05    # AdamW 표준값 (논문은 미언급)
NUM_WORKERS  = 0       # 윈도우 안전값
AUTO_RESUME  = True
LOG_EVERY    = 100

# ★ 조기 종료 (논문에는 없음 - 엔지니어링 추가) --------------------------------
EARLY_STOP_PATIENCE = 15   # val MAE 가 이 에폭 수만큼 개선 없으면 중단 (<0 이면 끔)
MIN_DELTA           = 0.01 # 개선으로 인정할 최소 MAE 감소(개월)
MIN_EPOCHS          = 65   # lr 1차 강하(60ep) 이후부터 조기종료 판단

# -- 모델 구조 ----------------------------------------------------------------
GENDER_EMB_DIM = 32    # [논문] Table III 최적
HEAD_RELU      = True  # [추론] Conv3x3 뒤 활성함수
GENDER_RELU    = True  # [추론] 성별 Dense 뒤 활성함수

# -- LDL ----------------------------------------------------------------------
LDL_DELTA  = 15.0      # [논문] 식(11) delta=15
LDL_LAMBDA = 0.5       # [논문] 최적 구간 lambda in [0.1, 1] 의 중앙값

ARCH = {"BACKBONE": REG_BACKBONE, "IMG_SIZE": REG_SIZE, "AGE_BINS": AGE_BINS,
        "GENDER_EMB_DIM": GENDER_EMB_DIM, "HEAD_RELU": HEAD_RELU,
        "GENDER_RELU": GENDER_RELU, "LDL_DELTA": LDL_DELTA, "LDL_LAMBDA": LDL_LAMBDA,
        "AGG_CHANNELS": list(AGG_CHANNELS), "NORMALIZE": NORMALIZE, "USE_AUG": USE_AUG,
        "TAU_R1": TAU_R1, "TAU_R2": TAU_R2_PAPER, "R2_METHOD": "paper_erase",
        "CLS_SIZE_REGION": CLS_SIZE_REGION,
        "PRE": PRE_PARAMS, "PRE_TAG": PRE_TAG, "HAND_SOURCE": str(HAND_CROP_DIR),
        "BOX_CFG": BOX_CFG, "MERGE_COMPONENTS": MERGE_COMPONENTS,
        "BANDS": BAND_CFG, "CLAMP_TO_BAND": CLAMP_TO_BAND,
        "RESIZE_MODE": RESIZE_MODE}
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
    """3단 스텝 LR 스케줄."""
    for upto, lr in LR_STEPS:
        if epoch <= upto:
            return lr
    return LR_STEPS[-1][1]


if EVAL_ONLY:
    log("--eval-only : Phase II 학습을 건너뛰고 best.pt 로 평가만 진행합니다.")
else:
    model = build_reg_model(ARCH, pretrained=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_at(1),
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
            log("  * 처음부터 다시 하려면 checkpoints_attention_v10 의 last.pt 삭제 후 실행")
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

results = {"method": "attention(R1/R2) + LDL, raw crop (no preprocessing)",
           "when": datetime.now().isoformat(timespec="seconds"),
           "arch": ck["arch"], "splits": {}}
lines = ["=" * 58,
         f"Attention + LDL 골연령 v10 (전처리 없음) | 채널 "
         f"{'+'.join(ck['arch'].get('AGG_CHANNELS', []))}",
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
        plt.axhline(4.3, ls="--", c="green", label="paper 4.3 (H+R1+R2, LDL)")
        plt.xlabel("Epoch"); plt.ylabel("months")
        plt.title("Learning curve (Attention+LDL, no preprocessing)")
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
#     내부에서 R1 / E / R2 를 CAM 으로 만들어 3채널을 구성합니다.
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
    """손 크롭 -> {kind: 512x512}. 학습과 동일:
       캔버스 -> (R1: 밴드 CAM) -> E -> (R2: E 로 학습한 cls-r2 의 CAM)."""
    out = {}
    src_full = to_gray8(hand_gray)
    ref, info = fit_canvas(src_full, REG_SIZE)
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
            out["r1"] = crop_roi(src_full, b1, info, REG_SIZE)
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
        out["r2"] = crop_roi(src_full, b2, info, REG_SIZE)
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
            im, _ = fit_canvas(to_gray8(g), REG_SIZE)
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
