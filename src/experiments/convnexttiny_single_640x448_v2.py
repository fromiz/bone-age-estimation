# -*- coding: utf-8 -*-
# =========================================================================
# 손 X-ray 골연령 - ConvNeXt-Tiny 단일 회귀  ★ v2
#   비정사각 640x448 · 이미지별 강도 정규화 · 코사인 완주 · TTA + 캘리브레이션
#
#   ▶ v1(convnexttiny_single_letterbox512_1.py) 결과가 기준선입니다
#       val MAE 5.94 / test MAE 4.53  (95% CI [4.07, 5.01], N=197)
#
#   ▶ v1 -> v2 변경점 5가지
#     ┌───┬────────────────┬────────────────┬───────────────────┬──────────────────────────┐
#     │ # │ 항목           │ v1             │ v2                │ 근거                     │
#     ├───┼────────────────┼────────────────┼───────────────────┼──────────────────────────┤
#     │ 1 │ EPOCHS/조기종료│ 60 / patience12│ 40 / 끔           │ v1은 29ep에 끊겨 LR이    │
#     │   │                │                │                   │ 피크의 84%인 채로 종료   │
#     │ 2 │ 캔버스         │ 512x512 정사각 │ 640x448 (HxW)     │ v1 여백 중앙 30.1%       │
#     │ 3 │ 화소 처리      │ 없음           │ p1p99 퍼센타일    │ 원본 std 6.9~55.6(8배)   │
#     │ 4 │ 증강/정칙화    │ Affine,dp 0.05 │ +shear/jitter/    │ train 5.26 vs val 6.66   │
#     │   │                │                │ erasing, dp 0.15  │ = 약한 과적합            │
#     │ 5 │ 추론           │ 단일 forward   │ 회전TTA + 캘리브  │ 0-4y bias +3.65,         │
#     │   │                │                │                   │ >16y bias -2.71 (축소)   │
#     └───┴────────────────┴────────────────┴───────────────────┴──────────────────────────┘
#
#   ▶ [중요] 이 파일은 더 이상 "전처리 없음" 실험이 아닙니다.
#       3번(퍼센타일 정규화) 때문에 v10 무전처리 대조군과 직접 비교할 수 없습니다.
#       2x2 ablation 표의 "전처리 없음 x 단일 회귀" 칸에는 v1 수치를 쓰세요.
#       NORM_MODE="none" 으로 두면 v1과 동일한 무전처리 조건이 됩니다.
#
#   ▶ 이미지 체인
#       crop_data 원본 --> [uint8 보장] --> [퍼센타일 스트레치] -->
#       비율유지 리사이즈 --> 0 패딩 --> 640x448
#       * 퍼센타일 스트레치는 '패딩 전, 원본 크롭 위에서' 수행합니다.
#         letterbox 이후에 하면 패딩 0이 통계에 섞여 스트레치가 망가집니다.
#
#   ▶ 백본 - 상업 사용 가능한 ConvNeXt V1 계열만
#       convnext_tiny.fb_in22k_ft_in1k_384  (기본, Apache-2.0 / MIT)
#       convnext_tiny.fb_in22k_ft_in1k      (224 ft. v8 과 동일 가중치)
#       convnext_tiny.in12k_ft_in1k         (timm 자체 학습, Apache-2.0)
#       ※ convnextv2_* 는 가중치가 CC BY-NC 4.0 -> 코드에서 차단
#
#   ▶ 실행: python convnexttiny_single_640x448_v2.py
#       - 창을 닫아도 서버에서 학습은 계속됩니다. 다시 실행하면 로그에 재부착.
#       - 각 스테이지는 완료 표식을 남기고 자동 스킵 -> 중단돼도 이어서 진행.
#
#   ▶ 실행 옵션
#       --fg             백그라운드 분리 없이 바로 실행(디버그용)
#       --eval-only      학습을 건너뛰고 best.pt 로 평가만
#       --rebuild-cache  캔버스 캐시를 강제로 재생성
#       --qc-only        QC 시트만 만들고 종료
#       --fresh          last.pt 를 무시하고 처음부터 학습
#       --no-tta         회전 TTA 끄기 (v1과 동일한 단일 forward 평가)
#       --no-calib       선형 캘리브레이션 끄기
#
#   ▶ 산출물 (모두 v1과 분리된 _v2 경로)
#       logs/convnext_single_v2_<타임스탬프>.log         실행 로그 전체
#       checkpoints_convnext_single_v2/env_<ts>.json     실행 환경 스냅샷
#       checkpoints_convnext_single_v2/best.pt           최고 검증 MAE 가중치
#       checkpoints_convnext_single_v2/last.pt           매 에폭 저장 (재개용)
#       checkpoints_convnext_single_v2/history.json      에폭별 지표
#       checkpoints_convnext_single_v2/calibration.json  배포용 캘리브레이션 계수
#       checkpoints_convnext_single_v2/results.txt|json  최종 성적표
#       checkpoints_convnext_single_v2/*.png             QC/곡선/산점도/GradCAM
# =========================================================================
from pathlib import Path
import os, sys, time, json, subprocess, platform
from datetime import datetime

PROJECT_DIR = Path(__file__).resolve().parent
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RUN_STATE = LOG_DIR / "convnext_single_v2_running.json"          # ★ v2 전용
_WORKER_ENV = "BONEAGE_CONVNEXT_SINGLE_V2_WORKER"                # ★ v2 전용

FOREGROUND    = "--fg" in sys.argv
EVAL_ONLY     = "--eval-only" in sys.argv
REBUILD_CACHE = "--rebuild-cache" in sys.argv
QC_ONLY       = "--qc-only" in sys.argv
FRESH         = "--fresh" in sys.argv
NO_TTA        = "--no-tta" in sys.argv
NO_CALIB      = "--no-calib" in sys.argv


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
    print(f" [v2] 학습이 백그라운드에서 실행 중입니다  (PID {pid})")
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
                        print("\n[프로세스 종료됨]", flush=True)
                        break
                    time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[로그 보기만 종료합니다 - 학습은 계속 진행 중]", flush=True)


def _spawn_detached():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"convnext_single_v2_{ts}.log"          # ★ v2 전용
    logf = open(log_path, "a", buffering=1, encoding="utf-8")
    env = dict(os.environ); env[_WORKER_ENV] = "1"
    env["PYTHONIOENCODING"] = "utf-8"; env["PYTHONUTF8"] = "1"
    env["BONEAGE_LOG_PATH"] = str(log_path)
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
            print("[v2] 이미 실행 중입니다 - 기존 로그에 다시 붙습니다.")
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

import random, math, copy
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")               # 백그라운드 실행 -> 창 없이 파일로만 저장
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF          # ★ v2: 회전 TTA

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


def _require(path, what):
    """경로가 없으면 즉시 중단. 잘못된 경로로 조용히 진행하지 않는다."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"[중단] {what} 경로가 없습니다:\n       {p}\n"
                         f"       실제 위치로 코드 상단 경로를 수정하세요.")
    return p


# -------------------------------------------------------------------------
# [B-1] 경로
#   ★ 데이터 입력 경로는 v1과 동일하게 유지합니다 (crop_data / crop_data_csv).
#     출력(캐시/체크포인트)만 _v2 로 분리됩니다.
# -------------------------------------------------------------------------
BASE_DIR      = Path(r"G:/Project/sinra_cho")
HAND_CROP_DIR = BASE_DIR / "crop_data_final_yolox_s_512"        # training / validation / test 하위폴더
CSV_DIR       = BASE_DIR / "crop_data_final_yolox_s_512"   # training.csv / validation.csv / test.csv

SPLIT_SUBDIR = {"train": "training", "val": "validation", "test": "test"}

_require(HAND_CROP_DIR, "손 크롭 폴더")
TRAIN_CSV = _require(CSV_DIR / "training.csv",   "training.csv")
VAL_CSV   = _require(CSV_DIR / "validation.csv", "validation.csv")
TEST_CSV  = _require(CSV_DIR / "test.csv",       "test.csv")


# =========================================================================
# [B-2] 재현 스위치
# =========================================================================
SEED = 42                   # 시드 앙상블 시 42 / 1337 / 2024 로 변경 (CKPT_DIR 도 함께)

# ── 이미지 준비 ★ v2 ────────────────────────────────────────────────
#   [v2-2] 비정사각 캔버스.
#     RSNA 손 크롭 종횡비(w/h) 중앙값 ≈ 0.65 -> 448/640 = 0.70
#     512x512 대비 픽셀 수 +9%, 여백은 중앙 30% -> 8~12% 로 감소
#     (ConvNeXt 는 완전 합성곱이라 비정사각 입력이 그대로 동작합니다)
IMG_H, IMG_W = 640, 448
IMG_SIZE     = (IMG_H, IMG_W)   # 하위 호환용 튜플

RESIZE_MODE = "letterbox"   # "letterbox"(비율유지+패딩, 권장) | "stretch"(강제 리사이즈)
PAD_VALUE   = 0             # 배경색 추정 제거 -> 상수 0 고정
PAD_ANCHOR  = "center"      # "center" | "topleft"

#   [v2-3][추론] 이미지별 강도 정규화. "none" 이면 v1과 동일한 무전처리 조건
NORM_MODE   = "p1p99"       # "none" | "p1p99" | "p2p98"

# ── 백본 (상업 사용 가능한 ConvNeXt V1 계열만) ───────────────────────
BACKBONE     = "convnext_tiny.fb_in22k_ft_in1k_384"
BACKBONE_ALT = ["convnext_tiny.fb_in22k_ft_in1k",   # 224 ft. (v8 과 동일 가중치)
                "convnext_tiny.in12k_ft_in1k"]      # timm 자체 학습 (Apache-2.0)
DROP_PATH    = 0.15         # [v2-4] 0.05 -> 0.15. ConvNeXt-T 권장 0.1~0.2

# ── 헤드 ─────────────────────────────────────────────────────────────
HEAD_TYPE      = "gap"      # "gap"(권장) | "paper"(Conv3x3+MaxPool+Flatten)
HEAD_DIM       = 512
GENDER_EMB_DIM = 32         # [논문] 식(15) k=32
DROPOUT        = 0.10       # [v2-4] 0.2 -> 0.10

# ── 최적화 ───────────────────────────────────────────────────────────
BATCH_SIZE    = 8           # 640x448 은 512x512 대비 픽셀 +9%. OOM 이면 6 또는 4
ACCUM_STEPS   = 4           # 유효 배치 = BATCH_SIZE * ACCUM_STEPS = 32
EPOCHS        = 40          # [v2-1] 코사인이 이 값에 맞춰 감깁니다
LR_HEAD       = 1e-4
LR_BACKBONE   = 4e-5        # L1 손실 전환에 맞춘 값 (1e-5 -> 4e-5)
WEIGHT_DECAY  = 0.05        # AdamW. norm/bias 에는 적용하지 않음
WARMUP_EPOCHS = 3
MIN_LR_RATIO  = 0.02        # cosine 최저 LR = base * 이 값
CLIP_GRAD     = 1.0         # None 이면 끔
USE_GRAD_CKPT = False       # OOM 이면 True (속도 -25%, VRAM -40%)

# ── 손실 ─────────────────────────────────────────────────────────────
#   [주의] HUBER_BETA 는 z-정규화 스케일 기준입니다. AGE_STD 가 크면
#          beta 0.5 가 사실상 순수 MSE 처럼 동작하므로 L1 을 기본으로 둡니다.
LOSS_TYPE  = "l1"           # "huber" | "l1"
HUBER_BETA = 0.5

# ── 정칙화 / 증강 ★ v2 강화 ─────────────────────────────────────────
USE_AUG       = True
AUG_ROT_DEG   = 12
AUG_TRANSLATE = 0.06
AUG_SCALE     = (0.92, 1.08)
AUG_SHEAR     = 4           # [v2-4] 신규
AUG_JITTER    = 0.18        # [v2-4] 신규. brightness/contrast (RSNA 노출 편차 모사)
AUG_ERASE_P   = 0.25        # [v2-4] 신규. scale 1~4% 로 작게 (크면 골단선을 지움)
# 좌우 반전은 넣지 않습니다 - RSNA 는 전부 좌수 촬영이라 해부학적으로 없는 입력입니다.

# ── EMA ──────────────────────────────────────────────────────────────
USE_EMA   = True
EMA_DECAY = 0.9995          # [v2-1] 0.999 -> 0.9995 (유효 창 ≈ 5에폭)

# ── 추론 ★ v2 신규 ──────────────────────────────────────────────────
TTA_ANGLES = () if NO_TTA else (0, -4, 4)   # 빈 튜플/(0,) 이면 TTA 끔
USE_CALIB_OPT = not NO_CALIB                # val 에서 개선될 때만 실제 적용

# ── 조기 종료 / 실행 ─────────────────────────────────────────────────
EARLY_STOP_PATIENCE = 0     # [v2-1] 0 = 끔. 코사인은 끝까지 가야 의미가 있습니다.
                            #   v1은 patience 12 로 29ep 에 끊겨 LR 이 피크의 84%
                            #   인 상태로 종료됐습니다 = 수렴 구간을 못 밟았습니다.
                            #   best.pt 는 계속 갱신되므로 끝까지 돌려도 안전합니다.
MIN_DELTA           = 0.0   # [v2-1] 0.01 은 노이즈 수준이라 무의미
MIN_EPOCHS          = EPOCHS
NUM_WORKERS         = 0     # [중요] Windows 는 반드시 0.
                            #   이 스크립트는 전체가 모듈 최상위에서 실행되므로,
                            #   workers>0 이면 spawn 된 워커가 스크립트를 다시 import 하면서
                            #   학습이 통째로 재실행됩니다. 아래에서 강제로 0 으로 고정합니다.
N_QC                = 8
BOOTSTRAP_N         = 2000

if os.name == "nt" and NUM_WORKERS > 0:
    NUM_WORKERS = 0
    _WORKERS_FORCED = True
else:
    _WORKERS_FORCED = False

# ── 제외 ID ──────────────────────────────────────────────────────────
#   True 로 두면 v8/v10 과 동일한 183건이 제외됩니다.
#   test 에 필터를 적용하면 낙관적 편향이 생기므로, 평가 단계에서는
#   '필터 적용' 과 '필터 없음' 두 수치를 모두 출력합니다.
USE_EXCLUDE = False

_EXCLUDE_FULL = {}
EXCLUDE_IDS = _EXCLUDE_FULL if USE_EXCLUDE else set()

# ── 출력 경로 ★ v2 전용 (설정이 다르면 캐시가 자동으로 분리됩니다) ──
PRE_TAG   = f"raw{IMG_H}x{IMG_W}_{RESIZE_MODE}_pad{PAD_VALUE}_{PAD_ANCHOR}_n{NORM_MODE}"
CACHE_DIR = BASE_DIR / "cache_convnext_single_v3" / PRE_TAG        # ★ v2
CKPT_DIR  = BASE_DIR / "checkpoints_convnext_single_v3"            # ★ v2
SPLITS    = ("train", "val", "test")

# v1 결과 파일 (있으면 평가 단계에서 자동 비교)
V1_RESULTS = BASE_DIR / "checkpoints_convnext_single" / "results.json"

if "_v2" not in CKPT_DIR.name:
    raise SystemExit("[중단] v1 체크포인트 경로를 덮어쓰려 합니다. CKPT_DIR 을 확인하세요.")

CACHE_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
for sp in SPLITS:
    (CACHE_DIR / sp).mkdir(parents=True, exist_ok=True)

CACHE_DONE   = CACHE_DIR / "_DONE_cache.json"
BEST_CKPT    = CKPT_DIR / "best.pt"
LAST_CKPT    = CKPT_DIR / "last.pt"
HISTORY_JSON = CKPT_DIR / "history.json"
RESULTS_TXT  = CKPT_DIR / "results.txt"
RESULTS_JSON = CKPT_DIR / "results.json"
CALIB_JSON   = CKPT_DIR / "calibration.json"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
# 정규화 공간에서 '픽셀값 0' 에 해당하는 값.
#   회전 TTA 의 여백을 여기로 채웁니다. 0 을 넣으면 중간 회색(≈124)이 들어가
#   TTA 가 오히려 손해가 됩니다.
PAD_NORM = [-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")


# =========================================================================
# [B-3] 실행 환경 기록 - 나중에 "그때 무슨 조건이었지?" 를 없애기 위한 스냅샷
# =========================================================================
def _pkg_versions():
    out = {}
    for name in ("torch", "torchvision", "timm", "numpy", "pandas", "cv2",
                 "matplotlib", "PIL"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "?")
        except Exception:
            out[name] = None
    return out


def dump_env(extra=None):
    """실행 환경 + 전체 설정을 JSON 한 파일로 남깁니다."""
    gpu = {}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        gpu = {"name": p.name, "vram_gb": round(p.total_memory / 1024 ** 3, 2),
               "capability": f"{p.major}.{p.minor}", "cuda": torch.version.cuda,
               "cudnn": torch.backends.cudnn.version()}
    info = {
        "version": "v2",
        "run_ts": RUN_TS,
        "script": str(Path(__file__).resolve()),
        "argv": sys.argv[1:],
        "pid": os.getpid(),
        "log_path": os.environ.get("BONEAGE_LOG_PATH"),
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "packages": _pkg_versions(),
        "device": str(device), "gpu": gpu, "amp": bool(USE_AMP),
        "seed": SEED,
        "paths": {"BASE_DIR": str(BASE_DIR), "HAND_CROP_DIR": str(HAND_CROP_DIR),
                  "CSV_DIR": str(CSV_DIR), "CACHE_DIR": str(CACHE_DIR),
                  "CKPT_DIR": str(CKPT_DIR)},
        "image_prep": {"preprocess": NORM_MODE, "img_h": IMG_H, "img_w": IMG_W,
                       "resize_mode": RESIZE_MODE, "pad_value": PAD_VALUE,
                       "pad_anchor": PAD_ANCHOR, "pre_tag": PRE_TAG},
        "model": {"backbone": BACKBONE, "backbone_alt": BACKBONE_ALT,
                  "drop_path": DROP_PATH, "head_type": HEAD_TYPE,
                  "head_dim": HEAD_DIM, "gender_emb_dim": GENDER_EMB_DIM,
                  "dropout": DROPOUT, "grad_ckpt": USE_GRAD_CKPT},
        "optim": {"batch_size": BATCH_SIZE, "accum_steps": ACCUM_STEPS,
                  "effective_batch": BATCH_SIZE * ACCUM_STEPS, "epochs": EPOCHS,
                  "lr_head": LR_HEAD, "lr_backbone": LR_BACKBONE,
                  "weight_decay": WEIGHT_DECAY, "warmup_epochs": WARMUP_EPOCHS,
                  "min_lr_ratio": MIN_LR_RATIO, "clip_grad": CLIP_GRAD,
                  "loss": LOSS_TYPE, "huber_beta": HUBER_BETA},
        "regularize": {"use_aug": USE_AUG, "rot_deg": AUG_ROT_DEG,
                       "translate": AUG_TRANSLATE, "scale": list(AUG_SCALE),
                       "shear": AUG_SHEAR, "jitter": AUG_JITTER,
                       "erase_p": AUG_ERASE_P,
                       "use_ema": USE_EMA, "ema_decay": EMA_DECAY},
        "inference": {"tta_angles": list(TTA_ANGLES), "calib_enabled": USE_CALIB_OPT},
        "earlystop": {"patience": EARLY_STOP_PATIENCE, "min_delta": MIN_DELTA,
                      "min_epochs": MIN_EPOCHS},
        "data": {"use_exclude": USE_EXCLUDE, "n_exclude": len(EXCLUDE_IDS)},
    }
    if extra:
        info.update(extra)
    p = CKPT_DIR / f"env_{RUN_TS}.json"
    json.dump(info, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return info, p


print("=" * 72)
log("ConvNeXt-Tiny 단일 회귀 골연령 v2 - 640x448 + 강도정규화 + TTA 시작")
log(f"Python {sys.version.split()[0]} | {platform.system()} {platform.release()}")
log(f"PyTorch {torch.__version__} | torchvision {torchvision.__version__} | timm {timm.__version__}")
log(f"OpenCV {cv2.__version__} | numpy {np.__version__} | pandas {pd.__version__}")
log(f"device {device} | CUDA {torch.cuda.is_available()} | AMP {USE_AMP}")
if torch.cuda.is_available():
    _p = torch.cuda.get_device_properties(0)
    log(f"GPU {_p.name} | VRAM {_p.total_memory/1024**3:.1f}GB | "
        f"sm_{_p.major}{_p.minor} | CUDA {torch.version.cuda}")
log(f"BASE_DIR       {BASE_DIR}")
log(f"HAND_CROP_DIR  {HAND_CROP_DIR}   (v1과 동일)")
log(f"CSV_DIR        {CSV_DIR}   (v1과 동일)")
log(f"CACHE_DIR      {CACHE_DIR}")
log(f"CKPT_DIR       {CKPT_DIR}")
log(f"이미지 준비    {RESIZE_MODE} -> {IMG_H}x{IMG_W}(HxW) | pad {PAD_VALUE} | "
    f"강도정규화 {NORM_MODE}")
log(f"백본           {BACKBONE} (drop_path {DROP_PATH})")
log(f"헤드           {HEAD_TYPE} | 성별임베딩 {GENDER_EMB_DIM} | dropout {DROPOUT}")
log(f"최적화         AdamW | lr {LR_BACKBONE:.0e}/{LR_HEAD:.0e} | wd {WEIGHT_DECAY} | "
    f"배치 {BATCH_SIZE}x{ACCUM_STEPS}={BATCH_SIZE*ACCUM_STEPS}")
log(f"스케줄         warmup {WARMUP_EPOCHS}ep + cosine(min ratio {MIN_LR_RATIO}) | "
    f"총 {EPOCHS}ep")
log(f"조기종료       {'끔 (코사인 완주)' if EARLY_STOP_PATIENCE <= 0 else EARLY_STOP_PATIENCE}"
    f" | min_delta {MIN_DELTA}")
log(f"손실 {LOSS_TYPE}"
    + (f"(beta {HUBER_BETA})" if LOSS_TYPE == "huber" else "")
    + f" | 증강 {USE_AUG} | EMA {USE_EMA}({EMA_DECAY})")
if USE_AUG:
    log(f"증강 상세      affine rot{AUG_ROT_DEG}° tr{AUG_TRANSLATE} sc{AUG_SCALE} "
        f"sh{AUG_SHEAR} | jitter ±{AUG_JITTER} | erasing p={AUG_ERASE_P}")
log(f"추론           TTA {list(TTA_ANGLES) if len(TTA_ANGLES) > 1 else '끔'} | "
    f"캘리브레이션 {'자동판정' if USE_CALIB_OPT else '끔'}")
log(f"제외 ID        {len(EXCLUDE_IDS)}개 (USE_EXCLUDE={USE_EXCLUDE})")
log(f"DataLoader     num_workers={NUM_WORKERS}"
    + ("  [Windows 안전장치로 0 강제]" if _WORKERS_FORCED else ""))
_flags = [f for f, on in [("--eval-only", EVAL_ONLY), ("--rebuild-cache", REBUILD_CACHE),
                          ("--qc-only", QC_ONLY), ("--fresh", FRESH),
                          ("--no-tta", NO_TTA), ("--no-calib", NO_CALIB),
                          ("--fg", FOREGROUND)] if on]
log(f"실행 옵션      {' '.join(_flags) if _flags else '(없음)'}")
if NORM_MODE != "none":
    log("[알림] NORM_MODE != 'none' 이므로 이 실행은 '전처리 없음' 대조군이 아닙니다.")
    log("       2x2 ablation 표의 무전처리 칸에는 v1 수치를 사용하세요.")
print("=" * 72, flush=True)
# [주의] RTX 5060(Blackwell, sm_120)은 최신 PyTorch 필요.
#   CUDA=False 또는 'no kernel image' 오류 시:
#   pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128


# =========================================================================
# [C] 한글 경로 I/O
#     cv2.imread/imwrite 는 경로에 한글이 있으면 '조용히' 실패합니다.
# =========================================================================
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
# [D] 이미지 준비 ★ v2
#       to_gray8()       : 8bit 보장 (이미 8bit 면 그대로 통과)
#       norm_intensity() : [v2 신규] 이미지별 퍼센타일 스트레치
#       fit_canvas()     : [v2 변경] 비정사각 캔버스 + 패딩
# =========================================================================
def to_gray8(g):
    """16bit·float 입력만 8bit 로 낮춥니다.
       입력이 이미 uint8 이면 아무 연산도 하지 않고 그대로 반환합니다."""
    if g.dtype == np.uint8:
        return g
    g = g.astype(np.float32)
    lo, hi = float(g.min()), float(g.max())
    if hi - lo < 1e-6:
        return np.zeros(g.shape, np.uint8)
    return np.clip((g - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def norm_intensity(g, mode=None):
    """[추론] 논문 미명시. 이미지별 퍼센타일 스트레치.

    v1 QC 에서 원본 std 가 6.9(id 2821) ~ 55.6(id 14141) 로 8배 차이났습니다.
    고정 ImageNet mean/std 로는 이 편차를 보정할 수 없어, 골단선 대비가
    이미지마다 완전히 다른 스케일로 네트워크에 들어갑니다.

    YOLO 크롭의 0-패딩 배경(g==0)은 통계에서 제외합니다.
    """
    mode = NORM_MODE if mode is None else mode
    if mode == "none":
        return g
    v = g[g > 0]
    if v.size < 1000:                       # 거의 빈 이미지는 건드리지 않음
        return g
    if mode == "p1p99":
        lo, hi = np.percentile(v, (1, 99))
    elif mode == "p2p98":
        lo, hi = np.percentile(v, (2, 98))
    else:
        raise SystemExit(f"[중단] 알 수 없는 NORM_MODE: {mode}")
    if hi - lo < 1e-3:
        return g
    out = (g.astype(np.float32) - lo) / (hi - lo) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def fit_canvas(g, h_out=None, w_out=None, mode=None,
               pad_val=None, anchor=None, norm=None):
    """h_out x w_out 캔버스로 맞춥니다. 반환 (canvas, info)

    [v2] 정사각 강제를 폐지하고 H/W 를 따로 받습니다.
         스케일 = min(h_out/h, w_out/w) -> 양 변 모두 넘치지 않는 최대 배율.
         강도 정규화는 '패딩 전' 에 수행합니다.
    """
    h_out   = IMG_H       if h_out   is None else h_out
    w_out   = IMG_W       if w_out   is None else w_out
    mode    = RESIZE_MODE if mode    is None else mode
    pad_val = PAD_VALUE   if pad_val is None else pad_val
    anchor  = PAD_ANCHOR  if anchor  is None else anchor

    g = to_gray8(g)
    g = norm_intensity(g, NORM_MODE if norm is None else norm)   # ★ 패딩 전
    h, w = g.shape[:2]

    if mode == "stretch":
        interp = cv2.INTER_AREA if (h_out < h and w_out < w) else cv2.INTER_CUBIC
        out = cv2.resize(g, (w_out, h_out), interpolation=interp)
        return out, {"sx": w_out / w, "sy": h_out / h, "src": (h, w),
                     "pad": (0, 0, 0, 0), "pad_frac": 0.0}

    # letterbox: 비율 왜곡 0
    s  = min(h_out / float(h), w_out / float(w))
    nh = max(1, min(h_out, int(round(h * s))))
    nw = max(1, min(w_out, int(round(w * s))))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    r = cv2.resize(g, (nw, nh), interpolation=interp)

    top  = 0 if anchor == "topleft" else (h_out - nh) // 2
    left = 0 if anchor == "topleft" else (w_out - nw) // 2
    out = cv2.copyMakeBorder(r, top, h_out - nh - top, left, w_out - nw - left,
                             cv2.BORDER_CONSTANT, value=int(pad_val))
    return out, {"sx": s, "sy": s, "src": (h, w),
                 "pad": (top, h_out - nh - top, left, w_out - nw - left),
                 "pad_frac": 1.0 - (nh * nw) / float(h_out * w_out)}


def _arch_hw(arch):
    """arch 에서 (H, W) 추출. v1 체크포인트(IMG_SIZE=정수 512)도 지원."""
    if "IMG_H" in arch and "IMG_W" in arch:
        return int(arch["IMG_H"]), int(arch["IMG_W"])
    s = arch.get("IMG_SIZE", 512)
    if isinstance(s, (list, tuple)):
        return int(s[0]), int(s[1])
    return int(s), int(s)


# =========================================================================
# [E] 라벨 로드 & 파일 인덱싱
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
    if not (age_col and sex_col):
        raise SystemExit(f"[중단] 컬럼 탐지 실패: {list(df.columns)}")

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


def index_files(split):
    """{id(stem): 파일경로} 인덱스. 하위 폴더까지 재귀 탐색."""
    d = HAND_CROP_DIR / SPLIT_SUBDIR[split]
    if not d.exists():
        raise SystemExit(f"[중단] {split} 이미지 폴더 없음: {d}")
    idx = {}
    for ext in ("*.png", "*.PNG", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
        for p in d.rglob(ext):
            idx.setdefault(p.stem.strip(), p)
    return idx


log("라벨 로드 중...")
SPLIT_DFS, FILE_INDEX = {}, {}
for _sp, _csv in [("train", TRAIN_CSV), ("val", VAL_CSV), ("test", TEST_CSV)]:
    _df  = load_labels(_csv)
    _idx = index_files(_sp)
    FILE_INDEX[_sp] = _idx
    _before = len(_df)
    if EXCLUDE_IDS:
        _df = _df[~_df["id"].isin(EXCLUDE_IDS)]
    _n_excl = _before - len(_df)
    _df = _df[_df["id"].isin(_idx.keys())].reset_index(drop=True)
    _df["path"] = _df["id"].map(lambda i: str(_idx[i]))
    SPLIT_DFS[_sp] = _df
    log(f"  {_sp:<5} 라벨 {_before:>6,} | 파일 {len(_idx):>6,} | "
        f"제외 {_n_excl:>4,} | 미크롭 {_before-_n_excl-len(_df):>4,} -> 사용 {len(_df):>6,}")

train_df, val_df, test_df = SPLIT_DFS["train"], SPLIT_DFS["val"], SPLIT_DFS["test"]
if len(train_df) == 0:
    raise SystemExit("[중단] train 사용 가능 이미지가 0장입니다. 경로/CSV 를 확인하세요.")

# 타깃 z-정규화 상수는 반드시 train 에서만 (val/test 누수 방지)
AGE_MEAN = float(train_df["boneage"].mean())
AGE_STD  = float(train_df["boneage"].std())
log(f"타깃 정규화 (train 기준) {AGE_MEAN:.2f} ± {AGE_STD:.2f} 개월")
log(f"남아 비율 {train_df.male.mean():.1%} | 연령 범위 "
    f"{train_df.boneage.min():.0f}~{train_df.boneage.max():.0f} 개월")

_env_info, _env_path = dump_env({"data_counts": {k: int(len(v)) for k, v in SPLIT_DFS.items()},
                                 "age_mean": AGE_MEAN, "age_std": AGE_STD})
log(f"실행 환경 스냅샷 저장: {_env_path}")


# =========================================================================
# [F] 스테이지 1 - 캔버스 캐시  (완료 표식 있으면 자동 스킵)
#     ★ v1 캐시는 재사용되지 않습니다 (PRE_TAG 에 크기·정규화 모드가 포함).
# =========================================================================
PRE_INFO = {"preprocess": NORM_MODE, "img_h": IMG_H, "img_w": IMG_W,
            "resize_mode": RESIZE_MODE, "pad_value": PAD_VALUE,
            "pad_anchor": PAD_ANCHOR,
            "n": {k: int(len(v)) for k, v in SPLIT_DFS.items()}}


def cache_valid():
    if REBUILD_CACHE or not CACHE_DONE.exists():
        return False
    try:
        return json.load(open(CACHE_DONE, encoding="utf-8")) == PRE_INFO
    except Exception:
        return False


def build_cache():
    if cache_valid():
        log("[스테이지1] 캐시 유효 - 스킵 (재생성: --rebuild-cache)")
        return
    log(f"[스테이지1] {IMG_H}x{IMG_W} 캔버스 캐시 생성 시작 (norm={NORM_MODE})")
    pad_fracs = []
    for sp in SPLITS:
        df = SPLIT_DFS[sp]
        if not len(df):
            continue
        made = skipped = failed = 0
        t0 = time.time()
        for n, (_, r) in enumerate(df.iterrows(), 1):
            dst = CACHE_DIR / sp / f"{r['id']}.png"
            if dst.exists() and not REBUILD_CACHE:
                skipped += 1
                continue
            g = imread_kr(r["path"], cv2.IMREAD_GRAYSCALE)
            if g is None:
                failed += 1
                log(f"    [경고] 읽기 실패: {r['path']}")
                continue
            canvas, info = fit_canvas(g)
            if not imwrite_kr(dst, canvas):
                failed += 1
                log(f"    [경고] 쓰기 실패: {dst}")
                continue
            pad_fracs.append(info["pad_frac"])
            made += 1
            if n % 2000 == 0:
                log(f"    [{sp}] {n:,}/{len(df):,} ({time.time()-t0:.0f}s)")
        log(f"  [{sp}] 생성 {made:,} · 스킵 {skipped:,} · 실패 {failed:,} "
            f"({time.time()-t0:.0f}s)")
    if pad_fracs:
        pf = np.array(pad_fracs)
        log(f"  여백 비율: 중앙 {np.median(pf):.1%} · 90%tile {np.percentile(pf,90):.1%} "
            f"· 최대 {pf.max():.1%}")
        log("  (v1 512x512 기준: 중앙 30.1% · 90%tile 39.5% · 최대 52.9%)")
        if np.median(pf) > 0.20:
            log("  [경고] 여백이 여전히 20%를 넘습니다 -> IMG_H/IMG_W 종횡비를 재검토하세요.")
        else:
            log("  [정상] 여백이 v1 대비 크게 줄었습니다.")
    json.dump(PRE_INFO, open(CACHE_DONE, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    log(f"[스테이지1] 완료 표식: {CACHE_DONE}")


build_cache()


# =========================================================================
# [G] QC ★ v2 - 검증 방향이 v1과 반대입니다
#     v1: "화소값을 안 건드렸음" 을 증명 (std비 ≈ 1.0)
#     v2: "이미지 간 대비 편차가 실제로 줄었음" 을 증명
#         -> 유효std 의 변동계수(CV)가 원본보다 뚜렷이 작아야 정상
# =========================================================================
def qc_sheet():
    log("[QC] 강도 정규화 검증 시트 생성")
    sample = train_df.sample(min(N_QC, len(train_df)), random_state=SEED)
    fig, axes = plt.subplots(len(sample), 2, figsize=(7, 3.0 * len(sample)))
    axes = np.atleast_2d(axes)
    rows = []
    for i, (_, r) in enumerate(sample.iterrows()):
        raw = imread_kr(r["path"], cv2.IMREAD_GRAYSCALE)
        cv_ = imread_kr(CACHE_DIR / "train" / f"{r['id']}.png", cv2.IMREAD_GRAYSCALE)
        if raw is None or cv_ is None:
            continue
        _, info = fit_canvas(raw)
        top, bot, left, right = info["pad"]
        eff = cv_[top:(cv_.shape[0] - bot if bot else None),
                  left:(cv_.shape[1] - right if right else None)]
        rows.append({"id": r["id"], "원본크기": f"{raw.shape[1]}x{raw.shape[0]}",
                     "원본범위": f"{raw.min()}~{raw.max()}",
                     "원본std": round(float(raw.std()), 1),
                     "유효std": round(float(eff.std()), 1),
                     "여백%": f"{info['pad_frac']:.0%}"})
        for ax, (t, im) in zip(axes[i], [(f"raw {r['id']}", raw),
                                         (f"canvas {IMG_H}x{IMG_W}", cv_)]):
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            ax.set_title(t, fontsize=9); ax.axis("off")
    plt.tight_layout()
    p = CKPT_DIR / "qc_intensity_norm.png"
    plt.savefig(p, dpi=110); plt.close()

    qc = pd.DataFrame(rows)
    if not len(qc):
        raise SystemExit("[중단] QC 샘플을 하나도 읽지 못했습니다. "
                         "캐시(PART 1)와 원본 경로를 확인하세요.")
    log(f"[QC] 시트 저장: {p}")
    log("\n" + qc.to_string(index=False))

    cv_raw = float(qc["원본std"].std() / max(1e-6, qc["원본std"].mean()))
    cv_eff = float(qc["유효std"].std() / max(1e-6, qc["유효std"].mean()))
    log(f"[QC] std 변동계수(CV):  원본 {cv_raw:.3f}  ->  캔버스 {cv_eff:.3f}")
    if NORM_MODE == "none":
        log("[QC] NORM_MODE='none' - 두 값이 비슷해야 정상입니다 (v1과 동일 조건).")
    elif cv_eff < cv_raw * 0.6:
        log("[QC][정상] 이미지 간 대비 편차가 크게 줄었습니다.")
    else:
        log("[QC][경고] 편차가 충분히 안 줄었습니다 -> norm_intensity 의 g>0 마스크가 "
            "제대로 동작하는지(배경이 정말 0인지) 확인하세요.")

    pf = qc["여백%"].str.rstrip("%").astype(float)
    log(f"[QC] 샘플 여백 중앙 {pf.median():.0f}% (v1 512정사각 기준 30%)")
    qc.to_csv(CKPT_DIR / "qc_intensity_norm.csv", index=False, encoding="utf-8-sig")


qc_sheet()
if QC_ONLY:
    log("--qc-only: QC 시트만 만들고 종료합니다.")
    raise SystemExit(0)


# =========================================================================
# [H] Dataset · Transform ★ v2 증강 강화
#   ToPILImage -> RandomAffine(rot/translate/scale/shear) -> ColorJitter
#              -> ToTensor -> Normalize -> RandomErasing
# =========================================================================
_aug = [
    transforms.RandomAffine(degrees=AUG_ROT_DEG,
                            translate=(AUG_TRANSLATE, AUG_TRANSLATE),
                            scale=AUG_SCALE, shear=AUG_SHEAR, fill=PAD_VALUE),
    transforms.ColorJitter(brightness=AUG_JITTER, contrast=AUG_JITTER),   # [v2]
]

train_tf = transforms.Compose(
    [transforms.ToPILImage()] + (_aug if USE_AUG else []) +
    [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)] +
    ([transforms.RandomErasing(p=AUG_ERASE_P, scale=(0.01, 0.04),
                               ratio=(0.5, 2.0), value=0.0)] if USE_AUG else [])
)
eval_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class BoneAgeDataset(Dataset):
    """캐시된 HxW 캔버스 -> 3채널 복제 -> 증강 -> ImageNet 정규화."""

    def __init__(self, df, split, tf):
        self.df, self.split, self.tf = df.reset_index(drop=True), split, tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        g = imread_kr(CACHE_DIR / self.split / f"{r['id']}.png", cv2.IMREAD_GRAYSCALE)
        if g is None:
            raise FileNotFoundError(
                f"캐시 없음: {self.split}/{r['id']}.png - --rebuild-cache 로 다시 만드세요.")
        if g.shape[:2] != (IMG_H, IMG_W):
            # 캐시본은 이미 강도 정규화가 적용된 상태 -> 재적용 금지 (이중 스트레치 방지)
            g, _ = fit_canvas(g, norm="none")
        x  = self.tf(np.stack([g] * 3, axis=-1))
        gd = torch.tensor([float(r["male"])], dtype=torch.float32)
        ym = torch.tensor([float(r["boneage"])], dtype=torch.float32)
        yn = (ym - AGE_MEAN) / AGE_STD
        return x, gd, yn, ym


# =========================================================================
# [I] 모델 - ConvNeXt-Tiny + 성별 임베딩
#     상업 사용 불가한 ConvNeXt V2 가중치는 차단합니다.
#     [v2] 입력 640x448 -> 최종 특징맵 20x14x768 (v1은 16x16x768)
# =========================================================================
def make_backbone(name=None, pretrained=True, drop_path=None):
    name      = BACKBONE  if name      is None else name
    drop_path = DROP_PATH if drop_path is None else drop_path
    last_err = None
    for cand in [name] + [c for c in BACKBONE_ALT if c != name]:
        if "convnextv2" in cand:
            raise SystemExit(f"[중단] {cand} 는 CC BY-NC 4.0 가중치입니다 (상업 사용 불가).")
        try:
            m = timm.create_model(cand, pretrained=pretrained, num_classes=0,
                                  global_pool="", drop_path_rate=drop_path)
            if cand != name:
                log(f"[backbone] '{name}' 실패 -> '{cand}' 로 폴백")
            log(f"[backbone] {cand} (pretrained={pretrained}, drop_path={drop_path})")
            return m, cand
        except Exception as e:
            last_err = e
            continue
    raise SystemExit(f"[중단] 백본 생성 실패: {last_err}")


class ConvNeXtRegressor(nn.Module):
    """gap  : GAP -> LayerNorm -> Dense(HEAD_DIM) -> 성별 concat -> Dense(128) -> 1
       paper: Conv3x3(256) -> MaxPool3x3 -> Flatten -> 성별 concat -> Dense(128) -> 1
              (논문 식 13~17 / Xception 노트북과 동일 구조)"""

    def __init__(self, backbone_name=BACKBONE, img_hw=None, head_type=HEAD_TYPE,
                 head_dim=HEAD_DIM, gender_dim=GENDER_EMB_DIM, dropout=DROPOUT,
                 drop_path=DROP_PATH, pretrained=True, verbose=True):
        super().__init__()
        img_hw = (IMG_H, IMG_W) if img_hw is None else img_hw
        self.backbone, self.backbone_name = make_backbone(backbone_name, pretrained, drop_path)
        self.head_type = head_type

        h, w = int(img_hw[0]), int(img_hw[1])
        with torch.no_grad():
            feat = self.backbone(torch.zeros(1, 3, h, w))       # [v2] 비정사각 프로브
        C, H, W = int(feat.shape[1]), int(feat.shape[2]), int(feat.shape[3])
        if verbose:
            log(f"[head] 입력 {h}x{w} -> 특징맵 {C}x{H}x{W} · head_type={head_type}")

        if head_type == "gap":
            self.norm = nn.LayerNorm(C)
            self.proj = nn.Sequential(nn.Linear(C, head_dim), nn.GELU(), nn.Dropout(dropout))
            img_out = head_dim
        elif head_type == "paper":
            self.conv = nn.Conv2d(C, 256, 3, padding=1)
            self.pool = nn.MaxPool2d(3, 3)
            self.drop = nn.Dropout(dropout)
            img_out = 256 * (H // 3) * (W // 3)
        else:
            raise SystemExit(f"[중단] 알 수 없는 HEAD_TYPE: {head_type}")

        self.gender = nn.Sequential(nn.Linear(1, gender_dim), nn.GELU())
        self.fc = nn.Sequential(nn.Linear(img_out + gender_dim, 128), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x, g):
        f = self.backbone(x)
        if self.head_type == "gap":
            z = F.adaptive_avg_pool2d(f, 1).flatten(1)
            z = self.proj(self.norm(z))
        else:
            z = self.drop(torch.flatten(self.pool(F.relu(self.conv(f))), 1))
        e = self.gender(g)
        return self.fc(torch.cat([z, e], dim=1)).squeeze(1)


def build_model(arch, pretrained=False, verbose=True):
    """ARCH 딕셔너리만으로 모델 복원 -> 체크포인트가 자체 완결."""
    m = ConvNeXtRegressor(
        backbone_name=arch.get("BACKBONE", BACKBONE),
        img_hw=_arch_hw(arch),
        head_type=arch.get("HEAD_TYPE", "gap"),
        head_dim=arch.get("HEAD_DIM", 512),
        gender_dim=arch.get("GENDER_EMB_DIM", 32),
        dropout=arch.get("DROPOUT", 0.10),
        drop_path=arch.get("DROP_PATH", 0.15),
        pretrained=pretrained, verbose=verbose,
    )
    if USE_GRAD_CKPT and hasattr(m.backbone, "set_grad_checkpointing"):
        m.backbone.set_grad_checkpointing(True)
        if verbose:
            log("[mem] gradient checkpointing ON")
    return m.to(device).to(memory_format=torch.channels_last)


ARCH = {"BACKBONE": BACKBONE, "IMG_H": IMG_H, "IMG_W": IMG_W,
        "IMG_SIZE": [IMG_H, IMG_W],
        "HEAD_TYPE": HEAD_TYPE, "HEAD_DIM": HEAD_DIM,
        "GENDER_EMB_DIM": GENDER_EMB_DIM, "DROPOUT": DROPOUT, "DROP_PATH": DROP_PATH,
        "PREPROCESS": NORM_MODE, "NORM_MODE": NORM_MODE,
        "RESIZE_MODE": RESIZE_MODE, "PAD_VALUE": PAD_VALUE, "PAD_ANCHOR": PAD_ANCHOR,
        "USE_AUG": USE_AUG, "USE_EXCLUDE": USE_EXCLUDE, "LOSS": LOSS_TYPE,
        "SEED": SEED, "VERSION": "v2"}


# =========================================================================
# [J] 옵티마이저 · 스케줄 · EMA
#     LayerNorm 가중치와 bias(ndim<=1)에는 weight decay 를 주지 않습니다.
# =========================================================================
def build_param_groups(model, lr_head, lr_backbone, wd):
    groups = {"bb_decay": [], "bb_nodecay": [], "hd_decay": [], "hd_nodecay": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_bb = n.startswith("backbone.")
        nodecay = (p.ndim <= 1) or n.endswith(".bias")
        groups[("bb_" if is_bb else "hd_") + ("nodecay" if nodecay else "decay")].append(p)

    cfg = [("bb_decay", lr_backbone, wd), ("bb_nodecay", lr_backbone, 0.0),
           ("hd_decay", lr_head, wd),     ("hd_nodecay", lr_head, 0.0)]
    out = []
    for k, lr, w in cfg:
        if groups[k]:
            n_par = sum(p.numel() for p in groups[k])
            out.append({"params": groups[k], "lr": lr, "base_lr": lr,
                        "weight_decay": w, "name": k})
            log(f"  {k:<11} tensors={len(groups[k]):>3} params={n_par/1e6:>6.2f}M "
                f"lr={lr:.1e} wd={w}")
    return out


def lr_scale_at(step, total_steps, warmup_steps):
    """warmup(선형) -> cosine(최저 MIN_LR_RATIO) 배율."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / float(warmup_steps)
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    prog = min(1.0, max(0.0, prog))
    return MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * prog))


class ModelEMA:
    """가중치 지수이동평균. 평가 시 원본보다 안정적인 경우가 많습니다."""

    def __init__(self, model, decay=0.9995):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])


def make_criterion():
    return nn.SmoothL1Loss(beta=HUBER_BETA) if LOSS_TYPE == "huber" else nn.L1Loss()


@torch.no_grad()
def evaluate(model, loader):
    """개월 단위 (MAE, RMSE) 반환. 학습 중 검증용 - TTA 미적용."""
    model.eval(); abs_sum, sq_sum, n = 0.0, 0.0, 0
    for x, g, yn, ym in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        g = g.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            p = model(x, g)
        d = (p.float().cpu() * AGE_STD + AGE_MEAN) - ym.squeeze(1)
        abs_sum += d.abs().sum().item(); sq_sum += (d ** 2).sum().item(); n += x.size(0)
    n = max(1, n)
    return abs_sum / n, math.sqrt(sq_sum / n)


# =========================================================================
# [K] 스테이지 2 - 학습
# =========================================================================
def make_loaders():
    tl = DataLoader(BoneAgeDataset(train_df, "train", train_tf),
                    batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                    pin_memory=True, drop_last=True,
                    persistent_workers=(NUM_WORKERS > 0))
    vl = DataLoader(BoneAgeDataset(val_df, "val", eval_tf),
                    batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                    pin_memory=True, persistent_workers=(NUM_WORKERS > 0))
    return tl, vl


def train_phase():
    train_loader, val_loader = make_loaders()
    log(f"[스테이지2] 배치 수: train {len(train_loader):,} | val {len(val_loader):,}")

    model = build_model(ARCH, pretrained=True)
    ARCH["BACKBONE_RESOLVED"] = model.backbone_name
    n_par = sum(p.numel() for p in model.parameters())
    log(f"[스테이지2] 파라미터 {n_par/1e6:.1f}M")

    log("[스테이지2] 파라미터 그룹:")
    optimizer = torch.optim.AdamW(build_param_groups(model, LR_HEAD, LR_BACKBONE, WEIGHT_DECAY))
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    ema = ModelEMA(model, EMA_DECAY) if USE_EMA else None

    steps_per_epoch = max(1, len(train_loader) // ACCUM_STEPS)
    total_steps  = steps_per_epoch * EPOCHS
    warmup_steps = steps_per_epoch * WARMUP_EPOCHS
    log(f"[스테이지2] 옵티마이저 스텝: 에폭당 {steps_per_epoch} · 총 {total_steps} "
        f"· warmup {warmup_steps}")
    if USE_EMA:
        log(f"[스테이지2] EMA 유효 창 ≈ {1/(1-EMA_DECAY)/steps_per_epoch:.1f} 에폭 "
            f"(decay {EMA_DECAY})")

    start_epoch, best_val, no_improve, global_step = 1, float("inf"), 0, 0
    history = {"train_mae": [], "val_mae": [], "val_rmse": [], "val_mae_ema": [],
               "lr": [], "lr_ratio": [], "sec": []}

    # ── 자동 재개: last.pt 가 있고 구조가 같으면 이어서 ─────────────
    if LAST_CKPT.exists() and not FRESH:
        ck = torch_load(LAST_CKPT, map_location=device)
        bad = [k for k in ("IMG_H", "IMG_W", "HEAD_TYPE", "HEAD_DIM",
                           "GENDER_EMB_DIM", "BACKBONE", "NORM_MODE")
               if ck.get("arch", {}).get(k) != ARCH.get(k)]
        if bad:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            alt = LAST_CKPT.with_name(f"last_incompatible_{ts}.pt")
            LAST_CKPT.rename(alt)
            log(f"[재개] 구조 불일치 {bad} -> 기존 체크포인트를 {alt.name} 으로 옮기고 새로 시작")
        else:
            model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            scaler.load_state_dict(ck["scaler"])
            if ema is not None and ck.get("ema") is not None:
                ema.module.load_state_dict(ck["ema"])
            start_epoch = ck["epoch"] + 1
            best_val    = ck.get("best_val", float("inf"))
            no_improve  = ck.get("no_improve", 0)
            global_step = ck.get("global_step", 0)
            history     = ck.get("history", history)
            history.setdefault("lr_ratio", [])
            log(f"[재개] epoch {start_epoch} 부터 · best val MAE {best_val:.2f}")
    else:
        log("[스테이지2] 새 학습 시작" + (" (--fresh)" if FRESH else ""))

    if start_epoch > EPOCHS:
        log(f"[스테이지2] 이미 {EPOCHS} 에폭 완료 - 학습 스킵")
        return

    criterion = make_criterion()

    def save_ckpt(path, epoch, val_mae, which):
        torch.save({"model": model.state_dict(),
                    "ema": (ema.module.state_dict() if ema is not None else None),
                    "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_val": best_val, "no_improve": no_improve,
                    "global_step": global_step, "history": history,
                    "arch": ARCH, "age_mean": AGE_MEAN, "age_std": AGE_STD,
                    "val_mae": val_mae, "best_from": which, "run_ts": RUN_TS}, path)

    epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, EPOCHS + 1):
            t0 = time.time()
            model.train()
            run_abs, seen = 0.0, 0
            optimizer.zero_grad(set_to_none=True)

            for it, (x, g, yn, ym) in enumerate(train_loader):
                x  = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                g  = g.to(device, non_blocking=True)
                yn = yn.to(device, non_blocking=True).squeeze(1)

                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    pred = model(x, g)
                    loss = criterion(pred, yn) / ACCUM_STEPS
                scaler.scale(loss).backward()

                if (it + 1) % ACCUM_STEPS == 0:
                    s = lr_scale_at(global_step, total_steps, warmup_steps)
                    for grp in optimizer.param_groups:
                        grp["lr"] = grp["base_lr"] * s
                    if CLIP_GRAD:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
                    scaler.step(optimizer); scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if ema is not None:
                        ema.update(model)
                    global_step += 1

                pm = pred.detach().float().cpu() * AGE_STD + AGE_MEAN
                run_abs += (pm - ym.squeeze(1)).abs().sum().item(); seen += x.size(0)

                if (it + 1) % max(1, len(train_loader) // 5) == 0:
                    log(f"    ep{epoch:02d} {it+1:>5,}/{len(train_loader):,} "
                        f"train_mae {run_abs/seen:.2f} lr {optimizer.param_groups[-1]['lr']:.2e}")

            tr_mae = run_abs / max(1, seen)
            va_mae, va_rmse = evaluate(model, val_loader)
            if ema is not None:
                va_ema, va_ema_rmse = evaluate(ema.module, val_loader)
            else:
                va_ema, va_ema_rmse = float("inf"), float("inf")
            which   = "ema" if va_ema < va_mae else "raw"
            va_best = min(va_mae, va_ema)
            dt = time.time() - t0

            cur_lr = optimizer.param_groups[-1]["lr"]
            lr_ratio = cur_lr / max(1e-12, optimizer.param_groups[-1]["base_lr"])

            history["train_mae"].append(tr_mae)
            history["val_mae"].append(va_mae)
            history["val_rmse"].append(va_rmse if which == "raw" else va_ema_rmse)
            history["val_mae_ema"].append(None if ema is None else va_ema)
            history["lr"].append(cur_lr)
            history["lr_ratio"].append(round(lr_ratio, 4))
            history["sec"].append(round(dt, 1))

            msg = (f"[{epoch:02d}/{EPOCHS}] train {tr_mae:.2f} | val {va_mae:.2f} "
                   f"(rmse {va_rmse:.2f})")
            if ema is not None:
                msg += f" | val(ema) {va_ema:.2f}"
            msg += f" | lr {lr_ratio:.0%} | {dt/60:.1f}분"

            if va_best < best_val - MIN_DELTA:
                best_val = va_best; no_improve = 0
                if which == "ema":
                    bak = copy.deepcopy(model.state_dict())
                    model.load_state_dict(ema.module.state_dict())
                    save_ckpt(BEST_CKPT, epoch, va_best, which)
                    model.load_state_dict(bak); del bak
                else:
                    save_ckpt(BEST_CKPT, epoch, va_best, which)
                msg += f"  ** best({which}) 저장 **"
            else:
                no_improve += 1
                if EARLY_STOP_PATIENCE > 0:
                    msg += f"  (개선 없음 {no_improve}/{EARLY_STOP_PATIENCE})"
            log(msg)

            save_ckpt(LAST_CKPT, epoch, va_best, which)
            json.dump(history, open(HISTORY_JSON, "w", encoding="utf-8"), ensure_ascii=False)

            if (EARLY_STOP_PATIENCE > 0 and epoch >= MIN_EPOCHS
                    and no_improve >= EARLY_STOP_PATIENCE):
                log(f"[스테이지2] 조기 종료: {EARLY_STOP_PATIENCE} 에폭 개선 없음 "
                    f"(best {best_val:.2f} 개월)")
                break

    except KeyboardInterrupt:
        save_ckpt(LAST_CKPT, epoch, float("nan"), "interrupt")
        json.dump(history, open(HISTORY_JSON, "w", encoding="utf-8"), ensure_ascii=False)
        log(f"[스테이지2] 중단됨 - last.pt 저장 (epoch {epoch}). 다시 실행하면 이어서 진행합니다.")
        raise SystemExit(130)

    log(f"[스테이지2] 완료 · 최종 best val MAE = {best_val:.2f} 개월  (v1 기준선 5.94)")
    if history["lr_ratio"]:
        log(f"[스테이지2] 종료 시점 LR = 피크의 {history['lr_ratio'][-1]:.1%}"
            + ("  [정상 - 코사인 완주]" if history["lr_ratio"][-1] < 0.08
               else "  [★ 아직 높음 - EPOCHS 를 실제 학습량에 맞추세요]"))


if not EVAL_ONLY:
    train_phase()
else:
    log("--eval-only: 학습을 건너뜁니다.")


# =========================================================================
# [L] 스테이지 3 - 평가  (best.pt 만 있으면 단독 실행 가능)
#     ★ v2: 회전 TTA + val 기반 선형 캘리브레이션
# =========================================================================
if not BEST_CKPT.exists():
    raise SystemExit(f"[중단] best.pt 가 없습니다: {BEST_CKPT}\n"
                     f"       --eval-only 없이 다시 실행해 학습을 먼저 진행하세요.")

_ck = torch_load(BEST_CKPT, map_location=device)
eval_model = build_model(_ck["arch"], pretrained=False)
eval_model.load_state_dict(_ck["model"]); eval_model.eval()
EM_MEAN, EM_STD = _ck["age_mean"], _ck["age_std"]
EM_H, EM_W = _arch_hw(_ck["arch"])

log("=" * 72)
log("[스테이지3] 평가 시작")
log(f"  백본     {_ck['arch'].get('BACKBONE_RESOLVED', _ck['arch']['BACKBONE'])}")
log(f"  헤드     {_ck['arch'].get('HEAD_TYPE')} | 화소처리 {_ck['arch'].get('NORM_MODE')}")
log(f"  가중치   {_ck.get('best_from')} (raw/ema 중 선택된 쪽)")
log(f"  정규화   {EM_MEAN:.1f} ± {EM_STD:.1f} | 입력 {EM_H}x{EM_W}")
log(f"  best val MAE {_ck.get('best_val', float('nan')):.2f} @ epoch {_ck.get('epoch')}")
log(f"  TTA      {list(TTA_ANGLES) if len(TTA_ANGLES) > 1 else '끔'}")


@torch.no_grad()
def predict_split(model, df, split, bs=16, angles=(0,)):
    """개월 단위 (preds, trues, ids) 반환. angles 길이>1 이면 회전 TTA."""
    if not angles:
        angles = (0,)
    loader = DataLoader(BoneAgeDataset(df, split, eval_tf), batch_size=bs,
                        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    P, T = [], []
    for x, g, yn, ym in loader:
        x = x.to(device).to(memory_format=torch.channels_last); g = g.to(device)
        acc = 0.0
        for a in angles:
            xa = x if a == 0 else TF.rotate(
                x, a, fill=PAD_NORM, interpolation=TF.InterpolationMode.BILINEAR)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                acc = acc + model(xa, g).float()
        P.append((acc / len(angles)).cpu() * EM_STD + EM_MEAN)
        T.append(ym.squeeze(1))
    return (torch.cat(P).numpy(), torch.cat(T).numpy(), df["id"].astype(str).values)


def metrics(preds, trues):
    return {"N": int(len(trues)),
            "mae": float(np.abs(preds - trues).mean()),
            "rmse": float(np.sqrt(np.mean((preds - trues) ** 2))),
            "bias": float(np.mean(preds - trues))}


def bootstrap_ci(preds, trues, n_boot=BOOTSTRAP_N, alpha=0.05, seed=SEED):
    """test 가 200장 수준이라 점추정만으로는 비교가 위험 -> MAE 신뢰구간."""
    rng = np.random.default_rng(seed); n = len(trues)
    b = np.empty(n_boot)
    for i in range(n_boot):
        j = rng.integers(0, n, n)
        b[i] = np.abs(preds[j] - trues[j]).mean()
    lo, hi = np.percentile(b, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def age_group_table(preds, trues):
    rows = []
    for lo, hi, lab in zip([0, 48, 96, 144, 192], [48, 96, 144, 192, 10 ** 5],
                           ["0-4y", "4-8y", "8-12y", "12-16y", ">16y"]):
        m = (trues >= lo) & (trues < hi)
        if m.sum():
            rows.append({"구간": lab, "N": int(m.sum()),
                         "MAE": round(float(np.abs(preds[m] - trues[m]).mean()), 2),
                         "bias": round(float(np.mean(preds[m] - trues[m])), 2)})
    return pd.DataFrame(rows)


def fit_calibration(pred_val, true_val):
    """[추론] 논문 미명시. 회귀 축소(regression to the mean) 보정.

    val 에서 pred ≈ a·true + b 를 적합하고 역함수를 반환합니다.
    a < 1 이면 예측이 평균 쪽으로 눌린 것이므로 펴 줍니다.
    v1 연령대별 bias: 0-4y +3.65 / 12-16y -1.80 / >16y -2.71 -> 전형적 축소.
    """
    a, b = np.polyfit(true_val, pred_val, 1)
    log(f"  [calib] slope a={a:.4f} intercept b={b:+.2f}"
        f"  ({'축소 존재 -> 보정 유효' if a < 0.98 else '축소 미미'})")
    return (lambda p: (np.asarray(p, dtype=np.float64) - b) / a), float(a), float(b)


results, lines = {}, []
lines.append("=" * 68)
lines.append(f" ConvNeXt-Tiny 단일 회귀 v2 / {IMG_H}x{IMG_W} / 강도정규화 {NORM_MODE}")
lines.append(f" 백본 {_ck['arch'].get('BACKBONE_RESOLVED', BACKBONE)} | "
             f"헤드 {_ck['arch'].get('HEAD_TYPE')}")
lines.append(f" 실행 {RUN_TS} | best epoch {_ck.get('epoch')} ({_ck.get('best_from')})")
lines.append(f" TTA {list(TTA_ANGLES) if len(TTA_ANGLES) > 1 else '끔'}")
lines.append("=" * 68)

# ── val: TTA 효과 측정 + 캘리브레이션 적합 ──────────────────────────
log("  [val] 예측 중...")
v_plain, v_true, _ = predict_split(eval_model, val_df, "val", angles=(0,))
results["val_plain"] = metrics(v_plain, v_true)
lines.append(f" {'VAL(단일)':<16} N={results['val_plain']['N']:>6,}  "
             f"MAE {results['val_plain']['mae']:5.2f}  "
             f"RMSE {results['val_plain']['rmse']:5.2f}  "
             f"bias {results['val_plain']['bias']:+5.2f}")

v_pred = v_plain
if len(TTA_ANGLES) > 1:
    v_pred, v_true, _ = predict_split(eval_model, val_df, "val", angles=TTA_ANGLES)
    results["val_tta"] = metrics(v_pred, v_true)
    lines.append(f" {'VAL(+TTA)':<16} N={results['val_tta']['N']:>6,}  "
                 f"MAE {results['val_tta']['mae']:5.2f}  "
                 f"RMSE {results['val_tta']['rmse']:5.2f}  "
                 f"bias {results['val_tta']['bias']:+5.2f}")

CAL_A, CAL_B, USE_CALIB = 1.0, 0.0, False
if USE_CALIB_OPT:
    calib_fn, CAL_A, CAL_B = fit_calibration(v_pred, v_true)
    v_cal = calib_fn(v_pred)
    results["val_calib"] = metrics(v_cal, v_true)
    base_mae = (results.get("val_tta") or results["val_plain"])["mae"]
    USE_CALIB = results["val_calib"]["mae"] < base_mae
    lines.append(f" {'VAL(+calib)':<16} N={results['val_calib']['N']:>6,}  "
                 f"MAE {results['val_calib']['mae']:5.2f}  "
                 f"RMSE {results['val_calib']['rmse']:5.2f}  "
                 f"bias {results['val_calib']['bias']:+5.2f}   "
                 f"-> {'채택' if USE_CALIB else '미채택'}")
    log(f"  [calib] 채택 여부: {USE_CALIB} "
        f"(val {base_mae:.2f} -> {results['val_calib']['mae']:.2f})")
else:
    calib_fn = (lambda p: np.asarray(p, dtype=np.float64))
    log("  [calib] --no-calib: 캘리브레이션을 사용하지 않습니다.")


def apply_calib(p):
    return calib_fn(p) if USE_CALIB else np.asarray(p, dtype=np.float64)


v_pred = apply_calib(v_pred)
results["val"] = metrics(v_pred, v_true)

# ── train (참고용, TTA 없이 단일 패스) ───────────────────────────────
log("  [train] 예측 중...")
_tr_p, _tr_t, _ = predict_split(eval_model, train_df, "train", angles=(0,))
results["train"] = metrics(apply_calib(_tr_p), _tr_t)
lines.append(f" {'TRAIN(단일)':<16} N={results['train']['N']:>6,}  "
             f"MAE {results['train']['mae']:5.2f}  "
             f"RMSE {results['train']['rmse']:5.2f}  "
             f"bias {results['train']['bias']:+5.2f}")
lines.append("-" * 68)

# ── test: 현재 설정 ──────────────────────────────────────────────────
log("  [test] 예측 중...")
t_raw, t_true, t_ids = predict_split(eval_model, test_df, "test", angles=TTA_ANGLES)
t_pred = apply_calib(t_raw)

key_a = "test_filtered" if USE_EXCLUDE else "test_raw"
results[key_a] = metrics(t_pred, t_true)
lo, hi = bootstrap_ci(t_pred, t_true)
results[key_a]["ci95"] = [round(lo, 2), round(hi, 2)]
tag_a = "TEST(제외적용)" if USE_EXCLUDE else "TEST(필터없음)"
lines.append(f" {tag_a:<16} N={results[key_a]['N']:>5,}  MAE {results[key_a]['mae']:5.2f}  "
             f"RMSE {results[key_a]['rmse']:5.2f}  bias {results[key_a]['bias']:+5.2f}  "
             f"CI95 [{lo:.2f}, {hi:.2f}]")

# 투명성: 캘리브레이션 미적용 수치도 함께 남깁니다
if USE_CALIB:
    results["test_uncalibrated"] = metrics(t_raw, t_true)
    lines.append(f" {'  (calib 미적용)':<16} MAE {results['test_uncalibrated']['mae']:5.2f}  "
                 f"bias {results['test_uncalibrated']['bias']:+5.2f}")

# ── test: 필터 없는 정직한 수치 (USE_EXCLUDE=True 일 때만 추가) ──────
if USE_EXCLUDE:
    full = load_labels(TEST_CSV)
    idx = FILE_INDEX["test"]
    full = full[full["id"].isin(idx.keys())].reset_index(drop=True)
    full["path"] = full["id"].map(lambda i: str(idx[i]))
    missing = [i for i in full["id"] if not (CACHE_DIR / "test" / f"{i}.png").exists()]
    if missing:
        log(f"  캐시 없는 test {len(missing)}장 생성 중...")
        for i in missing:
            g = imread_kr(idx[i], cv2.IMREAD_GRAYSCALE)
            if g is not None:
                imwrite_kr(CACHE_DIR / "test" / f"{i}.png", fit_canvas(g)[0])
    f_pred, f_true, _ = predict_split(eval_model, full, "test", angles=TTA_ANGLES)
    f_pred = apply_calib(f_pred)
    results["test_raw"] = metrics(f_pred, f_true)
    lo2, hi2 = bootstrap_ci(f_pred, f_true)
    results["test_raw"]["ci95"] = [round(lo2, 2), round(hi2, 2)]
    lines.append(f" {'TEST(필터없음)':<16} N={results['test_raw']['N']:>5,}  "
                 f"MAE {results['test_raw']['mae']:5.2f}  "
                 f"RMSE {results['test_raw']['rmse']:5.2f}  "
                 f"bias {results['test_raw']['bias']:+5.2f}  CI95 [{lo2:.2f}, {hi2:.2f}]")
    lines.append(f" 필터로 인한 MAE 차이 = "
                 f"{results['test_raw']['mae'] - results['test_filtered']['mae']:+.2f} 개월")

lines.append("-" * 68)
lines.append(" [연령대별 · test]")
_agrp = age_group_table(t_pred, t_true)
for _, r in _agrp.iterrows():
    lines.append(f"   {r['구간']:<7} N={int(r['N']):>4}  MAE {r['MAE']:5.2f}  bias {r['bias']:+5.2f}")
lines.append("   (v1 참고: 0-4y bias +3.65 / 8-12y MAE 7.27 / >16y bias -2.71)")
results["age_groups_test"] = _agrp.to_dict("records")

# ── v1 대비 비교 ─────────────────────────────────────────────────────
lines.append("-" * 68)
lines.append(" [v1 대비]")
_key = "test_raw" if "test_raw" in results else "test_filtered"
lines.append(f"   v2  test MAE {results[_key]['mae']:5.2f}  "
             f"CI95 {results[_key]['ci95']}  N={results[_key]['N']}")
if V1_RESULTS.exists():
    try:
        _r1 = json.load(open(V1_RESULTS, encoding="utf-8"))
        _k1 = "test_raw" if "test_raw" in _r1 else "test_filtered"
        _m1 = _r1[_k1]["mae"]
        _c1 = _r1[_k1].get("ci95", "?")
        lines.append(f"   v1  test MAE {_m1:5.2f}  CI95 {_c1}  N={_r1[_k1]['N']}")
        lines.append(f"   변화 {results[_key]['mae'] - _m1:+.2f} 개월")
        lines.append("   * CI 가 겹치면 통계적으로 유의한 개선이라 말할 수 없습니다.")
        lines.append("     시드 3개 앙상블로 재확인을 권장합니다.")
        results["v1_compare"] = {"v1_mae": _m1, "v2_mae": results[_key]["mae"],
                                "delta": results[_key]["mae"] - _m1}
    except Exception as e:
        lines.append(f"   v1 결과 로드 실패: {e}")
else:
    lines.append(f"   v1 결과 파일 없음: {V1_RESULTS}")
lines.append("=" * 68)


# ── 그림들 ───────────────────────────────────────────────────────────
def save_learning_curve():
    if not HISTORY_JSON.exists():
        return
    try:
        h = json.load(open(HISTORY_JSON, encoding="utf-8"))
        ep = range(1, len(h["train_mae"]) + 1)
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        ax[0].plot(ep, h["train_mae"], "-o", ms=3, label="train MAE")
        ax[0].plot(ep, h["val_mae"], "-o", ms=3, label="val MAE")
        if any(v is not None for v in h.get("val_mae_ema", [])):
            ax[0].plot(ep, h["val_mae_ema"], "-s", ms=3, alpha=.8, label="val MAE (EMA)")
        ax[0].axhline(5.94, ls="-.", c="gray", label="v1 baseline (val) 5.94")
        ax[0].axhline(4.10, ls="--", c="green", label="Zhang 2026 4.10")
        ax[0].axhline(4.30, ls=":", c="orange", label="Chen 2020 4.30")
        ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("MAE (months)")
        ax[0].set_title(f"ConvNeXt-Tiny v2 ({IMG_H}x{IMG_W} + {NORM_MODE})")
        ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
        ax[1].plot(ep, h["lr"], "-o", ms=3, c="crimson")
        ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("LR (head)")
        ax[1].set_title("warmup + cosine"); ax[1].set_yscale("log"); ax[1].grid(alpha=.3)
        plt.tight_layout(); plt.savefig(CKPT_DIR / "learning_curve.png", dpi=120); plt.close()
        log(f"  학습곡선 저장: {CKPT_DIR/'learning_curve.png'}")
        if h.get("lr_ratio"):
            log(f"  종료 시점 LR = 피크의 {h['lr_ratio'][-1]:.1%} "
                + ("[정상]" if h["lr_ratio"][-1] < 0.08 else "[★ 코사인 미완주]"))
    except Exception as e:
        log(f"  [경고] 학습곡선 저장 실패: {e}")


def save_scatter():
    try:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5.6))
        for a, (p, t, name) in zip(ax, [(v_pred, v_true, "Validation"),
                                        (t_pred, t_true, "Test")]):
            a.scatter(t, p, s=9, alpha=.45)
            lim = [0, max(t.max(), p.max()) + 5]
            a.plot(lim, lim, "r--"); a.set_xlim(lim); a.set_ylim(lim)
            a.set_xlabel("True (months)"); a.set_ylabel("Pred (months)")
            a.set_title(f"{name} · MAE {np.abs(p-t).mean():.2f}"); a.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(CKPT_DIR / "scatter.png", dpi=120); plt.close()
        log(f"  산점도 저장: {CKPT_DIR/'scatter.png'}")
    except Exception as e:
        log(f"  [경고] 산점도 저장 실패: {e}")


def save_worst_cases(k=8):
    try:
        err = np.abs(t_pred - t_true)
        order = np.argsort(-err)[:k]
        fig, axes = plt.subplots(2, (k + 1) // 2, figsize=(3.6 * ((k + 1) // 2), 9))
        for a, i in zip(np.atleast_1d(axes).ravel(), order):
            g = imread_kr(CACHE_DIR / "test" / f"{t_ids[i]}.png", cv2.IMREAD_GRAYSCALE)
            if g is not None:
                a.imshow(g, cmap="gray")
            a.axis("off")
            a.set_title(f"{t_ids[i]}\ntrue {t_true[i]:.0f} / pred {t_pred[i]:.0f} "
                        f"({t_pred[i]-t_true[i]:+.0f})", fontsize=9)
        plt.suptitle("Top errors - crop failure / label noise candidates", y=1.0)
        plt.tight_layout(); plt.savefig(CKPT_DIR / "worst_cases.png", dpi=120); plt.close()
        pd.DataFrame({"id": t_ids[order], "true": t_true[order].round(0),
                      "pred": t_pred[order].round(1), "err": err[order].round(1)}
                     ).to_csv(CKPT_DIR / "worst_cases.csv", index=False, encoding="utf-8-sig")
        log(f"  오차 상위 저장: {CKPT_DIR/'worst_cases.png'}")
    except Exception as e:
        log(f"  [경고] 오차 상위 저장 실패: {e}")


class GradCAM:
    """ConvNeXt 마지막 stage 특징맵 기준 Grad-CAM (회귀 출력)."""

    def __init__(self, model):
        self.model = model; self.feat = self.grad = None
        tgt, name = self._pick(model.backbone)
        log(f"  [Grad-CAM] target = {name}")
        tgt.register_forward_hook(lambda m, i, o: setattr(self, "feat", o.detach()))
        tgt.register_full_backward_hook(lambda m, gi, go: setattr(self, "grad", go[0].detach()))

    @staticmethod
    def _pick(backbone):
        mods = dict(backbone.named_modules())
        for n in ("stages.3", "stages_3", "norm_pre", "stages"):
            if n in mods:
                return mods[n], n
        return backbone, "backbone(output)"

    def __call__(self, x, g):
        self.model.eval(); self.model.zero_grad()
        out = self.model(x, g); out.sum().backward()
        w = self.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.feat).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        return cam / (cam.max() + 1e-8), float(out.item()) * EM_STD + EM_MEAN


def save_gradcam(k=4):
    """v2 는 여백이 크게 줄었으므로 관심이 성장판·수근골에 더 모여 있어야 정상입니다.
       여전히 여백·마커로 샌다면 그것이 Attention 파이프라인(v8/v10)을 정당화하는 증거."""
    try:
        engine = GradCAM(eval_model)
        samp = test_df.sample(min(k, len(test_df)), random_state=SEED)
        fig, axes = plt.subplots(1, len(samp), figsize=(3.6 * len(samp), 5.2))
        for a, (_, r) in zip(np.atleast_1d(axes), samp.iterrows()):
            g8 = imread_kr(CACHE_DIR / "test" / f"{r['id']}.png", cv2.IMREAD_GRAYSCALE)
            if g8 is None:
                continue
            x = eval_tf(np.stack([g8] * 3, -1)).unsqueeze(0).to(device)
            x = x.to(memory_format=torch.channels_last)
            gd = torch.tensor([[float(r["male"])]], device=device)
            cam, pred = engine(x, gd)
            a.imshow(g8, cmap="gray"); a.imshow(cam, cmap="jet", alpha=.42)
            a.set_title(f"{r['id']} · true {r['boneage']:.0f} / pred {pred:.1f}", fontsize=9)
            a.axis("off")
        plt.tight_layout(); plt.savefig(CKPT_DIR / "gradcam.png", dpi=120); plt.close()
        log(f"  Grad-CAM 저장: {CKPT_DIR/'gradcam.png'}")
    except Exception as e:
        log(f"  [경고] Grad-CAM 생략: {e}")


save_learning_curve()
save_scatter()
save_worst_cases()
save_gradcam()

# ── 캘리브레이션 계수 별도 저장 (배포용) ────────────────────────────
json.dump({"used": bool(USE_CALIB), "a": CAL_A, "b": CAL_B,
           "formula": "corrected = (pred - b) / a",
           "fitted_on": "validation", "tta_angles": list(TTA_ANGLES),
           "img_h": IMG_H, "img_w": IMG_W, "norm_mode": NORM_MODE,
           "run_ts": RUN_TS},
          open(CALIB_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
log(f"  캘리브레이션 계수 저장: {CALIB_JSON}")

results["arch"] = _ck["arch"]
results["run_ts"] = RUN_TS
results["env_file"] = str(_env_path)
results["best_epoch"] = _ck.get("epoch")
results["best_from"] = _ck.get("best_from")
results["tta_angles"] = list(TTA_ANGLES)
results["calib"] = {"used": bool(USE_CALIB), "a": CAL_A, "b": CAL_B}
results["version"] = "v2"
results["when"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

json.dump(results, open(RESULTS_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
open(RESULTS_TXT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)
log(f"결과 저장: {RESULTS_TXT}")
log(f"         {RESULTS_JSON}")


# =========================================================================
# [M] 단일 이미지 추론  (best.pt + 크롭된 이미지 1장이면 끝)
#     이미지 준비 설정을 체크포인트에서 읽어 학습과 동일한 처리를 보장합니다.
#     ★ v2: TTA 각도와 캘리브레이션 계수도 함께 적용할 수 있습니다.
# =========================================================================
_INFER_CACHE = {}


def predict_bone_age(image_path, is_male, ckpt_path=BEST_CKPT,
                     angles=None, calib_ab=None):
    """YOLO 로 손이 크롭된 X-ray 경로 + 성별(True=남) -> 골연령(개월).

    angles   : 회전 TTA 각도. None 이면 이 실행의 TTA_ANGLES 사용
    calib_ab : (a, b) 캘리브레이션 계수. None 이면 미적용
    """
    key = str(ckpt_path)
    if key not in _INFER_CACHE:
        c = torch_load(ckpt_path, map_location=device)
        m = build_model(c["arch"], pretrained=False, verbose=False)
        m.load_state_dict(c["model"]); m.eval()
        _INFER_CACHE[key] = (m, c)
    m, c = _INFER_CACHE[key]
    a = c["arch"]
    h_out, w_out = _arch_hw(a)
    angles = TTA_ANGLES if angles is None else angles
    if not angles:
        angles = (0,)

    g = imread_kr(image_path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise FileNotFoundError(image_path)
    canvas, _ = fit_canvas(g, h_out=h_out, w_out=w_out,
                          mode=a.get("RESIZE_MODE", "letterbox"),
                          pad_val=a.get("PAD_VALUE", 0),
                          anchor=a.get("PAD_ANCHOR", "center"),
                          norm=a.get("NORM_MODE", "none"))
    x = eval_tf(np.stack([canvas] * 3, -1)).unsqueeze(0).to(device)
    x = x.to(memory_format=torch.channels_last)
    gd = torch.tensor([[1.0 if is_male else 0.0]], dtype=torch.float32, device=device)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=USE_AMP):
        acc = 0.0
        for ang in angles:
            xa = x if ang == 0 else TF.rotate(
                x, ang, fill=PAD_NORM, interpolation=TF.InterpolationMode.BILINEAR)
            acc = acc + m(xa, gd).float()
        p = acc / len(angles)
    months = float(p.cpu().item()) * c["age_std"] + c["age_mean"]
    if calib_ab is not None:
        months = (months - calib_ab[1]) / calib_ab[0]
    return months


# 예시:
#   ab = (CAL_A, CAL_B) if USE_CALIB else None
#   months = predict_bone_age(HAND_CROP_DIR / "validation" / "1377.png",
#                             is_male=True, calib_ab=ab)
#   print(f"예측 골연령: {months:.1f} 개월")
try:
    _r = test_df.iloc[0]
    _ab = (CAL_A, CAL_B) if USE_CALIB else None
    _m = predict_bone_age(_r["path"], bool(_r["male"]), calib_ab=_ab)
    log(f"추론 함수 확인 [{_r['id']}] 예측 {_m:.1f}개월 / 실제 {_r['boneage']:.0f}개월")
except Exception as e:
    log(f"[경고] 추론 함수 확인 실패: {e}")

log(f"추론 함수 준비 완료 · best.pt: {BEST_CKPT}")
log(f"                     캘리브레이션: {CALIB_JSON}")
log("=== 전체 완료 (v2) ===")


# =========================================================================
# [부록] 시드 앙상블 (다음 단계, -0.3~-0.5 기대)
# =========================================================================
#   1) SEED 를 42 -> 1337 -> 2024 로 바꾸고 CKPT_DIR 도 함께 분리합니다.
#        CKPT_DIR = BASE_DIR / f"checkpoints_convnext_single_v2_s{SEED}"
#      (CACHE_DIR 는 시드와 무관하므로 공유해도 됩니다)
#
#   2) 3회 학습이 끝난 뒤 예측을 평균합니다.
#
#        SEEDS = [42, 1337, 2024]
#        preds = []
#        for s in SEEDS:
#            ckp = BASE_DIR / f"checkpoints_convnext_single_v2_s{s}" / "best.pt"
#            c = torch_load(ckp, map_location=device)
#            m = build_model(c["arch"], pretrained=False, verbose=False)
#            m.load_state_dict(c["model"]); m.eval()
#            EM_MEAN, EM_STD = c["age_mean"], c["age_std"]
#            p, t, ids = predict_split(m, test_df, "test", angles=TTA_ANGLES)
#            preds.append(p)
#        ens = np.mean(preds, axis=0)
#        print(metrics(ens, t), bootstrap_ci(ens, t))
#
#   앙상블은 val 에서 고른 단일 최고 모델보다 거의 항상 낫고,
#   test 200장 규모에서 CI 폭도 줄여줍니다 - 논문 비교표에서 이게 중요합니다.
#
# [부록] 여기서도 부족하면 만지는 순서
#   1. 시드 앙상블 3~5개                      (항상, 가장 확실)
#   2. NORM_MODE p1p99 -> p2p98 또는 none     (정규화가 되레 손해인지 A/B)
#   3. CLAHE 추가 (clip 2.0, tile 8)          (노이즈도 증폭되니 반드시 단독 측정)
#   4. IMG_H/IMG_W 640x448 -> 704x512         (USE_GRAD_CKPT=True 병행)
#   5. BACKBONE Tiny -> convnext_small...384  (용량 부족 = train 도 높을 때)
#   6. LR_BACKBONE 4e-5 -> 6e-5               (언더피팅일 때)
#   7. DROP_PATH 0.15 -> 0.25                 (여전히 train << val 일 때)
#   8. LDL 보조 헤드 (lambda ≈ 0.1)           (축소 편향을 구조적으로 줄일 때)
#   * 한 번에 하나씩만 바꾸고, 매번 CKPT_DIR 에 태그를 붙여 분리하세요.
# =========================================================================
