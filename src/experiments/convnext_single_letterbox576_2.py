# -*- coding: utf-8 -*-
# =========================================================================
# 손 X-ray 골연령 - ConvNeXt-Small 단일 회귀 / 전처리 없음 (레터박스 576만)
#   ▶▶ _2 = 1차(_1) 대비 성능 개선판. 시드 앙상블은 미포함(별도 실행).
#
#   ▶ 이 파일이 답하려는 질문
#       "Attention(R1/R2) + LDL 이 실제로 MAE 를 개선했는가?"
#       -> v10(전처리 없음 + Attention+LDL) 과 이 파일(전처리 없음 + 단일 회귀)의
#          test MAE 를 직접 비교하면 Attention+LDL 의 순수 기여도가 나옵니다.
#          (같은 split / 같은 EXCLUDE 조건 / 같은 레터박스 정책 / 같은 ConvNeXt V1 계열)
#       ※ [주의] _2 는 입력 576 + 백본 small 이므로 v10(512/tiny)과 직접 비교하려면
#          _1(512/tiny) 결과를 쓰세요. _2 는 '이 조건에서의 최고 성능' 을 재는 라인입니다.
#
#   ▶ 2x2 ablation 표에서 이 파일의 위치
#       ┌──────────────┬──────────────────────┬────────────────────────┐
#       │              │ Attention + LDL      │ 단일 회귀              │
#       ├──────────────┼──────────────────────┼────────────────────────┤
#       │ 전처리 있음  │ v8                   │ Xception TopHat/CLAHE  │
#       │ 전처리 없음  │ v10                  │ ★ 이 파일              │
#       └──────────────┴──────────────────────┴────────────────────────┘
#
#   ▶ 이미지 체인 (최종)
#       crop_data 원본 --> [uint8 변환] --> 비율유지 리사이즈 --> 0 패딩 --> 576x576
#       * uint8 변환은 16bit/float PNG 를 안전하게 다루기 위한 것이며,
#         입력이 이미 8bit 면 아무 일도 하지 않습니다 (완전 무손실 통과).
#       * 화소값을 바꾸는 전처리(정규화/TopHat/CLAHE/마커억제/회전정렬)는
#         함수 자체가 존재하지 않습니다.
#
#   ▶ 백본 - 상업 사용 가능한 ConvNeXt V1 계열만
#       convnext_small.fb_in22k_ft_in1k_384  (기본, Apache-2.0 / MIT)
#       convnext_small.fb_in22k_ft_in1k      (224 ft.)
#       convnext_tiny.fb_in22k_ft_in1k_384   (VRAM 부족 시 폴백)
#       ※ convnextv2_* 는 가중치가 CC BY-NC 4.0 -> 코드에서 즉시 중단
#
#   ▶ _1 -> _2 변경점 (성능 개선)
#       [1] 입력 512 -> 576            (성장판 유효 해상도 확대)
#       [2] 백본 tiny -> small(_384)   (용량 확대. 배치 4 x accum 8 = 유효 32 유지)
#       [3] Layer-wise LR Decay 0.75   (stage 깊이별 LR 감쇠. LR_BACKBONE 4e-5 -> 6e-5)
#       [4] EMA decay 0.999 -> 0.9997  (평균 창 ~2.5ep -> ~8ep)
#       [5] 회전 TTA [0, -4, +4]       (평가 전용. 학습/best 선택에는 미적용)
#       [6] gradient checkpointing 토글 + expandable_segments (8GB VRAM 대응)
#       [7] 조기종료 patience 12->15, MIN_EPOCHS 20->25 (EMA 창 확대에 맞춤)
#       ※ 시드 앙상블은 이 파일에 없습니다. SEED 와 CKPT_DIR 만 바꿔 3회 돌린 뒤
#         예측을 평균내는 방식으로 별도 진행하세요.
#
#   ▶ 실행: python convnext_single_letterbox576_2.py
#       - 창을 닫아도 서버에서 학습은 계속됩니다. 다시 실행하면 로그에 재부착.
#       - 각 스테이지는 완료 표식을 남기고 자동 스킵 -> 중단돼도 이어서 진행.
#
#   ▶ 실행 옵션
#       --fg             백그라운드 분리 없이 바로 실행(디버그용)
#       --eval-only      학습을 건너뛰고 best.pt 로 평가만
#       --rebuild-cache  576 캔버스 캐시를 강제로 재생성
#       --no-tta         평가에서 회전 TTA 를 끔
#       --qc-only        QC 시트만 만들고 종료 (전처리 통과 검증용)
#       --fresh          last.pt 를 무시하고 처음부터 학습
#
#   ▶ 산출물
#       logs/convnext_single_v2_<타임스탬프>.log       실행 로그 전체
#       checkpoints_convnext_single_2/env_<ts>.json    실행 환경 스냅샷
#       checkpoints_convnext_single_2/best.pt          최고 검증 MAE 가중치
#       checkpoints_convnext_single_2/last.pt          매 에폭 저장 (재개용)
#       checkpoints_convnext_single_2/history.json     에폭별 지표
#       checkpoints_convnext_single_2/results.txt|json 최종 성적표 (TTA/무TTA 병기)
#       checkpoints_convnext_single_2/*.png            QC/곡선/산점도/GradCAM
# =========================================================================
from pathlib import Path
import os, sys, time, json, subprocess, platform
from datetime import datetime

PROJECT_DIR = Path(__file__).resolve().parent
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RUN_STATE = LOG_DIR / "convnext_single_v2_running.json"
_WORKER_ENV = "BONEAGE_CONVNEXT_SINGLE_V2_WORKER"

FOREGROUND    = "--fg" in sys.argv
EVAL_ONLY     = "--eval-only" in sys.argv
REBUILD_CACHE = "--rebuild-cache" in sys.argv
QC_ONLY       = "--qc-only" in sys.argv
FRESH         = "--fresh" in sys.argv
NO_TTA        = "--no-tta" in sys.argv


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
    log_path = LOG_DIR / f"convnext_single_v2_{ts}.log"
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
            print("이미 실행 중입니다 - 기존 로그에 다시 붙습니다.")
            _follow(st["log"], st["pid"])
            sys.exit(0)
    _pid, _logp = _spawn_detached()
    _follow(_logp, _pid)
    sys.exit(0)


# =========================================================================
# [B] 본체
# =========================================================================
# 576 + small 은 8GB 에서 빠듯합니다. 할당자 파편화를 줄여 OOM 여유를 확보합니다.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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
import torchvision.transforms.functional as TF

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
# [B-1] 경로  (실제 폴더 구조에 맞춰 고정. 없으면 그 자리에서 바로 중단)
# -------------------------------------------------------------------------
BASE_DIR      = Path(r"G:/Project/sinra_cho")
HAND_CROP_DIR = BASE_DIR / "crop_data_final"        # training / validation / test 하위폴더
CSV_DIR       = BASE_DIR / "crop_data_final"    # training.csv / validation.csv / test.csv

SPLIT_SUBDIR = {"train": "training", "val": "validation", "test": "test"}

_require(HAND_CROP_DIR, "손 크롭 폴더")
TRAIN_CSV = _require(CSV_DIR / "training.csv",   "training.csv")
VAL_CSV   = _require(CSV_DIR / "validation.csv", "validation.csv")
TEST_CSV  = _require(CSV_DIR / "test.csv",       "test.csv")


# =========================================================================
# [B-2] 재현 스위치
# =========================================================================
SEED = 42

# ── 이미지 준비 = 기하 변환만 (화소값 조작 없음) ─────────────────────
IMG_SIZE    = 576           # [_2] 512 -> 576. 성장판 유효 해상도 확대
                            #   576/32 = 18 -> ConvNeXt 최종 특징맵 18x18
RESIZE_MODE = "letterbox"   # "letterbox"(비율유지+패딩, 권장) | "stretch"(정사각 강제)
PAD_VALUE   = 0             # 배경색 추정 제거 -> 상수 0 고정
PAD_ANCHOR  = "center"      # "center" | "topleft"

# ── 백본 (상업 사용 가능한 ConvNeXt V1 계열만) ───────────────────────
BACKBONE     = "convnext_small.fb_in22k_ft_in1k_384"   # [_2] tiny -> small
BACKBONE_ALT = ["convnext_small.fb_in22k_ft_in1k",     # 224 ft.
                "convnext_tiny.fb_in22k_ft_in1k_384",  # VRAM 부족 시 폴백
                "convnext_tiny.fb_in22k_ft_in1k"]
DROP_PATH    = 0.05          # ConvNeXt stochastic depth. 과적합 시 0.2까지

# ── 헤드 ─────────────────────────────────────────────────────────────
HEAD_TYPE      = "gap"      # "gap"(권장) | "paper"(Conv3x3+MaxPool+Flatten)
HEAD_DIM       = 512
GENDER_EMB_DIM = 32         # [논문] 식(15) k=32
DROPOUT        = 0.05

# ── 최적화 ───────────────────────────────────────────────────────────
BATCH_SIZE   = 4            # [_2] 576px + small + 8GB 기준
                            #   OOM 이면 (3, 11) 또는 (2, 16) 으로. 유효 배치는 32 유지
ACCUM_STEPS  = 8            # 유효 배치 = BATCH_SIZE * ACCUM_STEPS = 32
GRAD_CKPT    = False        # [_2] OOM 안전밸브. True 면 VRAM 크게 절약(속도 20~30% 손해)
EPOCHS       = 100
LR_HEAD      = 1e-4
LR_BACKBONE  = 6e-5         # [_2] 4e-5 -> 6e-5. 아래 LLRD 감쇠가 걸리므로 상향
LLRD         = 0.75         # [_2] Layer-wise LR Decay. stage 가 얕을수록 LR 감쇠
                            #   stem 은 실제 LR_BACKBONE * 0.75^3 수준으로 내려갑니다
                            #   1.0 으로 두면 _1 과 동일한 균일 LR 로 되돌아갑니다
WEIGHT_DECAY = 0.05         # AdamW. norm/bias 에는 적용하지 않음
WARMUP_EPOCHS= 3
MIN_LR_RATIO = 0.02         # cosine 최저 LR = base * 이 값
CLIP_GRAD    = 1.0          # None 이면 끔

# ── 손실 ─────────────────────────────────────────────────────────────
LOSS_TYPE  = "l1"        # "huber" | "l1"
HUBER_BETA = 0.5            # z-정규화 스케일 기준

# ── 정칙화 / 증강 ────────────────────────────────────────────────────
USE_AUG       = True
AUG_ROT_DEG   = 12
AUG_TRANSLATE = 0.06
AUG_SCALE     = (0.92, 1.08)
# 좌우 반전은 넣지 않습니다 - RSNA 는 전부 좌수 촬영이라 해부학적으로 없는 입력입니다.

# ── EMA ──────────────────────────────────────────────────────────────
USE_EMA   = True
EMA_DECAY = 0.9997          # [_2] 0.999 -> 0.9997. 평균 창 ~2.5ep -> ~8ep

# ── 조기 종료 / 실행 ─────────────────────────────────────────────────
EARLY_STOP_PATIENCE = 15    # [_2] 12 -> 15. EMA 창이 길어져 반영이 늦어짐
MIN_DELTA           = 0.01  # 개선으로 인정할 최소 MAE 감소(개월)
MIN_EPOCHS          = 25    # [_2] 20 -> 25. EMA 창(~8ep)이 안정될 때까지 대기
NUM_WORKERS         = 0     # [중요] Windows 는 반드시 0.
                            #   이 스크립트는 전체가 모듈 최상위에서 실행되므로,
                            #   workers>0 이면 spawn 된 워커가 스크립트를 다시 import 하면서
                            #   학습이 통째로 재실행됩니다. 아래에서 강제로 0 으로 고정합니다.
N_QC                = 6
BOOTSTRAP_N         = 2000

# ── [_2] 회전 TTA (평가 전용) ────────────────────────────────────────
#   학습 중 val 평가와 best 선택에는 쓰지 않습니다. 최종 성적표에서만
#   '무TTA' 와 'TTA' 를 나란히 출력해 이득을 확인할 수 있게 했습니다.
#   좌우 반전은 넣지 않습니다 - RSNA 는 전부 좌수 촬영입니다.
USE_TTA    = (not NO_TTA)
TTA_ANGLES = [0, -4, 4]

if os.name == "nt" and NUM_WORKERS > 0:
    NUM_WORKERS = 0   # 안전장치: Windows 재귀 실행 방지 (조용히 넘어가지 않고 아래에서 로그로 알림)
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

# ── 출력 경로 (설정이 다르면 캐시가 자동으로 분리됩니다) ─────────────
PRE_TAG   = f"raw{IMG_SIZE}_{RESIZE_MODE}_pad{PAD_VALUE}_{PAD_ANCHOR}"
CACHE_DIR = BASE_DIR / "cache_convnext_single_2" / PRE_TAG
CKPT_DIR  = BASE_DIR / "checkpoints_convnext_single_2"
SPLITS    = ("train", "val", "test")

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

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

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
        "image_prep": {"preprocess": "none", "img_size": IMG_SIZE,
                       "resize_mode": RESIZE_MODE, "pad_value": PAD_VALUE,
                       "pad_anchor": PAD_ANCHOR, "pre_tag": PRE_TAG},
        "variant": "_2 (576 / convnext_small / LLRD / EMA0.9997 / TTA)",
        "model": {"backbone": BACKBONE, "backbone_alt": BACKBONE_ALT,
                  "grad_checkpointing": GRAD_CKPT,
                  "drop_path": DROP_PATH, "head_type": HEAD_TYPE,
                  "head_dim": HEAD_DIM, "gender_emb_dim": GENDER_EMB_DIM,
                  "dropout": DROPOUT},
        "optim": {"batch_size": BATCH_SIZE, "accum_steps": ACCUM_STEPS,
                  "effective_batch": BATCH_SIZE * ACCUM_STEPS, "epochs": EPOCHS,
                  "lr_head": LR_HEAD, "lr_backbone": LR_BACKBONE,
                  "weight_decay": WEIGHT_DECAY, "warmup_epochs": WARMUP_EPOCHS,
                  "min_lr_ratio": MIN_LR_RATIO, "clip_grad": CLIP_GRAD,
                  "loss": LOSS_TYPE, "huber_beta": HUBER_BETA, "llrd": LLRD},
        "tta": {"enabled": USE_TTA, "angles": TTA_ANGLES},
        "regularize": {"use_aug": USE_AUG, "rot_deg": AUG_ROT_DEG,
                       "translate": AUG_TRANSLATE, "scale": list(AUG_SCALE),
                       "use_ema": USE_EMA, "ema_decay": EMA_DECAY},
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
log("ConvNeXt-Small 단일 회귀 골연령 - 전처리 없음 / 레터박스 576 [_2] 시작")
log(f"Python {sys.version.split()[0]} | {platform.system()} {platform.release()}")
log(f"PyTorch {torch.__version__} | torchvision {torchvision.__version__} | timm {timm.__version__}")
log(f"OpenCV {cv2.__version__} | numpy {np.__version__} | pandas {pd.__version__}")
log(f"device {device} | CUDA {torch.cuda.is_available()} | AMP {USE_AMP}")
if torch.cuda.is_available():
    _p = torch.cuda.get_device_properties(0)
    log(f"GPU {_p.name} | VRAM {_p.total_memory/1024**3:.1f}GB | "
        f"sm_{_p.major}{_p.minor} | CUDA {torch.version.cuda}")
log(f"BASE_DIR       {BASE_DIR}")
log(f"HAND_CROP_DIR  {HAND_CROP_DIR}")
log(f"CACHE_DIR      {CACHE_DIR}")
log(f"CKPT_DIR       {CKPT_DIR}")
log(f"이미지 준비    전처리 없음 | {RESIZE_MODE} -> {IMG_SIZE}px | pad {PAD_VALUE}")
log(f"백본           {BACKBONE} (drop_path {DROP_PATH}, grad_ckpt {GRAD_CKPT})")
log(f"헤드           {HEAD_TYPE} | 성별임베딩 {GENDER_EMB_DIM} | dropout {DROPOUT}")
log(f"최적화         AdamW | lr {LR_BACKBONE:.0e}/{LR_HEAD:.0e} | LLRD {LLRD} | "
    f"wd {WEIGHT_DECAY} | 배치 {BATCH_SIZE}x{ACCUM_STEPS}={BATCH_SIZE*ACCUM_STEPS}")
log(f"스케줄         warmup {WARMUP_EPOCHS}ep + cosine(min ratio {MIN_LR_RATIO}) | "
    f"최대 {EPOCHS}ep")
log(f"조기종료       patience {EARLY_STOP_PATIENCE} | min_delta {MIN_DELTA} | "
    f"min_epochs {MIN_EPOCHS}")
log(f"손실 {LOSS_TYPE}"
    + (f"(beta {HUBER_BETA})" if LOSS_TYPE == "huber" else "")
    + f" | 증강 {USE_AUG} | EMA {USE_EMA}({EMA_DECAY})")
log(f"평가 TTA       {'ON ' + str(TTA_ANGLES) + '도' if USE_TTA else 'OFF'} "
    f"(학습/best 선택에는 미적용)")
log(f"제외 ID        {len(EXCLUDE_IDS)}개 (USE_EXCLUDE={USE_EXCLUDE})")
log(f"DataLoader     num_workers={NUM_WORKERS}"
    + ("  [Windows 안전장치로 0 강제]" if _WORKERS_FORCED else ""))
_flags = [f for f, on in [("--eval-only", EVAL_ONLY), ("--rebuild-cache", REBUILD_CACHE),
                          ("--qc-only", QC_ONLY), ("--fresh", FRESH), ("--fg", FOREGROUND)] if on]
log(f"실행 옵션      {' '.join(_flags) if _flags else '(없음)'}")
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
# [D] 이미지 준비 - 화소값을 바꾸는 코드는 여기에 '없습니다'
#       to_gray8()  : 8bit 보장 (이미 8bit 면 그대로 통과)
#       fit_canvas(): 비율유지 리사이즈 + 0 패딩 (또는 stretch)
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


def fit_canvas(g, size=IMG_SIZE, mode=RESIZE_MODE, pad_val=PAD_VALUE, anchor=PAD_ANCHOR):
    """size x size 캔버스로 맞춥니다. 화소값 보정 없음. 반환 (canvas, info)"""
    g = to_gray8(g)
    h, w = g.shape[:2]

    if mode == "stretch":
        interp = cv2.INTER_AREA if (size < h and size < w) else cv2.INTER_CUBIC
        out = cv2.resize(g, (size, size), interpolation=interp)
        return out, {"sx": size / w, "sy": size / h, "src": (h, w),
                     "pad": (0, 0, 0, 0), "pad_frac": 0.0}

    # letterbox: 긴 변을 size 에 맞춤 -> 비율 왜곡 0
    s  = size / float(max(h, w))
    nh = max(1, min(size, int(round(h * s))))
    nw = max(1, min(size, int(round(w * s))))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    r = cv2.resize(g, (nw, nh), interpolation=interp)

    top  = 0 if anchor == "topleft" else (size - nh) // 2
    left = 0 if anchor == "topleft" else (size - nw) // 2
    out = cv2.copyMakeBorder(r, top, size - nh - top, left, size - nw - left,
                             cv2.BORDER_CONSTANT, value=int(pad_val))
    return out, {"sx": s, "sy": s, "src": (h, w),
                 "pad": (top, size - nh - top, left, size - nw - left),
                 "pad_frac": 1.0 - (nh * nw) / float(size * size)}


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
# [F] 스테이지 1 - 512 캔버스 캐시  (완료 표식 있으면 자동 스킵)
# =========================================================================
PRE_INFO = {"preprocess": "none", "img_size": IMG_SIZE, "resize_mode": RESIZE_MODE,
            "pad_value": PAD_VALUE, "pad_anchor": PAD_ANCHOR,
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
    log("[스테이지1] 512 캔버스 캐시 생성 시작")
    pad_fracs, upscaled = [], 0
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
            if max(g.shape[:2]) < IMG_SIZE:
                upscaled += 1
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
        if np.percentile(pf, 90) > 0.45:
            log("  [경고] 여백이 큽니다 -> 손 유효 해상도 손실. YOLO 크롭 마진을 점검하세요.")
    if upscaled:
        log(f"  [_2] 원본 긴 변 < {IMG_SIZE} 라 확대된 이미지: {upscaled:,}장")
        log(f"       확대는 새 정보를 만들지 못합니다. 이 비율이 높으면 576 의 이득이 줄어듭니다.")
    json.dump(PRE_INFO, open(CACHE_DONE, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    log(f"[스테이지1] 완료 표식: {CACHE_DONE}")


build_cache()


# =========================================================================
# [G] QC - 원본과 캔버스가 '화소값 기준으로' 동일한지 검증
#     전처리를 뺐다고 주장하려면 실제로 뺐는지 증명해야 합니다.
#     유효std / 원본std 가 0.95~1.05 밖이면 리샘플링 외의 조작이 끼어든 것.
# =========================================================================
def qc_sheet():
    log("[QC] 픽셀 통과 검증 시트 생성")
    sample = train_df.sample(min(N_QC, len(train_df)), random_state=SEED)
    fig, axes = plt.subplots(len(sample), 2, figsize=(7, 3.2 * len(sample)))
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
        ratio = float(eff.std()) / max(1e-6, float(raw.std()))
        rows.append({"id": r["id"], "원본크기": f"{raw.shape[1]}x{raw.shape[0]}",
                     "원본범위": f"{raw.min()}~{raw.max()}",
                     "원본std": round(float(raw.std()), 1),
                     "유효std": round(float(eff.std()), 1),
                     "std비": round(ratio, 3),
                     "여백%": f"{info['pad_frac']:.0%}"})
        for ax, (t, im) in zip(axes[i], [(f"raw {r['id']}", raw),
                                         (f"canvas {IMG_SIZE}", cv_)]):
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            ax.set_title(t, fontsize=9); ax.axis("off")
    plt.tight_layout()
    p = CKPT_DIR / "qc_pixel_passthrough.png"
    plt.savefig(p, dpi=110); plt.close()

    qc = pd.DataFrame(rows)
    log(f"[QC] 시트 저장: {p}")
    log("\n" + qc.to_string(index=False))
    bad = qc[(qc["std비"] < 0.95) | (qc["std비"] > 1.05)]
    if len(bad):
        log(f"[QC][경고] std비가 범위를 벗어난 {len(bad)}건 - 화소 조작이 끼어들었는지 확인하세요.")
    else:
        log("[QC] 전 샘플 std비 0.95~1.05 - 화소값 통과 확인")
    qc.to_csv(CKPT_DIR / "qc_pixel_passthrough.csv", index=False, encoding="utf-8-sig")


qc_sheet()
if QC_ONLY:
    log("--qc-only: QC 시트만 만들고 종료합니다.")
    raise SystemExit(0)


# =========================================================================
# [H] Dataset · Transform
# =========================================================================
_aug = [transforms.RandomAffine(degrees=AUG_ROT_DEG,
                                translate=(AUG_TRANSLATE, AUG_TRANSLATE),
                                scale=AUG_SCALE, fill=PAD_VALUE)]

train_tf = transforms.Compose(
    [transforms.ToPILImage()] + (_aug if USE_AUG else []) +
    [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
)
eval_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class BoneAgeDataset(Dataset):
    """캐시된 512 캔버스 -> 3채널 복제 -> 증강 -> ImageNet 정규화."""

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
        if g.shape[:2] != (IMG_SIZE, IMG_SIZE):
            g, _ = fit_canvas(g)
        x  = self.tf(np.stack([g] * 3, axis=-1))
        gd = torch.tensor([float(r["male"])], dtype=torch.float32)
        ym = torch.tensor([float(r["boneage"])], dtype=torch.float32)
        yn = (ym - AGE_MEAN) / AGE_STD
        return x, gd, yn, ym


# =========================================================================
# [I] 모델 - ConvNeXt-Tiny + 성별 임베딩
#     상업 사용 불가한 ConvNeXt V2 가중치는 assert 로 차단합니다.
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

    def __init__(self, backbone_name=BACKBONE, img_size=IMG_SIZE, head_type=HEAD_TYPE,
                 head_dim=HEAD_DIM, gender_dim=GENDER_EMB_DIM, dropout=DROPOUT,
                 drop_path=DROP_PATH, pretrained=True, verbose=True):
        super().__init__()
        self.backbone, self.backbone_name = make_backbone(backbone_name, pretrained, drop_path)
        self.head_type = head_type

        with torch.no_grad():
            feat = self.backbone(torch.zeros(1, 3, img_size, img_size))
        C, H, W = int(feat.shape[1]), int(feat.shape[2]), int(feat.shape[3])
        if verbose:
            log(f"[head] 특징맵 {C}x{H}x{W} · head_type={head_type}")

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
        img_size=arch["IMG_SIZE"],
        head_type=arch.get("HEAD_TYPE", "gap"),
        head_dim=arch.get("HEAD_DIM", 512),
        gender_dim=arch.get("GENDER_EMB_DIM", 32),
        dropout=arch.get("DROPOUT", 0.2),
        drop_path=arch.get("DROP_PATH", 0.1),
        pretrained=pretrained, verbose=verbose,
    )
    # [_2] OOM 안전밸브: 활성값을 저장하지 않고 역전파 때 재계산합니다.
    if GRAD_CKPT and hasattr(m.backbone, "set_grad_checkpointing"):
        m.backbone.set_grad_checkpointing(True)
        if verbose:
            log("[backbone] gradient checkpointing ON (VRAM 절약 / 속도 20~30% 손해)")
    return m.to(device).to(memory_format=torch.channels_last)


ARCH = {"BACKBONE": BACKBONE, "IMG_SIZE": IMG_SIZE, "HEAD_TYPE": HEAD_TYPE,
        "HEAD_DIM": HEAD_DIM, "GENDER_EMB_DIM": GENDER_EMB_DIM, "DROPOUT": DROPOUT,
        "DROP_PATH": DROP_PATH, "PREPROCESS": "none",
        "RESIZE_MODE": RESIZE_MODE, "PAD_VALUE": PAD_VALUE, "PAD_ANCHOR": PAD_ANCHOR,
        "USE_AUG": USE_AUG, "USE_EXCLUDE": USE_EXCLUDE, "LOSS": LOSS_TYPE,
        "LLRD": LLRD, "VARIANT": "_2"}


# =========================================================================
# [J] 옵티마이저 · 스케줄 · EMA
#     LayerNorm 가중치와 bias(ndim<=1)에는 weight decay 를 주지 않습니다.
# =========================================================================
_LLRD_MAX_DEPTH = 6   # 0=stem, 1~4=stages.0~3, 5=기타 백본, 6=헤드


def _depth_of(name):
    """파라미터 이름 -> 깊이. 얕을수록 작은 값 -> 더 낮은 LR."""
    if not name.startswith("backbone."):
        return _LLRD_MAX_DEPTH          # 헤드
    if ".stem" in name or name.startswith("backbone.stem"):
        return 0
    for i in range(4):
        if f"stages.{i}." in name or f"stages_{i}." in name:
            return i + 1
    return 5                            # norm_pre 등 백본 최상단


def build_param_groups(model, lr_head, lr_backbone, wd):
    """[_2] Layer-wise LR Decay.
       깊이 d 의 LR = lr_backbone * LLRD^(MAX-1-d)  (헤드는 lr_head 고정)
       LayerNorm 가중치와 bias(ndim<=1)에는 weight decay 를 주지 않습니다.
       LLRD=1.0 이면 _1 과 동일한 균일 LR 로 동작합니다."""
    buckets = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        d  = _depth_of(n)
        nd = (p.ndim <= 1) or n.endswith(".bias")
        lr = lr_head if d == _LLRD_MAX_DEPTH else lr_backbone * (LLRD ** (_LLRD_MAX_DEPTH - 1 - d))
        key = (d, nd)
        if key not in buckets:
            label = ("head" if d == _LLRD_MAX_DEPTH else
                     "stem" if d == 0 else
                     f"stage{d-1}" if 1 <= d <= 4 else "bbtop")
            buckets[key] = {"params": [], "lr": lr, "base_lr": lr,
                            "weight_decay": 0.0 if nd else wd,
                            "name": label + ("_nd" if nd else "")}
        buckets[key]["params"].append(p)

    out = [buckets[k] for k in sorted(buckets.keys())]
    for g in out:
        n_par = sum(p.numel() for p in g["params"])
        log(f"  {g['name']:<10} tensors={len(g['params']):>3} params={n_par/1e6:>6.2f}M "
            f"lr={g['lr']:.2e} wd={g['weight_decay']}")
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

    def __init__(self, model, decay=0.999):
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
    """개월 단위 (MAE, RMSE) 반환."""
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

    start_epoch, best_val, no_improve, global_step = 1, float("inf"), 0, 0
    history = {"train_mae": [], "val_mae": [], "val_rmse": [], "val_mae_ema": [],
               "lr": [], "sec": []}

    # ── 자동 재개: last.pt 가 있고 구조가 같으면 이어서 ─────────────
    if LAST_CKPT.exists() and not FRESH:
        ck = torch_load(LAST_CKPT, map_location=device)
        bad = [k for k in ("IMG_SIZE", "HEAD_TYPE", "HEAD_DIM", "GENDER_EMB_DIM", "BACKBONE")
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

            history["train_mae"].append(tr_mae)
            history["val_mae"].append(va_mae)
            history["val_rmse"].append(va_rmse if which == "raw" else va_ema_rmse)
            history["val_mae_ema"].append(None if ema is None else va_ema)
            history["lr"].append(optimizer.param_groups[-1]["lr"])
            history["sec"].append(round(dt, 1))

            msg = (f"[{epoch:02d}/{EPOCHS}] train {tr_mae:.2f} | val {va_mae:.2f} "
                   f"(rmse {va_rmse:.2f})")
            if ema is not None:
                msg += f" | val(ema) {va_ema:.2f}"
            msg += f" | {dt/60:.1f}분"

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

    log(f"[스테이지2] 완료 · 최종 best val MAE = {best_val:.2f} 개월")


if not EVAL_ONLY:
    train_phase()
else:
    log("--eval-only: 학습을 건너뜁니다.")


# =========================================================================
# [L] 스테이지 3 - 평가  (best.pt 만 있으면 단독 실행 가능)
# =========================================================================
if not BEST_CKPT.exists():
    raise SystemExit(f"[중단] best.pt 가 없습니다: {BEST_CKPT}\n"
                     f"       --eval-only 없이 다시 실행해 학습을 먼저 진행하세요.")

_ck = torch_load(BEST_CKPT, map_location=device)
eval_model = build_model(_ck["arch"], pretrained=False)
eval_model.load_state_dict(_ck["model"]); eval_model.eval()
EM_MEAN, EM_STD = _ck["age_mean"], _ck["age_std"]

log("=" * 72)
log("[스테이지3] 평가 시작")
log(f"  백본     {_ck['arch'].get('BACKBONE_RESOLVED', _ck['arch']['BACKBONE'])}")
log(f"  헤드     {_ck['arch'].get('HEAD_TYPE')} | 전처리 {_ck['arch'].get('PREPROCESS')}")
log(f"  가중치   {_ck.get('best_from')} (raw/ema 중 선택된 쪽)")
log(f"  정규화   {EM_MEAN:.1f} ± {EM_STD:.1f}")
log(f"  best val MAE {_ck.get('best_val', float('nan')):.2f} @ epoch {_ck.get('epoch')}")


# 정규화 후의 '패딩 0' 에 해당하는 값. TTA 회전 시 빈 곳을 이 값으로 채워야
# 학습 때 본 검은 여백과 같은 입력이 됩니다.
_TTA_FILL = [(PAD_VALUE / 255.0 - m) / sd for m, sd in zip(IMAGENET_MEAN, IMAGENET_STD)]


@torch.no_grad()
def predict_split(model, df, split, bs=16, tta=False):
    """개월 단위 (preds, trues, ids) 반환. tta=True 면 TTA_ANGLES 평균."""
    loader = DataLoader(BoneAgeDataset(df, split, eval_tf), batch_size=bs,
                        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    angles = TTA_ANGLES if tta else [0]
    P, T = [], []
    for x, g, yn, ym in loader:
        x = x.to(device).to(memory_format=torch.channels_last); g = g.to(device)
        acc = None
        for a in angles:
            xi = x if a == 0 else TF.rotate(
                x, a, interpolation=TF.InterpolationMode.BILINEAR, fill=_TTA_FILL)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                out = model(xi, g).float()
            acc = out if acc is None else acc + out
        p = acc / len(angles)
        P.append(p.cpu() * EM_STD + EM_MEAN); T.append(ym.squeeze(1))
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


results, lines = {}, []
lines.append("=" * 60)
lines.append(" [_2] ConvNeXt-Small 단일 회귀 / 전처리 없음 / 레터박스 576")
lines.append(f" 백본 {_ck['arch'].get('BACKBONE_RESOLVED', BACKBONE)} | "
             f"헤드 {_ck['arch'].get('HEAD_TYPE')}")
lines.append(f" 실행 {RUN_TS} | best epoch {_ck.get('epoch')} ({_ck.get('best_from')})")
lines.append(f" LLRD {_ck['arch'].get('LLRD')} | TTA {TTA_ANGLES if USE_TTA else 'OFF'}")
lines.append("=" * 60)

# ── train / val ──────────────────────────────────────────────────────
for sp, df in [("train", train_df), ("val", val_df)]:
    p, t, _ = predict_split(eval_model, df, sp, tta=USE_TTA)
    results[sp] = metrics(p, t)
    lines.append(f" {sp.upper():<6} N={results[sp]['N']:>6,}  MAE {results[sp]['mae']:5.2f}  "
                 f"RMSE {results[sp]['rmse']:5.2f}  bias {results[sp]['bias']:+5.2f}")
    if sp == "val":
        v_pred, v_true = p, t

# ── test: 무TTA (기준값) ─────────────────────────────────────────────
t_pred_ntta, t_true, t_ids = predict_split(eval_model, test_df, "test", tta=False)
results["test_noTTA"] = metrics(t_pred_ntta, t_true)
lines.append(f" {'TEST(무TTA)':<14} N={results['test_noTTA']['N']:>5,}  "
             f"MAE {results['test_noTTA']['mae']:5.2f}  "
             f"RMSE {results['test_noTTA']['rmse']:5.2f}  "
             f"bias {results['test_noTTA']['bias']:+5.2f}")

# ── test: 현재 설정 (TTA 적용 여부는 USE_TTA) ────────────────────────
if USE_TTA:
    t_pred, _t2, _i2 = predict_split(eval_model, test_df, "test", tta=True)
    lines.append(f" TTA 이득 = {results['test_noTTA']['mae'] - float(np.abs(t_pred-t_true).mean()):+.3f} 개월")
else:
    t_pred = t_pred_ntta
key_a = "test_filtered" if USE_EXCLUDE else "test_raw"
results[key_a] = metrics(t_pred, t_true)
lo, hi = bootstrap_ci(t_pred, t_true)
results[key_a]["ci95"] = [round(lo, 2), round(hi, 2)]
tag_a = "TEST(제외적용)" if USE_EXCLUDE else "TEST(필터없음)"
lines.append(f" {tag_a:<14} N={results[key_a]['N']:>5,}  MAE {results[key_a]['mae']:5.2f}  "
             f"RMSE {results[key_a]['rmse']:5.2f}  bias {results[key_a]['bias']:+5.2f}  "
             f"CI95 [{lo:.2f}, {hi:.2f}]")

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
    f_pred, f_true, _ = predict_split(eval_model, full, "test", tta=USE_TTA)
    results["test_raw"] = metrics(f_pred, f_true)
    lo2, hi2 = bootstrap_ci(f_pred, f_true)
    results["test_raw"]["ci95"] = [round(lo2, 2), round(hi2, 2)]
    lines.append(f" {'TEST(필터없음)':<14} N={results['test_raw']['N']:>5,}  "
                 f"MAE {results['test_raw']['mae']:5.2f}  "
                 f"RMSE {results['test_raw']['rmse']:5.2f}  "
                 f"bias {results['test_raw']['bias']:+5.2f}  CI95 [{lo2:.2f}, {hi2:.2f}]")
    lines.append(f" 필터로 인한 MAE 차이 = "
                 f"{results['test_raw']['mae'] - results['test_filtered']['mae']:+.2f} 개월")

lines.append("-" * 60)
lines.append(" [연령대별 · test]")
_agrp = age_group_table(t_pred, t_true)
for _, r in _agrp.iterrows():
    lines.append(f"   {r['구간']:<7} N={int(r['N']):>4}  MAE {r['MAE']:5.2f}  bias {r['bias']:+5.2f}")
results["age_groups_test"] = _agrp.to_dict("records")
lines.append("=" * 60)


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
        ax[0].axhline(4.10, ls="--", c="green", label="Zhang 2026 · 4.10")
        ax[0].axhline(4.30, ls=":", c="orange", label="Chen 2020 · 4.30")
        ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("MAE (months)")
        ax[0].set_title("[_2] ConvNeXt-Small 576 / no preprocessing")
        ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
        ax[1].plot(ep, h["lr"], "-o", ms=3, c="crimson")
        ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("LR (head)")
        ax[1].set_title("warmup + cosine"); ax[1].set_yscale("log"); ax[1].grid(alpha=.3)
        plt.tight_layout(); plt.savefig(CKPT_DIR / "learning_curve.png", dpi=120); plt.close()
        log(f"  학습곡선 저장: {CKPT_DIR/'learning_curve.png'}")
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
        fig, axes = plt.subplots(2, (k + 1) // 2, figsize=(4 * ((k + 1) // 2), 8))
        for a, i in zip(np.atleast_1d(axes).ravel(), order):
            g = imread_kr(CACHE_DIR / "test" / f"{t_ids[i]}.png", cv2.IMREAD_GRAYSCALE)
            if g is not None:
                a.imshow(g, cmap="gray")
            a.axis("off")
            a.set_title(f"{t_ids[i]}\ntrue {t_true[i]:.0f} / pred {t_pred[i]:.0f} "
                        f"({t_pred[i]-t_true[i]:+.0f})", fontsize=9)
        plt.suptitle("오차 상위 - 크롭 실패·라벨 오류 후보", y=1.0)
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
    """전처리도 attention 도 없으므로, 관심이 성장판이 아니라 여백·마커로
       새는지 확인합니다. 엉뚱한 곳이 밝다면 그것이 곧 개선 여지의 증거입니다."""
    try:
        engine = GradCAM(eval_model)
        samp = test_df.sample(min(k, len(test_df)), random_state=SEED)
        fig, axes = plt.subplots(1, len(samp), figsize=(4.2 * len(samp), 4.6))
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

results["arch"] = _ck["arch"]
results["run_ts"] = RUN_TS
results["tta"] = {"enabled": USE_TTA, "angles": TTA_ANGLES}
results["env_file"] = str(_env_path)
results["best_epoch"] = _ck.get("epoch")
results["best_from"] = _ck.get("best_from")
results["when"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

json.dump(results, open(RESULTS_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
open(RESULTS_TXT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)
log(f"결과 저장: {RESULTS_TXT}")
log(f"         {RESULTS_JSON}")


# =========================================================================
# [M] 단일 이미지 추론  (best.pt + 크롭된 이미지 1장이면 끝)
#     이미지 준비 설정을 체크포인트에서 읽어 학습과 동일한 처리를 보장합니다.
# =========================================================================
_INFER_CACHE = {}


def predict_bone_age(image_path, is_male, ckpt_path=BEST_CKPT):
    """YOLO 로 손이 크롭된 X-ray 경로 + 성별(True=남) -> 골연령(개월)."""
    key = str(ckpt_path)
    if key not in _INFER_CACHE:
        c = torch_load(ckpt_path, map_location=device)
        m = build_model(c["arch"], pretrained=False, verbose=False)
        m.load_state_dict(c["model"]); m.eval()
        _INFER_CACHE[key] = (m, c)
    m, c = _INFER_CACHE[key]
    a = c["arch"]

    g = imread_kr(image_path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise FileNotFoundError(image_path)
    canvas, _ = fit_canvas(g, size=a["IMG_SIZE"], mode=a.get("RESIZE_MODE", "letterbox"),
                          pad_val=a.get("PAD_VALUE", 0), anchor=a.get("PAD_ANCHOR", "center"))
    x = eval_tf(np.stack([canvas] * 3, -1)).unsqueeze(0).to(device)
    x = x.to(memory_format=torch.channels_last)
    gd = torch.tensor([[1.0 if is_male else 0.0]], dtype=torch.float32, device=device)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=USE_AMP):
        p = m(x, gd)
    return float(p.float().cpu().item()) * c["age_std"] + c["age_mean"]


# 예시:
#   months = predict_bone_age(HAND_CROP_DIR / "validation" / "1377.png", is_male=True)
#   print(f"예측 골연령: {months:.1f} 개월")
try:
    _r = test_df.iloc[0]
    _m = predict_bone_age(_r["path"], bool(_r["male"]))
    log(f"추론 함수 확인 [{_r['id']}] 예측 {_m:.1f}개월 / 실제 {_r['boneage']:.0f}개월")
except Exception as e:
    log(f"[경고] 추론 함수 확인 실패: {e}")

log(f"추론 함수 준비 완료 · best.pt: {BEST_CKPT}")
log("=== 전체 완료 ===")
