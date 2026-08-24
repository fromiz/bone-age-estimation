# -*- coding: utf-8 -*-
# =========================================================================
# 손 X-ray 골연령 추정 - ConvNeXt-Tiny 단일 회귀
#
# ┌─ 데이터 흐름 ────────────────────────────────────────────────────────┐
# │                                                                      │
# │  crop_data/{training,validation,test}/*.png   (YOLO 로 손만 크롭됨)  │
# │  crop_data_csv/*.csv                          (id, boneage, male)   │
# │            │                                                         │
# │            ├─[E]─ 라벨 로드 + 파일 인덱싱 → train/val/test DataFrame │
# │            │      · 타깃 z-정규화 상수(AGE_MEAN/STD)는 train 에서만  │
# │            ▼                                                         │
# │      [D] 이미지 준비  (한 장당 1회, 결과를 캐시에 저장)              │
# │            1. to_gray8        16bit/float → 8bit (이미 8bit면 통과)  │
# │            2. norm_intensity  이미지별 1~99 퍼센타일 스트레치        │
# │            3. fit_canvas      비율유지 리사이즈 + 0 패딩 → 640×448   │
# │            ▼                                                         │
# │      [F] 캐시  cache_convnext_single_v2/<PRE_TAG>/{train,val,test}/  │
# │            ▼                                                         │
# │      [H] Dataset  캐시 PNG 읽기 → 3채널 복제 → 증강 → ImageNet 정규화│
# │            ▼                                                         │
# │      [I] 모델  ConvNeXt-Tiny → GAP → Dense(512)                      │
# │                                    ⊕ 성별임베딩(32) → Dense(128) → 1 │
# │            ▼                                                         │
# │      [K] 학습  AdamW + warmup/cosine + EMA + grad accumulation       │
# │            ▼   매 에폭: 원본·EMA 둘 다 val 평가 → 좋은 쪽을 best.pt  │
# │            ▼                                                         │
# │      [L] 평가  회전 TTA → val 로 선형 캘리브레이션 적합 → test 적용  │
# │            ▼                                                         │
# │      [M] 단일 이미지 추론 함수                                       │
# └──────────────────────────────────────────────────────────────────────┘
#
# ▶ 코드 구성 (위에서 아래로 순차 실행되는 단일 스크립트)
#     [A] 런처          자기 자신을 백그라운드 프로세스로 분리
#     [B] 설정          경로 · 하이퍼파라미터 · 환경 스냅샷
#     [C] I/O 헬퍼      한글 경로 대응 imread/imwrite
#     [D] 이미지 준비   to_gray8 / norm_intensity / fit_canvas
#     [E] 라벨          CSV 파싱 + 파일 인덱싱 + z-정규화 상수
#     [F] 스테이지1     캔버스 캐시 생성 (완료 표식 있으면 스킵)
#     [G] QC            강도 정규화가 실제로 먹었는지 검증
#     [H] Dataset       Transform + Dataset 클래스
#     [I] 모델          백본 + 헤드 + build_model
#     [J] 학습 부품     param group / LR 스케줄 / EMA / 손실 / evaluate
#     [K] 스테이지2     학습 루프
#     [L] 스테이지3     평가 (TTA + 캘리브레이션 + 그림)
#     [M] 추론          predict_bone_age
#
# ▶ 백본 - 상업 사용 가능한 ConvNeXt V1 계열만
#     convnext_tiny.fb_in22k_ft_in1k_384  (기본, Apache-2.0 / MIT)
#     convnext_tiny.fb_in22k_ft_in1k      (224 ft.)
#     convnext_tiny.in12k_ft_in1k         (timm 자체 학습, Apache-2.0)
#     ※ convnextv2_* 는 가중치가 CC BY-NC 4.0 -> 코드에서 차단
#
# ▶ 실행: python convnexttiny_single_640x448_v2.py
#     - 창을 닫아도 서버에서 학습은 계속됩니다. 다시 실행하면 로그에 재부착.
#     - 각 스테이지는 완료 표식을 남기고 자동 스킵 -> 중단돼도 이어서 진행.
#
# ▶ 실행 옵션
#     --fg             백그라운드 분리 없이 바로 실행(디버그용)
#     --eval-only      학습을 건너뛰고 best.pt 로 평가만
#     --rebuild-cache  캔버스 캐시를 강제로 재생성
#     --qc-only        QC 시트만 만들고 종료
#     --fresh          last.pt 를 무시하고 처음부터 학습
#
# ▶ 산출물  checkpoints_convnext_single_v2/
#     best.pt              BEST_SELECT 규칙으로 고른 가중치 (평가·추론이 사용)
#     best_raw.pt          원본 가중치 기준 최고 (감사용)
#     best_ema.pt          EMA 가중치 기준 최고 (감사용)
#     last.pt              매 에폭 저장 (중단 시 재개용)
#     history.json         에폭별 train/val MAE, LR
#     calibration.json     배포용 캘리브레이션 계수
#     results.txt|json     최종 성적표
#     env_<ts>.json        실행 환경·설정 스냅샷
#     *.png|csv            QC / 학습곡선 / 산점도 / 오차상위 / GradCAM
#   logs/convnext_single_v2_<ts>.log   실행 로그 전체
# =========================================================================
from pathlib import Path
import os, sys, time, json, subprocess, platform
from datetime import datetime

PROJECT_DIR = Path(__file__).resolve().parent
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RUN_STATE = LOG_DIR / "convnext_single_v4_running.json"
_WORKER_ENV = "BONEAGE_CONVNEXT_SINGLE_V4_WORKER"

FOREGROUND    = "--fg" in sys.argv
EVAL_ONLY     = "--eval-only" in sys.argv
REBUILD_CACHE = "--rebuild-cache" in sys.argv
QC_ONLY       = "--qc-only" in sys.argv
FRESH         = "--fresh" in sys.argv


# =========================================================================
# [A] 런처
#     Windows 에서 이 스크립트를 그냥 실행하면, 자기 자신을 세션과 분리된
#     백그라운드 프로세스로 다시 띄우고 현재 창은 로그만 흘려보냅니다.
#     이미 실행 중이면 새로 띄우지 않고 기존 로그에 재부착합니다.
# =========================================================================
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
                        print("\n[프로세스 종료됨]", flush=True)
                        break
                    time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[로그 보기만 종료합니다 - 학습은 계속 진행 중]", flush=True)


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
# [B] 설정
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


# ── 입력 경로 ────────────────────────────────────────────────────────
BASE_DIR      = Path(r"G:/Project/sinra_cho")
HAND_CROP_DIR = BASE_DIR / "crop_data_final_yolox_s"        # training / validation / test 하위폴더
CSV_DIR       = BASE_DIR / "crop_data_final_yolox_s"   # training.csv / validation.csv / test.csv

SPLIT_SUBDIR = {"train": "training", "val": "validation", "test": "test"}
SPLITS       = ("train", "val", "test")

_require(HAND_CROP_DIR, "손 크롭 폴더")
TRAIN_CSV = _require(CSV_DIR / "training.csv",   "training.csv")
VAL_CSV   = _require(CSV_DIR / "validation.csv", "validation.csv")
TEST_CSV  = _require(CSV_DIR / "test.csv",       "test.csv")

SEED = 42

# ── 이미지 준비 ──────────────────────────────────────────────────────
#   손 크롭의 종횡비(w/h)는 중앙값 약 0.65 로 세로가 깁니다.
#   정사각 캔버스에 넣으면 픽셀의 30% 가 검은 패딩이 되므로,
#   448/640 = 0.70 인 비정사각 캔버스를 씁니다. ConvNeXt 는 완전
#   합성곱이라 비정사각 입력이 그대로 동작합니다.
IMG_H, IMG_W = 640, 448

RESIZE_MODE = "letterbox"   # "letterbox"(비율유지+패딩) | "stretch"(강제 리사이즈)
PAD_VALUE   = 0
PAD_ANCHOR  = "center"      # "center" | "topleft"

#   [추론] 논문 미명시. 이미지별 강도 정규화.
#   RSNA 원본은 촬영 노출이 제각각이라 이미지 std 가 7~56 까지 벌어집니다.
#   고정 ImageNet mean/std 로는 이 편차를 못 잡아서, 골단선 대비가
#   이미지마다 다른 스케일로 네트워크에 들어갑니다.
#   "none" 으로 두면 화소값을 전혀 건드리지 않는 조건이 됩니다.
NORM_MODE   = "p1p99"       # "none" | "p1p99" | "p2p98"

# ── 백본 ─────────────────────────────────────────────────────────────
BACKBONE     = "convnext_tiny.fb_in22k_ft_in1k_384"
BACKBONE_ALT = ["convnext_tiny.fb_in22k_ft_in1k",
                "convnext_tiny.in12k_ft_in1k"]
DROP_PATH    = 0.15         # ConvNeXt stochastic depth

# ── 헤드 ─────────────────────────────────────────────────────────────
HEAD_TYPE      = "gap"      # "gap" | "paper"(Conv3x3+MaxPool+Flatten)
HEAD_DIM       = 512
GENDER_EMB_DIM = 32         # [논문] 식(15) k=32
DROPOUT        = 0.10

# ── 최적화 ───────────────────────────────────────────────────────────
BATCH_SIZE    = 8           # 8GB VRAM 기준. OOM 이면 6 또는 4
ACCUM_STEPS   = 4           # 유효 배치 = BATCH_SIZE * ACCUM_STEPS = 32
EPOCHS        = 40          # 코사인 스케줄이 이 값에 맞춰 감깁니다
LR_HEAD       = 1e-4
LR_BACKBONE   = 4e-5        # 사전학습 특징 보존 -> 헤드보다 낮게
WEIGHT_DECAY  = 0.05        # AdamW. norm/bias 에는 적용하지 않음
WARMUP_EPOCHS = 3
MIN_LR_RATIO  = 0.02        # cosine 최저 LR = base * 이 값
CLIP_GRAD     = 1.0         # None 이면 끔
USE_GRAD_CKPT = False       # OOM 이면 True (속도 -25%, VRAM -40%)

# ── 손실 ─────────────────────────────────────────────────────────────
#   HUBER_BETA 는 z-정규화 스케일 기준입니다. AGE_STD 가 40 개월대라
#   beta 0.5 는 전환점이 20 개월이 되어 사실상 MSE 로 동작합니다.
LOSS_TYPE  = "l1"           # "huber" | "l1"
HUBER_BETA = 0.5

# ── 증강 ─────────────────────────────────────────────────────────────
#   좌우 반전은 넣지 않습니다 - RSNA 는 전부 좌수 촬영이라
#   반전 이미지는 해부학적으로 존재하지 않는 입력입니다.
USE_AUG       = True
AUG_ROT_DEG   = 12
AUG_TRANSLATE = 0.06
AUG_SCALE     = (0.92, 1.08)
AUG_SHEAR     = 4
AUG_JITTER    = 0.18        # brightness/contrast. 촬영 노출 편차를 모사
AUG_ERASE_P   = 0.25        # RandomErasing. scale 1~4% 로 작게
                            #   크게 잡으면 골단선을 통째로 지워 라벨 노이즈가 됨

# ── EMA ──────────────────────────────────────────────────────────────
#   학습 중 가중치는 미니배치마다 흔들립니다. 그 궤적의 지수이동평균을
#   따로 유지하면 더 안정적인 지점에 앉습니다. 매 에폭 원본과 EMA 를
#   둘 다 val 로 평가해서 좋은 쪽을 best.pt 에 저장합니다.
USE_EMA   = True
EMA_DECAY = 0.9995          # 유효 창 ≈ 1/(1-decay) 스텝

#   어느 가중치를 best.pt 로 쓸지는 학습 전에 고정합니다.
#   매 에폭 raw/EMA 중 좋은 쪽을 val 로 고르면, 40에폭 x 2후보 = 80개 중
#   최솟값을 val 에서 뽑는 셈이라 val MAE 가 낙관적으로 편향됩니다.
#   규칙을 미리 못박아 두면 "왜 그 체크포인트를 골랐나" 에 답할 수 있습니다.
#   raw / EMA 각각의 best 는 별도 파일로도 저장되어 평가 단계에서
#   같은 test 로 나란히 비교됩니다 (EMA 이득이 실제인지 감사).
BEST_SELECT = "ema"         # "ema"(권장) | "raw" | "auto"(에폭마다 좋은 쪽, 편향 있음)

# ── 추론 ─────────────────────────────────────────────────────────────
TTA_ANGLES = (0, -4, 4)     # 회전 TTA. (0,) 이면 단일 forward
USE_CALIB  = True           # val 에서 실제로 개선될 때만 자동 적용

# ── 학습 종료 조건 ───────────────────────────────────────────────────
#   코사인 스케줄은 끝까지 가야 LR 이 base*MIN_LR_RATIO 까지 내려가고
#   거기서 수렴이 일어납니다. 중간에 끊으면 그 구간을 통째로 버립니다.
#   best.pt 는 매 에폭 갱신되므로 끝까지 돌려도 과적합 가중치를 쓰지 않습니다.
EARLY_STOP_PATIENCE = 0     # 0 = 끔
MIN_DELTA           = 0.0
MIN_EPOCHS          = EPOCHS

NUM_WORKERS = 0             # [중요] Windows 는 반드시 0.
                            #   이 스크립트는 전체가 모듈 최상위에서 실행되므로
                            #   workers>0 이면 spawn 된 워커가 스크립트를 다시
                            #   import 하면서 학습이 통째로 재실행됩니다.
N_QC        = 8
BOOTSTRAP_N = 2000

if os.name == "nt" and NUM_WORKERS > 0:
    NUM_WORKERS = 0
    _WORKERS_FORCED = True
else:
    _WORKERS_FORCED = False

# ── 출력 경로 ────────────────────────────────────────────────────────
#   PRE_TAG 에 캔버스 크기와 정규화 모드가 들어가므로, 설정을 바꾸면
#   캐시 폴더가 자동으로 분리됩니다 (낡은 캐시를 조용히 재사용하지 않음).
PRE_TAG   = f"raw{IMG_H}x{IMG_W}_{RESIZE_MODE}_pad{PAD_VALUE}_{PAD_ANCHOR}_n{NORM_MODE}"
CACHE_DIR = BASE_DIR / "cache_convnext_single_v4" / PRE_TAG
CKPT_DIR  = BASE_DIR / "checkpoints_convnext_single_v4"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
for sp in SPLITS:
    (CACHE_DIR / sp).mkdir(parents=True, exist_ok=True)

CACHE_DONE   = CACHE_DIR / "_DONE_cache.json"
BEST_CKPT    = CKPT_DIR / "best.pt"          # BEST_SELECT 규칙으로 고른 최종본
BEST_RAW_CKPT = CKPT_DIR / "best_raw.pt"     # 원본 가중치 기준 최고 (감사용)
BEST_EMA_CKPT = CKPT_DIR / "best_ema.pt"     # EMA 가중치 기준 최고 (감사용)
LAST_CKPT    = CKPT_DIR / "last.pt"
HISTORY_JSON = CKPT_DIR / "history.json"
RESULTS_TXT  = CKPT_DIR / "results.txt"
RESULTS_JSON = CKPT_DIR / "results.json"
CALIB_JSON   = CKPT_DIR / "calibration.json"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
#   정규화 공간에서 '픽셀값 0' 에 해당하는 값. 회전 TTA 의 여백을 이걸로
#   채웁니다. 0 을 넣으면 중간 회색(≈124)이 들어가 TTA 가 손해가 됩니다.
PAD_NORM = [-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")


def _pkg_versions():
    out = {}
    for name in ("torch", "torchvision", "timm", "numpy", "pandas", "cv2",
                 "matplotlib", "PIL"):
        try:
            out[name] = getattr(__import__(name), "__version__", "?")
        except Exception:
            out[name] = None
    return out


def dump_env(extra=None):
    """실행 환경 + 전체 설정을 JSON 한 파일로 남깁니다.
       나중에 '그때 무슨 조건이었지?' 를 없애기 위한 스냅샷입니다."""
    gpu = {}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        gpu = {"name": p.name, "vram_gb": round(p.total_memory / 1024 ** 3, 2),
               "capability": f"{p.major}.{p.minor}", "cuda": torch.version.cuda,
               "cudnn": torch.backends.cudnn.version()}
    info = {
        "run_ts": RUN_TS, "script": str(Path(__file__).resolve()),
        "argv": sys.argv[1:], "pid": os.getpid(),
        "log_path": os.environ.get("BONEAGE_LOG_PATH"),
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "packages": _pkg_versions(),
        "device": str(device), "gpu": gpu, "amp": bool(USE_AMP), "seed": SEED,
        "paths": {"BASE_DIR": str(BASE_DIR), "HAND_CROP_DIR": str(HAND_CROP_DIR),
                  "CSV_DIR": str(CSV_DIR), "CACHE_DIR": str(CACHE_DIR),
                  "CKPT_DIR": str(CKPT_DIR)},
        "image_prep": {"img_h": IMG_H, "img_w": IMG_W, "norm_mode": NORM_MODE,
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
                       "use_ema": USE_EMA, "ema_decay": EMA_DECAY,
                       "best_select": BEST_SELECT},
        "inference": {"tta_angles": list(TTA_ANGLES), "calib": USE_CALIB},
        "stopping": {"patience": EARLY_STOP_PATIENCE, "min_delta": MIN_DELTA,
                     "min_epochs": MIN_EPOCHS},
    }
    if extra:
        info.update(extra)
    p = CKPT_DIR / f"env_{RUN_TS}.json"
    json.dump(info, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return info, p


print("=" * 72)
log("ConvNeXt-Tiny 단일 회귀 골연령 시작")
log(f"Python {sys.version.split()[0]} | {platform.system()} {platform.release()}")
log(f"PyTorch {torch.__version__} | torchvision {torchvision.__version__} | timm {timm.__version__}")
log(f"OpenCV {cv2.__version__} | numpy {np.__version__} | pandas {pd.__version__}")
log(f"device {device} | CUDA {torch.cuda.is_available()} | AMP {USE_AMP}")
if torch.cuda.is_available():
    _p = torch.cuda.get_device_properties(0)
    log(f"GPU {_p.name} | VRAM {_p.total_memory/1024**3:.1f}GB | "
        f"sm_{_p.major}{_p.minor} | CUDA {torch.version.cuda}")
log(f"HAND_CROP_DIR  {HAND_CROP_DIR}")
log(f"CSV_DIR        {CSV_DIR}")
log(f"CACHE_DIR      {CACHE_DIR}")
log(f"CKPT_DIR       {CKPT_DIR}")
log(f"이미지 준비    {RESIZE_MODE} -> {IMG_H}x{IMG_W}(HxW) | pad {PAD_VALUE} | "
    f"강도정규화 {NORM_MODE}")
log(f"백본           {BACKBONE} (drop_path {DROP_PATH})")
log(f"헤드           {HEAD_TYPE} | 성별임베딩 {GENDER_EMB_DIM} | dropout {DROPOUT}")
log(f"최적화         AdamW | lr {LR_BACKBONE:.0e}/{LR_HEAD:.0e} | wd {WEIGHT_DECAY} | "
    f"배치 {BATCH_SIZE}x{ACCUM_STEPS}={BATCH_SIZE*ACCUM_STEPS}")
log(f"스케줄         warmup {WARMUP_EPOCHS}ep + cosine(min ratio {MIN_LR_RATIO}) | "
    f"총 {EPOCHS}ep | 조기종료 "
    f"{'끔' if EARLY_STOP_PATIENCE <= 0 else EARLY_STOP_PATIENCE}")
log(f"손실 {LOSS_TYPE}" + (f"(beta {HUBER_BETA})" if LOSS_TYPE == "huber" else "")
    + f" | 증강 {USE_AUG} | EMA {USE_EMA}({EMA_DECAY})"
    + f" | best 선택 '{BEST_SELECT}'")
if USE_AUG:
    log(f"증강 상세      affine rot{AUG_ROT_DEG}° tr{AUG_TRANSLATE} sc{AUG_SCALE} "
        f"sh{AUG_SHEAR} | jitter ±{AUG_JITTER} | erasing p={AUG_ERASE_P}")
log(f"추론           TTA {list(TTA_ANGLES) if len(TTA_ANGLES) > 1 else '끔'} | "
    f"캘리브레이션 {'자동판정' if USE_CALIB else '끔'}")
log(f"DataLoader     num_workers={NUM_WORKERS}"
    + ("  [Windows 안전장치로 0 강제]" if _WORKERS_FORCED else ""))
_flags = [f for f, on in [("--eval-only", EVAL_ONLY), ("--rebuild-cache", REBUILD_CACHE),
                          ("--qc-only", QC_ONLY), ("--fresh", FRESH),
                          ("--fg", FOREGROUND)] if on]
log(f"실행 옵션      {' '.join(_flags) if _flags else '(없음)'}")
print("=" * 72, flush=True)
# [주의] RTX 5060(Blackwell, sm_120)은 최신 PyTorch 필요.
#   CUDA=False 또는 'no kernel image' 오류 시:
#   pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128


# =========================================================================
# [C] I/O 헬퍼
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
# [D] 이미지 준비
#     원본 크롭 한 장이 네트워크 입력 캔버스가 되기까지의 전 과정.
#     fit_canvas() 하나만 호출하면 아래 3단계가 순서대로 적용됩니다.
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
    """이미지별 퍼센타일 스트레치.

    YOLO 크롭의 0-패딩 배경(g==0)은 통계에서 제외합니다.
    반드시 '패딩 전, 원본 크롭 위에서' 호출해야 합니다.
    letterbox 이후에 하면 패딩 0 이 통계에 섞여 스트레치가 망가집니다.
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
    return np.clip((g.astype(np.float32) - lo) / (hi - lo) * 255.0,
                   0, 255).astype(np.uint8)


def fit_canvas(g, h_out=None, w_out=None, mode=None,
               pad_val=None, anchor=None, norm=None):
    """to_gray8 -> norm_intensity -> 리사이즈/패딩. 반환 (canvas, info)

    letterbox 스케일 = min(h_out/h, w_out/w) 이므로 양 변 모두 넘치지 않는
    최대 배율이 잡히고 종횡비 왜곡은 0 입니다.
    info["pad_frac"] 은 캔버스에서 패딩이 차지하는 비율입니다.
    """
    h_out   = IMG_H       if h_out   is None else h_out
    w_out   = IMG_W       if w_out   is None else w_out
    mode    = RESIZE_MODE if mode    is None else mode
    pad_val = PAD_VALUE   if pad_val is None else pad_val
    anchor  = PAD_ANCHOR  if anchor  is None else anchor

    g = to_gray8(g)
    g = norm_intensity(g, NORM_MODE if norm is None else norm)
    h, w = g.shape[:2]

    if mode == "stretch":
        interp = cv2.INTER_AREA if (h_out < h and w_out < w) else cv2.INTER_CUBIC
        out = cv2.resize(g, (w_out, h_out), interpolation=interp)
        return out, {"scale": (h_out / h, w_out / w), "src": (h, w),
                     "pad": (0, 0, 0, 0), "pad_frac": 0.0}

    s  = min(h_out / float(h), w_out / float(w))
    nh = max(1, min(h_out, int(round(h * s))))
    nw = max(1, min(w_out, int(round(w * s))))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    r = cv2.resize(g, (nw, nh), interpolation=interp)

    top  = 0 if anchor == "topleft" else (h_out - nh) // 2
    left = 0 if anchor == "topleft" else (w_out - nw) // 2
    out = cv2.copyMakeBorder(r, top, h_out - nh - top, left, w_out - nw - left,
                             cv2.BORDER_CONSTANT, value=int(pad_val))
    return out, {"scale": s, "src": (h, w),
                 "pad": (top, h_out - nh - top, left, w_out - nw - left),
                 "pad_frac": 1.0 - (nh * nw) / float(h_out * w_out)}


# =========================================================================
# [E] 라벨 로드 & 파일 인덱싱
#     CSV 를 표준 DataFrame(id / boneage / male / path)으로 만들고,
#     실제 크롭 파일이 존재하는 행만 남깁니다.
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
    _df = _df[_df["id"].isin(_idx.keys())].reset_index(drop=True)
    _df["path"] = _df["id"].map(lambda i: str(_idx[i]))
    SPLIT_DFS[_sp] = _df
    log(f"  {_sp:<5} 라벨 {_before:>6,} | 파일 {len(_idx):>6,} | "
        f"미크롭 {_before-len(_df):>4,} -> 사용 {len(_df):>6,}")

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
# [F] 스테이지 1 - 캔버스 캐시
#     모든 이미지를 한 번만 fit_canvas 로 변환해 PNG 로 저장합니다.
#     학습 중에는 이 캐시만 읽으므로 에폭마다 전처리를 반복하지 않습니다.
#     _DONE_cache.json 의 내용이 현재 설정과 같으면 통째로 스킵합니다.
# =========================================================================
PRE_INFO = {"norm_mode": NORM_MODE, "img_h": IMG_H, "img_w": IMG_W,
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
    log(f"[스테이지1] {IMG_H}x{IMG_W} 캔버스 캐시 생성 (norm={NORM_MODE})")
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
        if np.median(pf) > 0.20:
            log("  [경고] 여백이 20%를 넘습니다 -> IMG_H/IMG_W 종횡비를 재검토하세요.")
    json.dump(PRE_INFO, open(CACHE_DONE, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    log(f"[스테이지1] 완료 표식: {CACHE_DONE}")


build_cache()


# =========================================================================
# [G] QC
#     강도 정규화가 실제로 동작했는지 눈과 숫자로 확인합니다.
#     원본은 이미지마다 std 가 크게 다르고, 정규화 후에는 비슷한 값으로
#     수렴해야 정상입니다 -> 변동계수(CV) 가 뚜렷이 작아져야 합니다.
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
        raise SystemExit("[중단] QC 샘플을 하나도 읽지 못했습니다. 캐시와 원본 경로를 확인하세요.")
    log(f"[QC] 시트 저장: {p}")
    log("\n" + qc.to_string(index=False))

    cv_raw = float(qc["원본std"].std() / max(1e-6, qc["원본std"].mean()))
    cv_eff = float(qc["유효std"].std() / max(1e-6, qc["유효std"].mean()))
    log(f"[QC] std 변동계수(CV):  원본 {cv_raw:.3f}  ->  캔버스 {cv_eff:.3f}")
    if NORM_MODE == "none":
        log("[QC] NORM_MODE='none' - 두 값이 비슷해야 정상입니다.")
    elif cv_eff < cv_raw * 0.6:
        log("[QC] 정상 - 이미지 간 대비 편차가 크게 줄었습니다.")
    else:
        log("[QC] 경고 - 편차가 충분히 안 줄었습니다. norm_intensity 의 g>0 마스크가 "
            "제대로 동작하는지(배경이 정말 0인지) 확인하세요.")

    pf = qc["여백%"].str.rstrip("%").astype(float)
    log(f"[QC] 샘플 여백 중앙 {pf.median():.0f}%")
    qc.to_csv(CKPT_DIR / "qc_intensity_norm.csv", index=False, encoding="utf-8-sig")


qc_sheet()
if QC_ONLY:
    log("--qc-only: QC 시트만 만들고 종료합니다.")
    raise SystemExit(0)


# =========================================================================
# [H] Dataset · Transform
#     학습:  캐시 PNG -> PIL -> affine -> jitter -> tensor -> 정규화 -> erasing
#     평가:  캐시 PNG -> PIL -> tensor -> 정규화
# =========================================================================
_aug = [
    transforms.RandomAffine(degrees=AUG_ROT_DEG,
                            translate=(AUG_TRANSLATE, AUG_TRANSLATE),
                            scale=AUG_SCALE, shear=AUG_SHEAR, fill=PAD_VALUE),
    transforms.ColorJitter(brightness=AUG_JITTER, contrast=AUG_JITTER),
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
    """캐시된 HxW 캔버스 -> 3채널 복제 -> Transform.

    반환 (x, gender, y_norm, y_months)
      x        (3,H,W) 정규화된 텐서
      gender   (1,)    남=1 / 여=0
      y_norm   ()      z-정규화된 타깃 (손실 계산용)
      y_months ()      원 단위 개월 (MAE 계산용)
    """

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
            # 캐시본은 이미 강도 정규화가 적용된 상태 -> 재적용하면 이중 스트레치
            g, _ = fit_canvas(g, norm="none")
        x  = self.tf(np.stack([g] * 3, axis=-1))
        gd = torch.tensor([float(r["male"])], dtype=torch.float32)
        ym = torch.tensor([float(r["boneage"])], dtype=torch.float32)
        yn = (ym - AGE_MEAN) / AGE_STD
        return x, gd, yn, ym


# =========================================================================
# [I] 모델
#     ConvNeXt-Tiny 백본 -> 헤드 -> 성별 임베딩과 결합 -> 스칼라 1개
#     640x448 입력이면 백본 출력 특징맵은 768 x 20 x 14 입니다.
# =========================================================================
def make_backbone(name=None, pretrained=True, drop_path=None):
    """백본 생성. 실패하면 BACKBONE_ALT 후보로 순차 폴백."""
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

        # 실제 forward 로 특징맵 크기를 재서 헤드 입력 차원을 정합니다
        h, w = int(img_hw[0]), int(img_hw[1])
        with torch.no_grad():
            feat = self.backbone(torch.zeros(1, 3, h, w))
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
    """ARCH 딕셔너리만으로 모델을 복원합니다.
       체크포인트에 arch 가 함께 저장되므로 best.pt 하나로 자체 완결됩니다."""
    m = ConvNeXtRegressor(
        backbone_name=arch.get("BACKBONE", BACKBONE),
        img_hw=(arch["IMG_H"], arch["IMG_W"]),
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
        "HEAD_TYPE": HEAD_TYPE, "HEAD_DIM": HEAD_DIM,
        "GENDER_EMB_DIM": GENDER_EMB_DIM, "DROPOUT": DROPOUT, "DROP_PATH": DROP_PATH,
        "NORM_MODE": NORM_MODE, "RESIZE_MODE": RESIZE_MODE,
        "PAD_VALUE": PAD_VALUE, "PAD_ANCHOR": PAD_ANCHOR,
        "USE_AUG": USE_AUG, "LOSS": LOSS_TYPE, "SEED": SEED}


# =========================================================================
# [J] 학습 부품
# =========================================================================
def build_param_groups(model, lr_head, lr_backbone, wd):
    """(backbone / head) x (decay / no-decay) 4개 그룹.

    LayerNorm 가중치와 bias(ndim<=1)에 weight decay 를 주면 ConvNeXt 는
    성능이 떨어집니다. base_lr 을 그룹에 심어두고 스케줄러가 배율만 곱합니다.
    """
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
    """가중치 지수이동평균.

    옵티마이저 스텝마다  ema = decay*ema + (1-decay)*current  로 갱신합니다.
    학습 중 가중치는 미니배치마다 흔들리는데, 그 궤적의 평균은 더 안정적인
    지점에 앉습니다. 회귀 문제에서 별도 비용 없이 얻는 개선입니다.
    """

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
    """학습 루프에서 매 에폭 호출. 개월 단위 (MAE, RMSE) 반환. TTA 미적용."""
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
#
#   에폭 루프
#     ├ 미니배치마다 forward/backward, ACCUM_STEPS 마다 옵티마이저 스텝
#     │   스텝 직전에 LR = base_lr x lr_scale_at(...) 로 갱신
#     │   스텝 직후에 EMA 갱신
#     ├ 에폭 끝: 원본 가중치와 EMA 가중치를 둘 다 val 로 평가
#     ├ 더 좋은 쪽이 최고 기록을 경신했으면 그 가중치를 best.pt 로 저장
#     └ 매 에폭 last.pt 저장 (다음 실행에서 자동 재개)
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
    log(f"[스테이지2] 파라미터 {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

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
        log(f"[스테이지2] EMA 유효 창 ≈ {1/(1-EMA_DECAY)/steps_per_epoch:.1f} 에폭")

    # best_raw / best_ema 는 각 가중치의 최고 기록을 독립적으로 추적합니다.
    # best_val 은 BEST_SELECT 규칙이 가리키는 쪽의 기록으로, 조기종료 판단에 씁니다.
    start_epoch, no_improve, global_step = 1, 0, 0
    best_raw = best_ema = best_val = float("inf")
    history = {"train_mae": [], "val_mae": [], "val_rmse": [], "val_mae_ema": [],
               "lr": [], "lr_ratio": [], "sec": []}

    select = BEST_SELECT if (USE_EMA or BEST_SELECT == "raw") else "raw"
    log(f"[스테이지2] best.pt 선택 규칙 = '{select}' (학습 전 고정)")

    # 자동 재개: last.pt 가 있고 구조가 같으면 이어서, 다르면 옆으로 치우고 새로
    if LAST_CKPT.exists() and not FRESH:
        ck = torch_load(LAST_CKPT, map_location=device)
        bad = [k for k in ("IMG_H", "IMG_W", "HEAD_TYPE", "HEAD_DIM",
                           "GENDER_EMB_DIM", "BACKBONE", "NORM_MODE")
               if ck.get("arch", {}).get(k) != ARCH.get(k)]
        if bad:
            alt = LAST_CKPT.with_name(f"last_incompatible_{RUN_TS}.pt")
            LAST_CKPT.rename(alt)
            log(f"[재개] 구조 불일치 {bad} -> 기존 체크포인트를 {alt.name} 으로 옮기고 새로 시작")
        else:
            model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            scaler.load_state_dict(ck["scaler"])
            if ema is not None and ck.get("ema") is not None:
                ema.module.load_state_dict(ck["ema"])
            start_epoch = ck["epoch"] + 1
            best_raw    = ck.get("best_raw", float("inf"))
            best_ema    = ck.get("best_ema", float("inf"))
            best_val    = ck.get("best_val", float("inf"))
            no_improve  = ck.get("no_improve", 0)
            global_step = ck.get("global_step", 0)
            history     = ck.get("history", history)
            history.setdefault("lr_ratio", [])
            log(f"[재개] epoch {start_epoch} 부터 · best val MAE {best_val:.2f} "
                f"(raw {best_raw:.2f} / ema {best_ema:.2f})")
    else:
        log("[스테이지2] 새 학습 시작" + (" (--fresh)" if FRESH else ""))

    if start_epoch > EPOCHS:
        log(f"[스테이지2] 이미 {EPOCHS} 에폭 완료 - 학습 스킵")
        return

    criterion = make_criterion()

    def save_ckpt(path, epoch, val_mae, which, weights=None):
        """weights 를 주면 그 가중치를 'model' 자리에 넣어 저장합니다.
           덕분에 평가·추론 코드는 raw/EMA 구분 없이 ck["model"] 만 읽으면 됩니다."""
        torch.save({"model": (model.state_dict() if weights is None else weights),
                    "ema": (ema.module.state_dict() if ema is not None else None),
                    "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_val": best_val,
                    "best_raw": best_raw, "best_ema": best_ema,
                    "no_improve": no_improve,
                    "global_step": global_step, "history": history,
                    "arch": ARCH, "age_mean": AGE_MEAN, "age_std": AGE_STD,
                    "val_mae": val_mae, "best_from": which,
                    "best_select": select, "run_ts": RUN_TS}, path)

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
            dt = time.time() - t0

            cur_lr = optimizer.param_groups[-1]["lr"]
            lr_ratio = cur_lr / max(1e-12, optimizer.param_groups[-1]["base_lr"])

            history["train_mae"].append(tr_mae)
            history["val_mae"].append(va_mae)
            history["val_rmse"].append(va_rmse)
            history["val_mae_ema"].append(None if ema is None else va_ema)
            history["lr"].append(cur_lr)
            history["lr_ratio"].append(round(lr_ratio, 4))
            history["sec"].append(round(dt, 1))

            msg = (f"[{epoch:02d}/{EPOCHS}] train {tr_mae:.2f} | val {va_mae:.2f} "
                   f"(rmse {va_rmse:.2f})")
            if ema is not None:
                msg += f" | val(ema) {va_ema:.2f}"
            msg += f" | lr {lr_ratio:.0%} | {dt/60:.1f}분"

            # raw / EMA 각각의 최고 기록을 독립적으로 갱신.
            # 서로 비교하지 않으므로 두 파일 모두 편향 없는 자기 기준 최고입니다.
            improved = []
            if va_mae < best_raw - MIN_DELTA:
                best_raw = va_mae
                save_ckpt(BEST_RAW_CKPT, epoch, va_mae, "raw", model.state_dict())
                improved.append("raw")
            if ema is not None and va_ema < best_ema - MIN_DELTA:
                best_ema = va_ema
                save_ckpt(BEST_EMA_CKPT, epoch, va_ema, "ema",
                          ema.module.state_dict())
                improved.append("ema")

            # best.pt 는 학습 전에 고정한 규칙이 가리키는 쪽만 갱신합니다.
            if select == "auto":
                sel_mae, sel_which = ((va_ema, "ema") if va_ema < va_mae
                                      else (va_mae, "raw"))
            elif select == "ema":
                sel_mae, sel_which = va_ema, "ema"
            else:
                sel_mae, sel_which = va_mae, "raw"

            if sel_mae < best_val - MIN_DELTA:
                best_val = sel_mae; no_improve = 0
                w = (ema.module.state_dict() if sel_which == "ema"
                     else model.state_dict())
                save_ckpt(BEST_CKPT, epoch, sel_mae, sel_which, w)
                msg += f"  ** best.pt({sel_which}) {sel_mae:.2f} **"
            else:
                no_improve += 1
                if EARLY_STOP_PATIENCE > 0:
                    msg += f"  (개선 없음 {no_improve}/{EARLY_STOP_PATIENCE})"
            if improved:
                msg += f"  [{'/'.join(improved)} 갱신]"
            log(msg)

            save_ckpt(LAST_CKPT, epoch, sel_mae, sel_which)
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

    log(f"[스테이지2] 완료 · best val MAE = {best_val:.2f} 개월 (규칙 '{select}')")
    log(f"[스테이지2] 참고 · raw 최고 {best_raw:.2f} / ema 최고 {best_ema:.2f} "
        f"-> 각각 best_raw.pt / best_ema.pt")
    if history["lr_ratio"]:
        r = history["lr_ratio"][-1]
        log(f"[스테이지2] 종료 시점 LR = 피크의 {r:.1%}"
            + ("  [정상 - 코사인 완주]" if r < 0.08
               else "  [경고 - 코사인 미완주. EPOCHS 를 실제 학습량에 맞추세요]"))


if not EVAL_ONLY:
    train_phase()
else:
    log("--eval-only: 학습을 건너뜁니다.")


# =========================================================================
# [L] 스테이지 3 - 평가
#
#   1) best.pt 로드 -> eval_model
#   2) val 을 세 가지로 예측: 단일 / +TTA / +캘리브레이션
#      캘리브레이션은 val 에서 실제로 MAE 가 줄 때만 채택합니다.
#   3) 채택된 설정 그대로 train / test 예측
#   4) 지표 · 부트스트랩 CI · 연령대별 표 · 그림 저장
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
log(f"  헤드     {_ck['arch'].get('HEAD_TYPE')} | 화소처리 {_ck['arch'].get('NORM_MODE')}")
log(f"  가중치   {_ck.get('best_from')} (선택 규칙 '{_ck.get('best_select', BEST_SELECT)}')")
log(f"  정규화   {EM_MEAN:.1f} ± {EM_STD:.1f} | 입력 "
    f"{_ck['arch']['IMG_H']}x{_ck['arch']['IMG_W']}")
log(f"  best val MAE {_ck.get('best_val', float('nan')):.2f} @ epoch {_ck.get('epoch')}")


@torch.no_grad()
def predict_split(model, df, split, bs=16, angles=(0,)):
    """개월 단위 (preds, trues, ids) 반환.

    angles 가 여러 개면 각 각도로 회전한 입력의 예측을 평균합니다(TTA).
    좌우 반전은 좌수 데이터라 넣지 않습니다.
    """
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
    """회귀 축소(regression to the mean) 보정.

    회귀 모델은 어린 쪽을 과대, 나이 든 쪽을 과소 예측하는 경향이 있습니다.
    val 에서 pred ≈ a·true + b 를 적합하고 역함수 (pred-b)/a 를 적용하면
    눌린 기울기를 펴 줍니다. 적합은 반드시 val 에서만 합니다.
    """
    a, b = np.polyfit(true_val, pred_val, 1)
    log(f"  [calib] slope a={a:.4f} intercept b={b:+.2f}"
        f"  ({'축소 존재' if a < 0.98 else '축소 미미'})")
    return (lambda p: (np.asarray(p, dtype=np.float64) - b) / a), float(a), float(b)


results, lines = {}, []
lines.append("=" * 68)
lines.append(f" ConvNeXt-Tiny 단일 회귀 | {IMG_H}x{IMG_W} | 강도정규화 {NORM_MODE}")
lines.append(f" 백본 {_ck['arch'].get('BACKBONE_RESOLVED', BACKBONE)} | "
             f"헤드 {_ck['arch'].get('HEAD_TYPE')}")
lines.append(f" 실행 {RUN_TS} | best epoch {_ck.get('epoch')} ({_ck.get('best_from')})")
lines.append(f" TTA {list(TTA_ANGLES) if len(TTA_ANGLES) > 1 else '끔'}")
lines.append("=" * 68)

# 1) val - TTA 효과 측정 및 캘리브레이션 적합
log("  [val] 예측 중...")
v_plain, v_true, _ = predict_split(eval_model, val_df, "val", angles=(0,))
results["val_plain"] = metrics(v_plain, v_true)
lines.append(f" {'VAL(단일)':<14} N={results['val_plain']['N']:>6,}  "
             f"MAE {results['val_plain']['mae']:5.2f}  "
             f"RMSE {results['val_plain']['rmse']:5.2f}  "
             f"bias {results['val_plain']['bias']:+5.2f}")

v_pred = v_plain
if len(TTA_ANGLES) > 1:
    v_pred, v_true, _ = predict_split(eval_model, val_df, "val", angles=TTA_ANGLES)
    results["val_tta"] = metrics(v_pred, v_true)
    lines.append(f" {'VAL(+TTA)':<14} N={results['val_tta']['N']:>6,}  "
                 f"MAE {results['val_tta']['mae']:5.2f}  "
                 f"RMSE {results['val_tta']['rmse']:5.2f}  "
                 f"bias {results['val_tta']['bias']:+5.2f}")

CAL_A, CAL_B, CALIB_ON = 1.0, 0.0, False
calib_fn = (lambda p: np.asarray(p, dtype=np.float64))
if USE_CALIB:
    calib_fn, CAL_A, CAL_B = fit_calibration(v_pred, v_true)
    v_cal = calib_fn(v_pred)
    results["val_calib"] = metrics(v_cal, v_true)
    base_mae = (results.get("val_tta") or results["val_plain"])["mae"]
    CALIB_ON = results["val_calib"]["mae"] < base_mae
    lines.append(f" {'VAL(+calib)':<14} N={results['val_calib']['N']:>6,}  "
                 f"MAE {results['val_calib']['mae']:5.2f}  "
                 f"RMSE {results['val_calib']['rmse']:5.2f}  "
                 f"bias {results['val_calib']['bias']:+5.2f}   "
                 f"-> {'채택' if CALIB_ON else '미채택'}")
    log(f"  [calib] 채택 여부 {CALIB_ON} "
        f"(val {base_mae:.2f} -> {results['val_calib']['mae']:.2f})")


def apply_calib(p):
    return calib_fn(p) if CALIB_ON else np.asarray(p, dtype=np.float64)


v_pred = apply_calib(v_pred)
results["val"] = metrics(v_pred, v_true)

# 2) train - 과적합 정도 확인용 (TTA 없이 단일 패스)
log("  [train] 예측 중...")
_tr_p, _tr_t, _ = predict_split(eval_model, train_df, "train", angles=(0,))
results["train"] = metrics(apply_calib(_tr_p), _tr_t)
lines.append(f" {'TRAIN(단일)':<14} N={results['train']['N']:>6,}  "
             f"MAE {results['train']['mae']:5.2f}  "
             f"RMSE {results['train']['rmse']:5.2f}  "
             f"bias {results['train']['bias']:+5.2f}")
lines.append("-" * 68)

# 3) test - 최종 성적
log("  [test] 예측 중...")
t_raw, t_true, t_ids = predict_split(eval_model, test_df, "test", angles=TTA_ANGLES)
t_pred = apply_calib(t_raw)

results["test"] = metrics(t_pred, t_true)
lo, hi = bootstrap_ci(t_pred, t_true)
results["test"]["ci95"] = [round(lo, 2), round(hi, 2)]
lines.append(f" {'TEST':<14} N={results['test']['N']:>6,}  MAE {results['test']['mae']:5.2f}  "
             f"RMSE {results['test']['rmse']:5.2f}  bias {results['test']['bias']:+5.2f}  "
             f"CI95 [{lo:.2f}, {hi:.2f}]")
if CALIB_ON:
    results["test_uncalibrated"] = metrics(t_raw, t_true)
    lines.append(f" {'  calib 미적용':<14} MAE {results['test_uncalibrated']['mae']:5.2f}  "
                 f"bias {results['test_uncalibrated']['bias']:+5.2f}")

lines.append("-" * 68)

# 4) 가중치 감사 - raw 와 EMA 를 같은 test 로 각각 평가
#    EMA 가 val 에서만 좋아 보이는 것인지, test 에서도 실제로 좋은지 확인합니다.
#    비교를 오염시키지 않으려고 캘리브레이션은 적용하지 않습니다.
lines.append(" [가중치 감사 · test, TTA 적용 / calib 미적용]")
audit = {}
for _tag, _ckp in (("raw", BEST_RAW_CKPT), ("ema", BEST_EMA_CKPT)):
    if not _ckp.exists():
        continue
    _c = torch_load(_ckp, map_location=device)
    _m = build_model(_c["arch"], pretrained=False, verbose=False)
    _m.load_state_dict(_c["model"]); _m.eval()
    _p, _t, _ = predict_split(_m, test_df, "test", angles=TTA_ANGLES)
    _mm = metrics(_p, _t)
    _lo, _hi = bootstrap_ci(_p, _t)
    _mm["ci95"] = [round(_lo, 2), round(_hi, 2)]
    _mm["val_mae"] = float(_c.get("val_mae", float("nan")))
    _mm["epoch"] = _c.get("epoch")
    audit[_tag] = _mm
    lines.append(f"   {_tag:<4} ep{_c.get('epoch'):>3}  "
                 f"val {_mm['val_mae']:5.2f}  ->  test {_mm['mae']:5.2f}  "
                 f"CI95 [{_lo:.2f}, {_hi:.2f}]")
    del _m
    torch.cuda.empty_cache()
if "raw" in audit and "ema" in audit:
    _d = audit["ema"]["mae"] - audit["raw"]["mae"]
    lines.append(f"   EMA - raw = {_d:+.2f} 개월 (test 기준)")
    lines.append("   * 음수면 EMA 이득이 test 로 이어진 것, 양수면 val 에서만 좋았던 것.")
results["weight_audit"] = audit

lines.append("-" * 68)
_agrp = age_group_table(t_pred, t_true)
for _, r in _agrp.iterrows():
    lines.append(f"   {r['구간']:<7} N={int(r['N']):>4}  MAE {r['MAE']:5.2f}  bias {r['bias']:+5.2f}")
results["age_groups_test"] = _agrp.to_dict("records")
lines.append("=" * 68)


# ── 그림 ─────────────────────────────────────────────────────────────
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
        ax[0].axhline(4.10, ls="--", c="green", label="Zhang 2026 4.10")
        ax[0].axhline(4.30, ls=":", c="orange", label="Chen 2020 4.30")
        ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("MAE (months)")
        ax[0].set_title(f"ConvNeXt-Tiny single ({IMG_H}x{IMG_W}, {NORM_MODE})")
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
    """오차 상위 케이스. 크롭 실패나 라벨 오류 후보를 눈으로 확인합니다."""
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
        plt.suptitle("Top errors", y=1.0)
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
    """모델이 성장판·수근골을 보는지, 아니면 여백·마커로 새는지 확인합니다."""
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

json.dump({"used": bool(CALIB_ON), "a": CAL_A, "b": CAL_B,
           "formula": "corrected = (pred - b) / a", "fitted_on": "validation",
           "tta_angles": list(TTA_ANGLES), "img_h": IMG_H, "img_w": IMG_W,
           "norm_mode": NORM_MODE, "run_ts": RUN_TS},
          open(CALIB_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
log(f"  캘리브레이션 계수 저장: {CALIB_JSON}")

results["arch"] = _ck["arch"]
results["run_ts"] = RUN_TS
results["env_file"] = str(_env_path)
results["best_epoch"] = _ck.get("epoch")
results["best_from"] = _ck.get("best_from")
results["best_select"] = _ck.get("best_select", BEST_SELECT)
results["tta_angles"] = list(TTA_ANGLES)
results["calib"] = {"used": bool(CALIB_ON), "a": CAL_A, "b": CAL_B}
results["when"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

json.dump(results, open(RESULTS_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
open(RESULTS_TXT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)
log(f"결과 저장: {RESULTS_TXT}")
log(f"         {RESULTS_JSON}")


# =========================================================================
# [M] 단일 이미지 추론
#     이미지 준비 설정을 체크포인트에서 읽으므로 학습과 동일한 처리가
#     보장됩니다. 입력은 YOLO 로 손이 이미 크롭된 이미지입니다.
# =========================================================================
_INFER_CACHE = {}


def predict_bone_age(image_path, is_male, ckpt_path=BEST_CKPT,
                     angles=None, calib_ab=None):
    """크롭된 X-ray 경로 + 성별(True=남) -> 골연령(개월).

    angles   : 회전 TTA 각도. None 이면 이 실행의 TTA_ANGLES 사용
    calib_ab : (a, b) 캘리브레이션 계수. None 이면 미적용
               calibration.json 에 저장된 값을 그대로 넣으면 됩니다.
    """
    key = str(ckpt_path)
    if key not in _INFER_CACHE:
        c = torch_load(ckpt_path, map_location=device)
        m = build_model(c["arch"], pretrained=False, verbose=False)
        m.load_state_dict(c["model"]); m.eval()
        _INFER_CACHE[key] = (m, c)
    m, c = _INFER_CACHE[key]
    a = c["arch"]
    angles = TTA_ANGLES if angles is None else angles
    if not angles:
        angles = (0,)

    g = imread_kr(image_path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise FileNotFoundError(image_path)
    canvas, _ = fit_canvas(g, h_out=a["IMG_H"], w_out=a["IMG_W"],
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


# 사용 예:
#   ab = (CAL_A, CAL_B) if CALIB_ON else None
#   months = predict_bone_age(HAND_CROP_DIR / "validation" / "1377.png",
#                             is_male=True, calib_ab=ab)
try:
    _r = test_df.iloc[0]
    _ab = (CAL_A, CAL_B) if CALIB_ON else None
    _m = predict_bone_age(_r["path"], bool(_r["male"]), calib_ab=_ab)
    log(f"추론 함수 확인 [{_r['id']}] 예측 {_m:.1f}개월 / 실제 {_r['boneage']:.0f}개월")
except Exception as e:
    log(f"[경고] 추론 함수 확인 실패: {e}")

log(f"추론 함수 준비 완료 · best.pt: {BEST_CKPT}")
log("=== 전체 완료 ===")
