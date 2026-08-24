# -*- coding: utf-8 -*-
# =========================================================================
# 손 X-ray 골연령 추정 - v5 + v7 앙상블  (학습 없음 · 추론과 결합만)
#
# ┌─ 무엇을 하는가 ──────────────────────────────────────────────────────┐
# │  이미 학습이 끝난 체크포인트 두 개를 불러와 같은 val/test 에 대해     │
# │  각각 예측하고, 그 예측들을 어떻게 합칠지 val 에서 정한 뒤            │
# │  test 에 한 번만 적용해 최종 성적을 냅니다.                          │
# │                                                                      │
# │  멤버 1  v5  ConvNeXt-Tiny · 스칼라 회귀(L1)      640x448             │
# │  멤버 2  v7  Xception      · LDL 기대값 회귀      640x448             │
# │                                                                      │
# │  두 모델은 백본 계열도 손실 형태도 달라서 오차가 서로 다른 곳에서     │
# │  발생합니다. 앙상블 이득은 바로 그 '오차 비상관성'에서 나오므로,     │
# │  이 스크립트는 잔차 상관계수 r 을 반드시 함께 보고합니다.            │
# │  r 이 0.9 를 넘으면 두 모델이 사실상 같은 실수를 하고 있다는 뜻이고, │
# │  그때는 앙상블로 얻을 것이 거의 없습니다.                            │
# └──────────────────────────────────────────────────────────────────────┘
#
# ▶ 결합 방식 (전부 계산해서 표로 보여주고, val 최저를 headline 으로 채택)
#     single     각 멤버 단독
#     mean       단순 평균                       (자유도 0 - 과적합 위험 없음)
#     weight     가중 평균, 가중치는 val 격자탐색 (자유도 M-1)
#     stack      y ≈ c0 + Σ c_i·p_i 최소제곱     (자유도 M+1, 캘리브레이션 포함)
#   자유도가 큰 방식일수록 val 에 과적합될 수 있습니다. val 성능이 비슷하면
#   자유도가 작은 쪽을 고르도록 TIE_MARGIN 여유를 둡니다.
#
# ▶ 실행: python ensemble_single_v8.py
#     --fg          백그라운드 분리 없이 바로 실행
#     --rebuild-cache  멤버 캐시 강제 재생성
#     --no-detach   --fg 와 동일
#
# ▶ 전제조건
#     v5 와 v7 이 각각 학습을 마쳐 best.pt 가 존재해야 합니다.
#     없으면 즉시 중단합니다 (조용히 건너뛰지 않습니다).
#
# ▶ 산출물  checkpoints_ensemble_v8/
#     results.txt|json      최종 성적표 (모든 결합 방식 + 부트스트랩 CI)
#     ensemble.json         배포용 - 채택된 방식과 계수
#     preds_val.csv         멤버별/앙상블 val 예측 (재분석용)
#     preds_test.csv        멤버별/앙상블 test 예측
#     agreement.png         멤버 간 예측 산점도 + 잔차 상관
#     scatter.png           최종 앙상블 예측 대 실제
#     worst_cases.png|csv   오차 상위 케이스
#   logs/ensemble_v8_<ts>.log
# =========================================================================
from pathlib import Path
import os, sys, time, json, subprocess, platform, itertools
from datetime import datetime

PROJECT_DIR = Path(__file__).resolve().parent
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RUN_STATE = LOG_DIR / "ensemble_v8_running.json"
_WORKER_ENV = "BONEAGE_ENSEMBLE_V8_WORKER"

FOREGROUND    = ("--fg" in sys.argv) or ("--no-detach" in sys.argv)
REBUILD_CACHE = "--rebuild-cache" in sys.argv


# =========================================================================
# [A] 런처 - 세션과 분리된 백그라운드 프로세스
# =========================================================================
def _pid_alive(pid):
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True).stdout
        return str(pid) in out
    except Exception:
        return False


def _follow(log_path, pid):
    log_path = Path(log_path)
    for _ in range(200):
        if log_path.exists():
            break
        time.sleep(0.2)
    print("=" * 64)
    print(f" 앙상블 평가가 백그라운드에서 실행 중입니다  (PID {pid})")
    print(f" 로그 파일: {log_path}")
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
        print("\n[로그 보기만 종료합니다]", flush=True)


def _spawn_detached():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"ensemble_v8_{ts}.log"
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
            _follow(st["log"], st["pid"]); sys.exit(0)
    _pid, _logp = _spawn_detached()
    _follow(_logp, _pid); sys.exit(0)


# =========================================================================
# [B] 설정
# =========================================================================
os.environ["TORCH_HOME"] = str(PROJECT_DIR / ".torch_cache")
os.environ.setdefault("HF_HOME", str(PROJECT_DIR / ".hf_cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
import torchvision.transforms.functional as TF

try:
    import timm
except ImportError:
    raise SystemExit("timm 이 없습니다.  설치:  pip install timm")


def log(*a):
    print(f"[{datetime.now():%H:%M:%S}]", *a, flush=True)


def torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _require(path, what):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"[중단] {what} 경로가 없습니다:\n       {p}")
    return p


# ── 입력 경로 (v5/v6/v7 과 동일해야 합니다) ─────────────────────────
BASE_DIR      = Path(r"G:/Project/sinra_cho")
HAND_CROP_DIR = BASE_DIR / "crop_data_yolox_s"
CSV_DIR       = BASE_DIR / "crop_data_yolox_s"

SPLIT_SUBDIR = {"train": "training", "val": "validation", "test": "test"}
SPLITS       = ("val", "test")          # 앙상블은 학습을 안 하므로 train 불필요

_require(HAND_CROP_DIR, "손 크롭 폴더")
VAL_CSV  = _require(CSV_DIR / "validation.csv", "validation.csv")
TEST_CSV = _require(CSV_DIR / "test.csv",       "test.csv")

SEED = 42

# ── 앙상블 멤버 ──────────────────────────────────────────────────────
#   ckpt 안에 arch/age_mean/age_std 가 통째로 들어 있으므로, 여기서는
#   '어느 체크포인트냐' 만 지정하면 나머지는 전부 자동 복원됩니다.
#   멤버를 추가하려면 아래 리스트에 한 줄만 더 넣으면 됩니다.
MEMBERS = [
    {"tag": "v5_convnext_scalar",
     "ckpt": BASE_DIR / "checkpoints_convnext_single_v5" / "best.pt"},
    {"tag": "v7_xception_ldl",
     "ckpt": BASE_DIR / "checkpoints_xception_single_v7_ldl" / "best.pt"},
    # 3-way 로 넓히려면 주석을 푸세요 (v6 학습이 끝난 뒤)
    # {"tag": "v6_convnext_ldl",
    #  "ckpt": BASE_DIR / "checkpoints_convnext_single_v6_ldl" / "best.pt"},
]

# ── 추론 설정 ────────────────────────────────────────────────────────
TTA_ANGLES  = (0, -4, 4)    # 멤버 전체에 동일 적용
INFER_BS    = 16

# ── 결합 설정 ────────────────────────────────────────────────────────
#   가중치 격자 간격. 0.02 면 2멤버 기준 51개 후보를 봅니다.
WEIGHT_GRID = 0.02
#   val MAE 차이가 이 이내면 '동률'로 보고 자유도가 작은 방식을 택합니다.
#   val 200~1400장에서 0.02개월 차이는 잡음이지 실력이 아닙니다.
TIE_MARGIN  = 0.03
#   방식별 자유도 (작을수록 우선). single 은 멤버 수만큼 선택지가 있으므로 1.
METHOD_DOF  = {"mean": 0, "single": 1, "weight": None, "stack": None}

BOOTSTRAP_N = 2000
NUM_WORKERS = 0             # [중요] Windows 는 반드시 0
if os.name == "nt":
    NUM_WORKERS = 0

# ── 캐시 / 출력 ──────────────────────────────────────────────────────
#   멤버마다 전처리 설정(PRE_TAG)이 다를 수 있습니다. 아래 루트를 순서대로
#   뒤져 이미 만들어진 캐시를 재사용하고, 없으면 v8 전용 루트에 만듭니다.
CACHE_SEARCH_ROOTS = [
    BASE_DIR / "cache_convnext_single_v5",
    BASE_DIR / "cache_convnext_single_v6",
    BASE_DIR / "cache_xception_single_v7",
    BASE_DIR / "cache_ensemble_v8",
]
CACHE_OWN_ROOT = BASE_DIR / "cache_ensemble_v8"
CKPT_DIR       = BASE_DIR / "checkpoints_ensemble_v8"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_OWN_ROOT.mkdir(parents=True, exist_ok=True)

RESULTS_TXT   = CKPT_DIR / "results.txt"
RESULTS_JSON  = CKPT_DIR / "results.json"
ENSEMBLE_JSON = CKPT_DIR / "ensemble.json"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()
RUN_TS  = datetime.now().strftime("%Y%m%d_%H%M%S")

print("=" * 72)
log("골연령 앙상블 평가 v8 시작 (학습 없음)")
log(f"PyTorch {torch.__version__} | timm {timm.__version__} | device {device} | AMP {USE_AMP}")
log(f"멤버 {len(MEMBERS)}개 | TTA {list(TTA_ANGLES)} | 부트스트랩 {BOOTSTRAP_N}회")
log(f"CKPT_DIR  {CKPT_DIR}")
print("=" * 72, flush=True)


# =========================================================================
# [C] I/O 헬퍼 - 한글 경로 대응
# =========================================================================
def imread_kr(path, flags=cv2.IMREAD_GRAYSCALE):
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    except Exception:
        return None


def imwrite_kr(path, img):
    path = str(path); ext = os.path.splitext(path)[1] or ".png"
    try:
        ok, buf = cv2.imencode(ext, img)
        if ok:
            buf.tofile(path)
        return ok
    except Exception:
        return False


# =========================================================================
# [D] 이미지 준비 - v5/v6/v7 과 바이트 단위로 동일해야 합니다
#     (여기가 조금이라도 다르면 캐시를 공유해도 의미가 없습니다)
# =========================================================================
def to_gray8(g):
    if g.dtype == np.uint8:
        return g
    g = g.astype(np.float32)
    lo, hi = float(g.min()), float(g.max())
    if hi - lo < 1e-6:
        return np.zeros(g.shape, np.uint8)
    return np.clip((g - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def norm_intensity(g, mode):
    if mode == "none":
        return g
    v = g[g > 0]
    if v.size < 1000:
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


def fit_canvas(g, h_out, w_out, mode, pad_val, anchor, norm):
    g = to_gray8(g)
    g = norm_intensity(g, norm)
    h, w = g.shape[:2]
    if mode == "stretch":
        interp = cv2.INTER_AREA if (h_out < h and w_out < w) else cv2.INTER_CUBIC
        return cv2.resize(g, (w_out, h_out), interpolation=interp)
    s  = min(h_out / float(h), w_out / float(w))
    nh = max(1, min(h_out, int(round(h * s))))
    nw = max(1, min(w_out, int(round(w * s))))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    r = cv2.resize(g, (nw, nh), interpolation=interp)
    top  = 0 if anchor == "topleft" else (h_out - nh) // 2
    left = 0 if anchor == "topleft" else (w_out - nw) // 2
    return cv2.copyMakeBorder(r, top, h_out - nh - top, left, w_out - nw - left,
                              cv2.BORDER_CONSTANT, value=int(pad_val))


# =========================================================================
# [E] 라벨
# =========================================================================
def load_labels(csv_path):
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
    d = HAND_CROP_DIR / SPLIT_SUBDIR[split]
    if not d.exists():
        raise SystemExit(f"[중단] {split} 이미지 폴더 없음: {d}")
    idx = {}
    for ext in ("*.png", "*.PNG", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
        for p in d.rglob(ext):
            idx.setdefault(p.stem.strip(), p)
    return idx


log("라벨 로드 중...")
SPLIT_DFS = {}
for _sp, _csv in [("val", VAL_CSV), ("test", TEST_CSV)]:
    _df  = load_labels(_csv)
    _idx = index_files(_sp)
    _before = len(_df)
    _df = _df[_df["id"].isin(_idx.keys())].reset_index(drop=True)
    _df["path"] = _df["id"].map(lambda i: str(_idx[i]))
    SPLIT_DFS[_sp] = _df
    log(f"  {_sp:<5} 라벨 {_before:>6,} -> 사용 {len(_df):>6,}")

val_df, test_df = SPLIT_DFS["val"], SPLIT_DFS["test"]
if len(val_df) == 0 or len(test_df) == 0:
    raise SystemExit("[중단] val 또는 test 사용 가능 이미지가 0장입니다.")


# =========================================================================
# [F] 모델 복원
#
#   v5·v6·v7 의 모델 클래스는 모듈 이름이 완전히 같습니다
#   (backbone / norm / proj / conv / pool / gender / fc).
#   그래서 한 클래스로 세 체크포인트를 전부 복원할 수 있습니다.
#   백본이 ConvNeXt 인지 Xception 인지는 arch["BACKBONE"] 이 결정합니다.
# =========================================================================
BACKBONE_DENY = ("convnextv2",)


def resolve_norm_stats(name, fallback=(IMAGENET_MEAN, IMAGENET_STD)):
    """백본 이름 -> 사전학습 시 쓴 (mean, std).

    ConvNeXt 계열은 ImageNet 상수, Xception 계열은 0.5/0.5 입니다.
    멤버마다 다르므로 절대 하드코딩하면 안 됩니다.
    """
    try:
        cfg = timm.get_pretrained_cfg(name)
        m = getattr(cfg, "mean", None) or cfg["mean"]
        sd = getattr(cfg, "std", None) or cfg["std"]
        return [float(v) for v in m], [float(v) for v in sd]
    except Exception:
        return list(fallback[0]), list(fallback[1])


def make_backbone(name, drop_path=0.0):
    if any(d in name for d in BACKBONE_DENY):
        raise SystemExit(f"[중단] {name} 는 상업 사용 불가 가중치입니다.")
    kw_list = ([{"drop_path_rate": drop_path}] if drop_path else []) + [{}]
    last_err = None
    for kw in kw_list:
        try:
            # 평가 전용이므로 pretrained=False (어차피 체크포인트로 덮어씁니다)
            return timm.create_model(name, pretrained=False, num_classes=0,
                                     global_pool="", **kw)
        except Exception as e:
            last_err = e
    raise SystemExit(f"[중단] 백본 '{name}' 생성 실패: {last_err}")


class SingleRegressor(nn.Module):
    """v5/v6/v7 공통 구조. arch 만으로 완전히 복원됩니다."""

    def __init__(self, arch, verbose=True):
        super().__init__()
        bb_name = arch.get("BACKBONE_RESOLVED") or arch["BACKBONE"]
        h, w = int(arch["IMG_H"]), int(arch["IMG_W"])
        self.head_type  = arch.get("HEAD_TYPE", "gap")
        head_dim        = int(arch.get("HEAD_DIM", 512))
        gender_dim      = int(arch.get("GENDER_EMB_DIM", 32))
        dropout         = float(arch.get("DROPOUT", 0.10))
        self.use_ldl    = bool(arch.get("USE_LDL", False))
        self.n_bins     = int(arch.get("AGE_BINS", 240))

        self.backbone = make_backbone(bb_name, 0.0)   # 추론 시 drop_path 무의미
        self.backbone_name = bb_name
        with torch.no_grad():
            feat = self.backbone(torch.zeros(1, 3, h, w))
        C, H, W = int(feat.shape[1]), int(feat.shape[2]), int(feat.shape[3])
        if verbose:
            log(f"    특징맵 {C}x{H}x{W} · head={self.head_type} · "
                f"출력={'LDL ' + str(self.n_bins) if self.use_ldl else '스칼라 1'}")

        if self.head_type == "gap":
            self.norm = nn.LayerNorm(C)
            self.proj = nn.Sequential(nn.Linear(C, head_dim), nn.GELU(), nn.Dropout(dropout))
            img_out = head_dim
        elif self.head_type == "paper":
            self.conv = nn.Conv2d(C, 256, 3, padding=1)
            self.pool = nn.MaxPool2d(3, 3)
            self.drop = nn.Dropout(dropout)
            img_out = 256 * (H // 3) * (W // 3)
        else:
            raise SystemExit(f"[중단] 알 수 없는 HEAD_TYPE: {self.head_type}")

        n_out = self.n_bins if self.use_ldl else 1
        self.gender = nn.Sequential(nn.Linear(1, gender_dim), nn.GELU())
        self.fc = nn.Sequential(nn.Linear(img_out + gender_dim, 128), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(128, n_out))

    def forward(self, x, g):
        f = self.backbone(x)
        if self.head_type == "gap":
            z = F.adaptive_avg_pool2d(f, 1).flatten(1)
            z = self.proj(self.norm(z))
        else:
            z = self.drop(torch.flatten(self.pool(F.relu(self.conv(f))), 1))
        e = self.gender(g)
        out = self.fc(torch.cat([z, e], dim=1))
        return out if self.use_ldl else out.squeeze(1)


_KBINS = {}


def kbins(n, dev):
    key = (int(n), str(dev))
    if key not in _KBINS:
        _KBINS[key] = torch.arange(1, int(n) + 1, dtype=torch.float32, device=dev)
    return _KBINS[key]


def out_to_months(out, mean, std):
    """LDL 로짓이면 기대값, 스칼라면 z-역정규화. 개월 단위 반환."""
    out = out.float()
    if out.ndim == 2 and out.size(1) > 1:
        p = torch.softmax(out, dim=1)
        return (p * kbins(out.size(1), out.device)).sum(1)
    if out.ndim == 2:
        out = out.squeeze(1)
    return out * float(std) + float(mean)


# =========================================================================
# [G] 멤버 로드 + 멤버별 캐시 확보
# =========================================================================
PREP_KEYS = ("img_h", "img_w", "norm_mode", "resize_mode", "pad_value", "pad_anchor")


def prep_of(arch):
    return {"img_h": int(arch["IMG_H"]), "img_w": int(arch["IMG_W"]),
            "norm_mode": arch.get("NORM_MODE", "none"),
            "resize_mode": arch.get("RESIZE_MODE", "letterbox"),
            "pad_value": int(arch.get("PAD_VALUE", 0)),
            "pad_anchor": arch.get("PAD_ANCHOR", "center")}


def pre_tag_of(p):
    return (f"raw{p['img_h']}x{p['img_w']}_{p['resize_mode']}"
            f"_pad{p['pad_value']}_{p['pad_anchor']}_n{p['norm_mode']}")


def find_or_build_cache(prep, tag):
    """PRE_TAG 에 해당하는 캔버스 캐시를 찾고, 없거나 모자라면 채웁니다.

    기존 v5/v6/v7 캐시를 재사용하되 '완료 표식만 믿지' 않습니다.
    필요한 id 가 실제로 파일로 존재하는지 전부 확인하고, 빠진 것만 만듭니다.
    (표식은 있는데 파일이 지워진 상황에서 조용히 실패하는 것을 막습니다)
    """
    pt = pre_tag_of(prep)
    chosen = None
    if not REBUILD_CACHE:
        for root in CACHE_SEARCH_ROOTS:
            d = root / pt
            if d.exists():
                chosen = d
                log(f"  [캐시] '{tag}' 재사용 후보: {d}")
                break
    if chosen is None:
        chosen = CACHE_OWN_ROOT / pt
        log(f"  [캐시] '{tag}' 새로 생성: {chosen}")
    for sp in SPLITS:
        (chosen / sp).mkdir(parents=True, exist_ok=True)

    made = total = 0
    for sp in SPLITS:
        df = SPLIT_DFS[sp]
        t0 = time.time()
        for _, r in df.iterrows():
            total += 1
            dst = chosen / sp / f"{r['id']}.png"
            if dst.exists() and not REBUILD_CACHE:
                continue
            g = imread_kr(r["path"], cv2.IMREAD_GRAYSCALE)
            if g is None:
                raise SystemExit(f"[중단] 원본 읽기 실패: {r['path']}")
            canvas = fit_canvas(g, prep["img_h"], prep["img_w"],
                                prep["resize_mode"], prep["pad_value"],
                                prep["pad_anchor"], prep["norm_mode"])
            if not imwrite_kr(dst, canvas):
                raise SystemExit(f"[중단] 캐시 쓰기 실패: {dst}")
            made += 1
        if made:
            log(f"    [{sp}] 보충 생성 누적 {made:,} ({time.time()-t0:.0f}s)")
    log(f"  [캐시] '{tag}' 준비 완료 · 필요 {total:,}장 중 {made:,}장 생성")
    return chosen


MEM = []
log("-" * 72)
for spec in MEMBERS:
    ck_path = Path(spec["ckpt"])
    if not ck_path.exists():
        raise SystemExit(
            f"[중단] 멤버 '{spec['tag']}' 의 체크포인트가 없습니다:\n"
            f"       {ck_path}\n"
            f"       해당 버전 스크립트를 먼저 끝까지 학습시키세요.")
    log(f"[멤버] {spec['tag']}")
    ck   = torch_load(ck_path, map_location=device)
    arch = ck["arch"]
    model = SingleRegressor(arch)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if missing or unexpected:
        raise SystemExit(f"[중단] '{spec['tag']}' 가중치 불일치\n"
                         f"       missing={list(missing)[:5]}\n"
                         f"       unexpected={list(unexpected)[:5]}")
    model.eval().to(device).to(memory_format=torch.channels_last)

    prep = prep_of(arch)
    # 정규화 상수: arch 에 있으면 그것, 없으면 백본 pretrained_cfg 에서 조회
    if arch.get("NORM_MEAN") and arch.get("NORM_STD"):
        nmean, nstd = list(arch["NORM_MEAN"]), list(arch["NORM_STD"])
        src = "checkpoint"
    else:
        nmean, nstd = resolve_norm_stats(model.backbone_name)
        src = "timm pretrained_cfg"
    pad_norm = [-m / s for m, s in zip(nmean, nstd)]

    log(f"    백본 {model.backbone_name} | 입력 {prep['img_h']}x{prep['img_w']} "
        f"| 화소처리 {prep['norm_mode']}")
    log(f"    정규화 mean {[round(v,3) for v in nmean]} std {[round(v,3) for v in nstd]} "
        f"({src})")
    log(f"    best epoch {ck.get('epoch')} ({ck.get('best_from')}) "
        f"| 기록된 val MAE {ck.get('val_mae', float('nan')):.2f}")

    cache_dir = find_or_build_cache(prep, spec["tag"])

    MEM.append({
        "tag": spec["tag"], "ckpt": str(ck_path), "model": model, "arch": arch,
        "prep": prep, "cache": cache_dir, "pad_norm": pad_norm,
        "tf": transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                  transforms.Normalize(nmean, nstd)]),
        "age_mean": float(ck.get("age_mean", 0.0)),
        "age_std":  float(ck.get("age_std", 1.0)),
        "use_ldl": bool(arch.get("USE_LDL", False)),
        "ck_val_mae": float(ck.get("val_mae", float("nan"))),
        "epoch": ck.get("epoch"), "best_from": ck.get("best_from"),
    })
    log("-" * 72)

M = len(MEM)
if M < 2:
    raise SystemExit("[중단] 앙상블에는 멤버가 2개 이상 필요합니다.")


# =========================================================================
# [H] 멤버별 예측
# =========================================================================
class CanvasDataset(Dataset):
    def __init__(self, df, cache_dir, split, tf, hw):
        self.df = df.reset_index(drop=True)
        self.dir = Path(cache_dir) / split
        self.tf, self.hw = tf, hw

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        g = imread_kr(self.dir / f"{r['id']}.png", cv2.IMREAD_GRAYSCALE)
        if g is None:
            raise FileNotFoundError(f"캐시 없음: {self.dir/(str(r['id'])+'.png')}")
        if g.shape[:2] != tuple(self.hw):
            raise SystemExit(f"[중단] 캐시 크기 불일치 {g.shape[:2]} != {self.hw} "
                             f"({self.dir}) - --rebuild-cache 로 다시 만드세요.")
        x  = self.tf(np.stack([g] * 3, axis=-1))
        gd = torch.tensor([float(r["male"])], dtype=torch.float32)
        ym = torch.tensor([float(r["boneage"])], dtype=torch.float32)
        return x, gd, ym


@torch.no_grad()
def predict_member(m, df, split):
    """멤버 하나의 개월 단위 예측. 회전 TTA 는 '개월 환산 후' 평균합니다.

    LDL 은 softmax 가 비선형이라 로짓 평균이 성립하지 않고,
    스칼라는 환산이 아핀이라 어느 순서든 결과가 같습니다.
    두 경우를 한 규칙으로 처리하려면 개월 환산 후 평균이 유일하게 안전합니다.
    """
    loader = DataLoader(
        CanvasDataset(df, m["cache"], split, m["tf"],
                      (m["prep"]["img_h"], m["prep"]["img_w"])),
        batch_size=INFER_BS, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    P, T = [], []
    for x, g, ym in loader:
        x = x.to(device).to(memory_format=torch.channels_last); g = g.to(device)
        acc = 0.0
        for a in TTA_ANGLES:
            xa = x if a == 0 else TF.rotate(
                x, a, fill=m["pad_norm"], interpolation=TF.InterpolationMode.BILINEAR)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                out = m["model"](xa, g)
            acc = acc + out_to_months(out, m["age_mean"], m["age_std"])
        P.append((acc / len(TTA_ANGLES)).cpu())
        T.append(ym.squeeze(1))
    return torch.cat(P).numpy().astype(np.float64), torch.cat(T).numpy().astype(np.float64)


log("[예측] 멤버별 val / test 추론")
PV, PT = [], []
y_val = y_test = None
for m in MEM:
    t0 = time.time()
    pv, yv = predict_member(m, val_df, "val")
    pt, yt = predict_member(m, test_df, "test")
    PV.append(pv); PT.append(pt)
    if y_val is None:
        y_val, y_test = yv, yt
    else:
        # 같은 df 를 같은 순서로 돌았으므로 정답 벡터가 완전히 같아야 합니다.
        assert np.allclose(y_val, yv) and np.allclose(y_test, yt), \
            "멤버 간 정답 정렬이 어긋났습니다"
    log(f"  {m['tag']:<22} val MAE {np.abs(pv-yv).mean():5.2f} | "
        f"test MAE {np.abs(pt-yt).mean():5.2f} | {time.time()-t0:.0f}s")
    m["model"].to("cpu")
    torch.cuda.empty_cache()

PV = np.stack(PV)      # (M, N_val)
PT = np.stack(PT)      # (M, N_test)


# =========================================================================
# [I] 결합
# =========================================================================
def mae(p, y):
    return float(np.abs(p - y).mean())


def metrics(p, y):
    return {"N": int(len(y)), "mae": mae(p, y),
            "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
            "bias": float(np.mean(p - y))}


def bootstrap_ci(p, y, n_boot=BOOTSTRAP_N, alpha=0.05, seed=SEED):
    rng = np.random.default_rng(seed); n = len(y)
    b = np.empty(n_boot)
    for i in range(n_boot):
        j = rng.integers(0, n, n)
        b[i] = np.abs(p[j] - y[j]).mean()
    return float(np.percentile(b, 100 * alpha / 2)), \
           float(np.percentile(b, 100 * (1 - alpha / 2)))


def paired_bootstrap(pa, pb, y, n_boot=BOOTSTRAP_N, seed=SEED + 1):
    """MAE(a) - MAE(b) 의 신뢰구간. 같은 표본을 쓰므로 '짝지은' 부트스트랩입니다.

    test 200장에서 MAE 차이 0.2개월은 대개 잡음입니다. 이 구간이 0을 포함하면
    '앙상블이 더 좋다'고 말하면 안 됩니다.
    """
    rng = np.random.default_rng(seed); n = len(y)
    d = np.empty(n_boot)
    for i in range(n_boot):
        j = rng.integers(0, n, n)
        d[i] = np.abs(pa[j] - y[j]).mean() - np.abs(pb[j] - y[j]).mean()
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(np.mean(d)), float(lo), float(hi)


def simplex_grid(m, step):
    """합이 1인 비음수 가중치 격자. 멤버 수가 커지면 조합이 폭증하므로
       4개를 넘으면 격자를 성기게 잡습니다."""
    n = int(round(1.0 / step))
    if m > 4:
        n = min(n, 10)
    for c in itertools.combinations_with_replacement(range(m), n):
        w = np.zeros(m)
        for i in c:
            w[i] += 1.0
        yield w / n


# ── 후보 생성: 각 방식마다 (이름, val 예측, test 예측, 자유도, 계수) ──
cands = []

for i, m in enumerate(MEM):
    cands.append({"name": f"single:{m['tag']}", "kind": "single", "dof": 1,
                  "val": PV[i], "test": PT[i],
                  "params": {"member": m["tag"]}})

w_eq = np.full(M, 1.0 / M)
cands.append({"name": "mean", "kind": "mean", "dof": 0,
              "val": w_eq @ PV, "test": w_eq @ PT,
              "params": {"weights": [round(v, 4) for v in w_eq]}})

best_w, best_w_mae = None, float("inf")
for w in simplex_grid(M, WEIGHT_GRID):
    s = mae(w @ PV, y_val)
    if s < best_w_mae:
        best_w_mae, best_w = s, w
cands.append({"name": "weight", "kind": "weight", "dof": M - 1,
              "val": best_w @ PV, "test": best_w @ PT,
              "params": {"weights": [round(float(v), 4) for v in best_w]}})

#  stack: y ≈ c0 + Σ c_i·p_i  (val 최소제곱). 멤버별 캘리브레이션까지 흡수합니다.
A_val  = np.column_stack([np.ones(PV.shape[1])] + [PV[i] for i in range(M)])
A_test = np.column_stack([np.ones(PT.shape[1])] + [PT[i] for i in range(M)])
coef, *_ = np.linalg.lstsq(A_val, y_val, rcond=None)
cands.append({"name": "stack", "kind": "stack", "dof": M + 1,
              "val": A_val @ coef, "test": A_test @ coef,
              "params": {"intercept": float(coef[0]),
                         "coef": [float(v) for v in coef[1:]]}})

# ── val 최저 + 동률이면 자유도 작은 쪽 ───────────────────────────────
for c in cands:
    c["val_mae"] = mae(c["val"], y_val)
lo_mae = min(c["val_mae"] for c in cands)
pool = [c for c in cands if c["val_mae"] <= lo_mae + TIE_MARGIN]
chosen = sorted(pool, key=lambda c: (c["dof"], c["val_mae"]))[0]
log(f"[선택] '{chosen['name']}' (val MAE {chosen['val_mae']:.3f}, 자유도 {chosen['dof']}) "
    f"· 동률권 {len(pool)}개 중")


# =========================================================================
# [J] 결과 정리
# =========================================================================
results, lines = {}, []
lines.append("=" * 78)
lines.append(f" 골연령 앙상블 v8 | 멤버 {M}개 | TTA {list(TTA_ANGLES)} | {RUN_TS}")
for i, m in enumerate(MEM):
    lines.append(f"   [{i}] {m['tag']:<22} {m['arch'].get('BACKBONE_RESOLVED', m['arch']['BACKBONE'])}"
                 f"  {'LDL' if m['use_ldl'] else 'scalar'}"
                 f"  ep{m['epoch']}({m['best_from']})")
lines.append("=" * 78)

lines.append(" 방식별 성능  (val 은 선택에 쓰였으므로 낙관 편향이 있습니다)")
lines.append(f" {'방식':<26} {'dof':>3} {'val MAE':>8} {'test MAE':>9} {'test CI95':>16} {'bias':>7}")
lines.append("-" * 78)
table = []
for c in sorted(cands, key=lambda x: x["val_mae"]):
    tm = metrics(c["test"], y_test)
    lo, hi = bootstrap_ci(c["test"], y_test)
    tm["ci95"] = [round(lo, 2), round(hi, 2)]
    tm["val_mae"] = c["val_mae"]
    tm["dof"] = c["dof"]
    tm["params"] = c["params"]
    table.append({"name": c["name"], **tm})
    mark = "  <= 채택" if c["name"] == chosen["name"] else ""
    lines.append(f" {c['name']:<26} {c['dof']:>3} {c['val_mae']:>8.2f} {tm['mae']:>9.2f} "
                 f"  [{lo:5.2f}, {hi:5.2f}] {tm['bias']:>+7.2f}{mark}")
results["methods"] = table

# ── 멤버 간 오차 상관 - 앙상블 이득의 원천 ───────────────────────────
lines.append("-" * 78)
lines.append(" 멤버 잔차 상관 (test) - 낮을수록 앙상블 이득이 큽니다")
res_t = PT - y_test[None, :]
corr = np.corrcoef(res_t)
for i in range(M):
    for j in range(i + 1, M):
        r = float(corr[i, j])
        judge = ("거의 동일 - 이득 기대 어려움" if r > 0.90 else
                 "상당히 유사" if r > 0.75 else
                 "적당히 독립 - 앙상블 유리" if r > 0.5 else
                 "매우 독립 - 앙상블 크게 유리")
        lines.append(f"   r({MEM[i]['tag']}, {MEM[j]['tag']}) = {r:+.3f}   {judge}")
results["residual_corr"] = corr.round(4).tolist()

# ── 채택 방식 대 최고 단일 멤버: 짝지은 부트스트랩 ───────────────────
lines.append("-" * 78)
singles = [c for c in cands if c["kind"] == "single"]
best_single = min(singles, key=lambda c: c["val_mae"])       # val 기준으로 고름
d, dlo, dhi = paired_bootstrap(chosen["test"], best_single["test"], y_test)
sig = "유의" if (dlo < 0 and dhi < 0) else ("역전" if dlo > 0 else "판단 보류")
lines.append(f" 채택({chosen['name']}) - 최고단일({best_single['name']})")
lines.append(f"   test MAE 차이 {d:+.3f} 개월  CI95 [{dlo:+.3f}, {dhi:+.3f}]  -> {sig}")
lines.append(f"   * 구간이 0을 포함하면 개선을 주장할 수 없습니다 (test N={len(y_test)}).")
results["vs_best_single"] = {"chosen": chosen["name"], "best_single": best_single["name"],
                             "delta_mae": round(d, 4),
                             "ci95": [round(dlo, 4), round(dhi, 4)], "verdict": sig}

# ── 연령대별 ─────────────────────────────────────────────────────────
lines.append("-" * 78)
lines.append(f" 연령대별 (채택 방식 · test)")
rows = []
for lo_, hi_, lab in zip([0, 48, 96, 144, 192], [48, 96, 144, 192, 10 ** 5],
                         ["0-4y", "4-8y", "8-12y", "12-16y", ">16y"]):
    msk = (y_test >= lo_) & (y_test < hi_)
    if msk.sum():
        rows.append({"구간": lab, "N": int(msk.sum()),
                     "MAE": round(mae(chosen["test"][msk], y_test[msk]), 2),
                     "bias": round(float(np.mean(chosen["test"][msk] - y_test[msk])), 2)})
        lines.append(f"   {lab:<7} N={rows[-1]['N']:>4}  MAE {rows[-1]['MAE']:5.2f}  "
                     f"bias {rows[-1]['bias']:+5.2f}")
results["age_groups_test"] = rows

lines.append("=" * 78)
lines.append(f" 최종 (채택 '{chosen['name']}')  test MAE "
             f"{mae(chosen['test'], y_test):.2f} 개월")
lines.append(f" 참고 벤치마크  Zhang 2026 = 4.10 / Chen 2020 = 4.30")
lines.append("=" * 78)


# =========================================================================
# [K] 그림 · 파일
# =========================================================================
def save_agreement():
    try:
        n_pair = M * (M - 1) // 2
        fig, axes = plt.subplots(1, max(1, n_pair * 2), figsize=(5.6 * max(1, n_pair * 2), 5))
        axes = np.atleast_1d(axes)
        k = 0
        for i in range(M):
            for j in range(i + 1, M):
                a = axes[k]; k += 1
                a.scatter(PT[i], PT[j], s=10, alpha=.5)
                lim = [0, max(PT[i].max(), PT[j].max()) + 5]
                a.plot(lim, lim, "r--"); a.set_xlim(lim); a.set_ylim(lim)
                a.set_xlabel(f"{MEM[i]['tag']} pred"); a.set_ylabel(f"{MEM[j]['tag']} pred")
                a.set_title("member predictions (test)"); a.grid(alpha=.3)
                b = axes[k]; k += 1
                b.scatter(res_t[i], res_t[j], s=10, alpha=.5)
                b.axhline(0, c="k", lw=.7); b.axvline(0, c="k", lw=.7)
                b.set_xlabel(f"{MEM[i]['tag']} residual")
                b.set_ylabel(f"{MEM[j]['tag']} residual")
                b.set_title(f"residuals · r={corr[i,j]:+.3f}"); b.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(CKPT_DIR / "agreement.png", dpi=120); plt.close()
        log(f"  멤버 일치도 저장: {CKPT_DIR/'agreement.png'}")
    except Exception as e:
        log(f"  [경고] 일치도 그림 실패: {e}")


def save_scatter():
    try:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5.6))
        for a, (p, y, name) in zip(ax, [(chosen["val"], y_val, "Validation"),
                                        (chosen["test"], y_test, "Test")]):
            a.scatter(y, p, s=9, alpha=.45)
            lim = [0, max(y.max(), p.max()) + 5]
            a.plot(lim, lim, "r--"); a.set_xlim(lim); a.set_ylim(lim)
            a.set_xlabel("True (months)"); a.set_ylabel("Pred (months)")
            a.set_title(f"{name} · ensemble '{chosen['name']}' · MAE {mae(p,y):.2f}")
            a.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(CKPT_DIR / "scatter.png", dpi=120); plt.close()
        log(f"  산점도 저장: {CKPT_DIR/'scatter.png'}")
    except Exception as e:
        log(f"  [경고] 산점도 실패: {e}")


def save_worst(k=8):
    try:
        err = np.abs(chosen["test"] - y_test)
        order = np.argsort(-err)[:k]
        ids = test_df["id"].astype(str).values
        cdir = MEM[0]["cache"] / "test"
        fig, axes = plt.subplots(2, (k + 1) // 2, figsize=(3.6 * ((k + 1) // 2), 9))
        for a, i in zip(np.atleast_1d(axes).ravel(), order):
            g = imread_kr(cdir / f"{ids[i]}.png", cv2.IMREAD_GRAYSCALE)
            if g is not None:
                a.imshow(g, cmap="gray")
            a.axis("off")
            a.set_title(f"{ids[i]}\ntrue {y_test[i]:.0f} / ens {chosen['test'][i]:.0f}\n"
                        + " ".join(f"{MEM[j]['tag'][:6]}:{PT[j][i]:.0f}" for j in range(M)),
                        fontsize=8)
        plt.suptitle("Top ensemble errors", y=1.0)
        plt.tight_layout(); plt.savefig(CKPT_DIR / "worst_cases.png", dpi=120); plt.close()
        pd.DataFrame({"id": ids[order], "true": y_test[order].round(0),
                      "ens": chosen["test"][order].round(1),
                      **{MEM[j]["tag"]: PT[j][order].round(1) for j in range(M)},
                      "err": err[order].round(1)}
                     ).to_csv(CKPT_DIR / "worst_cases.csv", index=False, encoding="utf-8-sig")
        log(f"  오차 상위 저장: {CKPT_DIR/'worst_cases.png'}")
    except Exception as e:
        log(f"  [경고] 오차 상위 실패: {e}")


def save_preds():
    for split, df, Pm, yy, ens in (("val", val_df, PV, y_val, chosen["val"]),
                                   ("test", test_df, PT, y_test, chosen["test"])):
        d = {"id": df["id"].astype(str).values, "male": df["male"].values, "true": yy}
        for j in range(M):
            d[MEM[j]["tag"]] = Pm[j]
        d["ensemble"] = ens
        pd.DataFrame(d).to_csv(CKPT_DIR / f"preds_{split}.csv", index=False,
                               encoding="utf-8-sig")
    log(f"  예측값 저장: {CKPT_DIR/'preds_val.csv'} · {CKPT_DIR/'preds_test.csv'}")


save_agreement(); save_scatter(); save_worst(); save_preds()

deploy = {
    "method": chosen["name"], "kind": chosen["kind"], "dof": chosen["dof"],
    "params": chosen["params"],
    "members": [{"tag": m["tag"], "ckpt": m["ckpt"],
                 "backbone": m["arch"].get("BACKBONE_RESOLVED", m["arch"]["BACKBONE"]),
                 "use_ldl": m["use_ldl"], "prep": m["prep"],
                 "age_mean": m["age_mean"], "age_std": m["age_std"]} for m in MEM],
    "tta_angles": list(TTA_ANGLES),
    "formula": ("y = intercept + Σ coef_i·p_i" if chosen["kind"] == "stack"
                else "y = Σ w_i·p_i" if chosen["kind"] in ("mean", "weight")
                else "y = p_selected"),
    "fitted_on": "validation", "run_ts": RUN_TS,
}
json.dump(deploy, open(ENSEMBLE_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
log(f"  배포용 계수 저장: {ENSEMBLE_JSON}")

results.update({
    "run_ts": RUN_TS, "chosen": chosen["name"], "chosen_params": chosen["params"],
    "tta_angles": list(TTA_ANGLES), "tie_margin": TIE_MARGIN,
    "members": [{"tag": m["tag"], "ckpt": m["ckpt"], "epoch": m["epoch"],
                 "best_from": m["best_from"], "ck_val_mae": m["ck_val_mae"],
                 "use_ldl": m["use_ldl"],
                 "backbone": m["arch"].get("BACKBONE_RESOLVED", m["arch"]["BACKBONE"])}
                for m in MEM],
    "test_final": metrics(chosen["test"], y_test),
    "when": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "env": {"python": sys.version.split()[0], "torch": torch.__version__,
            "timm": timm.__version__, "platform": platform.platform()},
})
json.dump(results, open(RESULTS_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
open(RESULTS_TXT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)
log(f"결과 저장: {RESULTS_TXT}")
log(f"         {RESULTS_JSON}")


# =========================================================================
# [L] 단일 이미지 앙상블 추론
# =========================================================================
def predict_bone_age_ensemble(image_path, is_male, spec_json=ENSEMBLE_JSON):
    """크롭된 X-ray 경로 + 성별(True=남) -> 앙상블 골연령(개월).

    ensemble.json 하나만 있으면 멤버 체크포인트를 찾아 그대로 재현합니다.
    전처리 파라미터도 멤버별로 json 에 들어 있어 학습과 동일하게 처리됩니다.
    """
    spec = json.load(open(spec_json, encoding="utf-8"))
    g = imread_kr(image_path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise FileNotFoundError(image_path)

    preds = []
    for mi, msp in enumerate(spec["members"]):
        m = MEM[mi]
        if m["tag"] != msp["tag"]:
            raise SystemExit("[중단] ensemble.json 의 멤버 순서가 현재 로드와 다릅니다.")
        p = msp["prep"]
        canvas = fit_canvas(g, p["img_h"], p["img_w"], p["resize_mode"],
                            p["pad_value"], p["pad_anchor"], p["norm_mode"])
        x = m["tf"](np.stack([canvas] * 3, -1)).unsqueeze(0).to(device)
        x = x.to(memory_format=torch.channels_last)
        gd = torch.tensor([[1.0 if is_male else 0.0]], dtype=torch.float32, device=device)
        m["model"].to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=USE_AMP):
            acc = 0.0
            for a in spec["tta_angles"]:
                xa = x if a == 0 else TF.rotate(
                    x, a, fill=m["pad_norm"], interpolation=TF.InterpolationMode.BILINEAR)
                acc = acc + out_to_months(m["model"](xa, gd),
                                          msp["age_mean"], msp["age_std"])
            preds.append(float((acc / len(spec["tta_angles"])).cpu().item()))
        m["model"].to("cpu")

    kind, pr = spec["kind"], spec["params"]
    if kind == "single":
        idx = [m["tag"] for m in spec["members"]].index(pr["member"])
        return preds[idx]
    if kind in ("mean", "weight"):
        return float(np.dot(pr["weights"], preds))
    if kind == "stack":
        return float(pr["intercept"] + np.dot(pr["coef"], preds))
    raise SystemExit(f"[중단] 알 수 없는 결합 방식: {kind}")


try:
    _r = test_df.iloc[0]
    _m = predict_bone_age_ensemble(_r["path"], bool(_r["male"]))
    log(f"추론 함수 확인 [{_r['id']}] 앙상블 {_m:.1f}개월 / 실제 {_r['boneage']:.0f}개월")
except Exception as e:
    log(f"[경고] 추론 함수 확인 실패: {e}")

log("=== 전체 완료 ===")
