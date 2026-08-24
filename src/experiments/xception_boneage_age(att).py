# -*- coding: utf-8 -*-
# =========================================================================
# 손 X-ray 골연령 예측 - Xception 회귀 모델 (논문 §3.4)
#   | 노트북 "골연령_Xception_최종파이프라인_모델Xception.ipynb" 의 .py 이식본
#   | 파이프라인(전처리·캐시·학습레시피·평가) = 노트북 그대로
#   | 실행/경로/환경 설정      = inceptionv3_bilinear_512.py 방식 그대로
#
#   ▶ 실행: python xception_boneage_512.py   (또는 VSCode 실행 버튼)
#       - 실행하면 이 화면에 학습 로그가 실시간으로 계속 뜹니다.
#       - 창(터미널/VSCode)을 닫거나 노트북을 꺼도 서버에서 학습은 계속됩니다.
#       - 이 파일을 다시 실행하면 진행 중인 학습 로그에 자동으로 다시 붙습니다.
#
#   ▶ 실행 옵션 (선택)
#       python xception_boneage_512.py --fg          # 백그라운드 분리 없이 바로 실행(디버그용)
#       python xception_boneage_512.py --eval-only   # 학습 건너뛰고 best.pt 로 평가만
# =========================================================================

from pathlib import Path
import os, sys, time, json, subprocess
from datetime import datetime

PROJECT_DIR = Path(__file__).resolve().parent
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RUN_STATE = LOG_DIR / "boneage_xception_running.json"   # InceptionV3 런과 분리
_WORKER_ENV = "BONEAGE_XCEPTION_WORKER"

FOREGROUND = "--fg" in sys.argv
EVAL_ONLY  = "--eval-only" in sys.argv


# -------------------------------------------------------------------------
# [A] 런처: 실행되면 자기 자신을 '세션과 분리된' 백그라운드로 띄우고,
#     이 창에는 그 로그만 실시간으로 흘려보낸다. (윈도우 전용 처리)
# -------------------------------------------------------------------------
def _pid_alive(pid):
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True).stdout
        return str(pid) in out
    except Exception:
        return False


def _follow(log_path, pid):
    """로그 파일을 실시간으로 화면에 표시. 창을 닫거나 Ctrl+C 해도 학습은 계속됨."""
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
    print(" (아래는 실시간 로그 - 이 창만 종료해도 학습엔 영향 없음)")
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
                        print("\n[학습 프로세스가 종료되었습니다]", flush=True)
                        break
                    time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[로그 보기만 종료] - 학습은 백그라운드에서 계속됩니다.", flush=True)
        print(" 다시 보려면 이 파일을 그대로 실행하면 자동으로 다시 붙습니다.", flush=True)


def _spawn_detached():
    """자기 자신을 SSH 세션과 분리된 독립 프로세스로 재실행. (pid, log_path) 반환."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"boneage_xception_{ts}.log"
    logf = open(log_path, "a", buffering=1, encoding="utf-8")
    env = dict(os.environ); env[_WORKER_ENV] = "1"
    env["PYTHONIOENCODING"] = "utf-8"; env["PYTHONUTF8"] = "1"
    cmd = [sys.executable, "-u", "-X", "utf8",
           str(Path(__file__).resolve())] + sys.argv[1:]

    DETACHED_PROCESS = 0x00000008
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    CREATE_BREAKAWAY_FROM_JOB = 0x01000000     # SSH가 세션 종료 때 죽이는 job에서 이탈
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
    # 이미 학습이 돌고 있으면 새로 띄우지 말고 그 로그에 다시 붙는다
    if RUN_STATE.exists():
        try:
            st = json.loads(RUN_STATE.read_text(encoding="utf-8"))
        except Exception:
            st = {}
        if st.get("pid") and _pid_alive(st["pid"]):
            print("이미 학습이 실행 중입니다 - 기존 로그에 다시 붙습니다.")
            _follow(st["log"], st["pid"])
            sys.exit(0)
    _pid, _logp = _spawn_detached()
    _follow(_logp, _pid)
    sys.exit(0)


# =========================================================================
# [B] 실제 본체 - 위에서 분리 실행된 프로세스가 여기부터 실행한다
# =========================================================================
# 모델 가중치 캐시를 프로젝트 폴더 안에 둔다(사용자 홈 오염 방지, timm=HF 캐시 포함)
os.environ["TORCH_HOME"] = str(PROJECT_DIR / ".torch_cache")
os.environ.setdefault("HF_HOME", str(PROJECT_DIR / ".hf_cache"))

import random, math
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")            # 창 없는 환경에서 그림을 '파일로만' 저장
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

# Xception은 torchvision에 없습니다 -> timm 필요 (최초 1회만 설치: pip install timm)
try:
    import timm
except ImportError:
    raise SystemExit("timm 이 없습니다.  설치:  pip install timm")


def log(*a):
    print(f"[{datetime.now():%H:%M:%S}]", *a, flush=True)


def torch_load(path, map_location=None):
    """PyTorch 2.6+ 의 weights_only 기본값 변경에 대응하는 안전 로더."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:                       # 구버전 torch: 인자 자체가 없음
        return torch.load(path, map_location=map_location)


# -------------------------------------------------------------------------
# [B-1] 경로 (파일 위치 기준 자동 인식)
#   BASE_DIR 기본값 = 이 .py 가 있는 폴더.
#   데이터를 다른 곳에 두었다면 아래 한 줄만 바꾸거나
#   환경변수 BONEAGE_BASE_DIR 로 지정하면 됩니다.
# -------------------------------------------------------------------------
BASE_DIR = Path(os.environ.get("BONEAGE_BASE_DIR", PROJECT_DIR))


def _pick(*candidates):
    """존재하는 첫 경로를 반환(없으면 첫 후보를 그대로 반환 -> 로그에서 확인 가능)."""
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    return Path(candidates[0])


# 후보 1 = 노트북 폴더 구조 / 후보 2 = inceptionv3_bilinear_512.py 폴더 구조
TRAIN_IMG_DIR = _pick(BASE_DIR / "Bone+Age+Training+Set" / "boneage-training-dataset",
                      BASE_DIR / "boneage-training-dataset" / "boneage-training-dataset")
TRAIN_CSV     = _pick(BASE_DIR / "Bone+Age+Training+Set+Annotations" / "train.csv",
                      BASE_DIR / "boneage-training-dataset" / "train.csv")
VAL_IMG_DIR   = _pick(BASE_DIR / "Bone+Age+Validation+Set" / "Bone Age Validation Set" / "boneage-validation-dataset-1",
                      BASE_DIR / "boneage-validation-dataset" / "boneage-validation-dataset")
VAL_CSV       = _pick(BASE_DIR / "Bone+Age+Validation+Set" / "Bone Age Validation Set" / "Validation Dataset.csv",
                      BASE_DIR / "boneage-validation-dataset" / "Validation Dataset.csv")
TEST_IMG_DIR  = _pick(BASE_DIR / "Bone+Age+Test+Set" / "Test+Set+Images",
                      BASE_DIR / "Bone Age Test Set" / "Test Set Images")
TEST_CSV      = _pick(BASE_DIR / "Bone+Age+Test+Set" / "Bone age ground truth.csv",
                      BASE_DIR / "Bone Age Test Set" / "Bone age ground truth.csv")


# -------------------------------------------------------------------------
# [B-2] 재현 스위치 (전처리 · 캐시) — 노트북 셀 0-3 그대로
# -------------------------------------------------------------------------
SEED      = 42
IMG_SIZE  = 512          # 512 : 최종모델(InceptionV3+Bilinear)과 동일 - 캐시 공유·메모리 유리
                         # 608 : 논문 §3.4 Xception 스펙 - 백본 출력 19x19 -> z = 6x6x256 = 9,216
CROP_MODE = "advanced"   # "paper"    : 논문 표8식 단순 bbox 크롭 (재현 기준선)
                         # "plate"    : 2단계 크롭(플레이트 + 문자마커 제거)
                         # "advanced" : 전처리 노트북 파이프라인 이식본
                         #              CLAHE -> 마커소거 -> 손실루엣 크롭 -> 정사각 리사이즈
EQUALIZE  = "none"       # 마지막 히스토그램 평활화: "he" | "clahe" | "none"
USE_AUG   = False        # 논문 미언급. 재현 기준선은 False
FORCE_REBUILD = False    # True 면 전처리 캐시를 강제로 다시 생성
N_QC      = 8            # 크롭 QC 시트 장수 (0이면 생략) -> CKPT_DIR/qc_crop.png 로 저장

# 캐시는 (크롭모드 x 해상도 x 평활화)별로 분리 -> 동일 설정이면 최종모델 노트북과 그대로 재사용
CACHE_DIR = BASE_DIR / "cache_preprocessed_2" / f"{CROP_MODE}_{IMG_SIZE}_{EQUALIZE}"
CKPT_DIR  = BASE_DIR / "checkpoints_xception_2"          # 최종모델 체크포인트와 분리
for d in (CACHE_DIR, CACHE_DIR/"train", CACHE_DIR/"val", CACHE_DIR/"test", CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)

BEST_CKPT    = CKPT_DIR / "best.pt"         # 배포/추론용(가벼움, 자체완결)
LAST_CKPT    = CKPT_DIR / "last.pt"         # 재개용(옵티마이저 포함)
HISTORY_JSON = CKPT_DIR / "history.json"
DONE_MARKER  = CACHE_DIR / "_DONE.json"
RESULTS_TXT  = CKPT_DIR / "results.txt"
RESULTS_JSON = CKPT_DIR / "results.json"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# 모델 학습/평가 전 제외할 이상 이미지 (파일명 stem)
EXCLUDE_IDS = {"1521","1607","1779","2256","2414","2823","3883","3885",
               "3899","3905","3931","3964","3999","4004","4230"}

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

print("=" * 64)
log("골연령 Xception 회귀 - 프로세스 시작")
log(f"PyTorch {torch.__version__} | torchvision {torchvision.__version__} | timm {timm.__version__}")
log(f"device {device} | CUDA {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log(f"GPU {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda}")
log(f"BASE_DIR  {BASE_DIR}")
log(f"IMG_SIZE {IMG_SIZE} | CROP_MODE {CROP_MODE} | EQUALIZE {EQUALIZE} | USE_AUG {USE_AUG}")
log(f"CACHE     {CACHE_DIR}")
log(f"CKPT      {CKPT_DIR}")
print("=" * 64, flush=True)
# [주의] RTX 5060(Blackwell, sm_120)은 최신 PyTorch 필요.
#   CUDA=False 이거나 'no kernel image' 오류 시 CUDA 12.8+ 빌드 설치:
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
    ok, buf = cv2.imencode(ext, img)
    if ok: buf.tofile(path)
    return ok


# =========================================================================
# [C] 라벨 로드 (train / val / test) - 노트북 셀 0-4 그대로
# =========================================================================
def load_labels(csv_path, img_dir):
    """id/boneage/male 컬럼을 유연하게 탐지해 표준 DataFrame으로 반환."""
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(keys):
        for k in keys:
            for lc, orig in cols.items():
                if k in lc: return orig
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
        male = sv.map({"true":1,"false":0,"m":1,"f":0,"male":1,"female":0,"1":1,"0":0})
        if male.isna().any(): male = pd.to_numeric(s, errors="coerce")
    out["male"] = male.astype(float)
    out["path"] = out["id"].apply(lambda i: str(Path(img_dir) / f"{i}.png"))
    return out.dropna(subset=["boneage", "male"]).reset_index(drop=True)


for _name, _p in [("TRAIN_IMG", TRAIN_IMG_DIR), ("TRAIN_CSV", TRAIN_CSV),
                  ("VAL_IMG", VAL_IMG_DIR), ("VAL_CSV", VAL_CSV),
                  ("TEST_IMG", TEST_IMG_DIR), ("TEST_CSV", TEST_CSV)]:
    log(f"  {_name:<9} {'OK ' if Path(_p).exists() else '없음'} {_p}")

train_df = load_labels(TRAIN_CSV, TRAIN_IMG_DIR)
val_df   = load_labels(VAL_CSV,   VAL_IMG_DIR)

HAS_TEST = TEST_CSV.exists() and TEST_IMG_DIR.exists()
if HAS_TEST:
    test_df = load_labels(TEST_CSV, TEST_IMG_DIR)
else:
    test_df = pd.DataFrame(columns=["id", "boneage", "male", "path"])
    log("[경고] test 세트를 찾지 못했습니다 - TEST_IMG_DIR / TEST_CSV 경로를 확인하세요.")


def drop_excluded(df, name):
    """이상 이미지 제외 (모델 학습/평가 전)."""
    before = len(df)
    df = df[~df["id"].isin(EXCLUDE_IDS)].reset_index(drop=True)
    if before - len(df):
        log(f"  · {name}: 이상 이미지 {before-len(df)}장 제외")
    return df


train_df = drop_excluded(train_df, "train")
val_df   = drop_excluded(val_df,   "val")
test_df  = drop_excluded(test_df,  "test")

AGE_MEAN = float(train_df.boneage.mean())     # 회귀 안정화용(가정): 학습셋 통계
AGE_STD  = float(train_df.boneage.std())
log(f"라벨 | train {len(train_df):,} | val {len(val_df):,} | test {len(test_df):,} "
    f"| 나이 {AGE_MEAN:.2f}±{AGE_STD:.2f}개월 | 남 {train_df.male.mean():.1%}")


# =========================================================================
# [D] 전처리 (논문 표8의 3단계) + 캐시 - 노트북 셀 0-5 / PART 1 그대로
#     ① 흑백 + 자동 손 크롭  ② 리사이즈  ③ 히스토그램 평활화
# =========================================================================
def crop_hand_paper(gray, pad_ratio=0.03):
    """(A) 논문 표8식 크롭: Otsu -> 최대 blob의 bounding box.
       형태학 커널을 이미지 크기에 비례시켜 원본 해상도가 달라도 동작합니다."""
    H, W = gray.shape
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, m = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k  = max(3, int(min(H, W) * 0.02) | 1)          # 짧은 변의 2%, 홀수
    ks = max(3, (k // 3) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  np.ones((ks, ks), np.uint8))  # 잡티 제거
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k),  np.uint8))   # 손을 한 덩어리로
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return gray
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    px, py = int(w * pad_ratio), int(h * pad_ratio)
    roi = gray[max(0, y-py):min(H, y+h+py), max(0, x-px):min(W, x+w+px)]
    return roi if roi.size else gray


def crop_hand_roi(gray, pad_ratio=0.03):
    """(B) 2단계 크롭: (1) 촬영 플레이트 검출 -> (2) 판 안에서 손만 분리.
       프레임/글자마커(L/CLT/JCO 등)는 연결요소 필터로 제거됩니다.
       [주의] 논문에 없는 로직입니다 - 전수 시각 검증 전에는 기준선으로 쓰지 마세요."""
    H, W = gray.shape
    # (1) 밝은 촬영 플레이트 검출 -> 검정 배경 제거
    _, lit = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lit = cv2.morphologyEx(lit, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(lit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    g_in = gray
    if cnts:
        px, py, pw, ph = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        if pw * ph > 0.15 * H * W:                 # 판이 화면의 상당 부분일 때만 적용
            g_in = gray[py:py+ph, px:px+pw]
    Hi, Wi = g_in.shape

    # (2) 판 내부에서 밝은 손 분리
    blur = cv2.GaussianBlur(g_in, (7, 7), 0)
    _, hand = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hand = cv2.morphologyEx(hand, cv2.MORPH_OPEN,  np.ones((5, 5),   np.uint8))
    hand = cv2.morphologyEx(hand, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    n, _, stats, _ = cv2.connectedComponentsWithStats(hand, 8)
    best_i, best_area = -1, 0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w > 0.97 * Wi and h > 0.97 * Hi:        # 프레임/판 전체 -> 제외
            continue
        if area < 0.02 * Hi * Wi:                  # 글자마커/잡티 -> 제외
            continue
        if area > best_area:
            best_area, best_i = area, i
    if best_i < 0:
        return g_in
    x, y, w, h, _ = stats[best_i]
    px_, py_ = int(w * pad_ratio), int(h * pad_ratio)
    roi = g_in[max(0, y-py_):min(Hi, y+h+py_), max(0, x-px_):min(Wi, x+w+px_)]
    return roi if roi.size else g_in


CROP_FN = {"paper": crop_hand_paper, "plate": crop_hand_roi}

# -------------------------------------------------------------------------
# (C) "advanced" 크롭 = 전처리 노트북(bone_age_preprocess) 파이프라인 이식본.
#     [흑백 + 자동 손 크롭 + 리사이즈]를 담당 -> 최종모델의 동일 단계를 대체.
#     CLAHE 대비향상 -> 손 마스크 -> 마커 소거 -> 손 실루엣 -> bbox 크롭 -> 정사각 리사이즈
#     ※ 히스토그램 평활화는 이 파이프라인 '뒤'에 preprocess()에서 적용합니다.
# -------------------------------------------------------------------------
ADV_CLAHE_CLIP, ADV_CLAHE_TILE = 2.0, 8
ADV_PAD_FRAC        = 0.08
ADV_MIN_AREA_FRAC   = 0.05
ADV_MAX_AREA_FRAC   = 0.95
ADV_HAND_EXTENT_LO  = 0.30
ADV_HAND_EXTENT_HI  = 0.75
ADV_PLATE_EXTENT_TH = 0.85
ADV_MARKER_BRIGHT_TH= 200
ADV_MARKER_MAX_AREA = 4000
ADV_BG_MODE         = "keep"    # "keep"=배경 유지 | "mask"=실루엣 밖 0
ADV_SKIN_DELTA      = 12
ADV_SILH_DILATE     = 7
_adv_clahe = cv2.createCLAHE(clipLimit=ADV_CLAHE_CLIP,
                             tileGridSize=(ADV_CLAHE_TILE, ADV_CLAHE_TILE))


def _adv_enhance(img):
    return _adv_clahe.apply(img)


def _adv_hand_mask(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,  np.ones((5, 5),  np.uint8))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    return th


def _adv_select_hand(mask):
    """extent(=area/bbox)로 직사각 플레이트(≈0.9~1.0)와 손(≈0.4~0.6)을 구분해 손만 선택."""
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1: return None
    total = mask.size; comps = []
    for idx in range(1, n):
        x, y, w, h, area = stats[idx]
        comps.append((idx, area, area / max(w * h, 1)))
    comps.sort(key=lambda c: c[1], reverse=True)
    hand_like = [c for c in comps if ADV_HAND_EXTENT_LO <= c[2] <= ADV_HAND_EXTENT_HI
                 and c[1] >= ADV_MIN_AREA_FRAC * total]
    if hand_like:
        idx = hand_like[0][0]
    else:
        non_plate = [c for c in comps if c[2] < ADV_PLATE_EXTENT_TH]
        idx = non_plate[0][0] if non_plate else comps[0][0]
    hand = np.where(lbl == idx, 255, 0).astype(np.uint8)
    return cv2.morphologyEx(hand, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))


def _adv_remove_markers(img, hand):
    """손 마스크 밖의 '작고 밝은 blob'(L / CLT / JCO 등 번인 마커)을 배경값으로 덮어씀."""
    _, bright = cv2.threshold(img, ADV_MARKER_BRIGHT_TH, 255, cv2.THRESH_BINARY)
    outside = cv2.bitwise_and(bright, cv2.bitwise_not(hand))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(outside, 8)
    marker = np.zeros_like(img); removed = 0
    for idx in range(1, n):
        if stats[idx, cv2.CC_STAT_AREA] <= ADV_MARKER_MAX_AREA:
            marker[lbl == idx] = 255; removed += 1
    if removed:
        marker = cv2.dilate(marker, np.ones((5, 5), np.uint8))
        bg_px = img[hand == 0]; bg = int(np.median(bg_px)) if bg_px.size else 0
        img = img.copy(); img[marker == 255] = bg
    return img, removed


def _adv_fill_holes(mask):
    ff = mask.copy(); h, w = mask.shape
    m2 = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, m2, (0, 0), 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(ff))


def _adv_silhouette(img, seed):
    bg_px = img[seed == 0]; bg = int(np.median(bg_px)) if bg_px.size else 0
    thr = max(bg + ADV_SKIN_DELTA, 1)
    fg = np.where(img > thr, 255, 0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lbl, _, _ = cv2.connectedComponentsWithStats(fg, 8)
    keep = np.zeros_like(fg)
    for L in np.unique(lbl[seed > 0]):
        if L != 0: keep[lbl == L] = 255
    keep = _adv_fill_holes(keep)
    if ADV_SILH_DILATE > 0:
        keep = cv2.dilate(keep, np.ones((ADV_SILH_DILATE, ADV_SILH_DILATE), np.uint8))
    return keep


def _adv_square(img, size):
    h, w = img.shape; ss = max(h, w)
    canvas = np.zeros((ss, ss), np.uint8)
    oy, ox = (ss - h) // 2, (ss - w) // 2
    canvas[oy:oy + h, ox:ox + w] = img
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def _adv_crop(img, silh):
    ys, xs = np.where(silh > 0)
    if ys.size == 0: return img
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    H, W = img.shape
    ph, pw = int((y1 - y0) * ADV_PAD_FRAC), int((x1 - x0) * ADV_PAD_FRAC)
    y0, y1 = max(0, y0 - ph), min(H, y1 + ph + 1)
    x0, x1 = max(0, x0 - pw), min(W, x1 + pw + 1)
    out = img.copy()
    if ADV_BG_MODE == "mask":
        out[silh == 0] = 0
    return out[y0:y1, x0:x1]


def preprocess_advanced(gray, size):
    """전처리 노트북 파이프라인 = [흑백 + 자동 손 크롭 + 정사각 리사이즈]까지만 수행.
       히스토그램 평활화는 하지 않는다(뒤 단계 preprocess에서 적용). 검출 실패 시 rescue."""
    enh  = _adv_enhance(gray)
    hand = _adv_select_hand(_adv_hand_mask(enh))
    if hand is None:
        return _adv_square(enh, size)                       # rescue: 원본 정사각 리사이즈
    enh, _ = _adv_remove_markers(enh, hand)                 # 크롭 '전'에 마커 제거
    area_frac = float((hand > 0).sum()) / hand.size
    if not (ADV_MIN_AREA_FRAC <= area_frac <= ADV_MAX_AREA_FRAC):
        return _adv_square(enh, size)                       # rescue
    silh = _adv_silhouette(enh, hand)
    roi  = _adv_crop(enh, silh)                             # bbox 크롭
    return _adv_square(roi, size)                           # 정사각 리사이즈


# -- 마지막 히스토그램 평활화 (EQUALIZE 스위치) -------------------------------
FINAL_CLAHE_CLIP, FINAL_CLAHE_TILE = 2.0, 8
_final_clahe = cv2.createCLAHE(clipLimit=FINAL_CLAHE_CLIP,
                               tileGridSize=(FINAL_CLAHE_TILE, FINAL_CLAHE_TILE))


def _final_equalize(img, method):
    """"he"=전역 평활화 / "clahe"=국소 대비제한 평활화 / "none"=평활화 없음."""
    if method == "he":    return cv2.equalizeHist(img)
    if method == "clahe": return _final_clahe.apply(img)
    if method == "none":  return img
    raise ValueError(f"알 수 없는 EQUALIZE 값: {method!r} (he/clahe/none 중 하나)")


def preprocess(gray, size=None, mode=None, equalize=None):
    """size x size uint8 생성. 크롭 + 리사이즈 후 마지막에 _final_equalize 적용."""
    size = size or IMG_SIZE
    mode = mode or CROP_MODE
    equalize = equalize or EQUALIZE
    if mode == "advanced":
        roi_resized = preprocess_advanced(gray, size)                      # (1)(2)
        return _final_equalize(roi_resized, equalize)                      # (3)
    roi     = CROP_FN[mode](gray)                                          # (1) 크롭
    resized = cv2.resize(roi, (size, size), interpolation=cv2.INTER_AREA)  # (2) 리사이즈
    return _final_equalize(resized, equalize)                              # (3) 평활화


# -- 캐시 빌드 (1회만; 완료 표식이 있으면 자동 스킵) ---------------------------
def cache_is_valid():
    if FORCE_REBUILD or not DONE_MARKER.exists(): return False
    try:
        info = json.load(open(DONE_MARKER, encoding="utf-8"))
    except Exception:
        return False
    ok = (info.get("img_size") == IMG_SIZE and info.get("crop_mode") == CROP_MODE
          and info.get("equalize") == EQUALIZE)
    if ok and len(test_df) and not (CACHE_DIR/"test"/f"{test_df.iloc[0]['id']}.png").exists():
        return False   # test가 추가됐는데 캐시에 없으면 재빌드
    return ok


def build_cache(df, split):
    out_dir = CACHE_DIR / split; made = skipped = failed = 0; total = len(df)
    for i, (_, r) in enumerate(df.iterrows(), 1):
        dst = out_dir / f"{r['id']}.png"
        if dst.exists() and not FORCE_REBUILD:
            skipped += 1
        else:
            g = imread_kr(r["path"], cv2.IMREAD_GRAYSCALE)
            if g is None:
                failed += 1
            else:
                imwrite_kr(dst, preprocess(g)); made += 1
        if i % 2000 == 0:
            log(f"  전처리 {split} {i}/{total}")
    log(f"[{split}] 생성 {made} | 스킵 {skipped} | 실패 {failed}")


def filter_cached(df, split):
    if not len(df): return df
    ok = df["id"].apply(lambda i: (CACHE_DIR/split/f"{i}.png").exists())
    return df[ok].reset_index(drop=True)


if cache_is_valid():
    log(f"전처리 캐시 유효 ({CROP_MODE}/{IMG_SIZE}/{EQUALIZE}) - 스킵 "
        f"(다시 만들려면 FORCE_REBUILD=True)")
else:
    log("전처리 시작 (train/val/test)...")
    build_cache(train_df, "train")
    build_cache(val_df,   "val")
    if len(test_df): build_cache(test_df, "test")
    json.dump({"img_size": IMG_SIZE, "crop_mode": CROP_MODE, "equalize": EQUALIZE,
               "train": len(train_df), "val": len(val_df), "test": len(test_df)},
              open(DONE_MARKER, "w", encoding="utf-8"))
    log(f"전처리 완료 표식 저장: {DONE_MARKER}")

train_df = filter_cached(train_df, "train")
val_df   = filter_cached(val_df,   "val")
test_df  = filter_cached(test_df,  "test")
log(f"사용 가능 | train {len(train_df):,} | val {len(val_df):,} | test {len(test_df):,}")


# -- 크롭 QC 시트 (전/후 대조) : 창이 없으므로 PNG로 저장 ---------------------
if N_QC > 0 and len(train_df):
    try:
        sample = train_df.sample(min(N_QC, len(train_df)), random_state=SEED)
        cols = min(8, len(sample)); rows = int(math.ceil(len(sample) / cols))
        fig, axes = plt.subplots(rows*2, cols, figsize=(2.2*cols, 4.6*rows))
        axes = np.atleast_2d(axes)
        for j, (_, r) in enumerate(sample.iterrows()):
            rr, cc = (j // cols) * 2, j % cols
            raw = imread_kr(r["path"], cv2.IMREAD_GRAYSCALE)
            pre = imread_kr(CACHE_DIR/"train"/f"{r['id']}.png", cv2.IMREAD_GRAYSCALE)
            axes[rr,   cc].imshow(raw, cmap="gray"); axes[rr,   cc].set_title(f"raw {r['id']}", fontsize=8)
            axes[rr+1, cc].imshow(pre, cmap="gray"); axes[rr+1, cc].set_title(f"pre ({CROP_MODE}/{EQUALIZE})", fontsize=8)
        for ax in axes.ravel(): ax.axis("off")
        plt.tight_layout(); plt.savefig(CKPT_DIR / "qc_crop.png", dpi=110); plt.close()
        log(f"크롭 QC 시트 저장: {CKPT_DIR/'qc_crop.png'} (손/마커 상태를 눈으로 확인하세요)")
    except Exception as e:
        log(f"[경고] QC 시트 생성 실패: {e}")


# =========================================================================
# [E] Dataset · Transform · 모델(Xception) · 헬퍼 - 노트북 셀 0-6 그대로
# =========================================================================
# 증강은 논문에 언급이 없습니다 -> 재현 기준선(USE_AUG=False)은 정규화만 수행.
_aug = [transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.95, 1.05))]

train_tf = transforms.Compose(
    [transforms.ToPILImage()] + (_aug if USE_AUG else []) +
    [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
)
eval_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
log("train 증강: " + ("ON (rotation 15도 + affine)" if USE_AUG else "OFF (논문 재현 기준선)"))


class BoneAgeDataset(Dataset):
    def __init__(self, df, split, tf):
        self.df, self.split, self.tf = df.reset_index(drop=True), split, tf
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        g = imread_kr(CACHE_DIR/self.split/f"{r['id']}.png", cv2.IMREAD_GRAYSCALE)
        x = self.tf(np.stack([g, g, g], -1))                 # 흑백 -> 3채널(사전학습 백본 입력 규격)
        gender  = torch.tensor([r["male"]], dtype=torch.float32)
        y_norm  = torch.tensor([(r["boneage"]-AGE_MEAN)/AGE_STD], dtype=torch.float32)
        y_month = torch.tensor([r["boneage"]], dtype=torch.float32)
        return x, gender, y_norm, y_month


# =========================================================================
#  모델 - 논문 §3.4 식(13)~(17)
#     z = Flat( MaxPool_3x3( Conv_3x3(F) ) ),  e = φ_ψ(g) ∈ R^32,  ŷ = wᵀ[z; e] + b
#   · Bilinear pooling 없음 (최종모델과 갈리는 유일한 구조적 지점)
#   · Conv / Dense(32) 활성함수는 논문 미명시 -> ReLU 기본, 플래그로 분리
# =========================================================================
def make_xception_backbone(pretrained=True):
    """timm에서 원조 Xception(Chollet) 백본을 특징맵 형태로 생성."""
    last_err = None
    for name in ("legacy_xception", "xception"):
        try:
            m = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="")
            log(f"[backbone] timm '{name}' 로드 완료 (pretrained={pretrained})")
            return m
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Xception 백본 생성 실패: {last_err}")


class XceptionRegressor(nn.Module):
    """use_ldl=False -> 기존 스칼라 회귀 (ŷ ∈ R, z-정규화 타깃 + L1)
       use_ldl=True  -> Attention 논문 §III-B: 마지막 FC 출력 240 -> softmax -> 기대값 회귀
                        (forward는 '로짓'을 반환하고, 개월 환산은 out_to_months()가 담당)"""
    def __init__(self, img_size=512, head_relu=True, gender_relu=True, gender_dim=32,
                 pretrained=True, use_ldl=False, n_bins=240):
        super().__init__()
        self.backbone = make_xception_backbone(pretrained=pretrained)

        with torch.no_grad():                            # 백본 출력 채널/해상도 자동 탐지
            f = self.backbone(torch.zeros(1, 3, img_size, img_size))
        c, h, w = f.shape[1], f.shape[2], f.shape[3]

        self.conv = nn.Conv2d(c, 256, kernel_size=3, padding=1)   # padding='same'
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)         # floor 연산
        self.head_relu = head_relu

        with torch.no_grad():
            z = self.pool(self.conv(f))
        self.feat_dim = int(z.numel())

        self.gender      = nn.Linear(1, gender_dim)
        self.gender_relu = gender_relu
        # 논문 식(6): z_i = W^T f_i + b,  출력 차원 = 240 (데이터셋 최대 연령, 개월)
        self.use_ldl = use_ldl
        self.n_bins  = n_bins
        self.fc      = nn.Linear(self.feat_dim + gender_dim, n_bins if use_ldl else 1)

        n_par = sum(p.numel() for p in self.parameters()) / 1e6
        log(f"[model] head = {'LDL softmax ' + str(n_bins) + '-way (식 6~8)' if use_ldl else 'scalar regression'}")
        log(f"[model] backbone out = {c}x{h}x{w} -> z dim = {self.feat_dim:,} "
            f"(논문 9,216 = IMG_SIZE 608일 때)")
        log(f"[model] total params = {n_par:.1f}M (논문 표4: 22.9M)")

    def forward(self, x, g):
        f = self.backbone(x)
        z = self.conv(f)
        if self.head_relu:
            z = F.relu(z)
        z = torch.flatten(self.pool(z), 1)
        e = self.gender(g)
        if self.gender_relu:
            e = F.relu(e)
        out = self.fc(torch.cat([z, e], dim=1))
        # LDL: (B, 240) 로짓 그대로 반환 -> softmax/기대값은 손실·평가 쪽에서 fp32로 계산
        # 스칼라: (B,) 로 축소 (기존 동작 유지)
        return out if self.use_ldl else out.squeeze(1)


def build_model(arch, pretrained=False, freeze=False):
    """ARCH 딕셔너리만으로 모델을 복원 -> 체크포인트가 자체 완결이 됩니다."""
    m = XceptionRegressor(
        img_size    = arch["IMG_SIZE"],
        head_relu   = arch.get("HEAD_RELU", True),
        gender_relu = arch.get("GENDER_RELU", True),
        gender_dim  = arch.get("GENDER_EMB_DIM", 32),
        pretrained  = pretrained,
        use_ldl     = arch.get("USE_LDL", False),      # 체크포인트만으로 헤드 구조 복원
        n_bins      = arch.get("AGE_BINS", 240),
    )
    if freeze:                                   # 백본 동결(빠름·안정) - 헤드만 학습
        for p in m.backbone.parameters():
            p.requires_grad = False
    return m.to(device)


# =========================================================================
#  [E-2] 라벨 분포 학습(LDL) + 기대값 회귀
#        Chen et al., "Attention-Guided ... Label Distribution Learning" §III-B
#        식(7) p_k = softmax(z_k)
#        식(8) ŷ  = Σ_{k=1}^{240} k · p_k
#        식(9) ℓ_MAE = |y - ŷ|
#        식(11) G_k = N(k ; y, δ²),  δ = 15 (논문 명시)
#        식(10) ℓ_reg = D_KL(G‖p) = Σ_k G_k (ln G_k − ln p_k)
#        식(12) ℓ = ℓ_MAE + λ·ℓ_reg,  λ ∈ [0.1, 1] (논문 최적 구간)
#
#  [원문 표기 주의] 식(10)의 좌변은 D_KL(p‖G)로 적혀 있으나, 우변 전개식
#      −Σ G ln(p/G) = Σ G ln(G/p) 는 D_KL(G‖p) 입니다.
#      여기서는 '전개식(우변)'을 그대로 구현했습니다. 방향을 뒤집으면
#      목적함수 자체가 달라지므로 임의 수정 금지.
#
#  [논문 미명시 → 추론] 식(11)의 G는 1/(√2πδ) 계수만 있어 240구간 위에서
#      합이 정확히 1이 아닙니다(특히 y가 1 또는 240 근처면 절반이 잘림).
#      KL이 성립하려면 확률분포여야 하므로 합=1로 재정규화했습니다.
# =========================================================================
def logits_to_months(logits):
    """(B, 240) 로짓 -> (기대값 개월, 확률분포 p). AMP 하에서도 fp32로 계산."""
    logits = logits.float()                                    # fp16 softmax 오버플로 방지
    p = torch.softmax(logits, dim=1)                           # 식(7)
    k = torch.arange(1, logits.size(1) + 1,
                     device=logits.device, dtype=torch.float32)
    return (p * k).sum(dim=1), p                               # 식(8)


def out_to_months(out, mean, std):
    """모델 출력을 개월 단위 예측으로 환산.
       (B, 240) -> LDL 기대값 회귀 / (B,) -> 기존 z-정규화 역변환. 형태로 자동 분기."""
    if out.dim() == 2 and out.size(1) > 1:
        return logits_to_months(out)[0]
    return out.float() * std + mean


class LDLExpectationLoss(nn.Module):
    """ℓ = ℓ_MAE + λ·ℓ_reg  (식 9~12). 입력은 로짓, 타깃은 '개월 원단위'."""
    def __init__(self, n_bins=240, delta=15.0, lam=0.5):
        super().__init__()
        self.delta, self.lam = float(delta), float(lam)
        self.register_buffer("k", torch.arange(1, n_bins + 1, dtype=torch.float32))

    def forward(self, logits, y_month):
        logits = logits.float()
        logp = F.log_softmax(logits, dim=1)                    # ln p_k   (식 7)
        yhat = (logp.exp() * self.k).sum(dim=1)                # ŷ        (식 8)
        l_mae = (yhat - y_month).abs().mean()                  # ℓ_MAE    (식 9)

        # 식(11): 정답 나이 중심 가우시안 -> 합=1 재정규화(위 주석 참조)
        G = torch.exp(-((self.k[None, :] - y_month[:, None]) ** 2)
                      / (2.0 * self.delta ** 2))
        G = G / G.sum(dim=1, keepdim=True).clamp_min(1e-12)

        # 식(10): Σ_k G_k (ln G_k − ln p_k)
        l_reg = (G * (G.clamp_min(1e-12).log() - logp)).sum(dim=1).mean()
        return l_mae + self.lam * l_reg, l_mae.detach(), l_reg.detach()


def mae_months(out, y_month, mean, std):
    """배치 절대오차 '합'(개월). 스칼라/LDL 출력 모두 처리."""
    pm = out_to_months(out.detach(), mean, std).cpu()
    return (pm - y_month.cpu().squeeze(1)).abs().sum().item()


@torch.no_grad()
def evaluate(model, loader, mean, std, use_amp=True):
    model.eval(); tot, n = 0.0, 0
    for x, g, yn, ym in loader:
        x, g = x.to(device), g.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(x, g)
        tot += mae_months(out, ym, mean, std); n += x.size(0)
    return tot / n


@torch.no_grad()
def predict_months(model, loader, mean, std, use_amp=True):
    """개월 단위 예측/정답 배열 반환 (MAE·RMSE·산점도용)."""
    model.eval(); preds, trues = [], []
    for x, g, yn, ym in loader:
        x, g = x.to(device), g.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(x, g)
        preds.append(out_to_months(out, mean, std).cpu()); trues.append(ym.squeeze(1))
    return torch.cat(preds).numpy(), torch.cat(trues).numpy()


def mae_rmse(preds, trues):
    err = preds - trues
    return float(np.abs(err).mean()), float(np.sqrt((err ** 2).mean()))


def save_checkpoint(path, model, arch, mean, std, optimizer=None, scheduler=None,
                    scaler=None, epoch=None, best_val=None, history=None, no_improve=None):
    ck = {"model": model.state_dict(), "arch": arch, "age_mean": mean, "age_std": std}
    if optimizer  is not None: ck["optimizer"]  = optimizer.state_dict()
    if scheduler  is not None: ck["scheduler"]  = scheduler.state_dict()
    if scaler     is not None: ck["scaler"]     = scaler.state_dict()
    if epoch      is not None: ck["epoch"]      = epoch
    if best_val   is not None: ck["best_val"]   = best_val
    if history    is not None: ck["history"]    = history
    if no_improve is not None: ck["no_improve"] = no_improve   # 조기종료 카운터(재개용)
    torch.save(ck, path)


# =========================================================================
# [F] 학습 설정 (하이퍼파라미터는 학습 루프 '바로 앞'에 둔다)
# =========================================================================
# -- 하이퍼파라미터 (여기만 바꿔 재실험) --------------------------------------
BATCH_SIZE     = 16       # OOM 시 8 (IMG_SIZE=608이면 8 권장)
EPOCHS         = 100      # 넉넉히 - 조기종료가 알아서 멈춤
LR             = 1e-4
WEIGHT_DECAY   = 1e-5
FREEZE_BACKBONE= False    # 미세조정(False)=최고성능 / 동결(True)=빠름·안정
NUM_WORKERS    = 0        # 윈도우 안전값(리눅스면 4 이상 권장)
AUTO_RESUME    = True     # last.pt 가 있으면 자동으로 이어서, 없으면 새로 시작
EARLY_STOP_PATIENCE = 15  # 검증 MAE가 이 에폭 수만큼 개선 없으면 조기종료(<0이면 끔)
MIN_DELTA           = 0.01# 개선으로 인정할 최소 MAE 감소(개월)
LOG_EVERY      = 50       # 학습 중 몇 배치마다 진행 로그를 남길지

# -- 모델 구조 (논문 §3.4 · 미명시 항목은 주석으로 표시) ----------------------
GENDER_EMB_DIM = 32       # 논문 명시 (식 15, k=32)
HEAD_RELU      = True     # 논문 미명시 - Conv3x3 뒤 활성함수 (기본 ReLU)
GENDER_RELU    = True     # 논문 미명시 - 성별 Dense 뒤 활성함수 (기본 ReLU)

# -- 라벨 분포 학습(LDL) + 기대값 회귀 : attention 논문 §III-B ----------------
USE_LDL    = True   # True  = 240-way softmax 분포 + 기대값 회귀 (식 6~12)
                    #         → 타깃 z-정규화를 쓰지 않음(출력이 이미 개월 원단위)
                    # False = 기존 스칼라 회귀(z-정규화 + L1). 두 설정 비교용 토글
AGE_BINS   = 240    # 논문 명시 - 데이터셋 최대 연령(개월)
LDL_DELTA  = 15.0   # 논문 명시 - 식(11) 가우시안 폭 δ (성능이 δ에 둔감하다고 보고)
LDL_LAMBDA = 0.5    # 논문 명시 최적 구간 λ ∈ [0.1, 1] 의 중앙값
                    #   λ=0  → 정규화 없는 순수 기대값 회귀(그래도 ℓ1보다 우수, Fig.5b)
                    #   λ↑↑  → 분포가 지나치게 가우시안에 끌려가 MAE 악화(종 모양 곡선)

# 참고) USE_LDL=False 일 때만 타깃을 학습셋 통계로 z-정규화 후 L1 학습
#       (논문 미명시 · 수렴 안정화용 가정). 평가·저장·추론은 항상 개월 단위.
if USE_LDL:
    AGE_MEAN, AGE_STD = 0.0, 1.0     # 항등 변환 - out_to_months()가 그대로 통과시킴
    log("[LDL] 라벨분포학습 ON - 타깃 z-정규화 비활성화 (출력이 개월 원단위)")

ARCH = {"GENDER_EMB_DIM": GENDER_EMB_DIM, "HEAD_RELU": HEAD_RELU, "GENDER_RELU": GENDER_RELU,
        "BACKBONE": "xception", "IMG_SIZE": IMG_SIZE,
        "CROP_MODE": CROP_MODE, "USE_AUG": USE_AUG, "EQUALIZE": EQUALIZE,
        "USE_LDL": USE_LDL, "AGE_BINS": AGE_BINS,
        "LDL_DELTA": LDL_DELTA, "LDL_LAMBDA": LDL_LAMBDA}
log(f"설정: {ARCH}")
log(f"batch {BATCH_SIZE} | epochs {EPOCHS} | lr {LR} | auto_resume {AUTO_RESUME}")

# -- 데이터로더 --------------------------------------------------------------
train_loader = DataLoader(BoneAgeDataset(train_df, "train", train_tf),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
train_eval_loader = DataLoader(BoneAgeDataset(train_df, "train", eval_tf),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(BoneAgeDataset(val_df, "val", eval_tf),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(BoneAgeDataset(test_df, "test", eval_tf),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True) if len(test_df) else None
log(f"배치 수 | train {len(train_loader)} | val {len(val_loader)} "
    f"| test {len(test_loader) if test_loader else 0}")


# =========================================================================
# [G] 학습 (best 즉시 저장 · 조기 종료 · 중단 재개)
#   - best.pt : 검증 MAE 최저 순간 저장 -> [H] 평가가 이걸 로드
#   - last.pt : 매 에폭 저장(조기종료 카운터 포함) -> 다음 실행 시 자동 재개
# =========================================================================
if EVAL_ONLY:
    log("--eval-only : 학습을 건너뛰고 best.pt 로 평가만 진행합니다.")
else:
    model     = build_model(ARCH, pretrained=True, freeze=FREEZE_BACKBONE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=3)
    scaler    = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    if USE_LDL:
        criterion = LDLExpectationLoss(AGE_BINS, LDL_DELTA, LDL_LAMBDA).to(device)
        log(f"[LDL] loss = ℓ_MAE + {LDL_LAMBDA}·D_KL(G‖p) | bins {AGE_BINS} | δ {LDL_DELTA}")
    else:
        criterion = nn.L1Loss()

    if AUTO_RESUME and LAST_CKPT.exists():
        ck = torch_load(LAST_CKPT, map_location=device)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1; best_val = ck["best_val"]
        history = ck["history"]; epochs_no_improve = ck.get("no_improve", 0)
        history.setdefault("val_rmse", [])
        log(f"이어서 학습: epoch {start_epoch}부터 (이전 best val MAE {best_val:.2f}, "
            f"개선정체 {epochs_no_improve})")
        log("  * 처음부터 다시 하려면 checkpoints_xception 폴더의 last.pt 를 지우고 실행")
    else:
        start_epoch, best_val, epochs_no_improve = 1, float("inf"), 0
        history = {"train_mae": [], "val_mae": [], "val_rmse": []}
        log("새로 학습 시작")

    log(f"파라미터 {sum(p.numel() for p in model.parameters())/1e6:.1f}M (논문 표4 ≈22.9M)")

    best_epoch = start_epoch - epochs_no_improve
    epoch = start_epoch - 1
    n_batches = len(train_loader)
    try:
        for epoch in range(start_epoch, EPOCHS + 1):
            model.train(); run_abs, seen = 0.0, 0
            run_reg, run_nb = 0.0, 0
            t0 = time.time()
            for step, (x, g, yn, ym) in enumerate(train_loader, 1):
                x, g = x.to(device), g.to(device)
                yn_d = yn.to(device).squeeze(1)              # z-정규화 타깃(스칼라 경로)
                ym_d = ym.to(device).squeeze(1)              # 개월 원단위 타깃(LDL 경로)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    out = model(x, g)
                if USE_LDL:
                    # 손실은 autocast 밖에서 fp32로 (softmax·KL 수치 안정성)
                    loss, l_mae, l_reg = criterion(out, ym_d)
                    run_reg += float(l_reg); run_nb += 1
                else:
                    loss = criterion(out.float(), yn_d)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                run_abs += mae_months(out, ym, AGE_MEAN, AGE_STD); seen += x.size(0)
                if step % LOG_EVERY == 0 or step == n_batches:
                    extra = f"  kl {run_reg/max(run_nb,1):.3f}" if USE_LDL else ""
                    log(f"  Epoch {epoch:02d}  {step}/{n_batches}  "
                        f"train_mae {run_abs/seen:.2f}{extra}  ({time.time()-t0:.0f}s)")

            tr_mae = run_abs / seen
            vp, vt = predict_months(model, val_loader, AGE_MEAN, AGE_STD, USE_AMP)
            va_mae, va_rmse = mae_rmse(vp, vt)
            scheduler.step(va_mae)
            history["train_mae"].append(tr_mae)
            history["val_mae"].append(va_mae); history["val_rmse"].append(va_rmse)

            # 개선 판정(MIN_DELTA 이상 감소해야 개선으로 인정)
            if va_mae < best_val - MIN_DELTA:
                best_val = va_mae; epochs_no_improve = 0; best_epoch = epoch
                save_checkpoint(BEST_CKPT, model, ARCH, AGE_MEAN, AGE_STD, best_val=best_val)
                flag = f"* best 저장 (val MAE {best_val:.2f})"
            else:
                epochs_no_improve += 1
                flag = f"개선없음 {epochs_no_improve}/{EARLY_STOP_PATIENCE}"
            log(f"[Epoch {epoch:02d} 완료] train MAE {tr_mae:.2f} | val MAE {va_mae:.2f} "
                f"| val RMSE {va_rmse:.2f} | lr {optimizer.param_groups[0]['lr']:.1e} | {flag}")

            # 매 에폭 last 저장(조기종료 카운터 포함) + history 저장
            save_checkpoint(LAST_CKPT, model, ARCH, AGE_MEAN, AGE_STD,
                            optimizer, scheduler, scaler, epoch, best_val, history,
                            no_improve=epochs_no_improve)
            json.dump(history, open(HISTORY_JSON, "w"))

            # 조기 종료 (EARLY_STOP_PATIENCE < 0 이면 비활성)
            if EARLY_STOP_PATIENCE >= 0 and epochs_no_improve >= EARLY_STOP_PATIENCE:
                log(f"조기종료: {EARLY_STOP_PATIENCE}에폭 연속 개선 없음 "
                    f"| 최고 val MAE {best_val:.2f} @ epoch {best_epoch}")
                break
        else:
            log(f"학습 완료(전체 {EPOCHS}에폭) | 최고 val MAE {best_val:.2f} @ epoch {best_epoch}")
    except KeyboardInterrupt:
        save_checkpoint(LAST_CKPT, model, ARCH, AGE_MEAN, AGE_STD,
                        optimizer, scheduler, scaler, epoch, best_val, history,
                        no_improve=epochs_no_improve)
        json.dump(history, open(HISTORY_JSON, "w"))
        log(f"중단됨 - last.pt 저장(epoch {epoch}). 다시 실행하면 이어서 학습합니다.")


# =========================================================================
# [H] 최종 평가 (best.pt 만 로드 - 학습 없이 이 부분만 재실행 가능)
#     train=학습 / val=검증(best 선택) / test=최종 평가 | MAE + RMSE + 연령대별
# =========================================================================
print("=" * 64)
log("최종 평가 (best.pt 로드)")
if not BEST_CKPT.exists():
    log(f"[중단] best.pt 가 없습니다: {BEST_CKPT}")
    sys.exit(0)

ck = torch_load(BEST_CKPT, map_location=device)
eval_model = build_model(ck["arch"], pretrained=False)
eval_model.load_state_dict(ck["model"]); eval_model.eval()
EM_MEAN, EM_STD, EM_IMG = ck["age_mean"], ck["age_std"], ck["arch"]["IMG_SIZE"]
log(f"best.pt 로드 완료 | 정규화 {EM_MEAN:.1f}±{EM_STD:.1f} | IMG {EM_IMG}")

results = {"backbone": "xception", "img_size": IMG_SIZE, "crop_mode": CROP_MODE,
           "equalize": EQUALIZE, "when": datetime.now().isoformat(timespec="seconds"),
           "head": "LDL+expectation" if ck["arch"].get("USE_LDL") else "scalar",
           "ldl": {"bins": ck["arch"].get("AGE_BINS"), "delta": ck["arch"].get("LDL_DELTA"),
                   "lambda": ck["arch"].get("LDL_LAMBDA")} if ck["arch"].get("USE_LDL") else None,
           "splits": {}}
lines = ["=" * 54,
         f"골연령 Xception 회귀 (IMG={IMG_SIZE}, {CROP_MODE}, eq={EQUALIZE})",
         f"{datetime.now():%Y-%m-%d %H:%M}", "=" * 54,
         f"{'split':>6} | {'N':>6} | {'MAE(mo)':>8} | {'RMSE(mo)':>9} | {'bias':>7}"]
log(lines[-1])

split_loaders = [("train", train_eval_loader), ("val", val_loader)]
if test_loader is not None:
    split_loaders.append(("test", test_loader))

for name, loader in split_loaders:
    preds, trues = predict_months(eval_model, loader, EM_MEAN, EM_STD, USE_AMP)
    mae, rmse = mae_rmse(preds, trues); bias = float(np.mean(preds - trues))
    results["splits"][name] = {"N": int(len(trues)), "mae": mae, "rmse": rmse, "bias": bias}
    row = f"{name:>6} | {len(trues):>6,} | {mae:>8.2f} | {rmse:>9.2f} | {bias:>+7.2f}"
    lines.append(row); log(row)
    plt.figure(figsize=(6, 6)); plt.scatter(trues, preds, s=8, alpha=.4)
    lim = [0, max(trues.max(), preds.max()) + 5]; plt.plot(lim, lim, "r--")
    plt.xlabel("True (months)"); plt.ylabel("Pred (months)")
    plt.title(f"{name} | MAE={mae:.2f} · RMSE={rmse:.2f} mo"); plt.tight_layout()
    plt.savefig(CKPT_DIR / f"scatter_{name}.png", dpi=120); plt.close()

# -- 연령대별 표 (논문 표6) : test 가 있으면 test, 없으면 val 기준 -------------
grp_name, grp_loader = ("test", test_loader) if test_loader is not None else ("val", val_loader)
gp, gt = predict_months(eval_model, grp_loader, EM_MEAN, EM_STD, USE_AMP)
lines += ["-" * 54, f"[{grp_name} 연령대별]",
          f"{'group':>7} | {'N':>5} | {'MAE':>6} | {'RMSE':>6} | {'bias':>6}"]
grp = {}
for lo, hi, lab in zip([0, 48, 96, 144, 192], [48, 96, 144, 192, 10**5],
                       ["0-4y", "4-8y", "8-12y", "12-16y", ">16y"]):
    m = (gt >= lo) & (gt < hi)
    if m.sum():
        gm, gr = mae_rmse(gp[m], gt[m]); gb = float(np.mean(gp[m] - gt[m]))
        grp[lab] = {"N": int(m.sum()), "mae": gm, "rmse": gr, "bias": gb}
        lines.append(f"{lab:>7} | {m.sum():>5} | {gm:>6.2f} | {gr:>6.2f} | {gb:>+6.2f}")
results[f"{grp_name}_by_age"] = grp
lines.append("=" * 54)

# -- 학습된 나이 분포 (attention 논문 Fig.4) : LDL일 때만 ----------------------
if ck["arch"].get("USE_LDL"):
    try:
        samp = val_df.sample(min(4, len(val_df)), random_state=7)
        plt.figure(figsize=(7, 4.5))
        for _, r in samp.iterrows():
            gimg = imread_kr(CACHE_DIR/"val"/f"{r['id']}.png", cv2.IMREAD_GRAYSCALE)
            xx = eval_tf(np.stack([gimg]*3, -1)).unsqueeze(0).to(device)
            gg = torch.tensor([[r["male"]]], dtype=torch.float32, device=device)
            with torch.no_grad():
                yh, pp = logits_to_months(eval_model(xx, gg))
            plt.plot(np.arange(1, pp.size(1)+1), pp[0].cpu().numpy(),
                     label=f"y={int(r.boneage)}, ŷ={yh.item():.1f}")
        plt.xlabel("Age (months)"); plt.ylabel("Probability")
        plt.title(f"Learned age distribution (λ={ck['arch'].get('LDL_LAMBDA')}, "
                  f"δ={ck['arch'].get('LDL_DELTA')})")
        plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(CKPT_DIR / "age_distribution.png", dpi=120); plt.close()
        log(f"나이 분포 저장: {CKPT_DIR/'age_distribution.png'}")
    except Exception as e:
        log(f"[경고] 나이 분포 그림 생략: {e}")

# -- 학습 곡선 (논문 그림 8) --------------------------------------------------
if HISTORY_JSON.exists():
    try:
        h = json.load(open(HISTORY_JSON)); ep = range(1, len(h["train_mae"]) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(ep, h["train_mae"], "-o", ms=3, label="train MAE")
        plt.plot(ep, h["val_mae"],   "-o", ms=3, label="val MAE")
        if len(h.get("val_rmse", [])) == len(h["train_mae"]):   # 구버전 재개 시 길이 불일치 방지
            plt.plot(ep, h["val_rmse"], "-s", ms=3, label="val RMSE", alpha=.7)
        plt.axhline(4.10, ls="--", c="green", label="paper 4.10")
        plt.xlabel("Epoch"); plt.ylabel("months"); plt.title("Learning curve (Xception)")
        plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(CKPT_DIR / "learning_curve.png", dpi=120); plt.close()
    except Exception as e:
        log(f"[경고] 학습곡선 저장 실패: {e}")

json.dump(results, open(RESULTS_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
open(RESULTS_TXT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)
log(f"결과 저장: {RESULTS_TXT}")


# =========================================================================
# [H-2] Grad-CAM (논문 그림 11) - 백본 마지막 특징맵 기준, PNG로 저장
# =========================================================================
def _pick_cam_layer(backbone):
    """Xception 백본에서 Grad-CAM 대상 레이어를 자동 선택 (없으면 백본 전체)."""
    mods = dict(backbone.named_modules())
    for name in ("act4", "bn4", "conv4", "block12"):
        if name in mods:
            return mods[name], name
    return backbone, "backbone(output)"


class GradCAM:
    """백본 마지막 특징맵 기준 Grad-CAM (회귀 출력)."""
    def __init__(self, model):
        self.model = model; self.feat = self.grad = None
        tgt, name = _pick_cam_layer(model.backbone)
        log(f"[Grad-CAM] target layer = {name}")
        tgt.register_forward_hook(lambda m, i, o: setattr(self, "feat", o.detach()))
        tgt.register_full_backward_hook(lambda m, gi, go: setattr(self, "grad", go[0].detach()))

    def __call__(self, x, g):
        self.model.eval(); out = self.model(x, g); self.model.zero_grad()
        # LDL이면 240개 로짓 합이 아니라 '기대값 ŷ'을 역전파해야 의미 있는 CAM이 나옴
        months = out_to_months(out, EM_MEAN, EM_STD)
        months.sum().backward()
        w = self.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.feat).sum(1))[0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), float(months.item())


try:
    cam_engine = GradCAM(eval_model)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, (_, r) in zip(axes, val_df.sample(min(4, len(val_df)), random_state=1).iterrows()):
        g = imread_kr(CACHE_DIR/"val"/f"{r['id']}.png", cv2.IMREAD_GRAYSCALE)
        x = eval_tf(np.stack([g]*3, -1)).unsqueeze(0).to(device)
        gd = torch.tensor([[r["male"]]], dtype=torch.float32, device=device)
        cam, pred_month = cam_engine(x, gd)   # 이미 개월 단위
        heat = cv2.applyColorMap(np.uint8(255*cv2.resize(cam, (EM_IMG, EM_IMG))), cv2.COLORMAP_JET)
        over = cv2.addWeighted(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), 0.55, heat, 0.45, 0)
        ax.imshow(cv2.cvtColor(over, cv2.COLOR_BGR2RGB))
        ax.set_title(f"pred {pred_month:.0f}/true {int(r.boneage)}mo"); ax.axis("off")
    plt.tight_layout(); plt.savefig(CKPT_DIR / "gradcam.png", dpi=120); plt.close()
    log(f"Grad-CAM 저장: {CKPT_DIR/'gradcam.png'}")
except Exception as e:
    log(f"[경고] Grad-CAM 생략: {e}")


# =========================================================================
# [I] 단일 이미지 추론 (완전 독립 - best.pt + 이미지 1장이면 끝)
# =========================================================================
def predict_bone_age(image_path, is_male, ckpt_path=BEST_CKPT, return_dist=False):
    """원본 X-ray 경로 + 성별(True=남) -> 골연령(개월).
       전처리(해상도·크롭모드·평활화)·모델·정규화상수를 모두 체크포인트에서 읽어 자체 완결."""
    ck = torch_load(ckpt_path, map_location=device)
    m = build_model(ck["arch"], pretrained=False); m.load_state_dict(ck["model"]); m.eval()
    g = imread_kr(image_path, cv2.IMREAD_GRAYSCALE)
    if g is None: raise FileNotFoundError(image_path)
    pre = preprocess(g, size=ck["arch"]["IMG_SIZE"],
                     mode=ck["arch"].get("CROP_MODE", "paper"),
                     equalize=ck["arch"].get("EQUALIZE", "he"))   # 학습과 동일한 전처리 보장
    x  = eval_tf(np.stack([pre]*3, -1)).unsqueeze(0).to(device)
    gd = torch.tensor([[float(is_male)]], dtype=torch.float32, device=device)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
        out = m(x, gd)
    months = float(out_to_months(out, ck["age_mean"], ck["age_std"]).item())
    if not return_dist:
        return months
    # LDL이면 논문 Fig.4 처럼 나이 분포도 함께 반환 (스칼라 회귀면 None)
    dist = logits_to_months(out)[1][0].cpu().numpy() if out.dim() == 2 and out.size(1) > 1 else None
    return months, dist


# 예시:
#   months = predict_bone_age(VAL_IMG_DIR / "1386.png", is_male=True)
#   print(f"예측 골연령: {months:.1f} 개월")
log(f"추론 함수 준비 완료 · best.pt: {BEST_CKPT}")
log("=== 전체 완료 ===")
