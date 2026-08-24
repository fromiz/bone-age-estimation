# -*- coding: utf-8 -*-
# =========================================================================
# 손 X-ray 골연령 - Attention-Guided RoI Localization + Label Distribution Learning
#   논문: Chen, Chen, Jin, Li, Speier, Arnold,
#         "Attention-Guided Discriminative Region Localization and Label
#          Distribution Learning for Bone Age Assessment" (arXiv:2006.00202)
#   공식 구현(Keras): https://github.com/chenchao666/Bone-Age-Assessment
#
#   ▶ 논문의 2단계 구조를 그대로 이식했습니다.
#       Phase I  : 분류모델(InceptionV3 + soft label) 학습 -> CAM -> RoI 자동 크롭
#                  · Hand : 300x300 입력 + GAP + tau=20
#                  · R1   : 560x560 입력 + GMP + tau=50   (수근골)
#                  · E    : R1 영역을 랜덤값으로 덮은 전체 이미지
#                  · R2   : E 로 재학습한 모델의 CAM      (중수골)
#       Phase II : Xception + 성별(±1) + 240-way softmax -> 기대값 회귀
#                  손실 = l_MAE + lambda * D_KL(G||p),  delta=15
#
#   ▶ 실행: python attention_ldl_boneage.py     (또는 VSCode 실행 버튼)
#       - 실행하면 이 화면에 로그가 실시간으로 계속 뜹니다.
#       - 창(터미널/VSCode)을 닫거나 노트북을 꺼도 서버에서 학습은 계속됩니다.
#       - 이 파일을 다시 실행하면 진행 중인 로그에 자동으로 다시 붙습니다.
#       - 각 스테이지는 완료 표식을 남기고 자동 스킵 -> 중단돼도 이어서 진행.
#
#   ▶ 실행 옵션 (선택)
#       --fg          백그라운드 분리 없이 바로 실행(디버그용)
#       --eval-only   Phase II 학습을 건너뛰고 best.pt 로 평가만
#       --rebuild-roi RoI 크롭 캐시를 강제로 다시 생성 (CAM부터 재실행)
#
#   ▶ 표기 규칙
#       [논문] = 논문/공식코드에 명시된 값      [추론] = 미명시 -> 합리적 추정
#       PAPER_STRICT=True 면 [추론] 항목 중 성능에 영향 주는 것들을 논문값으로 고정
# =========================================================================

from pathlib import Path
import os, sys, time, json, subprocess
from datetime import datetime

PROJECT_DIR = Path(__file__).resolve().parent
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RUN_STATE = LOG_DIR / "boneage_attention_running.json"      # 다른 런과 분리
_WORKER_ENV = "BONEAGE_ATTENTION_WORKER"

FOREGROUND  = "--fg" in sys.argv
EVAL_ONLY   = "--eval-only" in sys.argv
REBUILD_ROI = "--rebuild-roi" in sys.argv


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
    """자기 자신을 SSH 세션과 분리된 독립 프로세스로 재실행. (pid, log_path) 반환."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"boneage_attention_{ts}.log"
    logf = open(log_path, "a", buffering=1, encoding="utf-8")
    env = dict(os.environ); env[_WORKER_ENV] = "1"
    env["PYTHONIOENCODING"] = "utf-8"; env["PYTHONUTF8"] = "1"
    cmd = [sys.executable, "-u", "-X", "utf8", str(Path(__file__).resolve())] + sys.argv[1:]

    DETACHED_PROCESS = 0x00000008
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    CREATE_BREAKAWAY_FROM_JOB = 0x01000000      # SSH 종료 시 job kill 회피
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
# [B] 본체 - 분리 실행된 프로세스가 여기부터 실행
# =========================================================================
os.environ["TORCH_HOME"] = str(PROJECT_DIR / ".torch_cache")
os.environ.setdefault("HF_HOME", str(PROJECT_DIR / ".hf_cache"))

import random, math
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")                       # 창 없는 환경 -> 파일로만 저장
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

try:
    import timm                             # InceptionV3 / Xception 백본 공급
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
# [B-1] 경로 (파일 위치 기준 자동 인식)
# -------------------------------------------------------------------------
BASE_DIR = Path(os.environ.get("BONEAGE_BASE_DIR", PROJECT_DIR))


def _pick(*candidates):
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    return Path(candidates[0])


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


# =========================================================================
# [B-2] 재현 스위치
# =========================================================================
SEED = 42

# -- 논문 명시 해상도 ---------------------------------------------------------
CLS_SIZE_REGION = 560     # [논문] R1/R2 국소화용 분류모델 입력 (특징맵 17~18x18)
CLS_SIZE_HAND   = 300     # [논문] 손 전체 국소화용 분류모델 입력 (특징맵 8x8)
REG_SIZE        = 560     # [논문] Table IV - 560 이상은 개선 없음

# -- CAM 임계값 (0~255 스케일. 공식코드가 heatmap/max*255 로 정규화하므로) -----
TAU_HAND = 20             # [논문] Table I: 손 mIoU 0.757 / AP50 0.995 최적
TAU_R1   = 50             # [논문] Table I: 수근골 mIoU 0.722 / AP50 0.965 최적
TAU_R2   = 50             # [논문] R1 과 동일 방식

# -- 소프트 라벨 (식 5) -------------------------------------------------------
SOFT_L   = 50             # [논문] l=50, 삼각형 소프트 라벨 폭
AGE_BINS = 240            # [논문] 데이터셋 최대 연령(개월)

# -- Phase II 집계 채널 -------------------------------------------------------
#    [논문] Table V 최고 성능 조합: H+R1+E (4.3) / H+R1+R2 (4.3)
#    사용 가능: "hand", "r1", "r2", "erased", "orig"
AGG_CHANNELS = ("hand", "r1", "erased")

# -- 논문 충실도 토글 ---------------------------------------------------------
PAPER_STRICT = True       # True  : 증강 OFF, /255 정규화, 조기종료 OFF (논문 그대로)
                          # False : ImageNet 정규화 + 증강 + 조기종료 허용(엔지니어링)
USE_AUG   = (not PAPER_STRICT)
NORMALIZE = "div255" if PAPER_STRICT else "imagenet"   # 공식코드는 x/255. 만 수행

# -- 기타 --------------------------------------------------------------------
CROP_PAD_FRAC = 0.0       # [추론] bbox 여유. 논문 미명시 -> 0 이 기준선
ERASE_FILL    = "noise"   # [논문] "replacing the pixels in Region-1 with random values"
USE_EXCLUDE   = True      # 이상 이미지(팔찌/프레임 등) 제외 여부
N_QC          = 6         # 논문 Fig.3 형태 QC 시트 장수 (0이면 생략)

CACHE_DIR = BASE_DIR / "cache_attention" / f"{CLS_SIZE_REGION}_{CLS_SIZE_HAND}"
CKPT_DIR  = BASE_DIR / "checkpoints_attention"
SPLITS    = ("train", "val", "test")
ROI_KINDS = ("hand", "r1", "r2", "erased")

for d in [CACHE_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)
for sub in ("base560", "base300") + ROI_KINDS:
    for sp in SPLITS:
        (CACHE_DIR / sub / sp).mkdir(parents=True, exist_ok=True)

BASE_DONE   = CACHE_DIR / "_DONE_base.json"
ROI_DONE    = CACHE_DIR / "_DONE_roi.json"
BBOX_CSV    = CKPT_DIR / "roi_bboxes.csv"

CLS_R1_CKPT   = CKPT_DIR / "cls_r1.pt"       # 560 + GMP (원본 이미지로 학습)
CLS_R2_CKPT   = CKPT_DIR / "cls_r2.pt"       # 560 + GMP (E 이미지로 학습)
CLS_HAND_CKPT = CKPT_DIR / "cls_hand.pt"     # 300 + GAP

BEST_CKPT    = CKPT_DIR / "best.pt"          # Phase II 배포/추론용 (자체완결)
LAST_CKPT    = CKPT_DIR / "last.pt"          # 재개용 (옵티마이저 포함)
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
               "10278", "10573", "10715", "10720", "11043", "11079", "11152", "11367", "11863", "11910", 
               "11917", "11971", "11987", "12036", "12074", "12192", "12296", "12335", "12351", "12684", 
               "13130", "14011", "14086", "14152", "14179", "14234", "14235", "14281", "14343", "14552",
               "14595", "14742", "14770", "15035", "15114", "15234", "15398", "15413",

               "1397", "1450", "1537", "1583", "1687", "1740", "2190", "2945", "3131", "3319", 
               "3326", "3404", "3625", "3853", "3868", "4022", "4119", "4274", "4800", "5618", 
               "6165", "6393", "7629", "7790", "8549", "9607", "10389", "10543", "11146", "11183", 
               "11312", "11559", "11774", "11839", "12110", "13308", "13806", "14325", "14746",

               "4389", "4423", "4432", "4455", "4483",

               "4127", "5801"
               } if USE_EXCLUDE else set()

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

print("=" * 64)
log("Attention-Guided RoI + LDL 골연령 - 프로세스 시작")
log(f"PyTorch {torch.__version__} | torchvision {torchvision.__version__} | timm {timm.__version__}")
log(f"device {device} | CUDA {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log(f"GPU {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda}")
log(f"BASE_DIR {BASE_DIR}")
log(f"PAPER_STRICT {PAPER_STRICT} | 정규화 {NORMALIZE} | 증강 {USE_AUG}")
log(f"집계 채널 {AGG_CHANNELS} | tau(H/R1/R2) {TAU_HAND}/{TAU_R1}/{TAU_R2}")
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
# [C] 라벨 로드
# =========================================================================
def load_labels(csv_path, img_dir):
    """id/boneage/male 컬럼을 유연하게 탐지해 표준 DataFrame으로 반환."""
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
    log("[경고] test 세트를 찾지 못했습니다 - 경로를 확인하세요.")


def drop_excluded(df, name):
    if not len(df):
        return df
    before = len(df)
    df = df[~df["id"].isin(EXCLUDE_IDS)].reset_index(drop=True)
    if before - len(df):
        log(f"  · {name}: 이상 이미지 {before - len(df)}장 제외")
    return df


train_df = drop_excluded(train_df, "train")
val_df   = drop_excluded(val_df,   "val")
test_df  = drop_excluded(test_df,  "test")

log(f"라벨 | train {len(train_df):,} | val {len(val_df):,} | test {len(test_df):,} "
    f"| 나이 {train_df.boneage.mean():.1f}±{train_df.boneage.std():.1f}개월 "
    f"| 남 {train_df.male.mean():.1%}")

SPLIT_DFS = {"train": train_df, "val": val_df, "test": test_df}


# =========================================================================
# [D] 기본 캐시 - 원본을 560 / 300 정사각으로 리사이즈해 저장
#     * 논문 Phase I 은 손 검출/CLAHE 없이 '원본 리사이즈'만 사용합니다.
#     * 매 에폭 2000x1500 PNG를 디코딩하면 GPU가 놀기 때문에 1회만 만들어 둡니다.
# =========================================================================
def base_cache_valid():
    if not BASE_DONE.exists():
        return False
    try:
        info = json.load(open(BASE_DONE, encoding="utf-8"))
    except Exception:
        return False
    return (info.get("s560") == CLS_SIZE_REGION and info.get("s300") == CLS_SIZE_HAND
            and info.get("test") == len(test_df))


def build_base_cache():
    for sp in SPLITS:
        df = SPLIT_DFS[sp]
        if not len(df):
            continue
        made = skipped = failed = 0
        for i, (_, r) in enumerate(df.iterrows(), 1):
            d560 = CACHE_DIR / "base560" / sp / f"{r['id']}.png"
            d300 = CACHE_DIR / "base300" / sp / f"{r['id']}.png"
            if d560.exists() and d300.exists():
                skipped += 1
            else:
                g = imread_kr(r["path"], cv2.IMREAD_GRAYSCALE)
                if g is None:
                    failed += 1
                else:
                    imwrite_kr(d560, cv2.resize(g, (CLS_SIZE_REGION, CLS_SIZE_REGION),
                                                interpolation=cv2.INTER_AREA))
                    imwrite_kr(d300, cv2.resize(g, (CLS_SIZE_HAND, CLS_SIZE_HAND),
                                                interpolation=cv2.INTER_AREA))
                    made += 1
            if i % 2000 == 0:
                log(f"  기본캐시 {sp} {i}/{len(df)}")
        log(f"[base:{sp}] 생성 {made} | 스킵 {skipped} | 실패 {failed}")
    json.dump({"s560": CLS_SIZE_REGION, "s300": CLS_SIZE_HAND,
               "train": len(train_df), "val": len(val_df), "test": len(test_df)},
              open(BASE_DONE, "w", encoding="utf-8"))


if base_cache_valid():
    log("기본 리사이즈 캐시 유효 - 스킵")
else:
    log("기본 리사이즈 캐시 생성 중 (원본 -> 560 / 300)...")
    build_base_cache()
    log("기본 리사이즈 캐시 완료")


def filter_cached(df, sub, sp):
    """캐시 파일이 실제로 존재하는 행만 남긴다."""
    if not len(df):
        return df
    ok = df["id"].apply(lambda i: (CACHE_DIR / sub / sp / f"{i}.png").exists())
    return df[ok].reset_index(drop=True)


for _sp in SPLITS:
    SPLIT_DFS[_sp] = filter_cached(SPLIT_DFS[_sp], "base560", _sp)
train_df, val_df, test_df = SPLIT_DFS["train"], SPLIT_DFS["val"], SPLIT_DFS["test"]
log(f"사용 가능 | train {len(train_df):,} | val {len(val_df):,} | test {len(test_df):,}")


# =========================================================================
# [E] Phase I : 분류 모델 (InceptionV3 + soft label) - 논문 §III-A
#     식(1)~(3): Y_t = (1/HW) * sum_ij sum_k W_kt F_ijk
#       -> 마지막 FC 는 반드시 '선형'(활성함수 없음)이어야 CAM 수식이 성립
#     식(5)   : 삼각형 소프트 라벨,  Y_i = max(0, 1 - |i-t|/l),  l=50
#               원-핫으로는 수렴하지 않는다고 논문이 명시
# =========================================================================
def make_inception_backbone(pretrained=True):
    """timm InceptionV3 백본을 특징맵(B,2048,h,w) 형태로 생성."""
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
    """CAM 추출이 가능한 분류 헤드.
       pool='gmp' -> 작고 뾰족한 영역(R1/R2) / pool='gap' -> 넓은 영역(Hand)
       fc 는 bias 를 두되 CAM 계산에서는 무시한다(논문도 bias 무시)."""

    def __init__(self, pool="gmp", n_bins=AGE_BINS, pretrained=True):
        super().__init__()
        self.backbone = make_inception_backbone(pretrained)
        self.pool_kind = pool
        with torch.no_grad():
            f = self.backbone(torch.zeros(1, 3, CLS_SIZE_HAND if pool == "gap"
                                          else CLS_SIZE_REGION,
                                          CLS_SIZE_HAND if pool == "gap"
                                          else CLS_SIZE_REGION))
        self.c = f.shape[1]
        self.fc = nn.Linear(self.c, n_bins)      # [논문] 활성함수 없음
        log(f"[cls-{pool}] feat {tuple(f.shape[1:])} -> FC {self.c}->{n_bins}")

    def features(self, x):
        return self.backbone(x)                  # (B, C, h, w)

    def forward(self, x):
        f = self.features(x)
        p = (F.adaptive_avg_pool2d(f, 1) if self.pool_kind == "gap"
             else F.adaptive_max_pool2d(f, 1))
        return self.fc(torch.flatten(p, 1))      # (B, 240) 로짓(=소프트라벨 회귀 대상)


def soft_labels(y_month, n_bins=AGE_BINS, l=SOFT_L):
    """식(5) 삼각형 소프트 라벨. y_month: (B,) float 텐서 -> (B, n_bins)"""
    k = torch.arange(1, n_bins + 1, device=y_month.device, dtype=torch.float32)
    d = (k[None, :] - y_month[:, None]).abs()
    return torch.clamp(1.0 - d / float(l), min=0.0)


class ClsDataset(Dataset):
    """Phase I 학습용. sub 로 어떤 캐시(base560/base300/erased)를 쓸지 지정."""

    def __init__(self, df, split, sub, size, train=False):
        self.df = df.reset_index(drop=True)
        self.split, self.sub, self.size, self.train = split, sub, size, train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        g = imread_kr(CACHE_DIR / self.sub / self.split / f"{r['id']}.png",
                      cv2.IMREAD_GRAYSCALE)
        if g is None:                                     # 방어: 손상 파일
            g = np.zeros((self.size, self.size), np.uint8)
        if g.shape[0] != self.size or g.shape[1] != self.size:
            g = cv2.resize(g, (self.size, self.size), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(np.stack([g, g, g], 0)).float() / 255.0   # [논문] x/255
        return x, torch.tensor(float(r["boneage"]), dtype=torch.float32)


# -- Phase I 하이퍼파라미터 (학습 루프 바로 앞) --------------------------------
CLS_BATCH   = 8      # [논문] 32. 8GB 노트북 GPU + 560 입력이면 8 권장 (OOM 시 4)
CLS_EPOCHS  = 70     # [논문] 70 (lr 3e-4 x 50 epoch -> 1e-4 x 20 epoch)
CLS_LR1, CLS_LR2 = 3e-4, 1e-4
CLS_SWITCH  = 50     # [논문] lr 전환 시점
CLS_WORKERS = 0      # 윈도우 안전값 (리눅스면 4 이상)
CLS_LOG_EVERY = 100

# 빠른 점검용 프리셋: 전체 파이프라인을 하루 안에 한 바퀴 돌려보고 싶을 때만 True
CLS_QUICK = False
if CLS_QUICK:
    CLS_EPOCHS, CLS_SWITCH = 12, 8
    log("[주의] CLS_QUICK=True - 논문 스케줄(70 epoch)이 아닙니다. RoI 품질 저하 가능")


def train_cam_classifier(tag, ckpt_path, pool, size, sub, epochs=CLS_EPOCHS):
    """Phase I 분류 모델 학습. 이미 ckpt 가 있으면 그대로 로드하고 스킵."""
    if ckpt_path.exists():
        log(f"[{tag}] 기존 체크포인트 사용 - 학습 스킵 ({ckpt_path.name})")
        ck = torch_load(ckpt_path, map_location=device)
        m = CAMClassifier(pool=pool, pretrained=False).to(device)
        m.load_state_dict(ck["model"]); m.eval()
        return m

    log(f"[{tag}] 분류 모델 학습 시작 | pool={pool} size={size} src={sub} epochs={epochs}")
    model = CAMClassifier(pool=pool, pretrained=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=CLS_LR1, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    crit = nn.L1Loss()                            # [논문] soft label 에 대한 MAE

    tr_loader = DataLoader(ClsDataset(train_df, "train", sub, size, train=True),
                           batch_size=CLS_BATCH, shuffle=True, num_workers=CLS_WORKERS,
                           pin_memory=True, drop_last=True)
    va_loader = DataLoader(ClsDataset(val_df, "val", sub, size),
                           batch_size=CLS_BATCH, shuffle=False, num_workers=CLS_WORKERS,
                           pin_memory=True)
    best = float("inf")
    n_b = len(tr_loader)
    for ep in range(1, epochs + 1):
        lr = CLS_LR1 if ep <= CLS_SWITCH else CLS_LR2     # [논문] 2단 스텝 스케줄
        for pg in opt.param_groups:
            pg["lr"] = lr
        model.train(); run, seen, t0 = 0.0, 0, time.time()
        for step, (x, y) in enumerate(tr_loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            tgt = soft_labels(y)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                out = model(x)
            loss = crit(out.float(), tgt)                 # softmax 없이 직접 회귀
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += float(loss) * x.size(0); seen += x.size(0)
            if step % CLS_LOG_EVERY == 0 or step == n_b:
                log(f"  [{tag}] ep{ep:02d} {step}/{n_b} loss {run/seen:.4f} "
                    f"lr {lr:.0e} ({time.time()-t0:.0f}s)")

        # 검증: argmax 를 나이로 본 대략적 MAE (논문 지표는 아니고 모니터링용)
        model.eval(); errs = []
        with torch.no_grad():
            for x, y in va_loader:
                x = x.to(device)
                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    out = model(x)
                pred = out.float().argmax(1).cpu().numpy() + 1
                errs.append(np.abs(pred - y.numpy()))
        vmae = float(np.concatenate(errs).mean()) if errs else float("nan")
        log(f"  [{tag}] Epoch {ep:02d} 완료 | train L1 {run/seen:.4f} | val argmax-MAE {vmae:.2f}")
        if vmae < best:
            best = vmae
            torch.save({"model": model.state_dict(), "pool": pool, "size": size,
                        "val_argmax_mae": best, "epoch": ep}, ckpt_path)
            log(f"  [{tag}] * best 저장 (argmax-MAE {best:.2f})")
    log(f"[{tag}] 학습 완료 | best argmax-MAE {best:.2f}")

    ck = torch_load(ckpt_path, map_location=device)
    model.load_state_dict(ck["model"]); model.eval()
    return model


# =========================================================================
# [F] CAM -> 마스크 -> bbox -> 원본에서 고해상도 크롭  (식 2 / 4)
#
#  [공식코드 참고] func_utils.GAPAttention()
#     · 클래스 인덱스 t = argmax(prediction)
#     · 가중치는 W[:, t-5:t+5] 의 '평균'  -> 인접 나이 채널 노이즈 완화
#     · heatmap = heatmap / max ; heatmap = uint8(255*heatmap)
#       -> 그래서 논문의 tau (10~100) 는 0~255 축 위의 값입니다.
#
#  [원본 코드 결함 → 수정] 위 uint8 변환은 음수 값에서 언더플로 래핑을 일으켜
#     배경이 밝게 튑니다. 여기서는 clip(0, None) 후 정규화합니다.
#  [논문 미명시 → 추론] 마스크가 여러 조각일 때의 처리 -> '최대 연결성분의 bbox'
# =========================================================================
@torch.no_grad()
def compute_cam(model, x, span=5):
    """x: (1,3,H,W) -> cam (h,w) float(0 이상), pred_class(int)"""
    feat = model.features(x).float()                              # (1,C,h,w)
    p = (F.adaptive_avg_pool2d(feat, 1) if model.pool_kind == "gap"
         else F.adaptive_max_pool2d(feat, 1))
    logits = model.fc(torch.flatten(p, 1))                        # (1,240)
    t = int(logits.argmax(1).item())
    W = model.fc.weight                                           # (240, C)
    lo, hi = max(0, t - span), min(W.size(0), t + span)
    w = W[lo:hi].mean(0)                                          # (C,)  [공식코드]
    cam = (feat[0] * w[:, None, None]).sum(0)                     # 식(2)
    cam = torch.clamp(cam, min=0)                                 # 위 주석 참조
    return cam.cpu().numpy(), t + 1


def cam_to_bbox(cam, tau, out_w, out_h, pad_frac=CROP_PAD_FRAC):
    """CAM -> (x0,y0,x1,y1) in (out_w,out_h) 좌표. 실패 시 전체 영역 반환."""
    m = cv2.resize(cam.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    mx = float(m.max())
    if mx <= 1e-8:
        return (0, 0, out_w, out_h), False
    heat = np.uint8(255.0 * m / mx)                               # 0~255 정규화
    mask = (heat >= tau).astype(np.uint8)                         # 식(4)
    if mask.sum() == 0:
        return (0, 0, out_w, out_h), False
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return (0, 0, out_w, out_h), False
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))           # 최대 연결성분 [추론]
    x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                  int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
    if pad_frac > 0:
        px, py = int(w * pad_frac), int(h * pad_frac)
        x, y, w, h = x - px, y - py, w + 2 * px, h + 2 * py
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(out_w, x + w), min(out_h, y + h)
    if x1 - x0 < 8 or y1 - y0 < 8:                                # 너무 작으면 실패 처리
        return (0, 0, out_w, out_h), False
    return (x0, y0, x1, y1), True


def crop_and_save(orig_gray, box_in_ref, ref_w, ref_h, dst, size=REG_SIZE):
    """ref(=CAM을 계산한 좌표계) 의 bbox 를 '원본 해상도'로 역스케일해서 크롭.
       논문: crop the high-resolution local patches from the original images."""
    H, W = orig_gray.shape[:2]
    sx, sy = W / float(ref_w), H / float(ref_h)
    x0, y0, x1, y1 = box_in_ref
    X0, Y0 = int(round(x0 * sx)), int(round(y0 * sy))
    X1, Y1 = int(round(x1 * sx)), int(round(y1 * sy))
    X0, Y0 = max(0, X0), max(0, Y0)
    X1, Y1 = min(W, max(X0 + 1, X1)), min(H, max(Y0 + 1, Y1))
    patch = orig_gray[Y0:Y1, X0:X1]
    if patch.size == 0:
        patch = orig_gray
    return imwrite_kr(dst, cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA))


def roi_cache_valid():
    if REBUILD_ROI or not ROI_DONE.exists():
        return False
    try:
        info = json.load(open(ROI_DONE, encoding="utf-8"))
    except Exception:
        return False
    return (info.get("tau_hand") == TAU_HAND and info.get("tau_r1") == TAU_R1
            and info.get("tau_r2") == TAU_R2 and info.get("reg_size") == REG_SIZE
            and info.get("test") == len(test_df))


@torch.no_grad()
def localize_and_crop(model, kind, tau, src_sub, src_size, records):
    """kind = 'hand' | 'r1' | 'r2'
       src_sub = CAM 계산에 쓸 캐시(base300 / base560 / erased)
       - hand/r1/r2 : 원본에서 고해상도 크롭 -> CACHE_DIR/<kind>/<split>/<id>.png
       - r1 의 경우 추가로 'erased'(=E) 전체 이미지도 함께 생성 (Fig.3(f))
    """
    model.eval()
    for sp in SPLITS:
        df = SPLIT_DFS[sp]
        if not len(df):
            continue
        ok_cnt = fail_cnt = 0
        for i, (_, r) in enumerate(df.iterrows(), 1):
            dst = CACHE_DIR / kind / sp / f"{r['id']}.png"
            need_e = (kind == "r1")
            dst_e = CACHE_DIR / "erased" / sp / f"{r['id']}.png"
            if dst.exists() and (not need_e or dst_e.exists()) and not REBUILD_ROI:
                continue

            ref = imread_kr(CACHE_DIR / src_sub / sp / f"{r['id']}.png", cv2.IMREAD_GRAYSCALE)
            if ref is None:
                fail_cnt += 1
                continue
            if ref.shape[0] != src_size:
                ref = cv2.resize(ref, (src_size, src_size), interpolation=cv2.INTER_AREA)
            x = torch.from_numpy(np.stack([ref] * 3, 0)).float().div(255.0).unsqueeze(0).to(device)
            cam, _ = compute_cam(model, x)
            box, ok = cam_to_bbox(cam, tau, src_size, src_size)
            ok_cnt += int(ok); fail_cnt += int(not ok)

            orig = imread_kr(r["path"], cv2.IMREAD_GRAYSCALE)
            if orig is None:
                orig = ref
            crop_and_save(orig, box, src_size, src_size, dst)

            # 논문 Fig.3(f): R1 픽셀을 랜덤 값으로 덮은 '전체 이미지' = E
            if need_e:
                e = ref.copy()
                x0, y0, x1, y1 = box
                if ERASE_FILL == "noise":
                    e[y0:y1, x0:x1] = np.random.randint(0, 256, (y1 - y0, x1 - x0), dtype=np.uint8)
                else:
                    e[y0:y1, x0:x1] = int(ref.mean())
                imwrite_kr(dst_e, e)

            records.append({"split": sp, "id": r["id"], "kind": kind,
                            "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3],
                            "ref_size": src_size, "found": int(ok)})
            if i % 1000 == 0:
                log(f"  [{kind}] {sp} {i}/{len(df)}  (마스크실패 {fail_cnt})")
        log(f"[{kind}:{sp}] 완료 | 정상 {ok_cnt} | 마스크실패(전체영역 대체) {fail_cnt}")


if roi_cache_valid():
    log("RoI 크롭 캐시 유효 - Phase I 전체 스킵 (다시 만들려면 --rebuild-roi)")
else:
    bbox_records = []

    # --- (1) Hand : 300 입력 + GAP + tau=20 ---------------------------------
    log("### Phase I-a : Hand 영역 국소화 ###")
    m_hand = train_cam_classifier("cls-hand", CLS_HAND_CKPT, "gap",
                                  CLS_SIZE_HAND, "base300")
    localize_and_crop(m_hand, "hand", TAU_HAND, "base300", CLS_SIZE_HAND, bbox_records)
    del m_hand; torch.cuda.empty_cache()

    # --- (2) Region-1 : 560 입력 + GMP + tau=50, 동시에 E 생성 ---------------
    log("### Phase I-b : Region-1(수근골) 국소화 + Erased 생성 ###")
    m_r1 = train_cam_classifier("cls-r1", CLS_R1_CKPT, "gmp",
                                CLS_SIZE_REGION, "base560")
    localize_and_crop(m_r1, "r1", TAU_R1, "base560", CLS_SIZE_REGION, bbox_records)
    del m_r1; torch.cuda.empty_cache()

    # --- (3) Region-2 : E 로 재학습한 모델의 CAM ------------------------------
    #     논문: R1 이 지워진 이미지로 학습시키면 네트워크가 '그 외 영역'에
    #           근거해 예측하게 되고, 같은 방식으로 R2 를 찾을 수 있다.
    if "r2" in AGG_CHANNELS:
        log("### Phase I-c : Region-2(중수골) 국소화 ###")
        m_r2 = train_cam_classifier("cls-r2", CLS_R2_CKPT, "gmp",
                                    CLS_SIZE_REGION, "erased")
        localize_and_crop(m_r2, "r2", TAU_R2, "erased", CLS_SIZE_REGION, bbox_records)
        del m_r2; torch.cuda.empty_cache()
    else:
        log("AGG_CHANNELS 에 'r2' 가 없어 Phase I-c 를 건너뜁니다 (논문 최고 조합 H+R1+E)")

    if bbox_records:
        pd.DataFrame(bbox_records).to_csv(BBOX_CSV, index=False, encoding="utf-8-sig")
        log(f"bbox 로그 저장: {BBOX_CSV}")
    json.dump({"tau_hand": TAU_HAND, "tau_r1": TAU_R1, "tau_r2": TAU_R2,
               "reg_size": REG_SIZE, "train": len(train_df), "val": len(val_df),
               "test": len(test_df), "channels": list(AGG_CHANNELS)},
              open(ROI_DONE, "w", encoding="utf-8"))
    log("RoI 크롭 캐시 완료")


# -- 논문 Fig.3 형태 QC 시트 (원본 / 손 / R1 / E / R2) -------------------------
if N_QC > 0 and len(val_df):
    try:
        samp = val_df.sample(min(N_QC, len(val_df)), random_state=SEED)
        kinds = ["base560", "hand", "r1", "erased"] + (["r2"] if "r2" in AGG_CHANNELS else [])
        fig, axes = plt.subplots(len(samp), len(kinds), figsize=(2.4 * len(kinds), 2.4 * len(samp)))
        axes = np.atleast_2d(axes)
        for rr, (_, r) in enumerate(samp.iterrows()):
            for cc, k in enumerate(kinds):
                im = imread_kr(CACHE_DIR / k / "val" / f"{r['id']}.png", cv2.IMREAD_GRAYSCALE)
                if im is not None:
                    axes[rr, cc].imshow(im, cmap="gray")
                axes[rr, cc].set_title(f"{k} {r['id']}", fontsize=7)
                axes[rr, cc].axis("off")
        plt.tight_layout(); plt.savefig(CKPT_DIR / "qc_roi.png", dpi=110); plt.close()
        log(f"RoI QC 시트 저장: {CKPT_DIR/'qc_roi.png'}  <- 손/수근골이 제대로 잡혔는지 반드시 눈으로 확인")
    except Exception as e:
        log(f"[경고] QC 시트 생성 실패: {e}")


# =========================================================================
# [G] Phase II : Xception + 성별 + LDL 기대값 회귀 - 논문 §III-B
#     구조: Xception(no top) -> Conv2d(256,3x3) -> MaxPool(3x3) -> Flatten
#           gender(±1) -> Linear(32)
#           concat -> Linear(240) -> softmax -> 기대값
# =========================================================================
_aug = [transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.95, 1.05))]

if NORMALIZE == "imagenet":
    _norm = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
else:
    _norm = [transforms.ToTensor()]          # ToTensor 가 이미 /255 수행 [논문 공식코드]

train_tf = transforms.Compose([transforms.ToPILImage()] + (_aug if USE_AUG else []) + _norm)
eval_tf  = transforms.Compose([transforms.ToPILImage()] + _norm)
log(f"Phase II 정규화 {NORMALIZE} | 증강 {'ON' if USE_AUG else 'OFF (논문 기준선)'}")


class AggDataset(Dataset):
    """서로 다른 국소 패치를 '입력 채널'에 하나씩 꽂는다 (공식 main_aggregation.py 방식).
       ch0 = AGG_CHANNELS[0], ch1 = [1], ch2 = [2]"""

    def __init__(self, df, split, tf, channels=AGG_CHANNELS):
        self.df = df.reset_index(drop=True)
        self.split, self.tf, self.ch = split, tf, channels

    def __len__(self):
        return len(self.df)

    def _load(self, kind, iid):
        sub = "base560" if kind == "orig" else kind
        g = imread_kr(CACHE_DIR / sub / self.split / f"{iid}.png", cv2.IMREAD_GRAYSCALE)
        if g is None:
            g = np.zeros((REG_SIZE, REG_SIZE), np.uint8)
        if g.shape[0] != REG_SIZE or g.shape[1] != REG_SIZE:
            g = cv2.resize(g, (REG_SIZE, REG_SIZE), interpolation=cv2.INTER_AREA)
        return g

    def __getitem__(self, i):
        r = self.df.iloc[i]
        chans = [self._load(k, r["id"]) for k in self.ch]
        while len(chans) < 3:                       # 3채널 미만이면 첫 채널로 채움
            chans.append(chans[0])
        x = self.tf(np.stack(chans[:3], -1))
        gender = torch.tensor([1.0 if r["male"] >= 0.5 else -1.0],  # [논문] 남 +1 / 여 -1
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
        self.conv = nn.Conv2d(c, 256, kernel_size=3)      # [논문] Conv 256, 3x3
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)  # [논문] MaxPool 3x3
        self.head_relu, self.gender_relu = head_relu, gender_relu
        with torch.no_grad():
            z = self.pool(self.conv(f))
        self.feat_dim = int(z.numel())
        self.gender = nn.Linear(1, gender_dim)             # [논문] Table III: 1->32 최적
        self.fc = nn.Linear(self.feat_dim + gender_dim, n_bins)   # 식(6), 출력 240
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
        return self.fc(torch.cat([z, e], 1))               # (B,240) 로짓


def build_reg_model(arch, pretrained=False):
    """ARCH 딕셔너리만으로 복원 -> 체크포인트가 자체 완결이 됩니다."""
    m = AttentionLDLRegressor(img_size=arch["IMG_SIZE"],
                              gender_dim=arch.get("GENDER_EMB_DIM", 32),
                              n_bins=arch.get("AGE_BINS", 240),
                              pretrained=pretrained,
                              head_relu=arch.get("HEAD_RELU", True),
                              gender_relu=arch.get("GENDER_RELU", True))
    return m.to(device)


# -------------------------------------------------------------------------
#  라벨 분포 학습(LDL) + 기대값 회귀  - 식(7)~(12)
#    식(7)  p_k = softmax(z_k)
#    식(8)  yhat = sum_k k * p_k
#    식(9)  l_MAE = |y - yhat|
#    식(11) G_k = N(k ; y, delta^2),  delta = 15 [논문 명시]
#    식(10) l_reg = sum_k G_k (ln G_k - ln p_k) = D_KL(G||p)
#    식(12) l = l_MAE + lambda * l_reg,  lambda in [0.1, 1] [논문 최적 구간]
#
#  [원문 표기 주의] 식(10) 좌변은 D_KL(p||G) 로 적혀 있으나 우변 전개식
#      -sum G ln(p/G) = sum G ln(G/p) 는 D_KL(G||p) 입니다.
#      전개식(우변)을 그대로 구현했습니다. 방향을 뒤집으면 목적함수가 달라짐.
#  [논문 미명시 -> 추론] 식(11)의 G 는 1/sqrt(2*pi*delta) 계수만 있어 240구간
#      합이 1이 아닙니다(y 가 1 또는 240 근처면 절반이 잘림). KL 이 성립하려면
#      확률분포여야 하므로 합=1 로 재정규화했습니다.
# -------------------------------------------------------------------------
def logits_to_months(logits):
    """(B,240) 로짓 -> (기대값 개월, 확률분포 p). AMP 하에서도 fp32 로 계산."""
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
        logp = F.log_softmax(logits, dim=1)                           # ln p_k
        yhat = (logp.exp() * self.k).sum(dim=1)                       # 식(8)
        l_mae = (yhat - y_month).abs().mean()                         # 식(9)
        G = torch.exp(-((self.k[None, :] - y_month[:, None]) ** 2) / (2.0 * self.delta ** 2))
        G = G / G.sum(dim=1, keepdim=True).clamp_min(1e-12)           # 재정규화(위 주석)
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


def save_checkpoint(path, model, arch, optimizer=None, scheduler_lr=None, scaler=None,
                    epoch=None, best_val=None, history=None, no_improve=None):
    ck = {"model": model.state_dict(), "arch": arch}
    if optimizer is not None:    ck["optimizer"] = optimizer.state_dict()
    if scaler is not None:       ck["scaler"] = scaler.state_dict()
    if scheduler_lr is not None: ck["lr"] = scheduler_lr
    if epoch is not None:        ck["epoch"] = epoch
    if best_val is not None:     ck["best_val"] = best_val
    if history is not None:      ck["history"] = history
    if no_improve is not None:   ck["no_improve"] = no_improve
    torch.save(ck, path)


# =========================================================================
# [H] Phase II 학습 설정 (하이퍼파라미터는 학습 루프 '바로 앞')
# =========================================================================
# -- 하이퍼파라미터 (여기만 바꿔 재실험) --------------------------------------
BATCH_SIZE  = 8       # [논문] 16. 8GB GPU + 560 입력이면 8 권장 (OOM 시 4)
EPOCHS      = 120     # [논문] 120 epoch, 3단 스텝 스케줄
LR_STEPS    = [(60, 3e-4), (90, 1e-4), (120, 1e-5)]   # [논문] (누적epoch, lr)
WEIGHT_DECAY = 0.0    # [논문] 미언급 -> 0 (Adam 기본)
NUM_WORKERS = 0       # 윈도우 안전값 (리눅스면 4 이상)
AUTO_RESUME = True    # last.pt 가 있으면 자동으로 이어서 학습
EARLY_STOP_PATIENCE = -1 if PAPER_STRICT else 20   # [추론] 논문에 조기종료 없음
MIN_DELTA   = 0.01
LOG_EVERY   = 100

# -- 모델 구조 ----------------------------------------------------------------
GENDER_EMB_DIM = 32   # [논문] Table III 최적
HEAD_RELU      = True # [추론] Conv3x3 뒤 활성함수 (Keras 원코드는 활성함수 미지정)
GENDER_RELU    = True # [추론] 성별 Dense 뒤 활성함수

# -- LDL --------------------------------------------------------------------
LDL_DELTA  = 15.0     # [논문] 식(11) delta=15, 성능이 delta 에 둔감하다고 보고
LDL_LAMBDA = 0.5      # [논문] 최적 구간 lambda in [0.1, 1] 의 중앙값
                      #   lambda=0  -> 정규화 없는 순수 기대값 회귀(그래도 l1 보다 우수)
                      #   lambda 과대 -> 분포가 가우시안에 과하게 끌려가 MAE 악화

ARCH = {"BACKBONE": "xception", "IMG_SIZE": REG_SIZE, "AGE_BINS": AGE_BINS,
        "GENDER_EMB_DIM": GENDER_EMB_DIM, "HEAD_RELU": HEAD_RELU,
        "GENDER_RELU": GENDER_RELU, "LDL_DELTA": LDL_DELTA, "LDL_LAMBDA": LDL_LAMBDA,
        "AGG_CHANNELS": list(AGG_CHANNELS), "NORMALIZE": NORMALIZE,
        "USE_AUG": USE_AUG, "TAU": [TAU_HAND, TAU_R1, TAU_R2],
        "CLS_SIZE_REGION": CLS_SIZE_REGION, "CLS_SIZE_HAND": CLS_SIZE_HAND}
log(f"Phase II 설정: {ARCH}")

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
    """[논문] 3단 스텝 스케줄: 3e-4(1~60) -> 1e-4(61~90) -> 1e-5(91~120)"""
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

    if AUTO_RESUME and LAST_CKPT.exists():
        ck = torch_load(LAST_CKPT, map_location=device)
        prev = ck.get("arch", {})
        bad = [k for k in ("IMG_SIZE", "AGE_BINS", "GENDER_EMB_DIM", "AGG_CHANNELS")
               if prev.get(k) != ARCH.get(k)]
        if bad:                                   # 구조가 바뀌면 이어서 학습 불가
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            LAST_CKPT.rename(LAST_CKPT.with_name(f"last_stale_{ts}.pt"))
            log(f"[경고] 체크포인트 구조 불일치 {bad} -> last.pt 를 보관 후 새로 학습")
            start_epoch, best_val, epochs_no_improve = 1, float("inf"), 0
            history = {"train_mae": [], "val_mae": [], "val_rmse": [], "kl": []}
        else:
            model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
            scaler.load_state_dict(ck["scaler"])
            start_epoch = ck["epoch"] + 1; best_val = ck["best_val"]
            history = ck["history"]; epochs_no_improve = ck.get("no_improve", 0)
            history.setdefault("kl", [])
            log(f"이어서 학습: epoch {start_epoch}부터 (이전 best val MAE {best_val:.2f})")
            log("  * 처음부터 다시 하려면 checkpoints_attention 폴더의 last.pt 삭제 후 실행")
    else:
        start_epoch, best_val, epochs_no_improve = 1, float("inf"), 0
        history = {"train_mae": [], "val_mae": [], "val_rmse": [], "kl": []}
        log("Phase II 새로 학습 시작")

    best_epoch = start_epoch
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
                # 손실은 autocast 밖 fp32 로 (softmax/KL 수치 안정성)
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
                best_val = va_mae; epochs_no_improve = 0; best_epoch = epoch
                save_checkpoint(BEST_CKPT, model, ARCH, best_val=best_val)
                flag = f"* best 저장 (val MAE {best_val:.2f})"
            else:
                epochs_no_improve += 1
                flag = f"개선없음 {epochs_no_improve}"
            log(f"[Epoch {epoch:03d} 완료] train MAE {tr_mae:.2f} | val MAE {va_mae:.2f} "
                f"| val RMSE {va_rmse:.2f} | {flag}")

            save_checkpoint(LAST_CKPT, model, ARCH, optimizer, lr, scaler,
                            epoch, best_val, history, epochs_no_improve)
            json.dump(history, open(HISTORY_JSON, "w"))

            if EARLY_STOP_PATIENCE >= 0 and epochs_no_improve >= EARLY_STOP_PATIENCE:
                log(f"조기종료 | 최고 val MAE {best_val:.2f} @ epoch {best_epoch}")
                break
        else:
            log(f"학습 완료(전체 {EPOCHS}에폭) | 최고 val MAE {best_val:.2f} @ epoch {best_epoch}")
    except KeyboardInterrupt:
        save_checkpoint(LAST_CKPT, model, ARCH, optimizer, lr_at(max(epoch, 1)), scaler,
                        epoch, best_val, history, epochs_no_improve)
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
log(f"best.pt 로드 완료 | 채널 {ck['arch'].get('AGG_CHANNELS')} "
    f"| lambda {ck['arch'].get('LDL_LAMBDA')} | delta {ck['arch'].get('LDL_DELTA')}")

results = {"method": "attention-guided RoI + LDL", "when": datetime.now().isoformat(timespec="seconds"),
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

# -- 연령대별 표 --------------------------------------------------------------
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
# [J] 단일 이미지 end-to-end 추론
#     원본 1장 -> (분류모델 CAM으로 Hand/R1/E 크롭) -> 회귀모델 -> 개월
#     Phase I 체크포인트 3개 중 필요한 것만 로드합니다.
# =========================================================================
_CLS_CACHE = {}


def _load_cls(ckpt_path, pool):
    key = str(ckpt_path)
    if key not in _CLS_CACHE:
        c = torch_load(ckpt_path, map_location=device)
        m = CAMClassifier(pool=pool, pretrained=False).to(device)
        m.load_state_dict(c["model"]); m.eval()
        _CLS_CACHE[key] = m
    return _CLS_CACHE[key]


def make_rois(orig_gray, channels=AGG_CHANNELS):
    """원본 그레이스케일 -> {kind: 560x560 uint8} 딕셔너리."""
    out = {}
    ref560 = cv2.resize(orig_gray, (CLS_SIZE_REGION, CLS_SIZE_REGION), interpolation=cv2.INTER_AREA)
    ref300 = cv2.resize(orig_gray, (CLS_SIZE_HAND, CLS_SIZE_HAND), interpolation=cv2.INTER_AREA)
    tmp = CKPT_DIR / "_tmp_roi.png"

    def _crop(ref, size, model, tau, kind):
        x = torch.from_numpy(np.stack([ref] * 3, 0)).float().div(255.).unsqueeze(0).to(device)
        cam, _ = compute_cam(model, x)
        box, _ok = cam_to_bbox(cam, tau, size, size)
        crop_and_save(orig_gray, box, size, size, tmp)
        img = imread_kr(tmp, cv2.IMREAD_GRAYSCALE)
        out[kind] = img
        return box, ref

    if "hand" in channels:
        _crop(ref300, CLS_SIZE_HAND, _load_cls(CLS_HAND_CKPT, "gap"), TAU_HAND, "hand")
    if "r1" in channels or "erased" in channels:
        box, _ = _crop(ref560, CLS_SIZE_REGION, _load_cls(CLS_R1_CKPT, "gmp"), TAU_R1, "r1")
        if "erased" in channels:
            e = ref560.copy()
            x0, y0, x1, y1 = box
            e[y0:y1, x0:x1] = np.random.randint(0, 256, (y1 - y0, x1 - x0), dtype=np.uint8)
            out["erased"] = e
    if "r2" in channels:
        e = out.get("erased")
        if e is None:
            e = ref560
        _crop(e, CLS_SIZE_REGION, _load_cls(CLS_R2_CKPT, "gmp"), TAU_R2, "r2")
    if "orig" in channels:
        out["orig"] = ref560
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass
    return out


def predict_bone_age(image_path, is_male, ckpt_path=BEST_CKPT, return_dist=False):
    """원본 X-ray 경로 + 성별(True=남) -> 골연령(개월).
       전처리 규격/채널 구성/모델을 전부 체크포인트에서 읽어 자체 완결로 동작."""
    c = torch_load(ckpt_path, map_location=device)
    m = build_reg_model(c["arch"], pretrained=False)
    m.load_state_dict(c["model"]); m.eval()
    g = imread_kr(image_path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise FileNotFoundError(image_path)
    chans_kind = c["arch"].get("AGG_CHANNELS", list(AGG_CHANNELS))
    rois = make_rois(g, chans_kind)
    chans = []
    for k in chans_kind:
        im = rois.get(k)
        if im is None:
            im = cv2.resize(g, (REG_SIZE, REG_SIZE), interpolation=cv2.INTER_AREA)
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
#   months = predict_bone_age(VAL_IMG_DIR / "1386.png", is_male=True)
#   print(f"예측 골연령: {months:.1f} 개월")
log(f"추론 함수 준비 완료 · best.pt: {BEST_CKPT}")
log("=== 전체 완료 ===")
