# -*- coding: utf-8 -*-
# =========================================================================
# 손 X-ray 골연령 예측 - InceptionV3 + Bilinear Pooling (논문 최종 모델)
#   | 이미지 512 | train 학습 / val 검증 / test 최종평가 | MAE + RMSE
#
#   ▶ 실행: python inceptionv3_bilinear_512.py   (또는 VSCode 실행 버튼)
#       - 실행하면 이 화면에 학습 로그가 실시간으로 계속 뜹니다.
#       - 창(터미널/VSCode)을 닫거나 노트북을 꺼도 서버에서 학습은 계속됩니다.
#       - 이 파일을 다시 실행하면 진행 중인 학습 로그에 자동으로 다시 붙습니다.
#   추가로 칠 명령도, 수정할 것도 없습니다.
# =========================================================================

from pathlib import Path
import os, sys, time, json, subprocess
from datetime import datetime

PROJECT_DIR = Path(__file__).resolve().parent
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RUN_STATE = LOG_DIR / "boneage_running.json"
_WORKER_ENV = "BONEAGE_WORKER"


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
    log_path = LOG_DIR / f"boneage_{ts}.log"
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


if os.name == "nt" and os.environ.get(_WORKER_ENV) != "1":
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
# [B] 실제 학습 코드 - 위에서 분리 실행된 프로세스가 여기부터 실행한다
# =========================================================================
os.environ["TORCH_HOME"] = str(PROJECT_DIR / ".torch_cache")

import random
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
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models.feature_extraction import create_feature_extractor


def log(*a):
    print(f"[{datetime.now():%H:%M:%S}]", *a, flush=True)


# -- 경로 (파일 위치 기준 자동 인식) ------------------------------------------
TRAIN_IMG_DIR = PROJECT_DIR / "boneage-training-dataset"   / "boneage-training-dataset"
TRAIN_CSV     = PROJECT_DIR / "boneage-training-dataset"   / "train.csv"
VAL_IMG_DIR   = PROJECT_DIR / "boneage-validation-dataset" / "boneage-validation-dataset"
VAL_CSV       = PROJECT_DIR / "boneage-validation-dataset" / "Validation Dataset.csv"
TEST_IMG_DIR  = PROJECT_DIR / "Bone Age Test Set" / "Test Set Images"
TEST_CSV      = PROJECT_DIR / "Bone Age Test Set" / "Bone age ground truth.csv"

# -- 설정 (필요하면 이 값들만 바꾸면 됨) -------------------------------------
SEED           = 42
IMG_SIZE       = 512
CROP_MODE      = "paper"      # "paper"(논문 재현) / "plate"(개선 실험)
USE_AUG        = False
BATCH_SIZE     = 16           # OOM 시 8 또는 4
EPOCHS         = 100
LR             = 1e-4
WEIGHT_DECAY   = 1e-5
FREEZE_BACKBONE= False
BILINEAR_NORM  = False
GENDER_EMB_DIM = 16
REDUCE_1, REDUCE_2 = 512, 128
EARLY_STOP_PATIENCE = 12
MIN_DELTA           = 0.01
AUTO_RESUME    = True         # last.pt 있으면 자동으로 이어서, 없으면 새로 시작
FORCE_REBUILD  = False        # True 면 전처리 캐시 다시 생성
LOG_EVERY      = 50           # 학습 중 몇 배치마다 진행 로그 출력

CACHE_DIR = PROJECT_DIR / "cache_preprocessed" / f"{CROP_MODE}_{IMG_SIZE}"
CKPT_DIR  = PROJECT_DIR / "models" / f"inceptionv3_bilinear_{IMG_SIZE}_{CROP_MODE}"
for d in (CACHE_DIR/"train", CACHE_DIR/"val", CACHE_DIR/"test", CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)
BEST_CKPT    = CKPT_DIR / "best.pt"
LAST_CKPT    = CKPT_DIR / "last.pt"
HISTORY_JSON = CKPT_DIR / "history.json"
DONE_MARKER  = CACHE_DIR / "_DONE.json"
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

print("=" * 60)
log("골연령 InceptionV3 + Bilinear (512) - 학습 프로세스 시작")
log(f"PyTorch {torch.__version__} | device {device} | CUDA {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log(f"GPU {torch.cuda.get_device_name(0)}")
log(f"IMG_SIZE {IMG_SIZE} | CROP_MODE {CROP_MODE} | USE_AUG {USE_AUG}")
print("=" * 60, flush=True)


def imread_kr(path, flags=cv2.IMREAD_GRAYSCALE):
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    except Exception:
        return None


def imwrite_kr(path, img):
    path = str(path); ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok: buf.tofile(path)
    return ok


# -- 라벨 (train/val/test) ---------------------------------------------------
def load_labels(csv_path, img_dir):
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(keys):
        for k in keys:
            for lc, orig in cols.items():
                if k in lc: return orig
        return None

    id_col  = pick(["case", "image", "id"]) or df.columns[0]
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
        if male.isna().any(): male = pd.to_numeric(s, errors="coerce")
    out["male"] = male.astype(float)
    out["path"] = out["id"].apply(lambda i: str(Path(img_dir) / f"{i}.png"))
    return out.dropna(subset=["boneage", "male"]).reset_index(drop=True)


train_df = load_labels(TRAIN_CSV, TRAIN_IMG_DIR)
val_df   = load_labels(VAL_CSV,   VAL_IMG_DIR)
test_df  = load_labels(TEST_CSV,  TEST_IMG_DIR)
AGE_MEAN = float(train_df.boneage.mean())
AGE_STD  = float(train_df.boneage.std())
log(f"라벨 | train {len(train_df):,} | val {len(val_df):,} | test {len(test_df):,} "
    f"| 나이 {AGE_MEAN:.1f}±{AGE_STD:.1f}개월")


# -- 전처리 (논문 표8 3단계) + 캐시 ------------------------------------------
def crop_hand_paper(gray, pad_ratio=0.03):
    H, W = gray.shape
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, m = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k  = max(3, int(min(H, W) * 0.02) | 1); ks = max(3, (k // 3) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  np.ones((ks, ks), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k),  np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return gray
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    px, py = int(w * pad_ratio), int(h * pad_ratio)
    roi = gray[max(0, y-py):min(H, y+h+py), max(0, x-px):min(W, x+w+px)]
    return roi if roi.size else gray


def crop_hand_plate(gray, pad_ratio=0.03):
    H, W = gray.shape
    _, lit = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lit = cv2.morphologyEx(lit, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(lit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    g_in = gray
    if cnts:
        px, py, pw, ph = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        if pw * ph > 0.15 * H * W: g_in = gray[py:py+ph, px:px+pw]
    Hi, Wi = g_in.shape
    blur = cv2.GaussianBlur(g_in, (7, 7), 0)
    _, hand = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hand = cv2.morphologyEx(hand, cv2.MORPH_OPEN,  np.ones((5, 5),   np.uint8))
    hand = cv2.morphologyEx(hand, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(hand, 8)
    best_i, best_area = -1, 0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w > 0.97 * Wi and h > 0.97 * Hi: continue
        if area < 0.02 * Hi * Wi:           continue
        if area > best_area: best_area, best_i = area, i
    if best_i < 0: return g_in
    x, y, w, h, _ = stats[best_i]
    px_, py_ = int(w * pad_ratio), int(h * pad_ratio)
    roi = g_in[max(0, y-py_):min(Hi, y+h+py_), max(0, x-px_):min(Wi, x+w+px_)]
    return roi if roi.size else g_in


CROP_FN = {"paper": crop_hand_paper, "plate": crop_hand_plate}


def preprocess(gray, size=IMG_SIZE, mode=CROP_MODE):
    roi = CROP_FN[mode](gray)
    resized = cv2.resize(roi, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(resized)


def cache_is_valid():
    if FORCE_REBUILD or not DONE_MARKER.exists(): return False
    info = json.load(open(DONE_MARKER, encoding="utf-8"))
    return (info.get("img_size") == IMG_SIZE and info.get("crop_mode") == CROP_MODE
            and info.get("has_test") is True)


def build_cache(df, split):
    out_dir = CACHE_DIR / split; made = skipped = failed = 0; total = len(df)
    for i, (_, r) in enumerate(df.iterrows(), 1):
        dst = out_dir / f"{r['id']}.png"
        if dst.exists() and not FORCE_REBUILD:
            skipped += 1
        else:
            g = imread_kr(r["path"], cv2.IMREAD_GRAYSCALE)
            if g is None: failed += 1
            else: imwrite_kr(dst, preprocess(g)); made += 1
        if i % 2000 == 0:
            log(f"  전처리 {split} {i}/{total}")
    log(f"[{split}] 생성 {made} | 스킵 {skipped} | 실패 {failed}")


def filter_cached(df, split):
    ok = df["id"].apply(lambda i: (CACHE_DIR/split/f"{i}.png").exists())
    return df[ok].reset_index(drop=True)


if cache_is_valid():
    log(f"전처리 캐시 유효 ({CROP_MODE}/{IMG_SIZE}) - 스킵")
else:
    log("전처리 시작 (train/val/test)...")
    build_cache(train_df, "train")
    build_cache(val_df,   "val")
    build_cache(test_df,  "test")
    json.dump({"img_size": IMG_SIZE, "crop_mode": CROP_MODE, "has_test": True},
              open(DONE_MARKER, "w", encoding="utf-8"))
train_df = filter_cached(train_df, "train")
val_df   = filter_cached(val_df,   "val")
test_df  = filter_cached(test_df,  "test")
log(f"사용 가능 | train {len(train_df):,} | val {len(val_df):,} | test {len(test_df):,}")


# -- Dataset | Model ---------------------------------------------------------
_aug = [transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.95, 1.05))]
train_tf = transforms.Compose(
    [transforms.ToPILImage()] + (_aug if USE_AUG else []) +
    [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
eval_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class BoneAgeDataset(Dataset):
    def __init__(self, df, split, tf):
        self.df, self.split, self.tf = df.reset_index(drop=True), split, tf
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        g = imread_kr(CACHE_DIR/self.split/f"{r['id']}.png", cv2.IMREAD_GRAYSCALE)
        x = self.tf(np.stack([g, g, g], -1))
        gender  = torch.tensor([r["male"]], dtype=torch.float32)
        y_norm  = torch.tensor([(r["boneage"]-AGE_MEAN)/AGE_STD], dtype=torch.float32)
        y_month = torch.tensor([r["boneage"]], dtype=torch.float32)
        return x, gender, y_norm, y_month


class InceptionV3Bilinear(nn.Module):
    def __init__(self, reduce1=512, reduce2=128, gender_dim=16,
                 pretrained=True, freeze=False, bilinear_norm=False):
        super().__init__()
        self.bilinear_norm = bilinear_norm
        weights = Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
        net = inception_v3(weights=weights, aux_logits=True)
        net.transform_input = False
        self.backbone = create_feature_extractor(net, return_nodes={"Mixed_7c": "feat"})
        if freeze:
            for p in self.backbone.parameters(): p.requires_grad = False
        self.reduce = nn.Sequential(nn.Conv2d(2048, reduce1, 1),
                                    nn.Conv2d(reduce1, reduce2, 1))
        self.gender_fc = nn.Linear(1, gender_dim)
        self.regressor = nn.Linear(reduce2*reduce2 + gender_dim, 1)

    def bilinear_pool(self, f2):
        with torch.autocast(device_type="cuda", enabled=False):
            f2 = f2.float()
            B, C, H, W = f2.shape
            X = f2.view(B, C, H*W)
            z = torch.bmm(X, X.transpose(1, 2)).view(B, C*C) / (H*W)
            if self.bilinear_norm:
                z = torch.sign(z) * torch.sqrt(torch.abs(z) + 1e-8)
                z = F.normalize(z, dim=1)
        return z

    def forward(self, x, gender):
        f2 = self.reduce(self.backbone(x)["feat"])
        z  = self.bilinear_pool(f2)
        e  = self.gender_fc(gender).float()
        return self.regressor(torch.cat([z, e], dim=1)).squeeze(1)


def build_model(pretrained=False, freeze=False):
    return InceptionV3Bilinear(REDUCE_1, REDUCE_2, GENDER_EMB_DIM,
                               pretrained, freeze, BILINEAR_NORM).to(device)


ARCH = {"REDUCE_1": REDUCE_1, "REDUCE_2": REDUCE_2, "GENDER_EMB_DIM": GENDER_EMB_DIM,
        "BILINEAR_NORM": BILINEAR_NORM, "IMG_SIZE": IMG_SIZE, "CROP_MODE": CROP_MODE}


def save_ckpt(path, model, optimizer=None, scheduler=None, scaler=None,
              epoch=None, best_val=None, history=None, no_improve=None):
    ck = {"model": model.state_dict(), "arch": ARCH,
          "age_mean": AGE_MEAN, "age_std": AGE_STD}
    for key, v in [("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)]:
        if v is not None: ck[key] = v.state_dict()
    for key, v in [("epoch", epoch), ("best_val", best_val),
                   ("history", history), ("no_improve", no_improve)]:
        if v is not None: ck[key] = v
    torch.save(ck, path)


@torch.no_grad()
def predict_months(model, loader):
    model.eval(); preds, trues = [], []
    for x, g, yn, ym in loader:
        x, g = x.to(device), g.to(device)
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            p = model(x, g)
        preds.append(p.float().cpu() * AGE_STD + AGE_MEAN)
        trues.append(ym.squeeze(1))
    return torch.cat(preds).numpy(), torch.cat(trues).numpy()


def mae_rmse(preds, trues):
    err = preds - trues
    return float(np.abs(err).mean()), float(np.sqrt((err**2).mean()))


# NUM_WORKERS=0 : 윈도우 안전값
train_loader = DataLoader(BoneAgeDataset(train_df, "train", train_tf),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
train_eval_loader = DataLoader(BoneAgeDataset(train_df, "train", eval_tf),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
val_loader = DataLoader(BoneAgeDataset(val_df, "val", eval_tf),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(BoneAgeDataset(test_df, "test", eval_tf),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
log(f"배치 수 | train {len(train_loader)} | val {len(val_loader)} | test {len(test_loader)}")


# -- 학습 --------------------------------------------------------------------
model     = build_model(pretrained=True, freeze=FREEZE_BACKBONE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=3)
scaler    = torch.amp.GradScaler("cuda", enabled=USE_AMP)
criterion = nn.L1Loss()

if AUTO_RESUME and LAST_CKPT.exists():
    ck = torch.load(LAST_CKPT, map_location=device)
    model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
    scheduler.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
    start_epoch = ck["epoch"] + 1; best_val = ck["best_val"]
    history = ck["history"]; no_improve = ck.get("no_improve", 0)
    log(f"이어서 학습: epoch {start_epoch}부터 (이전 best val MAE {best_val:.2f})")
    log("  * 처음부터 다시 하려면 models 폴더의 last.pt 를 지우고 실행")
else:
    start_epoch, best_val, no_improve = 1, float("inf"), 0
    history = {"train_mae": [], "val_mae": [], "val_rmse": []}
    log("새로 학습 시작")

log(f"파라미터 {sum(p.numel() for p in model.parameters())/1e6:.1f}M | "
    f"batch {BATCH_SIZE} | epochs {EPOCHS} | lr {LR}")

best_epoch = start_epoch - no_improve
epoch = start_epoch - 1
n_batches = len(train_loader)
try:
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train(); run_abs, seen = 0.0, 0
        t0 = time.time()
        for step, (x, g, yn, ym) in enumerate(train_loader, 1):
            x, g, yn = x.to(device), g.to(device), yn.to(device).squeeze(1)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                pred = model(x, g); loss = criterion(pred, yn)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            pm = pred.detach().float().cpu() * AGE_STD + AGE_MEAN
            run_abs += (pm - ym.cpu().squeeze(1)).abs().sum().item(); seen += x.size(0)
            if step % LOG_EVERY == 0 or step == n_batches:
                log(f"  Epoch {epoch:02d}  {step}/{n_batches}  "
                    f"train_mae {run_abs/seen:.2f}  ({time.time()-t0:.0f}s)")

        tr_mae = run_abs / seen
        vp, vt = predict_months(model, val_loader)
        va_mae, va_rmse = mae_rmse(vp, vt)
        scheduler.step(va_mae)
        history["train_mae"].append(tr_mae)
        history["val_mae"].append(va_mae); history["val_rmse"].append(va_rmse)

        if va_mae < best_val - MIN_DELTA:
            best_val = va_mae; no_improve = 0; best_epoch = epoch
            save_ckpt(BEST_CKPT, model, best_val=best_val)
            flag = f"* best 저장 (val MAE {best_val:.2f})"
        else:
            no_improve += 1
            flag = f"개선없음 {no_improve}/{EARLY_STOP_PATIENCE}"
        log(f"[Epoch {epoch:02d} 완료] train MAE {tr_mae:.2f} | val MAE {va_mae:.2f} "
            f"| val RMSE {va_rmse:.2f} | lr {optimizer.param_groups[0]['lr']:.1e} | {flag}")

        save_ckpt(LAST_CKPT, model, optimizer, scheduler, scaler,
                  epoch, best_val, history, no_improve)
        json.dump(history, open(HISTORY_JSON, "w"))

        if EARLY_STOP_PATIENCE >= 0 and no_improve >= EARLY_STOP_PATIENCE:
            log(f"조기종료 | 최고 val MAE {best_val:.2f} @ epoch {best_epoch}")
            break
    else:
        log(f"학습 완료 | 최고 val MAE {best_val:.2f} @ epoch {best_epoch} (논문 4.10)")
except KeyboardInterrupt:
    save_ckpt(LAST_CKPT, model, optimizer, scheduler, scaler,
              epoch, best_val, history, no_improve)
    json.dump(history, open(HISTORY_JSON, "w"))
    log(f"중단됨 - last.pt 저장(epoch {epoch}). 다시 실행하면 이어서 학습합니다.")


# -- 최종 평가 (train/val/test | MAE + RMSE) + 그래프 저장 --------------------
print("=" * 60)
log("최종 평가 (best.pt 로드)")
ck = torch.load(BEST_CKPT, map_location=device)
best_model = build_model(pretrained=False)
best_model.load_state_dict(ck["model"]); best_model.eval()

results = {"img_size": IMG_SIZE, "crop_mode": CROP_MODE,
           "when": datetime.now().isoformat(timespec="seconds"), "splits": {}}
lines = ["=" * 54,
         f"골연령 InceptionV3+Bilinear (IMG={IMG_SIZE}, {CROP_MODE})",
         f"{datetime.now():%Y-%m-%d %H:%M}", "=" * 54,
         f"{'split':>6} | {'N':>6} | {'MAE(mo)':>8} | {'RMSE(mo)':>9} | {'bias':>7}"]
log(lines[-1])
for name, loader in [("train", train_eval_loader), ("val", val_loader), ("test", test_loader)]:
    preds, trues = predict_months(best_model, loader)
    mae, rmse = mae_rmse(preds, trues); bias = float(np.mean(preds - trues))
    results["splits"][name] = {"N": int(len(trues)), "mae": mae, "rmse": rmse, "bias": bias}
    row = f"{name:>6} | {len(trues):>6,} | {mae:>8.2f} | {rmse:>9.2f} | {bias:>+7.2f}"
    lines.append(row); log(row)
    plt.figure(figsize=(6, 6)); plt.scatter(trues, preds, s=8, alpha=.4)
    lim = [0, max(trues.max(), preds.max()) + 5]; plt.plot(lim, lim, "r--")
    plt.xlabel("True (months)"); plt.ylabel("Pred (months)")
    plt.title(f"{name} | MAE={mae:.2f}mo"); plt.tight_layout()
    plt.savefig(CKPT_DIR / f"scatter_{name}.png", dpi=120); plt.close()

tp, tt = predict_months(best_model, test_loader)
lines += ["-" * 54, "[test 연령대별]",
          f"{'group':>7} | {'N':>5} | {'MAE':>6} | {'RMSE':>6} | {'bias':>6}"]
grp = {}
for lo, hi, lab in zip([0,48,96,144,192], [48,96,144,192,10**5],
                       ["0-4y","4-8y","8-12y","12-16y",">16y"]):
    m = (tt >= lo) & (tt < hi)
    if m.sum():
        gm, gr = mae_rmse(tp[m], tt[m]); gb = float(np.mean(tp[m]-tt[m]))
        grp[lab] = {"N": int(m.sum()), "mae": gm, "rmse": gr, "bias": gb}
        lines.append(f"{lab:>7} | {m.sum():>5} | {gm:>6.2f} | {gr:>6.2f} | {gb:>+6.2f}")
results["test_by_age"] = grp
lines.append("=" * 54)

if HISTORY_JSON.exists():
    h = json.load(open(HISTORY_JSON)); ep = range(1, len(h["train_mae"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(ep, h["train_mae"], "-o", ms=3, label="train MAE")
    plt.plot(ep, h["val_mae"],   "-o", ms=3, label="val MAE")
    plt.plot(ep, h["val_rmse"],  "-s", ms=3, label="val RMSE", alpha=.7)
    plt.axhline(4.10, ls="--", c="green", label="paper MAE 4.10")
    plt.xlabel("Epoch"); plt.ylabel("months"); plt.title("Learning curve")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(CKPT_DIR / "learning_curve.png", dpi=120); plt.close()

json.dump(results, open(RESULTS_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
open(RESULTS_TXT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)
log(f"결과 저장: {RESULTS_TXT}")
log("=== 전체 완료 ===")
