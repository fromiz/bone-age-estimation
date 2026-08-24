# -*- coding: utf-8 -*-
"""
attention_r1_current512_v1.py

현재 tight YOLOX-S + 512 INTER_AREA 데이터를 기준으로 R1 Attention ROI만
처음부터 다시 만드는 1단계 코드입니다.

중요:
- official validation / held-out test는 사용하지 않습니다.
- training 12,440장 내부에서 attention tune 300장을 따로 고정합니다.
- R1 localizer는 나머지 training으로 학습하고 tune 300장으로 early stopping/시각 확인합니다.
- R1이 확정된 뒤 R2 코드를 별도로 붙입니다.

순서:
1) python attention_r1_current512_v1.py --task init
2) python attention_r1_current512_v1.py --task train --device cuda:0
3) python attention_r1_current512_v1.py --task debug --debug_n 100 --device cuda:0
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm

NUM_CLASSES = 240
SOFT_L = 50.0

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def seed_worker(worker_id: int):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def norm_col(x: str) -> str:
    return str(x).strip().lower().replace("_", " ").replace("-", " ")


def standardize_df(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    cmap = {norm_col(c): c for c in raw.columns}

    def pick(cands: Sequence[str], required=True) -> Optional[str]:
        for c in cands:
            if norm_col(c) in cmap:
                return cmap[norm_col(c)]
        if required:
            raise KeyError(f"필수 컬럼 없음: {cands}, columns={list(raw.columns)}")
        return None

    id_col = pick(["id", "image id", "case id", "imageid", "caseid"])
    age_col = pick(["boneage", "bone age", "bone age months", "bone age (months)"])
    male_col = pick(["male", "sex", "gender"], required=False)

    ids = pd.to_numeric(raw[id_col], errors="coerce")
    if ids.isna().any():
        ext = raw[id_col].astype(str).str.extract(r"(\d+)", expand=False)
        ids = ids.fillna(pd.to_numeric(ext, errors="coerce"))

    ages = pd.to_numeric(raw[age_col], errors="coerce")

    if male_col is None:
        male = pd.Series(np.zeros(len(raw)), index=raw.index)
    else:
        def to_male(v):
            if isinstance(v, str):
                s = v.strip().lower()
                if s in {"m", "male", "true", "1", "남", "남자"}:
                    return 1.0
                if s in {"f", "female", "false", "0", "여", "여자"}:
                    return 0.0
            try:
                return float(v)
            except Exception:
                return 0.0
        male = raw[male_col].map(to_male)

    df = pd.DataFrame({"id": ids, "boneage": ages, "male": male})
    df = df.dropna(subset=["id", "boneage"]).reset_index(drop=True)
    df["id"] = df["id"].astype(int)
    df["boneage"] = df["boneage"].astype(float)
    df["male"] = (df["male"].astype(float) >= 0.5).astype(int)

    if (~df["boneage"].between(1, 240)).any():
        raise ValueError("boneage가 1~240 범위를 벗어났습니다.")
    if df["id"].duplicated().any():
        raise ValueError("training.csv에 중복 ID가 있습니다.")
    return df


def find_image(image_id: int, image_dir: Path) -> Path:
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        p = image_dir / f"{image_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"ID={image_id} 없음: {image_dir}")


def make_tune_split(df: pd.DataFrame, tune_n: int, seed: int) -> list[int]:
    work = df.copy()
    work["age_band"] = ((work["boneage"] - 1) // 24).astype(int)
    work["stratum"] = work["male"].astype(str) + "_" + work["age_band"].astype(str)

    counts = work["stratum"].value_counts().sort_index()
    exact = counts / len(work) * tune_n
    take = np.floor(exact).astype(int)

    # 가능하면 각 stratum에 localizer training sample 1장 이상 남김
    max_take = (counts - 1).clip(lower=0)
    take = np.minimum(take, max_take)

    remaining = int(tune_n - take.sum())
    frac_order = (exact - np.floor(exact)).sort_values(ascending=False).index.tolist()

    while remaining > 0:
        changed = False
        for key in frac_order:
            if remaining == 0:
                break
            if take.loc[key] < max_take.loc[key]:
                take.loc[key] += 1
                remaining -= 1
                changed = True
        if not changed:
            raise RuntimeError("tune subset을 정확히 구성하지 못했습니다.")

    rng = np.random.default_rng(seed)
    ids = []
    for key, n in take.items():
        if n <= 0:
            continue
        pool = work.loc[work["stratum"] == key, "id"].to_numpy()
        ids.extend(rng.choice(pool, size=int(n), replace=False).astype(int).tolist())

    ids = sorted(ids)
    if len(ids) != tune_n:
        raise RuntimeError(f"tune={len(ids)} != {tune_n}")
    return ids


def ensure_split(dataset_dir: Path, work_dir: Path, tune_n: int, seed: int):
    train_df = standardize_df(dataset_dir / "training.csv")
    tune_file = work_dir / "attention_tune_ids.csv"
    work_dir.mkdir(parents=True, exist_ok=True)

    if tune_file.exists():
        tune_ids = set(pd.read_csv(tune_file)["id"].astype(int).tolist())
        if len(tune_ids) != tune_n:
            raise RuntimeError(
                f"기존 tune IDs={len(tune_ids)}. "
                f"--tune_n {len(tune_ids)}로 실행하거나 파일 삭제 후 재생성하세요."
            )
    else:
        ids = make_tune_split(train_df, tune_n, seed)
        pd.DataFrame({"id": ids}).to_csv(tune_file, index=False, encoding="utf-8-sig")
        tune_ids = set(ids)

    localizer_train = train_df[~train_df["id"].isin(tune_ids)].reset_index(drop=True)
    tune_df = train_df[train_df["id"].isin(tune_ids)].reset_index(drop=True)

    localizer_train.to_csv(work_dir / "localizer_training.csv", index=False, encoding="utf-8-sig")
    tune_df.to_csv(work_dir / "attention_tune.csv", index=False, encoding="utf-8-sig")

    print(f"전체 training : {len(train_df)}")
    print(f"R1 학습       : {len(localizer_train)}")
    print(f"attention tune: {len(tune_df)}")
    print("official validation/test: NOT USED")
    return localizer_train, tune_df


class InceptionFeatures(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.Inception_V3_Weights.DEFAULT if pretrained else None
        base = models.inception_v3(weights=weights, aux_logits=True, transform_input=False)
        base.aux_logits = False
        base.AuxLogits = None

        self.Conv2d_1a_3x3 = base.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = base.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = base.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = base.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = base.Conv2d_4a_3x3
        self.Mixed_5b = base.Mixed_5b
        self.Mixed_5c = base.Mixed_5c
        self.Mixed_5d = base.Mixed_5d
        self.Mixed_6a = base.Mixed_6a
        self.Mixed_6b = base.Mixed_6b
        self.Mixed_6c = base.Mixed_6c
        self.Mixed_6d = base.Mixed_6d
        self.Mixed_6e = base.Mixed_6e
        self.Mixed_7a = base.Mixed_7a
        self.Mixed_7b = base.Mixed_7b
        self.Mixed_7c = base.Mixed_7c

    def forward(self, x):
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, 3, 2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, 3, 2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        return self.Mixed_7c(x)


class R1Localizer(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.features = InceptionFeatures(pretrained)
        self.fc = nn.Linear(2048, NUM_CLASSES)

    def forward(self, x):
        feat = self.features(x)
        pooled = torch.amax(feat, dim=(2, 3))  # GMP
        return self.fc(pooled), feat


def soft_label(age: float):
    bins = torch.arange(1, 241, dtype=torch.float32)
    return torch.clamp(1.0 - torch.abs(bins - float(age)) / SOFT_L, min=0.0)


def preprocess(image_bgr: np.ndarray, size: int):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (size, size):
        inter = cv2.INTER_AREA if max(rgb.shape[:2]) >= size else cv2.INTER_CUBIC
        rgb = cv2.resize(rgb, (size, size), interpolation=inter)
    x = torch.from_numpy(rgb.astype(np.float32).transpose(2, 0, 1)) / 255.0
    return (x - IMAGENET_MEAN) / IMAGENET_STD


class R1Dataset(Dataset):
    def __init__(self, df, image_dir, size):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.size = size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        image_id = int(r["id"])
        age = float(r["boneage"])
        img = cv2.imread(str(find_image(image_id, self.image_dir)), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"imread 실패 ID={image_id}")
        return preprocess(img, self.size), soft_label(age), image_id, age


def loader(df, image_dir, size, batch, workers, shuffle, seed):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        R1Dataset(df, image_dir, size),
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(workers > 0),
        worker_init_fn=seed_worker if workers > 0 else None,
        generator=g,
    )


def run_epoch(model, dl, device, optimizer, amp, scaler):
    train = optimizer is not None
    model.train(train)
    crit = nn.L1Loss()
    total_loss = total_ae = n = 0

    for x, y, _, ages in tqdm(dl, leave=False, desc="train" if train else "tune"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        ages = ages.to(device, dtype=torch.float32, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(amp and device.type == "cuda"),
        ):
            logits, _ = model(x)
            loss = crit(logits, y)

        if train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        pred = torch.argmax(logits.detach(), dim=1).float() + 1
        bs = x.size(0)
        total_loss += float(loss.detach().cpu()) * bs
        total_ae += float(torch.abs(pred - ages).sum().detach().cpu())
        n += bs

    return total_loss / n, total_ae / n


def train_r1(args, train_df, tune_df, image_dir, work_dir, device):
    ckpt_dir = work_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "r1_best.pt"
    last_path = ckpt_dir / "r1_last.pt"
    history_path = ckpt_dir / "r1_history.csv"

    if args.retrain:
        for p in [best_path, last_path, history_path]:
            if p.exists():
                p.unlink()

    model = R1Localizer(pretrained=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    amp = (not args.no_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if amp else None

    tr_dl = loader(train_df, image_dir, args.localizer_size, args.batch_size, args.workers, True, args.seed + 1)
    va_dl = loader(tune_df, image_dir, args.localizer_size, args.batch_size, args.workers, False, args.seed + 2)

    start = 1
    best = math.inf
    best_epoch = 0
    no_improve = 0
    history = []

    if args.resume and last_path.exists() and not args.retrain:
        c = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["optimizer"])
        if scaler is not None and c.get("scaler"):
            scaler.load_state_dict(c["scaler"])
        start = int(c["epoch"]) + 1
        best = float(c["best"])
        best_epoch = int(c["best_epoch"])
        no_improve = int(c["no_improve"])
        history = list(c.get("history", []))
        print(f"resume epoch {start}, best tune L1={best:.6f} @ {best_epoch}")

    for epoch in range(start, args.epochs + 1):
        tr_loss, tr_mae = run_epoch(model, tr_dl, device, opt, amp, scaler)
        with torch.no_grad():
            va_loss, va_mae = run_epoch(model, va_dl, device, None, amp, None)

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_age_mae": tr_mae,
            "tune_loss": va_loss,
            "tune_age_mae": va_mae,
        })

        improved = va_loss < best - args.min_delta
        if improved:
            best = va_loss
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "input_size": args.localizer_size,
                "best_tune_loss": va_loss,
                "tune_age_mae": va_mae,
            }, best_path)
            mark = "BEST"
        else:
            no_improve += 1
            mark = f"no improve {no_improve}/{args.patience}"

        torch.save({
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "best": best,
            "best_epoch": best_epoch,
            "no_improve": no_improve,
            "history": history,
        }, last_path)

        pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8-sig")

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train L1={tr_loss:.6f}, MAE={tr_mae:.3f} | "
            f"tune L1={va_loss:.6f}, MAE={va_mae:.3f} | {mark}"
        )

        if no_improve >= args.patience:
            print(f"Early stopping | best tune L1={best:.6f} @ epoch {best_epoch}")
            break


@torch.inference_mode()
def cam_bbox(model, image_bgr, size, tau, device):
    h0, w0 = image_bgr.shape[:2]
    x = preprocess(image_bgr, size).unsqueeze(0).to(device)
    logits, feat = model(x)
    cls = int(torch.argmax(logits, dim=1).item())

    w = model.fc.weight[cls]
    cam = (feat[0] * w.view(-1, 1, 1)).sum(dim=0).float().cpu().numpy()
    cam = np.maximum(cam, 0)

    mx = float(cam.max())
    if mx <= 1e-8:
        return None, None, None, cls + 1

    cam100 = cam / mx * 100.0
    cam100 = cv2.resize(cam100.astype(np.float32), (w0, h0), interpolation=cv2.INTER_CUBIC)

    binary = (cam100 >= tau).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return None, cam100, binary * 255, cls + 1

    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x0 = int(stats[idx, cv2.CC_STAT_LEFT])
    y0 = int(stats[idx, cv2.CC_STAT_TOP])
    bw = int(stats[idx, cv2.CC_STAT_WIDTH])
    bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
    mask = (labels == idx).astype(np.uint8) * 255

    return (x0, y0, bw, bh), cam100, mask, cls + 1


def square_roi(core, image_w, image_h, scale, min_frac, max_frac):
    x, y, w, h = map(float, core)
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * scale
    base = min(image_w, image_h)
    side = np.clip(side, base * min_frac, base * max_frac)
    side = int(round(min(side, image_w, image_h)))

    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x0 = max(0, min(x0, image_w - side))
    y0 = max(0, min(y0, image_h - side))
    return x0, y0, side, side


def debug_r1(args, tune_df, image_dir, work_dir, device):
    best_path = work_dir / "checkpoints" / "r1_best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"R1 best 없음: {best_path}")

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model = R1Localizer(pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_root = work_dir / "debug_r1"
    if out_root.exists() and args.clear_debug:
        import shutil
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    subset = tune_df.iloc[: args.debug_n]

    for _, r in tqdm(subset.iterrows(), total=len(subset), desc="R1 debug"):
        image_id = int(r["id"])
        age = float(r["boneage"])

        p = find_image(image_id, image_dir)
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue

        core, cam100, mask, pred = cam_bbox(
            model, img, args.localizer_size, args.r1_tau, device
        )

        rec = {"id": image_id, "boneage": age, "pred_month": pred}

        if core is None:
            rec.update({"status": "fail", "reason": "no_core"})
            rows.append(rec)
            continue

        roi = square_roi(
            core,
            img.shape[1],
            img.shape[0],
            args.r1_crop_scale,
            args.r1_min_side,
            args.r1_max_side,
        )

        x, y, w, h = roi
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        crop = gray[y:y+h, x:x+w]
        crop = cv2.resize(
            crop,
            (args.roi_output_size, args.roi_output_size),
            interpolation=cv2.INTER_AREA if max(crop.shape) >= args.roi_output_size else cv2.INTER_CUBIC,
        )

        case = out_root / str(image_id)
        case.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(case / "00_H.png"), img)

        heat = np.clip(cam100 / 100 * 255, 0, 255).astype(np.uint8)
        color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        cv2.imwrite(str(case / "01_heatmap.png"), color)
        cv2.imwrite(str(case / "02_mask.png"), mask)

        overlay = cv2.addWeighted(img, 0.60, color, 0.40, 0)
        cx, cy, cw, ch = core
        cv2.rectangle(overlay, (cx, cy), (cx+cw, cy+ch), (255, 128, 0), 2)  # blue: core
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 255), 3)        # yellow: ROI
        cv2.imwrite(str(case / "03_overlay.png"), overlay)
        cv2.imwrite(str(case / "04_R1.png"), crop)

        rec.update({
            "status": "ok",
            "core_x": cx, "core_y": cy, "core_w": cw, "core_h": ch,
            "roi_x": x, "roi_y": y, "roi_w": w, "roi_h": h,
            "core_area_ratio": cw * ch / float(img.shape[0] * img.shape[1]),
            "roi_area_ratio": w * h / float(img.shape[0] * img.shape[1]),
        })
        rows.append(rec)

    report = pd.DataFrame(rows)
    report.to_csv(out_root / "r1_debug_report.csv", index=False, encoding="utf-8-sig")

    ok = report[report["status"] == "ok"]
    print(f"\nR1 debug OK: {len(ok)}/{len(report)}")
    if len(ok):
        print(f"core area mean   = {ok['core_area_ratio'].mean():.3f}")
        print(f"R1 crop area mean= {ok['roi_area_ratio'].mean():.3f}")
    print("debug:", out_root)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset_dir",
        default=r"G:\Project\sinra_cho\crop_data_yolox_s_512_area",
    )
    p.add_argument(
        "--work_dir",
        default=r"G:\Project\sinra_cho\attention_current512",
    )
    p.add_argument("--task", required=True, choices=["init", "train", "debug"])
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tune_n", type=int, default=300)

    p.add_argument("--localizer_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--min_delta", type=float, default=1e-4)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--no_amp", action="store_true")

    p.add_argument("--r1_tau", type=float, default=50.0)
    p.add_argument("--r1_crop_scale", type=float, default=1.80)
    p.add_argument("--r1_min_side", type=float, default=0.28)
    p.add_argument("--r1_max_side", type=float, default=0.42)
    p.add_argument("--roi_output_size", type=int, default=512)

    p.add_argument("--debug_n", type=int, default=100)
    p.add_argument("--clear_debug", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed, True)

    dataset_dir = Path(args.dataset_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    image_dir = dataset_dir / "training"

    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)

    train_df, tune_df = ensure_split(dataset_dir, work_dir, args.tune_n, args.seed)

    print("=" * 80)
    print("CURRENT 512 R1 ATTENTION")
    print("task           :", args.task)
    print("dataset        :", dataset_dir)
    print("work           :", work_dir)
    print("R1 tau         :", args.r1_tau)
    print(
        "R1 crop        :",
        f"scale={args.r1_crop_scale}, "
        f"min_side={args.r1_min_side}, max_side={args.r1_max_side}"
    )
    print("official val   : NOT USED")
    print("held-out test  : NOT USED")
    print("=" * 80)

    if args.task == "init":
        print("init 완료. 다음: --task train")
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 사용 불가")

    if args.task == "train":
        train_r1(args, train_df, tune_df, image_dir, work_dir, device)
        return

    if args.task == "debug":
        debug_r1(args, tune_df, image_dir, work_dir, device)
        return


if __name__ == "__main__":
    main()