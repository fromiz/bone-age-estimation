# -*- coding: utf-8 -*-
r"""
fit_v11_sex_calibration_from_cache.py

V11의 기존 모델 가중치는 건드리지 않고,
내부 validation 예측만 사용해 calibration을 비교/생성합니다.

비교:
1) RAW V11
2) Global affine calibration (5-fold OOF)
3) Sex-specific affine calibration (5-fold OOF)
4) Sex-specific median-shift calibration (5-fold OOF)

중요:
- Enterprise/Test GT는 절대 사용하지 않습니다.
- V11 validation cache 이미지를 그대로 읽습니다.
- TTA = [0, -6, -3, +3, +6] 유지
- OOF 결과로 calibration 방식의 일반화 가능성을 먼저 확인합니다.
- 최종 calibration 계수는 validation 전체에 다시 fit하여 JSON 저장합니다.

기본 경로:
G:\Project\sinra_cho\cache_convnext_single_v11\
  raw768x512_letterbox_pad0_center_mp1p99\val

실행:
python .\fit_v11_sex_calibration_from_cache.py --device cuda:0

checkpoint/csv 자동탐색 실패 시:
python .\fit_v11_sex_calibration_from_cache.py `
  --checkpoint "G:\...\best_model.pt" `
  --val_csv "G:\...\validation.csv" `
  --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import timm
from tqdm import tqdm


BASE = Path(r"G:\Project\sinra_cho")

DEFAULT_CACHE = (
    BASE
    / "cache_convnext_single_v11"
    / "raw768x512_letterbox_pad0_center_mp1p99"
    / "val"
)

DEFAULT_CKPT_ROOT = BASE / "checkpoints_convnext_single_v11_ldl"

OUTPUT_DIR = BASE / "v11_sex_calibration"

TTA_ANGLES = (0, -6, -3, 3, 6)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PAD_NORM = [-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]

EVAL_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(s):
    if s == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    d = torch.device(s)
    if d.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 사용 불가")
    return d


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def restore_float32(state):
    out = {}
    for k, v in state.items():
        if torch.is_tensor(v) and v.dtype == torch.float16:
            out[k] = v.float()
        else:
            out[k] = v
    return out


def create_backbone(model_name: str, drop_path: float = 0.0):
    return timm.create_model(
        model_name,
        pretrained=False,
        num_classes=0,
        global_pool="",
        drop_path_rate=drop_path,
    )


class GenderFiLM(nn.Module):
    def __init__(self, n_ch: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_ch * 2),
        )

    def forward(self, feat, g):
        gamma, beta = self.net(g).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return feat * (1.0 + gamma) + beta


class ConvNeXtRegressor(nn.Module):
    def __init__(
        self,
        *,
        backbone_name: str,
        img_hw: Tuple[int, int],
        head_type: str = "gap",
        head_dim: int = 512,
        gender_dim: int = 32,
        dropout: float = 0.10,
        drop_path: float = 0.15,
        use_ldl: bool = True,
        n_bins: int = 240,
        use_film: bool = True,
        film_stages: Iterable[int] = (2, 3),
        film_hidden: int = 64,
    ):
        super().__init__()

        self.backbone = create_backbone(backbone_name, drop_path)
        self.head_type = head_type

        self.use_film = bool(use_film) and hasattr(self.backbone, "stages")
        self.film = nn.ModuleDict()

        if self.use_film:
            with torch.no_grad():
                z = self.backbone.stem(torch.zeros(1, 3, 64, 64))
                channels = []
                for stage in self.backbone.stages:
                    z = stage(z)
                    channels.append(int(z.shape[1]))

            for i in film_stages:
                if 0 <= i < len(channels):
                    self.film[str(i)] = GenderFiLM(
                        channels[i], hidden=film_hidden
                    )

        height, width = int(img_hw[0]), int(img_hw[1])

        with torch.no_grad():
            feature = self._backbone_forward(
                torch.zeros(1, 3, height, width),
                torch.zeros(1, 1),
            )

        C = int(feature.shape[1])
        feat_h = int(feature.shape[2])
        feat_w = int(feature.shape[3])

        if head_type == "gap":
            self.norm = nn.LayerNorm(C)
            self.proj = nn.Sequential(
                nn.Linear(C, head_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            image_out = head_dim

        elif head_type == "paper":
            self.conv = nn.Conv2d(C, 256, 3, padding=1)
            self.pool = nn.MaxPool2d(3, 3)
            self.drop = nn.Dropout(dropout)
            image_out = 256 * (feat_h // 3) * (feat_w // 3)

        else:
            raise ValueError(f"지원하지 않는 head_type: {head_type}")

        self.use_ldl = bool(use_ldl)
        self.n_bins = int(n_bins)
        n_out = self.n_bins if self.use_ldl else 1

        self.gender = nn.Sequential(
            nn.Linear(1, gender_dim),
            nn.GELU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(image_out + gender_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_out),
        )

    def _backbone_forward(self, x, g):
        if not self.use_film:
            return self.backbone(x)

        z = self.backbone.stem(x)

        for i, stage in enumerate(self.backbone.stages):
            z = stage(z)
            key = str(i)
            if key in self.film:
                z = self.film[key](z, g)

        z = self.backbone.norm_pre(z)

        if hasattr(self.backbone, "forward_head"):
            z = self.backbone.forward_head(z)
        elif hasattr(self.backbone, "head"):
            z = self.backbone.head(z)

        return z

    def forward(self, x, g):
        f = self._backbone_forward(x, g)

        if self.head_type == "gap":
            z = F.adaptive_avg_pool2d(f, 1).flatten(1)
            z = self.proj(self.norm(z))
        else:
            z = self.drop(
                torch.flatten(
                    self.pool(F.relu(self.conv(f))),
                    1,
                )
            )

        e = self.gender(g)
        return self.fc(torch.cat([z, e], dim=1))


def out_to_months(out, age_mean, age_std):
    out = out.float()

    if out.ndim == 2 and out.size(1) > 1:
        p = torch.softmax(out, dim=1)
        bins = torch.arange(
            1,
            out.size(1) + 1,
            dtype=torch.float32,
            device=out.device,
        )
        return (p * bins.unsqueeze(0)).sum(1)

    if out.ndim == 2:
        out = out.squeeze(1)

    return out * age_std + age_mean


def is_v11_checkpoint(path: Path):
    try:
        c = torch_load(path)
    except Exception:
        return None

    if not isinstance(c, dict):
        return None

    if "model" not in c or "arch" not in c:
        return None

    a = c["arch"]
    if int(a.get("IMG_H", -1)) != 768 or int(a.get("IMG_W", -1)) != 512:
        return None

    return c


def find_checkpoint(explicit: str | None):
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(p)
        return p

    candidates = []

    roots = [
        DEFAULT_CKPT_ROOT,
        BASE / "cache_convnext_single_v11",
        BASE,
    ]

    seen = set()

    for root in roots:
        if not root.exists():
            continue

        pattern = "*.pt" if root == BASE else "**/*.pt"

        for p in root.glob(pattern):
            if p in seen:
                continue
            seen.add(p)

            c = is_v11_checkpoint(p)
            if c is None:
                continue

            val_mae = float(c.get("val_mae", 9999.0))
            candidates.append((val_mae, p))

    if not candidates:
        raise FileNotFoundError(
            "V11 checkpoint 자동탐색 실패. --checkpoint로 지정하세요."
        )

    candidates.sort(key=lambda x: x[0])
    print("[AUTO] V11 checkpoint 후보:")
    for mae, p in candidates[:5]:
        print(f"  val_mae={mae:.6f} | {p}")

    return candidates[0][1]


def load_model(checkpoint_path: Path, device):
    c = torch_load(checkpoint_path)
    a = dict(c["arch"])

    backbone = (
        a.get("BACKBONE_RESOLVED")
        or a.get("BACKBONE")
        or "convnext_tiny.fb_in22k_ft_in1k_384"
    )

    model = ConvNeXtRegressor(
        backbone_name=backbone,
        img_hw=(int(a["IMG_H"]), int(a["IMG_W"])),
        head_type=a.get("HEAD_TYPE", "gap"),
        head_dim=int(a.get("HEAD_DIM", 512)),
        gender_dim=int(a.get("GENDER_EMB_DIM", 32)),
        dropout=float(a.get("DROPOUT", 0.10)),
        drop_path=float(a.get("DROP_PATH", 0.15)),
        use_ldl=bool(a.get("USE_LDL", True)),
        n_bins=int(a.get("AGE_BINS", 240)),
        use_film=bool(a.get("USE_FILM", True)),
        film_stages=tuple(a.get("FILM_STAGES", (2, 3))),
        film_hidden=int(a.get("FILM_HIDDEN", 64)),
    )

    model.load_state_dict(
        restore_float32(c["model"]),
        strict=True,
    )
    model.eval().to(device)
    model.to(memory_format=torch.channels_last)

    return model, c


def normalize_name(x):
    return (
        str(x).strip().lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def find_col(df, aliases):
    mapping = {normalize_name(c): c for c in df.columns}

    for a in aliases:
        key = normalize_name(a)
        if key in mapping:
            return mapping[key]

    return None


def parse_male(v):
    if pd.isna(v):
        return np.nan

    s = str(v).strip().lower()

    if s in {"1", "1.0", "m", "male", "남", "남성", "boy"}:
        return 1.0
    if s in {"0", "0.0", "f", "female", "여", "여성", "girl"}:
        return 0.0

    try:
        z = float(v)
        if z in (0.0, 1.0):
            return z
    except Exception:
        pass

    return np.nan


def load_label_csv(path: Path):
    df = pd.read_csv(path)

    id_col = find_col(
        df,
        ["id", "image_id", "imageid", "case_id", "patient_id"],
    )
    age_col = find_col(
        df,
        ["boneage", "bone_age", "bone age", "label", "target"],
    )
    sex_col = find_col(
        df,
        ["male", "sex", "gender"],
    )

    if id_col is None or age_col is None or sex_col is None:
        raise ValueError(
            f"필수 열을 못 찾음: {path}\ncolumns={list(df.columns)}"
        )

    out = pd.DataFrame({
        "id": df[id_col].astype(str).str.strip(),
        "boneage": pd.to_numeric(df[age_col], errors="coerce"),
        "male": df[sex_col].map(parse_male),
    })

    out = out.dropna().copy()

    # 1234.0 형태 방지
    out["id"] = out["id"].str.replace(r"\.0$", "", regex=True)

    return out


def collect_cache(cache_dir: Path):
    files = [
        p for p in cache_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    if not files:
        raise FileNotFoundError(f"cache 이미지 없음: {cache_dir}")

    rows = []
    for p in files:
        rows.append({
            "id": p.stem,
            "image_path": str(p),
        })

    return pd.DataFrame(rows)


def auto_find_val_csv(cache_df, explicit: str | None):
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(p)
        return p

    candidates = [
        BASE / "crop_yolo_seg" / "csv" / "validation.csv",
        BASE / "crop_yolo_seg_maskedp_bgremove_512" / "csv" / "validation.csv",
        BASE / "crop_yolo_seg_bgremove_512" / "csv" / "validation.csv",
        BASE / "boneage-validation-dataset" / "Validation Dataset.csv",
        BASE / "boneage-validation-dataset" / "validation.csv",
    ]

    cache_ids = set(cache_df["id"].astype(str))
    scored = []

    for p in candidates:
        if not p.is_file():
            continue
        try:
            labels = load_label_csv(p)
        except Exception:
            continue

        match = len(cache_ids & set(labels["id"].astype(str)))
        scored.append((match, p))

    if not scored:
        raise FileNotFoundError(
            "validation CSV 자동탐색 실패. --val_csv로 지정하세요."
        )

    scored.sort(reverse=True, key=lambda x: x[0])

    print("[AUTO] validation CSV 후보:")
    for n, p in scored:
        print(f"  matched={n}/{len(cache_ids)} | {p}")

    return scored[0][1]


class CacheDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]

        gray = cv2.imread(
            str(r["image_path"]),
            cv2.IMREAD_GRAYSCALE,
        )

        if gray is None:
            raise RuntimeError(r["image_path"])

        if gray.shape != (768, 512):
            raise ValueError(
                f"V11 cache shape expected 768x512, got {gray.shape}: "
                f"{r['image_path']}"
            )

        rgb = np.stack([gray] * 3, axis=-1)
        x = EVAL_TRANSFORM(rgb)

        return {
            "image": x,
            "male": torch.tensor(
                [float(r["male"])],
                dtype=torch.float32,
            ),
            "age": torch.tensor(
                float(r["boneage"]),
                dtype=torch.float32,
            ),
            "id": str(r["id"]),
        }


@torch.no_grad()
def predict_validation(
    model,
    loader,
    device,
    age_mean,
    age_std,
    amp=True,
):
    rows = []
    use_amp = bool(amp and device.type == "cuda")

    for b in tqdm(loader, desc="V11 validation TTA", ncols=120):
        images = b["image"].to(
            device,
            non_blocking=True,
        ).to(memory_format=torch.channels_last)

        male = b["male"].to(
            device,
            non_blocking=True,
        )

        accumulated = 0.0

        with torch.amp.autocast(
            "cuda",
            enabled=use_amp,
        ):
            for angle in TTA_ANGLES:
                rotated = (
                    images
                    if angle == 0
                    else TF.rotate(
                        images,
                        angle,
                        fill=PAD_NORM,
                        interpolation=TF.InterpolationMode.BILINEAR,
                    )
                )

                accumulated = accumulated + out_to_months(
                    model(rotated, male),
                    age_mean,
                    age_std,
                )

        pred = (
            accumulated / len(TTA_ANGLES)
        ).float().cpu().numpy()

        ages = b["age"].numpy()
        males = b["male"].numpy().reshape(-1)
        ids = list(b["id"])

        for iid, s, y, p in zip(ids, males, ages, pred):
            rows.append({
                "id": str(iid),
                "male": float(s),
                "true_age": float(y),
                "pred_raw": float(p),
            })

    return pd.DataFrame(rows)


def metrics(y, p):
    e = p - y
    ae = np.abs(e)

    return {
        "MAE": float(ae.mean()),
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
        "Bias": float(e.mean()),
        "MedianAE": float(np.median(ae)),
        "P90": float(np.percentile(ae, 90)),
    }


def fit_affine_pred_on_true(y, p):
    """
    기존 V11 convention:
        pred ~= a * true + b
        corrected = (pred - b) / a
    """
    A = np.column_stack([y, np.ones_like(y)])
    a, b = np.linalg.lstsq(A, p, rcond=None)[0]

    if abs(a) < 1e-6:
        a = 1.0

    return float(a), float(b)


def apply_affine(p, a, b):
    return (p - b) / a


def make_stratified_folds(df, n_splits=5, seed=42):
    rng = np.random.default_rng(seed)
    folds = np.full(len(df), -1, dtype=int)

    for sex in (0.0, 1.0):
        idx = np.where(
            np.isclose(df["male"].to_numpy(float), sex)
        )[0]
        rng.shuffle(idx)

        for j, i in enumerate(idx):
            folds[i] = j % n_splits

    if np.any(folds < 0):
        raise RuntimeError("fold assignment 실패")

    return folds


def oof_calibrations(df, n_splits=5, seed=42):
    y = df["true_age"].to_numpy(float)
    p = df["pred_raw"].to_numpy(float)
    male = df["male"].to_numpy(float)

    folds = make_stratified_folds(
        df,
        n_splits=n_splits,
        seed=seed,
    )

    pred_global = np.zeros_like(p)
    pred_sex_affine = np.zeros_like(p)
    pred_sex_shift = np.zeros_like(p)

    for fold in range(n_splits):
        tr = folds != fold
        va = folds == fold

        # global affine
        a, b = fit_affine_pred_on_true(
            y[tr], p[tr]
        )
        pred_global[va] = apply_affine(
            p[va], a, b
        )

        # sex-specific
        for sex in (0.0, 1.0):
            trs = tr & np.isclose(male, sex)
            vas = va & np.isclose(male, sex)

            a_s, b_s = fit_affine_pred_on_true(
                y[trs], p[trs]
            )
            pred_sex_affine[vas] = apply_affine(
                p[vas], a_s, b_s
            )

            # MAE-friendly additive correction:
            # median(pred - true) is the L1-optimal constant shift.
            shift = float(
                np.median(
                    p[trs] - y[trs]
                )
            )
            pred_sex_shift[vas] = p[vas] - shift

    return {
        "raw": p,
        "global_affine": pred_global,
        "sex_affine": pred_sex_affine,
        "sex_median_shift": pred_sex_shift,
    }


def fit_full_calibrators(df):
    y = df["true_age"].to_numpy(float)
    p = df["pred_raw"].to_numpy(float)
    male = df["male"].to_numpy(float)

    ga, gb = fit_affine_pred_on_true(y, p)

    result = {
        "global_affine": {
            "a": ga,
            "b": gb,
        },
        "sex_affine": {},
        "sex_median_shift": {},
    }

    for sex, name in [(0.0, "female"), (1.0, "male")]:
        m = np.isclose(male, sex)

        a, b = fit_affine_pred_on_true(
            y[m], p[m]
        )

        shift = float(
            np.median(
                p[m] - y[m]
            )
        )

        result["sex_affine"][name] = {
            "a": a,
            "b": b,
        }

        result["sex_median_shift"][name] = {
            "shift": shift,
        }

    return result


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--cache_val",
        default=str(DEFAULT_CACHE),
    )
    ap.add_argument(
        "--checkpoint",
        default=None,
    )
    ap.add_argument(
        "--val_csv",
        default=None,
    )
    ap.add_argument(
        "--output_dir",
        default=str(OUTPUT_DIR),
    )
    ap.add_argument(
        "--device",
        default="auto",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
    )
    ap.add_argument(
        "--folds",
        type=int,
        default=5,
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    ap.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = ap.parse_args()

    set_seed(args.seed)

    cache_dir = Path(args.cache_val).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not cache_dir.is_dir():
        raise FileNotFoundError(cache_dir)

    cache_df = collect_cache(cache_dir)

    ckpt_path = find_checkpoint(args.checkpoint)
    val_csv = auto_find_val_csv(
        cache_df,
        args.val_csv,
    )

    labels = load_label_csv(val_csv)

    merged = cache_df.merge(
        labels,
        on="id",
        how="inner",
    )

    print()
    print("=" * 100)
    print("V11 VALIDATION-ONLY CALIBRATION")
    print("=" * 100)
    print("cache       :", cache_dir)
    print("cache N     :", len(cache_df))
    print("label CSV   :", val_csv)
    print("matched N   :", len(merged))
    print("checkpoint  :", ckpt_path)
    print("Enterprise  : NOT USED")
    print("Test        : NOT USED")
    print("=" * 100)

    if len(merged) < 100:
        raise RuntimeError(
            "매칭 수가 너무 적습니다. validation CSV를 확인하세요."
        )

    device = resolve_device(args.device)

    model, ckpt = load_model(
        ckpt_path,
        device,
    )

    ds = CacheDataset(merged)

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    pred_df = predict_validation(
        model,
        dl,
        device,
        float(ckpt.get("age_mean", 0.0)),
        float(ckpt.get("age_std", 1.0)),
        amp=args.amp,
    )

    oof = oof_calibrations(
        pred_df,
        n_splits=args.folds,
        seed=args.seed,
    )

    y = pred_df["true_age"].to_numpy(float)

    comparison_rows = []

    for name, pred in oof.items():
        m = metrics(y, pred)
        comparison_rows.append({
            "method": name,
            **m,
        })
        pred_df[f"pred_{name}"] = pred

    comp = pd.DataFrame(
        comparison_rows
    ).sort_values("MAE")

    print()
    print("5-fold OOF comparison")
    print(
        comp.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # Do not choose RAW as calibration output.
    candidates = comp[
        comp["method"] != "raw"
    ].copy()

    best_method = str(
        candidates.iloc[0]["method"]
    )

    raw_mae = float(
        comp.loc[
            comp["method"] == "raw",
            "MAE",
        ].iloc[0]
    )

    best_oof_mae = float(
        candidates.iloc[0]["MAE"]
    )

    full = fit_full_calibrators(
        pred_df
    )

    payload = {
        "used": True,
        "type": best_method,
        "fitted_on": "validation_only",
        "selection": "5-fold sex-stratified OOF MAE",
        "raw_validation_mae": raw_mae,
        "best_oof_mae": best_oof_mae,
        "oof_improvement_months": raw_mae - best_oof_mae,
        "tta_angles": list(TTA_ANGLES),
        "img_h": 768,
        "img_w": 512,
        "global_affine": full["global_affine"],
        "sex_affine": full["sex_affine"],
        "sex_median_shift": full["sex_median_shift"],
        "enterprise_used": False,
        "test_used": False,
    }

    # For direct inference patch, include active coefficients at top level.
    if best_method == "global_affine":
        payload.update(
            full["global_affine"]
        )

    elif best_method == "sex_affine":
        payload["female"] = full["sex_affine"]["female"]
        payload["male"] = full["sex_affine"]["male"]

    elif best_method == "sex_median_shift":
        payload["female"] = full["sex_median_shift"]["female"]
        payload["male"] = full["sex_median_shift"]["male"]

    pred_df.to_csv(
        out_dir / "v11_validation_predictions_calibration.csv",
        index=False,
        encoding="utf-8-sig",
    )

    comp.to_csv(
        out_dir / "calibration_oof_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    json_path = out_dir / "calibration_v11_best.json"

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 100)
    print("RESULT")
    print("=" * 100)
    print("RAW Val MAE      :", f"{raw_mae:.4f}")
    print("Best OOF method  :", best_method)
    print("Best OOF Val MAE :", f"{best_oof_mae:.4f}")
    print("OOF improvement  :", f"{raw_mae - best_oof_mae:+.4f} months")
    print()
    print("JSON :", json_path)
    print(
        "※ OOF에서도 RAW보다 좋아질 때만 최종 적용하는 것을 권장합니다."
    )
    print("※ Enterprise/Test GT는 사용하지 않았습니다.")
    print("=" * 100)


if __name__ == "__main__":
    main()