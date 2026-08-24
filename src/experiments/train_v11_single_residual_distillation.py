# -*- coding: utf-8 -*-
r"""
train_v11_single_residual_distillation.py

목적
----
V11 + Run138 50:50 ensemble의 보완 이득을
기업 추론에서는 V11 계열 단일 모델 하나로 흡수하는 빠른 distillation 실험.

핵심 구조
--------
V11 base (고정)
  └─ 5-angle TTA에서 얻은 128-d hidden feature 평균
  └─ V11 TTA prediction
                ↓
        작은 residual head
                ↓
final_pred = V11_pred + correction

학습 teacher
-----------
teacher_pred = 0.5 * V11_pred + 0.5 * Run138_pred

즉 residual head는
  0.5 * (Run138_pred - V11_pred)
를 V11 feature만 보고 추정하도록 학습한다.

중요
----
- V11 / Run138 backbone은 둘 다 freeze.
- Run138은 TRAIN teacher prediction 생성에만 사용.
- 최종 배포 시 Run138은 필요 없음.
- Enterprise/Test GT 절대 사용 안 함.
- validation으로 student 후보를 선택.
- student가 V11 RAW보다 개선되지 않으면 버린다.

자동 비교 후보
-------------
1) linear_distill
2) linear_hybrid025
3) mlp32_distill
4) mlp64_distill
5) mlp64_hybrid025
6) mlp64_hybrid050

hybrid loss:
  distill_loss
  + lambda_gt * GT SmoothL1
  + weak correction regularization

출력
----
G:\Project\sinra_cho\v11_single_residual_distillation\
  feature_cache_train.npz
  feature_cache_val.npz
  candidate_results.csv
  best_student.pt
  best_validation_predictions.csv
  summary.json

필요 파일
--------
같은 서버 폴더:
  fit_v11_sex_calibration_from_cache_FIXED.py
  train_run138_male_head_only_bias_correction.py

기본 경로
--------
V11 checkpoint:
  G:\Project\sinra_cho\checkpoints_convnext_single_v11_ldl\best.pt

Run138 checkpoint:
  G:\Project\sinra_cho\convnext_tiny_512_results\
  138_male_head_only_bias_correction\best_mae_model.pt

V11 cache:
  G:\Project\sinra_cho\cache_convnext_single_v11\...\train, val

Run138 data:
  G:\Project\sinra_cho\crop_yolo_seg_maskedp_bgremove_512

실행
----
python .\train_v11_single_residual_distillation.py --device cuda:0
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
import timm
from timm.data import resolve_model_data_config
from tqdm import tqdm

import fit_v11_sex_calibration_from_cache_FIXED as v11
import train_run138_male_head_only_bias_correction as r138


BASE = Path(r"G:\Project\sinra_cho")

V11_CACHE_ROOT = BASE / "cache_convnext_single_v11"

V11_CKPT = (
    BASE
    / "checkpoints_convnext_single_v11_ldl"
    / "best.pt"
)

RUN138_ROOT = (
    BASE
    / "convnext_tiny_512_results"
    / "138_male_head_only_bias_correction"
)

RUN138_CKPT = RUN138_ROOT / "best_mae_model.pt"

RUN138_DATA = (
    BASE
    / "crop_yolo_seg_maskedp_bgremove_512"
)

OUTPUT_ROOT = (
    BASE
    / "v11_single_residual_distillation"
)

TTA_ANGLES = (0, -6, -3, 3, 6)


# =============================================================================
# General
# =============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True


def resolve_device(spec):
    if spec == "auto":
        return torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

    d = torch.device(spec)

    if d.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 사용 불가")

    return d


def norm_ids(s):
    return (
        s.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def metrics(y, p):
    e = p - y
    ae = np.abs(e)

    return {
        "MAE": float(ae.mean()),
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
        "Bias": float(e.mean()),
        "MedianAE": float(np.median(ae)),
        "P90": float(np.percentile(ae, 90)),
        "P95": float(np.percentile(ae, 95)),
        "MaxAE": float(ae.max()),
    }


def subgroup_metrics(df, pred_col):
    male = df["male"].to_numpy(float)
    y = df["true_age"].to_numpy(float)
    p = df[pred_col].to_numpy(float)

    groups = [
        ("Overall", np.ones(len(df), dtype=bool)),
        ("Female", male < 0.5),
        ("Male", male >= 0.5),
        (
            "Male_le60",
            (male >= 0.5) & (y <= 60),
        ),
        (
            "Male_61_96",
            (male >= 0.5) & (y > 60) & (y <= 96),
        ),
        (
            "Male_97_144",
            (male >= 0.5) & (y > 96) & (y <= 144),
        ),
        (
            "Male_gt144",
            (male >= 0.5) & (y > 144),
        ),
    ]

    rows = []

    for name, m in groups:
        if not np.any(m):
            continue

        mm = metrics(
            y[m],
            p[m],
        )

        rows.append({
            "group": name,
            "N": int(m.sum()),
            **mm,
        })

    return pd.DataFrame(rows)


# =============================================================================
# Locate V11 caches
# =============================================================================

def find_v11_split(split_name):
    aliases = {
        "train": {"train"},
        "val": {"val", "validation"},
    }[split_name]

    candidates = []

    if not V11_CACHE_ROOT.is_dir():
        raise FileNotFoundError(V11_CACHE_ROOT)

    for p in V11_CACHE_ROOT.rglob("*"):
        if not p.is_dir():
            continue

        if p.name.lower() not in aliases:
            continue

        parent = str(p.parent).lower()

        score = 0.0

        for token in (
            "raw768x512",
            "letterbox",
            "mp1p99",
        ):
            if token in parent:
                score += 10.0

        try:
            n = sum(
                1
                for f in p.iterdir()
                if f.is_file()
                and f.suffix.lower()
                in {
                    ".png", ".jpg", ".jpeg",
                    ".bmp", ".tif", ".tiff",
                }
            )
        except Exception:
            n = 0

        if n > 0:
            score += min(n, 20000) / 1000.0
            candidates.append(
                (score, n, p)
            )

    if not candidates:
        raise FileNotFoundError(
            f"V11 {split_name} cache 자동 탐색 실패"
        )

    candidates.sort(
        key=lambda x: (x[0], x[1]),
        reverse=True,
    )

    print(
        f"[AUTO] V11 {split_name} cache:"
    )

    for score, n, p in candidates[:5]:
        print(
            f"  score={score:.3f} N={n} | {p}"
        )

    return candidates[0][2]


def collect_cache(cache_dir):
    rows = []

    for p in cache_dir.iterdir():
        if (
            p.is_file()
            and p.suffix.lower()
            in {
                ".png", ".jpg", ".jpeg",
                ".bmp", ".tif", ".tiff",
            }
        ):
            rows.append({
                "id": p.stem,
                "image_path": str(p),
            })

    if not rows:
        raise RuntimeError(
            f"cache 이미지 없음: {cache_dir}"
        )

    out = pd.DataFrame(rows)
    out["id"] = norm_ids(out["id"])
    return out


def load_labels(csv_path):
    df = v11.load_label_csv(
        csv_path
    )

    df["id"] = norm_ids(
        df["id"]
    )

    return df


# =============================================================================
# V11 hidden feature / prediction cache
# =============================================================================

def v11_hidden_and_output(model, x, g):
    """
    Exact V11 forward path + penultimate 128-d feature.
    model.eval() 상태에서 dropout은 no-op.
    """
    f = model._backbone_forward(
        x,
        g,
    )

    if model.head_type == "gap":
        z = F.adaptive_avg_pool2d(
            f,
            1,
        ).flatten(1)

        z = model.proj(
            model.norm(z)
        )

    else:
        z = model.drop(
            torch.flatten(
                model.pool(
                    F.relu(
                        model.conv(f)
                    )
                ),
                1,
            )
        )

    gender = model.gender(g)

    fusion = torch.cat(
        [z, gender],
        dim=1,
    )

    h = model.fc[0](fusion)
    h = model.fc[1](h)
    h = model.fc[2](h)

    out = model.fc[3](h)

    return h, out


@torch.no_grad()
def cache_v11_features(
    model,
    loader,
    device,
    age_mean,
    age_std,
    out_npz,
    use_amp,
):
    all_id = []
    all_male = []
    all_age = []
    all_pred = []
    all_hidden = []

    amp_enabled = bool(
        use_amp
        and device.type == "cuda"
    )

    for b in tqdm(
        loader,
        desc="V11 TTA feature cache",
        ncols=125,
    ):
        x = (
            b["image"]
            .to(device, non_blocking=True)
            .to(memory_format=torch.channels_last)
        )

        male = b["male"].to(
            device,
            non_blocking=True,
        )

        pred_sum = 0.0
        hidden_sum = 0.0

        with torch.amp.autocast(
            "cuda",
            enabled=amp_enabled,
        ):
            for angle in TTA_ANGLES:
                xx = (
                    x
                    if angle == 0
                    else TF.rotate(
                        x,
                        angle,
                        fill=v11.PAD_NORM,
                        interpolation=TF.InterpolationMode.BILINEAR,
                    )
                )

                h, out = v11_hidden_and_output(
                    model,
                    xx,
                    male,
                )

                pred = v11.out_to_months(
                    out,
                    age_mean,
                    age_std,
                )

                pred_sum = pred_sum + pred.float()
                hidden_sum = hidden_sum + h.float()

        pred = (
            pred_sum / len(TTA_ANGLES)
        ).cpu().numpy()

        hidden = (
            hidden_sum / len(TTA_ANGLES)
        ).cpu().numpy()

        all_id.extend(
            list(b["id"])
        )
        all_male.append(
            b["male"].numpy().reshape(-1)
        )
        all_age.append(
            b["age"].numpy().reshape(-1)
        )
        all_pred.append(pred)
        all_hidden.append(hidden)

    ids = np.array(
        [str(x) for x in all_id],
        dtype=str,
    )

    male = np.concatenate(
        all_male
    ).astype(np.float32)

    age = np.concatenate(
        all_age
    ).astype(np.float32)

    pred = np.concatenate(
        all_pred
    ).astype(np.float32)

    hidden = np.concatenate(
        all_hidden
    ).astype(np.float32)

    np.savez_compressed(
        out_npz,
        id=ids,
        male=male,
        age=age,
        pred_v11=pred,
        hidden=hidden,
    )

    return {
        "id": ids,
        "male": male,
        "age": age,
        "pred_v11": pred,
        "hidden": hidden,
    }


def load_npz_dict(path):
    d = np.load(
        path,
        allow_pickle=False,
    )

    return {
        k: d[k]
        for k in d.files
    }


# =============================================================================
# Run138 teacher prediction cache
# =============================================================================

def load_run138_model(
    ckpt_path,
    device,
):
    c = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    config = c.get(
        "config",
        {},
    )

    model = (
        r138.run106
        .ConvNeXtTinyDistributionRegression(
            model_name=config.get(
                "model_name",
                "convnext_tiny.fb_in1k",
            ),
            image_dim=int(
                config.get(
                    "image_dim",
                    512,
                )
            ),
            sex_dim=int(
                config.get(
                    "sex_dim",
                    32,
                )
            ),
            fusion_dim=int(
                config.get(
                    "fusion_dim",
                    128,
                )
            ),
            image_dropout=float(
                config.get(
                    "image_dropout",
                    0.20,
                )
            ),
            fusion_dropout=float(
                config.get(
                    "fusion_dropout",
                    0.20,
                )
            ),
            pretrained=False,
            num_bins=int(
                config.get(
                    "num_bins",
                    240,
                )
            ),
        )
        .to(device)
    )

    model.load_state_dict(
        c["model_state_dict"],
        strict=True,
    )

    model.eval()

    return model, config


def build_run138_eval_transform(
    model_name,
    image_size,
):
    probe = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=0,
        global_pool="avg",
    )

    cfg = resolve_model_data_config(
        probe
    )

    del probe

    mean = cfg.get(
        "mean",
        (0.5, 0.5, 0.5),
    )
    std = cfg.get(
        "std",
        (0.5, 0.5, 0.5),
    )

    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std,
        ),
    ])


@torch.no_grad()
def predict_run138_split(
    split,
    model,
    config,
    device,
    batch_size,
    workers,
    amp,
):
    csv_path = (
        RUN138_DATA
        / "csv"
        / (
            "train.csv"
            if split == "train"
            else "validation.csv"
        )
    )

    img_dir = (
        RUN138_DATA
        / (
            "train"
            if split == "train"
            else "validation"
        )
        / "images"
    )

    df = (
        r138.run106
        .attach_image_paths(
            r138.run106
            .standardize_dataframe(
                csv_path
            ),
            img_dir,
            f"Run138 {split} teacher",
        )
    )

    tfm = build_run138_eval_transform(
        config.get(
            "model_name",
            "convnext_tiny.fb_in1k",
        ),
        int(
            config.get(
                "image_size",
                512,
            )
        ),
    )

    ds = r138.BoneAgeDataset(
        df,
        tfm,
    )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )

    criterion = (
        r138.run106
        .DistributionAgeLoss(
            sigma=float(
                config.get(
                    "sigma",
                    10.0,
                )
            ),
            lambda_kl=float(
                config.get(
                    "lambda_kl",
                    0.025,
                )
            ),
        )
        .to(device)
    )

    _, pred_df = r138.evaluate(
        model,
        dl,
        criterion,
        device,
        bool(
            amp
            and device.type == "cuda"
        ),
        f"Run138 {split} teacher",
    )

    pred_df["id"] = norm_ids(
        pred_df["id"]
    )

    return pred_df


# =============================================================================
# Merge feature/teacher tables
# =============================================================================

def v11_cache_to_df(cache):
    df = pd.DataFrame({
        "id": norm_ids(
            pd.Series(
                cache["id"].astype(str)
            )
        ),
        "male": cache["male"].astype(float),
        "true_age": cache["age"].astype(float),
        "pred_v11": cache["pred_v11"].astype(float),
    })

    return df


def merge_teacher(
    v11_cache,
    run138_pred,
):
    left = v11_cache_to_df(
        v11_cache
    )

    right = pd.DataFrame({
        "id": norm_ids(
            run138_pred["id"]
        ),
        "true_r138": pd.to_numeric(
            run138_pred["true_boneage"],
            errors="raise",
        ),
        "male_r138": pd.to_numeric(
            run138_pred["male"],
            errors="raise",
        ),
        "pred_run138": pd.to_numeric(
            run138_pred["pred_boneage"],
            errors="raise",
        ),
    })

    merged = left.merge(
        right,
        on="id",
        how="inner",
    )

    if len(merged) != len(left):
        raise RuntimeError(
            f"V11/Run138 ID merge 실패: "
            f"V11={len(left)}, merged={len(merged)}"
        )

    gt_diff = np.abs(
        merged["true_age"]
        - merged["true_r138"]
    ).max()

    sex_diff = np.abs(
        merged["male"]
        - merged["male_r138"]
    ).max()

    if gt_diff > 1e-4:
        raise RuntimeError(
            f"GT mismatch max={gt_diff}"
        )

    if sex_diff > 1e-4:
        raise RuntimeError(
            f"Sex mismatch max={sex_diff}"
        )

    # Preserve V11 cache order.
    index_map = {
        str(iid): i
        for i, iid in enumerate(
            v11_cache["id"].astype(str)
        )
    }

    merged["_idx"] = (
        merged["id"]
        .map(index_map)
    )

    merged = (
        merged
        .sort_values("_idx")
        .reset_index(drop=True)
    )

    return merged


# =============================================================================
# Residual student
# =============================================================================

class ResidualHead(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
    ):
        super().__init__()

        if hidden_dim <= 0:
            self.net = nn.Linear(
                input_dim,
                1,
            )

            nn.init.zeros_(
                self.net.weight
            )
            nn.init.zeros_(
                self.net.bias
            )

        else:
            self.net = nn.Sequential(
                nn.Linear(
                    input_dim,
                    hidden_dim,
                ),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(
                    hidden_dim,
                    1,
                ),
            )

            nn.init.zeros_(
                self.net[-1].weight
            )
            nn.init.zeros_(
                self.net[-1].bias
            )

    def forward(self, x):
        return self.net(x).squeeze(1)


def build_student_features(
    cache,
):
    hidden = cache[
        "hidden"
    ].astype(np.float32)

    pred = cache[
        "pred_v11"
    ].astype(np.float32).reshape(-1, 1)

    male = cache[
        "male"
    ].astype(np.float32).reshape(-1, 1)

    # hidden + base age + sex
    X = np.concatenate(
        [
            hidden,
            pred,
            male,
        ],
        axis=1,
    )

    return X


def standardize_train_val(
    X_train,
    X_val,
):
    mu = X_train.mean(
        axis=0,
    ).astype(np.float32)

    sd = X_train.std(
        axis=0,
    ).astype(np.float32)

    sd[
        sd < 1e-6
    ] = 1.0

    return (
        (X_train - mu) / sd,
        (X_val - mu) / sd,
        mu,
        sd,
    )


def train_candidate(
    name,
    hidden_dim,
    lambda_gt,
    X_train,
    y_train,
    v11_train,
    r138_train,
    X_val,
    y_val,
    v11_val,
    device,
    seed,
    max_epochs,
    patience,
    lr,
    batch_size,
):
    set_seed(seed)

    model = ResidualHead(
        X_train.shape[1],
        hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-3,
    )

    # Teacher correction exactly corresponds to 50:50 ensemble.
    target_corr = (
        0.5
        * (
            r138_train
            - v11_train
        )
    ).astype(np.float32)

    train_ds = TensorDataset(
        torch.from_numpy(
            X_train.astype(np.float32)
        ),
        torch.from_numpy(
            y_train.astype(np.float32)
        ),
        torch.from_numpy(
            v11_train.astype(np.float32)
        ),
        torch.from_numpy(
            target_corr
        ),
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    X_val_t = torch.from_numpy(
        X_val.astype(np.float32)
    ).to(device)

    best = None
    bad = 0
    history = []

    for epoch in range(
        1,
        max_epochs + 1,
    ):
        model.train()

        loss_sum = 0.0
        n_steps = 0

        for xb, yb, vb, cb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            vb = vb.to(device)
            cb = cb.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            corr = model(xb)

            # Conservative correction range.
            final = vb + torch.clamp(
                corr,
                -12.0,
                12.0,
            )

            distill = F.smooth_l1_loss(
                corr,
                cb,
                beta=2.0,
            )

            gt_loss = F.smooth_l1_loss(
                final,
                yb,
                beta=2.0,
            )

            reg = torch.mean(
                corr.square()
            )

            loss = (
                distill
                + float(lambda_gt) * gt_loss
                + 0.001 * reg
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

            loss_sum += float(
                loss.detach().cpu()
            )
            n_steps += 1

        model.eval()

        with torch.no_grad():
            corr_val = (
                model(X_val_t)
                .cpu()
                .numpy()
            )

        corr_val = np.clip(
            corr_val,
            -12.0,
            12.0,
        )

        pred_val = (
            v11_val
            + corr_val
        )

        vm = metrics(
            y_val,
            pred_val,
        )

        history.append({
            "epoch": epoch,
            "loss": loss_sum / max(
                1,
                n_steps,
            ),
            "val_mae": vm["MAE"],
            "val_rmse": vm["RMSE"],
            "val_bias": vm["Bias"],
            "mean_abs_correction": float(
                np.mean(
                    np.abs(corr_val)
                )
            ),
        })

        if (
            best is None
            or vm["MAE"]
            < best["metrics"]["MAE"]
            - 1e-5
        ):
            best = {
                "epoch": epoch,
                "metrics": vm,
                "state_dict": copy.deepcopy(
                    model.state_dict()
                ),
                "pred_val": pred_val.copy(),
                "corr_val": corr_val.copy(),
            }
            bad = 0
        else:
            bad += 1

        if bad >= patience:
            break

    model.load_state_dict(
        best["state_dict"]
    )

    return {
        "name": name,
        "hidden_dim": int(
            hidden_dim
        ),
        "lambda_gt": float(
            lambda_gt
        ),
        "best_epoch": int(
            best["epoch"]
        ),
        "metrics": best[
            "metrics"
        ],
        "state_dict": best[
            "state_dict"
        ],
        "pred_val": best[
            "pred_val"
        ],
        "corr_val": best[
            "corr_val"
        ],
        "history": history,
    }


# =============================================================================
# Main
# =============================================================================

def build_parser():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--device",
        default="auto",
    )

    ap.add_argument(
        "--v11_checkpoint",
        default=str(
            V11_CKPT
        ),
    )

    ap.add_argument(
        "--run138_checkpoint",
        default=str(
            RUN138_CKPT
        ),
    )

    ap.add_argument(
        "--output_dir",
        default=str(
            OUTPUT_ROOT
        ),
    )

    ap.add_argument(
        "--feature_batch_size",
        type=int,
        default=8,
    )

    ap.add_argument(
        "--teacher_batch_size",
        type=int,
        default=64,
    )

    ap.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--head_batch_size",
        type=int,
        default=512,
    )

    ap.add_argument(
        "--head_lr",
        type=float,
        default=3e-4,
    )

    ap.add_argument(
        "--head_epochs",
        type=int,
        default=150,
    )

    ap.add_argument(
        "--head_patience",
        type=int,
        default=12,
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

    ap.add_argument(
        "--rebuild_cache",
        action="store_true",
    )

    return ap


def main():
    args = build_parser().parse_args()

    set_seed(
        args.seed
    )

    device = resolve_device(
        args.device
    )

    out_dir = Path(
        args.output_dir
    ).resolve()

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    v11_ckpt = Path(
        args.v11_checkpoint
    ).resolve()

    r138_ckpt = Path(
        args.run138_checkpoint
    ).resolve()

    if not v11_ckpt.is_file():
        raise FileNotFoundError(
            v11_ckpt
        )

    if not r138_ckpt.is_file():
        raise FileNotFoundError(
            r138_ckpt
        )

    train_cache_path = (
        out_dir
        / "feature_cache_train.npz"
    )

    val_cache_path = (
        out_dir
        / "feature_cache_val.npz"
    )

    # ---------------------------------------------------------
    # V11 feature cache
    # ---------------------------------------------------------
    train_dir = find_v11_split(
        "train"
    )
    val_dir = find_v11_split(
        "val"
    )

    train_csv = (
        RUN138_DATA
        / "csv"
        / "train.csv"
    )

    val_csv = (
        RUN138_DATA
        / "csv"
        / "validation.csv"
    )

    train_labels = load_labels(
        train_csv
    )
    val_labels = load_labels(
        val_csv
    )

    def build_v11_df(
        cache_dir,
        labels,
    ):
        c = collect_cache(
            cache_dir
        )

        m = c.merge(
            labels,
            on="id",
            how="inner",
        )

        if len(m) != len(c):
            print(
                f"[WARNING] cache={len(c)} matched={len(m)}"
            )

        return m

    train_v11_df = build_v11_df(
        train_dir,
        train_labels,
    )

    val_v11_df = build_v11_df(
        val_dir,
        val_labels,
    )

    v11_model, v11_checkpoint = (
        v11.load_model(
            v11_ckpt,
            device,
        )
    )

    print()
    print("=" * 112)
    print("V11 SINGLE RESIDUAL DISTILLATION")
    print("=" * 112)
    print("V11 checkpoint :", v11_ckpt)
    print(
        "V11 ckpt valMAE:",
        v11_checkpoint.get(
            "val_mae"
        ),
    )
    print("V11 train N    :", len(train_v11_df))
    print("V11 val N      :", len(val_v11_df))
    print("Run138 ckpt    :", r138_ckpt)
    print("Enterprise     : NOT USED")
    print("Test           : NOT USED")
    print("=" * 112)

    if (
        args.rebuild_cache
        or not train_cache_path.is_file()
    ):
        train_ds = v11.CacheDataset(
            train_v11_df
        )

        train_dl = DataLoader(
            train_ds,
            batch_size=args.feature_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )

        train_cache = cache_v11_features(
            v11_model,
            train_dl,
            device,
            float(
                v11_checkpoint.get(
                    "age_mean",
                    0.0,
                )
            ),
            float(
                v11_checkpoint.get(
                    "age_std",
                    1.0,
                )
            ),
            train_cache_path,
            args.amp,
        )
    else:
        print(
            "[CACHE] V11 train feature cache 재사용"
        )
        train_cache = load_npz_dict(
            train_cache_path
        )

    if (
        args.rebuild_cache
        or not val_cache_path.is_file()
    ):
        val_ds = v11.CacheDataset(
            val_v11_df
        )

        val_dl = DataLoader(
            val_ds,
            batch_size=args.feature_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )

        val_cache = cache_v11_features(
            v11_model,
            val_dl,
            device,
            float(
                v11_checkpoint.get(
                    "age_mean",
                    0.0,
                )
            ),
            float(
                v11_checkpoint.get(
                    "age_std",
                    1.0,
                )
            ),
            val_cache_path,
            args.amp,
        )
    else:
        print(
            "[CACHE] V11 val feature cache 재사용"
        )
        val_cache = load_npz_dict(
            val_cache_path
        )

    # V11 no longer needed on GPU.
    del v11_model
    torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # Run138 deterministic teacher prediction
    # ---------------------------------------------------------
    run138_train_csv = (
        out_dir
        / "run138_teacher_train.csv"
    )

    run138_val_csv = (
        out_dir
        / "run138_teacher_val.csv"
    )

    if (
        args.rebuild_cache
        or not run138_train_csv.is_file()
        or not run138_val_csv.is_file()
    ):
        run138_model, run138_config = (
            load_run138_model(
                r138_ckpt,
                device,
            )
        )

        if (
            args.rebuild_cache
            or not run138_train_csv.is_file()
        ):
            r138_train = predict_run138_split(
                "train",
                run138_model,
                run138_config,
                device,
                args.teacher_batch_size,
                args.workers,
                args.amp,
            )

            r138_train.to_csv(
                run138_train_csv,
                index=False,
                encoding="utf-8-sig",
            )
        else:
            r138_train = pd.read_csv(
                run138_train_csv
            )

        if (
            args.rebuild_cache
            or not run138_val_csv.is_file()
        ):
            r138_val = predict_run138_split(
                "val",
                run138_model,
                run138_config,
                device,
                args.teacher_batch_size,
                args.workers,
                args.amp,
            )

            r138_val.to_csv(
                run138_val_csv,
                index=False,
                encoding="utf-8-sig",
            )
        else:
            r138_val = pd.read_csv(
                run138_val_csv
            )

        del run138_model
        torch.cuda.empty_cache()

    else:
        print(
            "[CACHE] Run138 teacher predictions 재사용"
        )

        r138_train = pd.read_csv(
            run138_train_csv
        )

        r138_val = pd.read_csv(
            run138_val_csv
        )

    # ---------------------------------------------------------
    # Align teacher predictions with V11 feature cache
    # ---------------------------------------------------------
    train_merge = merge_teacher(
        train_cache,
        r138_train,
    )

    val_merge = merge_teacher(
        val_cache,
        r138_val,
    )

    X_train_raw = build_student_features(
        train_cache
    )

    X_val_raw = build_student_features(
        val_cache
    )

    # merge_teacher preserves V11 cache order.
    X_train, X_val, feat_mu, feat_sd = (
        standardize_train_val(
            X_train_raw,
            X_val_raw,
        )
    )

    y_train = train_merge[
        "true_age"
    ].to_numpy(np.float32)

    y_val = val_merge[
        "true_age"
    ].to_numpy(np.float32)

    v11_train = train_merge[
        "pred_v11"
    ].to_numpy(np.float32)

    v11_val = val_merge[
        "pred_v11"
    ].to_numpy(np.float32)

    r138_train = train_merge[
        "pred_run138"
    ].to_numpy(np.float32)

    r138_val = val_merge[
        "pred_run138"
    ].to_numpy(np.float32)

    # Baselines.
    teacher_val = (
        0.5 * v11_val
        + 0.5 * r138_val
    )

    base_v11 = metrics(
        y_val,
        v11_val,
    )

    base_r138 = metrics(
        y_val,
        r138_val,
    )

    base_teacher = metrics(
        y_val,
        teacher_val,
    )

    print()
    print("Validation baselines")
    print(
        f"V11 RAW       : "
        f"MAE={base_v11['MAE']:.4f} "
        f"RMSE={base_v11['RMSE']:.4f}"
    )
    print(
        f"Run138        : "
        f"MAE={base_r138['MAE']:.4f} "
        f"RMSE={base_r138['RMSE']:.4f}"
    )
    print(
        f"50:50 Teacher : "
        f"MAE={base_teacher['MAE']:.4f} "
        f"RMSE={base_teacher['RMSE']:.4f}"
    )

    # ---------------------------------------------------------
    # Fast residual-head candidate search
    # ---------------------------------------------------------
    candidates = [
        (
            "linear_distill",
            0,
            0.00,
        ),
        (
            "linear_hybrid025",
            0,
            0.25,
        ),
        (
            "mlp32_distill",
            32,
            0.00,
        ),
        (
            "mlp64_distill",
            64,
            0.00,
        ),
        (
            "mlp64_hybrid025",
            64,
            0.25,
        ),
        (
            "mlp64_hybrid050",
            64,
            0.50,
        ),
    ]

    results = []

    for idx, (
        name,
        hidden_dim,
        lambda_gt,
    ) in enumerate(candidates):
        print()
        print(
            f"[STUDENT] {name} "
            f"hidden={hidden_dim} "
            f"lambda_gt={lambda_gt}"
        )

        result = train_candidate(
            name=name,
            hidden_dim=hidden_dim,
            lambda_gt=lambda_gt,
            X_train=X_train,
            y_train=y_train,
            v11_train=v11_train,
            r138_train=r138_train,
            X_val=X_val,
            y_val=y_val,
            v11_val=v11_val,
            device=device,
            seed=args.seed + idx * 100,
            max_epochs=args.head_epochs,
            patience=args.head_patience,
            lr=args.head_lr,
            batch_size=args.head_batch_size,
        )

        results.append(
            result
        )

        print(
            f"  best epoch={result['best_epoch']} "
            f"Val MAE={result['metrics']['MAE']:.4f} "
            f"RMSE={result['metrics']['RMSE']:.4f} "
            f"Bias={result['metrics']['Bias']:+.4f} "
            f"mean|corr|="
            f"{np.mean(np.abs(result['corr_val'])):.4f}"
        )

    # ---------------------------------------------------------
    # Candidate selection
    # ---------------------------------------------------------
    result_rows = []

    for r in results:
        result_rows.append({
            "name": r["name"],
            "hidden_dim": r["hidden_dim"],
            "lambda_gt": r["lambda_gt"],
            "best_epoch": r["best_epoch"],
            **r["metrics"],
            "mean_abs_correction": float(
                np.mean(
                    np.abs(
                        r["corr_val"]
                    )
                )
            ),
        })

    result_df = pd.DataFrame(
        result_rows
    ).sort_values(
        ["MAE", "RMSE"],
        ascending=[True, True],
    ).reset_index(drop=True)

    best_name = str(
        result_df.iloc[0]["name"]
    )

    best = next(
        r
        for r in results
        if r["name"] == best_name
    )

    best_mae = float(
        best["metrics"]["MAE"]
    )

    improves_v11 = (
        best_mae
        < base_v11["MAE"]
        - 1e-4
    )

    closes_teacher_gap = (
        (
            base_v11["MAE"]
            - best_mae
        )
        /
        max(
            1e-8,
            base_v11["MAE"]
            - base_teacher["MAE"],
        )
    )

    # ---------------------------------------------------------
    # Save validation predictions / diagnostics
    # ---------------------------------------------------------
    pred_df = pd.DataFrame({
        "id": val_merge["id"],
        "male": val_merge["male"],
        "true_age": y_val,
        "pred_v11": v11_val,
        "pred_run138": r138_val,
        "pred_teacher50": teacher_val,
        "pred_student": best["pred_val"],
        "student_correction": best["corr_val"],
    })

    pred_df.to_csv(
        out_dir
        / "best_validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    result_df.to_csv(
        out_dir
        / "candidate_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sub_frames = []

    for col in [
        "pred_v11",
        "pred_run138",
        "pred_teacher50",
        "pred_student",
    ]:
        sub = subgroup_metrics(
            pred_df,
            col,
        )

        sub.insert(
            0,
            "prediction",
            col,
        )

        sub_frames.append(
            sub
        )

    sub_df = pd.concat(
        sub_frames,
        ignore_index=True,
    )

    sub_df.to_csv(
        out_dir
        / "subgroup_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Save student deployment checkpoint only if it improves V11.
    student_path = (
        out_dir
        / "best_student.pt"
    )

    torch.save(
        {
            "used": bool(
                improves_v11
            ),
            "architecture": (
                "V11 frozen base + scalar residual distillation head"
            ),
            "base_v11_checkpoint": str(
                v11_ckpt
            ),
            "teacher": (
                "0.5*V11 + 0.5*Run138"
            ),
            "candidate_name": best[
                "name"
            ],
            "input_dim": int(
                X_train.shape[1]
            ),
            "hidden_dim": int(
                best["hidden_dim"]
            ),
            "feature_mean": torch.tensor(
                feat_mu,
                dtype=torch.float32,
            ),
            "feature_std": torch.tensor(
                feat_sd,
                dtype=torch.float32,
            ),
            "residual_state_dict": best[
                "state_dict"
            ],
            "tta_angles": list(
                TTA_ANGLES
            ),
            "correction_clip_months": 12.0,
            "val_metrics": best[
                "metrics"
            ],
            "v11_val_metrics": base_v11,
            "teacher50_val_metrics": base_teacher,
            "enterprise_used": False,
            "test_used": False,
        },
        student_path,
    )

    summary = {
        "v11_raw": base_v11,
        "run138": base_r138,
        "teacher50": base_teacher,
        "best_student_name": best[
            "name"
        ],
        "best_student": best[
            "metrics"
        ],
        "improves_v11": bool(
            improves_v11
        ),
        "v11_mae_gain_months": float(
            base_v11["MAE"]
            - best_mae
        ),
        "teacher_gap_recovered_fraction": float(
            closes_teacher_gap
        ),
        "enterprise_used": False,
        "test_used": False,
    }

    with open(
        out_dir
        / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 112)
    print("FINAL RESULT")
    print("=" * 112)
    print(
        f"V11 RAW       : "
        f"MAE={base_v11['MAE']:.4f} "
        f"RMSE={base_v11['RMSE']:.4f}"
    )
    print(
        f"Teacher 50:50 : "
        f"MAE={base_teacher['MAE']:.4f} "
        f"RMSE={base_teacher['RMSE']:.4f}"
    )
    print(
        f"Best Student  : "
        f"{best['name']} | "
        f"MAE={best['metrics']['MAE']:.4f} "
        f"RMSE={best['metrics']['RMSE']:.4f} "
        f"Bias={best['metrics']['Bias']:+.4f}"
    )
    print(
        f"V11 gain      : "
        f"{base_v11['MAE'] - best_mae:+.4f} months"
    )
    print(
        f"Teacher gain recovered: "
        f"{closes_teacher_gap*100:.1f}%"
    )

    if improves_v11:
        print(
            "판정           : SUCCESS - ensemble 없이 단일 V11 student 후보 유지"
        )
        print(
            "student ckpt   :",
            student_path,
        )
    else:
        print(
            "판정           : FAIL - V11 RAW보다 못함, student 버림"
        )

    print(
        "Enterprise/Test: NOT USED"
    )
    print("=" * 112)


if __name__ == "__main__":
    main()