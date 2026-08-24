# -*- coding: utf-8 -*-
r"""
fit_v11_run138_validation_stacker.py

목적
----
V11 + Run138을 단순 평균하지 않고, 내부 validation에서만 결합 규칙을 학습/검증한다.

사용 데이터
---------
- V11: 실제 기업 V11 패키지와 같은 checkpoint(best.pt, val_mae≈5.8125)를 사용하여
       V11 validation cache(768x512, 5-angle TTA)에서 prediction 생성
- Run138: 기존 validation_predictions_best_mae.csv 사용
- Enterprise/Test GT: 절대 사용하지 않음

비교 방법 (5-fold OOF)
---------------------
1) raw_v11
2) raw_run138
3) mean50
4) global_convex
   pred = alpha*V11 + (1-alpha)*Run138
   alpha는 각 outer-train에서만 선택
5) sex_convex
   Female/Male alpha를 각 outer-train에서만 따로 선택
6) sex_predage_convex
   성별 + V11 predicted-age 구간별 alpha
   구간: <=60, 61~96, 97~144, >144 (GT가 아니라 V11 prediction으로 routing)
   작은 subgroup는 sex-level alpha로 fallback
7) ridge_residual
   V11을 base로 두고 Run138과의 disagreement/sex를 이용한 작은 ridge residual stacker
   lambda는 outer-train 내부 CV로 선택

선택 원칙
--------
- 전체 validation 5-fold OOF MAE 기준
- V11 RAW보다 실제로 낮아질 때만 결합 채택 권장
- 최종 full-validation fit config를 JSON으로 저장
- 이 JSON은 Enterprise 결과를 전혀 보지 않고 생성된다.

필요 파일
--------
같은 폴더에:
  fit_v11_sex_calibration_from_cache_FIXED.py

기본 서버 경로
------------
V11 checkpoint:
  G:\Project\sinra_cho\checkpoints_convnext_single_v11_ldl\best.pt

Run138 validation prediction:
  G:\Project\sinra_cho\convnext_tiny_512_results\
  138_male_head_only_bias_correction\validation_predictions_best_mae.csv

실행
----
python .\fit_v11_run138_validation_stacker.py --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import fit_v11_sex_calibration_from_cache_FIXED as v11


BASE = Path(r"G:\Project\sinra_cho")

DEFAULT_V11_CKPT = (
    BASE
    / "checkpoints_convnext_single_v11_ldl"
    / "best.pt"
)

DEFAULT_RUN138_PRED = (
    BASE
    / "convnext_tiny_512_results"
    / "138_male_head_only_bias_correction"
    / "validation_predictions_best_mae.csv"
)

DEFAULT_OUTPUT = (
    BASE
    / "v11_run138_validation_stacker"
)

PRED_AGE_BINS = [
    (0.0, 60.0, "le60"),
    (60.0, 96.0, "61_96"),
    (96.0, 144.0, "97_144"),
    (144.0, float("inf"), "gt144"),
]

ALPHA_GRID = np.linspace(0.0, 1.0, 101)
RIDGE_LAMBDAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def norm_id_series(s: pd.Series):
    return (
        s.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def find_col(df, aliases):
    mapping = {
        str(c).strip().lower().replace("_", "").replace(" ", ""): c
        for c in df.columns
    }

    for a in aliases:
        key = str(a).strip().lower().replace("_", "").replace(" ", "")
        if key in mapping:
            return mapping[key]

    return None


def load_run138_predictions(path: Path):
    df = pd.read_csv(path)

    id_col = find_col(df, ["id", "image_id", "imageid"])
    male_col = find_col(df, ["male", "sex", "gender"])
    true_col = find_col(
        df,
        ["true_boneage", "true_age", "boneage", "bone_age", "target"],
    )
    pred_col = find_col(
        df,
        ["pred_boneage", "pred_age", "predicted_age", "prediction"],
    )

    if None in (id_col, male_col, true_col, pred_col):
        raise ValueError(
            "Run138 validation prediction columns를 찾지 못했습니다.\n"
            f"columns={list(df.columns)}"
        )

    out = pd.DataFrame({
        "id": norm_id_series(df[id_col]),
        "male_r138": pd.to_numeric(df[male_col], errors="coerce"),
        "true_r138": pd.to_numeric(df[true_col], errors="coerce"),
        "pred_run138": pd.to_numeric(df[pred_col], errors="coerce"),
    })

    return out.dropna().copy()


def metrics(y, p):
    e = p - y
    ae = np.abs(e)

    return {
        "MAE": float(np.mean(ae)),
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
        "Bias": float(np.mean(e)),
        "MedianAE": float(np.median(ae)),
        "P90": float(np.percentile(ae, 90)),
    }


def subgroup_metrics(df, pred_col):
    rows = []

    groups = [
        ("Overall", np.ones(len(df), dtype=bool)),
        ("Female", df["male"].to_numpy(float) < 0.5),
        ("Male", df["male"].to_numpy(float) >= 0.5),
        (
            "Male_true_le60",
            (df["male"].to_numpy(float) >= 0.5)
            & (df["true_age"].to_numpy(float) <= 60.0),
        ),
        (
            "Male_true_61_96",
            (df["male"].to_numpy(float) >= 0.5)
            & (df["true_age"].to_numpy(float) > 60.0)
            & (df["true_age"].to_numpy(float) <= 96.0),
        ),
        (
            "Male_true_97_144",
            (df["male"].to_numpy(float) >= 0.5)
            & (df["true_age"].to_numpy(float) > 96.0)
            & (df["true_age"].to_numpy(float) <= 144.0),
        ),
        (
            "Male_true_gt144",
            (df["male"].to_numpy(float) >= 0.5)
            & (df["true_age"].to_numpy(float) > 144.0),
        ),
    ]

    y_all = df["true_age"].to_numpy(float)
    p_all = df[pred_col].to_numpy(float)

    for name, mask in groups:
        n = int(np.sum(mask))
        if n == 0:
            continue

        m = metrics(y_all[mask], p_all[mask])
        rows.append({
            "group": name,
            "N": n,
            **m,
        })

    return pd.DataFrame(rows)


def make_folds(df, n_splits=5, seed=42):
    """
    sex x true-age band stratified round-robin.
    Fold 구성에 GT age를 쓰는 것은 validation CV의 균형을 위한 것이며
    inference routing에는 GT를 사용하지 않는다.
    """
    rng = np.random.default_rng(seed)
    folds = np.full(len(df), -1, dtype=int)

    age = df["true_age"].to_numpy(float)
    male = df["male"].to_numpy(float)

    true_bins = np.select(
        [
            age <= 60,
            (age > 60) & (age <= 96),
            (age > 96) & (age <= 144),
            age > 144,
        ],
        [0, 1, 2, 3],
        default=4,
    )

    for sex in (0, 1):
        for age_bin in range(4):
            idx = np.where(
                (male.astype(int) == sex)
                & (true_bins == age_bin)
            )[0]

            if len(idx) == 0:
                continue

            rng.shuffle(idx)

            for j, i in enumerate(idx):
                folds[i] = j % n_splits

    if np.any(folds < 0):
        leftover = np.where(folds < 0)[0]
        rng.shuffle(leftover)
        for j, i in enumerate(leftover):
            folds[i] = j % n_splits

    return folds


def best_alpha_mae(y, v11p, r138p):
    best = None

    for alpha in ALPHA_GRID:
        p = alpha * v11p + (1.0 - alpha) * r138p
        mae = float(np.mean(np.abs(p - y)))

        if best is None or mae < best[0]:
            best = (mae, float(alpha))

    return best[1]


def apply_global_convex(v11p, r138p, alpha):
    return alpha * v11p + (1.0 - alpha) * r138p


def fit_sex_alpha(y, v11p, r138p, male):
    out = {}

    global_alpha = best_alpha_mae(y, v11p, r138p)

    for sex, name in [(0, "female"), (1, "male")]:
        m = male.astype(int) == sex
        if int(np.sum(m)) >= 20:
            out[name] = best_alpha_mae(
                y[m],
                v11p[m],
                r138p[m],
            )
        else:
            out[name] = global_alpha

    out["global"] = global_alpha
    return out


def apply_sex_convex(v11p, r138p, male, params):
    alpha = np.where(
        male >= 0.5,
        float(params["male"]),
        float(params["female"]),
    )

    return alpha * v11p + (1.0 - alpha) * r138p


def pred_age_band_labels(v11p):
    labels = np.empty(len(v11p), dtype=object)

    for low, high, name in PRED_AGE_BINS:
        if math.isinf(high):
            m = v11p > low
        elif low <= 0:
            m = v11p <= high
        else:
            m = (v11p > low) & (v11p <= high)

        labels[m] = name

    return labels


def fit_sex_predage_alpha(
    y,
    v11p,
    r138p,
    male,
    min_local_n=35,
):
    """
    deployable routing:
      sex + V11 predicted age only
    """
    sex_params = fit_sex_alpha(
        y,
        v11p,
        r138p,
        male,
    )

    bands = pred_age_band_labels(v11p)
    local = {}

    for sex, sex_name in [(0, "female"), (1, "male")]:
        sex_mask = male.astype(int) == sex

        for _, _, band_name in PRED_AGE_BINS:
            m = sex_mask & (bands == band_name)
            key = f"{sex_name}_{band_name}"

            if int(np.sum(m)) >= min_local_n:
                local[key] = {
                    "alpha": best_alpha_mae(
                        y[m],
                        v11p[m],
                        r138p[m],
                    ),
                    "n": int(np.sum(m)),
                    "fallback": False,
                }
            else:
                local[key] = {
                    "alpha": float(sex_params[sex_name]),
                    "n": int(np.sum(m)),
                    "fallback": True,
                }

    return {
        "sex": sex_params,
        "local": local,
        "min_local_n": int(min_local_n),
    }


def apply_sex_predage_alpha(v11p, r138p, male, params):
    bands = pred_age_band_labels(v11p)
    pred = np.empty_like(v11p, dtype=float)

    for i in range(len(v11p)):
        sex_name = "male" if male[i] >= 0.5 else "female"
        key = f"{sex_name}_{bands[i]}"
        alpha = float(params["local"][key]["alpha"])
        pred[i] = alpha * v11p[i] + (1.0 - alpha) * r138p[i]

    return pred


def ridge_features(v11p, r138p, male):
    """
    V11을 anchor로 두고 작은 residual만 학습.
    deployable features only.
    """
    diff = r138p - v11p
    meanp = (v11p + r138p) / 2.0
    male01 = male.astype(float)

    # raw features; train-fold statistics로 standardize
    X = np.column_stack([
        diff,
        np.abs(diff),
        meanp,
        male01,
        diff * male01,
        np.abs(diff) * male01,
    ])

    return X


def fit_ridge(
    X,
    target,
    lam,
):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0

    Z = (X - mu) / sd
    Z = np.column_stack([
        np.ones(len(Z)),
        Z,
    ])

    penalty = np.eye(Z.shape[1])
    penalty[0, 0] = 0.0

    beta = np.linalg.solve(
        Z.T @ Z + float(lam) * penalty,
        Z.T @ target,
    )

    return {
        "mu": mu,
        "sd": sd,
        "beta": beta,
        "lambda": float(lam),
    }


def predict_ridge(model, X):
    Z = (X - model["mu"]) / model["sd"]
    Z = np.column_stack([
        np.ones(len(Z)),
        Z,
    ])
    return Z @ model["beta"]


def select_ridge_lambda(
    y,
    v11p,
    r138p,
    male,
    seed,
    n_splits=4,
):
    """
    inner CV on outer-train only.
    residual target = y - V11
    """
    tmp = pd.DataFrame({
        "true_age": y,
        "male": male,
    })
    inner_folds = make_folds(
        tmp,
        n_splits=n_splits,
        seed=seed,
    )

    X = ridge_features(
        v11p,
        r138p,
        male,
    )
    target = y - v11p

    scores = []

    for lam in RIDGE_LAMBDAS:
        fold_maes = []

        for f in range(n_splits):
            tr = inner_folds != f
            va = inner_folds == f

            if int(np.sum(va)) == 0:
                continue

            model = fit_ridge(
                X[tr],
                target[tr],
                lam,
            )

            corr = predict_ridge(
                model,
                X[va],
            )

            # conservative guard against extreme correction
            corr = np.clip(corr, -12.0, 12.0)
            pred = v11p[va] + corr

            fold_maes.append(
                float(
                    np.mean(
                        np.abs(
                            pred - y[va]
                        )
                    )
                )
            )

        scores.append(
            (
                float(np.mean(fold_maes)),
                float(lam),
            )
        )

    scores.sort(key=lambda x: x[0])
    return scores[0][1]


def oof_predictions(
    df,
    n_splits=5,
    seed=42,
):
    y = df["true_age"].to_numpy(float)
    p1 = df["pred_v11"].to_numpy(float)
    p2 = df["pred_run138"].to_numpy(float)
    male = df["male"].to_numpy(float)

    folds = make_folds(
        df,
        n_splits=n_splits,
        seed=seed,
    )

    outputs = {
        "raw_v11": p1.copy(),
        "raw_run138": p2.copy(),
        "mean50": 0.5 * p1 + 0.5 * p2,
        "global_convex": np.zeros_like(p1),
        "sex_convex": np.zeros_like(p1),
        "sex_predage_convex": np.zeros_like(p1),
        "ridge_residual": np.zeros_like(p1),
    }

    fold_configs = []

    for fold in range(n_splits):
        tr = folds != fold
        va = folds == fold

        # global convex
        alpha = best_alpha_mae(
            y[tr],
            p1[tr],
            p2[tr],
        )
        outputs["global_convex"][va] = apply_global_convex(
            p1[va],
            p2[va],
            alpha,
        )

        # sex convex
        sex_params = fit_sex_alpha(
            y[tr],
            p1[tr],
            p2[tr],
            male[tr],
        )
        outputs["sex_convex"][va] = apply_sex_convex(
            p1[va],
            p2[va],
            male[va],
            sex_params,
        )

        # sex + predicted-age convex
        band_params = fit_sex_predage_alpha(
            y[tr],
            p1[tr],
            p2[tr],
            male[tr],
        )
        outputs["sex_predage_convex"][va] = apply_sex_predage_alpha(
            p1[va],
            p2[va],
            male[va],
            band_params,
        )

        # nested-CV ridge residual
        lam = select_ridge_lambda(
            y[tr],
            p1[tr],
            p2[tr],
            male[tr],
            seed=seed + fold * 100,
            n_splits=4,
        )

        X_tr = ridge_features(
            p1[tr],
            p2[tr],
            male[tr],
        )
        X_va = ridge_features(
            p1[va],
            p2[va],
            male[va],
        )

        ridge_model = fit_ridge(
            X_tr,
            y[tr] - p1[tr],
            lam,
        )

        corr = predict_ridge(
            ridge_model,
            X_va,
        )
        corr = np.clip(corr, -12.0, 12.0)

        outputs["ridge_residual"][va] = p1[va] + corr

        fold_configs.append({
            "fold": int(fold),
            "global_alpha": float(alpha),
            "sex_alpha": {
                k: float(v)
                for k, v in sex_params.items()
            },
            "ridge_lambda": float(lam),
        })

    return outputs, folds, fold_configs


def choose_full_ridge_lambda(
    y,
    p1,
    p2,
    male,
    seed,
):
    return select_ridge_lambda(
        y,
        p1,
        p2,
        male,
        seed=seed + 9999,
        n_splits=5,
    )


def serialize_ridge(model):
    return {
        "mu": [float(x) for x in model["mu"]],
        "sd": [float(x) for x in model["sd"]],
        "beta": [float(x) for x in model["beta"]],
        "lambda": float(model["lambda"]),
        "clip_correction_months": 12.0,
        "feature_order": [
            "run138_minus_v11",
            "abs_run138_minus_v11",
            "mean_prediction",
            "male",
            "diff_x_male",
            "absdiff_x_male",
        ],
    }


def fit_full_config(
    method,
    df,
    seed,
):
    y = df["true_age"].to_numpy(float)
    p1 = df["pred_v11"].to_numpy(float)
    p2 = df["pred_run138"].to_numpy(float)
    male = df["male"].to_numpy(float)

    if method == "raw_v11":
        return {
            "method": "raw_v11",
        }

    if method == "raw_run138":
        return {
            "method": "raw_run138",
        }

    if method == "mean50":
        return {
            "method": "mean50",
            "alpha_v11": 0.5,
        }

    if method == "global_convex":
        alpha = best_alpha_mae(
            y,
            p1,
            p2,
        )
        return {
            "method": method,
            "alpha_v11": float(alpha),
            "alpha_run138": float(1.0 - alpha),
        }

    if method == "sex_convex":
        params = fit_sex_alpha(
            y,
            p1,
            p2,
            male,
        )
        return {
            "method": method,
            "alpha_v11": {
                k: float(v)
                for k, v in params.items()
            },
        }

    if method == "sex_predage_convex":
        params = fit_sex_predage_alpha(
            y,
            p1,
            p2,
            male,
        )
        return {
            "method": method,
            "routing_age_source": "V11 predicted age",
            "age_bins": [
                "<=60",
                "61-96",
                "97-144",
                ">144",
            ],
            "params": params,
        }

    if method == "ridge_residual":
        lam = choose_full_ridge_lambda(
            y,
            p1,
            p2,
            male,
            seed,
        )
        X = ridge_features(
            p1,
            p2,
            male,
        )
        model = fit_ridge(
            X,
            y - p1,
            lam,
        )
        return {
            "method": method,
            "base": "V11",
            "ridge": serialize_ridge(model),
        }

    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--v11_checkpoint",
        default=str(DEFAULT_V11_CKPT),
    )
    ap.add_argument(
        "--cache_val",
        default=None,
    )
    ap.add_argument(
        "--val_csv",
        default=None,
    )
    ap.add_argument(
        "--run138_pred",
        default=str(DEFAULT_RUN138_PRED),
    )
    ap.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT),
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

    v11_ckpt = Path(
        args.v11_checkpoint
    ).resolve()

    run138_path = Path(
        args.run138_pred
    ).resolve()

    out_dir = Path(
        args.output_dir
    ).resolve()

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not v11_ckpt.is_file():
        raise FileNotFoundError(
            f"V11 checkpoint 없음: {v11_ckpt}"
        )

    if not run138_path.is_file():
        raise FileNotFoundError(
            f"Run138 validation prediction 없음: {run138_path}"
        )

    # --------------------------------------------------------
    # Exact V11 validation prediction
    # --------------------------------------------------------
    cache_dir = v11.resolve_cache_val(
        args.cache_val
    )

    cache_df = v11.collect_cache(
        cache_dir
    )

    val_csv = v11.auto_find_val_csv(
        cache_df,
        args.val_csv,
    )

    labels = v11.load_label_csv(
        val_csv
    )

    v11_input = cache_df.merge(
        labels,
        on="id",
        how="inner",
    )

    device = v11.resolve_device(
        args.device
    )

    model, checkpoint = v11.load_model(
        v11_ckpt,
        device,
    )

    print()
    print("=" * 112)
    print("V11 + RUN138 VALIDATION-ONLY STACKER")
    print("=" * 112)
    print("V11 checkpoint :", v11_ckpt)
    print("V11 ckpt valMAE:", checkpoint.get("val_mae"))
    print("V11 cache      :", cache_dir)
    print("V11 matched N  :", len(v11_input))
    print("Run138 pred    :", run138_path)
    print("Enterprise     : NOT USED")
    print("Test           : NOT USED")
    print("=" * 112)

    ds = v11.CacheDataset(
        v11_input
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    v11_pred = v11.predict_validation(
        model,
        dl,
        device,
        float(checkpoint.get("age_mean", 0.0)),
        float(checkpoint.get("age_std", 1.0)),
        amp=args.amp,
    )

    v11_pred = v11_pred.rename(
        columns={
            "true_age": "true_v11",
            "pred_raw": "pred_v11",
            "male": "male_v11",
        }
    )

    v11_pred["id"] = norm_id_series(
        v11_pred["id"]
    )

    # --------------------------------------------------------
    # Run138 prediction merge
    # --------------------------------------------------------
    run138 = load_run138_predictions(
        run138_path
    )

    merged = v11_pred.merge(
        run138,
        on="id",
        how="inner",
    )

    if len(merged) != len(v11_pred):
        print(
            f"[WARNING] V11 N={len(v11_pred)}, "
            f"merged N={len(merged)}"
        )

    true_diff = np.abs(
        merged["true_v11"].to_numpy(float)
        - merged["true_r138"].to_numpy(float)
    )

    male_diff = np.abs(
        merged["male_v11"].to_numpy(float)
        - merged["male_r138"].to_numpy(float)
    )

    print(
        f"GT max difference      : {true_diff.max():.6f}"
    )
    print(
        f"Sex max difference     : {male_diff.max():.6f}"
    )

    if float(true_diff.max()) > 1e-4:
        raise RuntimeError(
            "V11/Run138 validation GT가 일치하지 않습니다."
        )

    if float(male_diff.max()) > 1e-4:
        raise RuntimeError(
            "V11/Run138 validation sex가 일치하지 않습니다."
        )

    df = pd.DataFrame({
        "id": merged["id"],
        "male": merged["male_v11"].astype(float),
        "true_age": merged["true_v11"].astype(float),
        "pred_v11": merged["pred_v11"].astype(float),
        "pred_run138": merged["pred_run138"].astype(float),
    })

    # --------------------------------------------------------
    # OOF evaluation
    # --------------------------------------------------------
    outputs, folds, fold_configs = oof_predictions(
        df,
        n_splits=args.folds,
        seed=args.seed,
    )

    df["fold"] = folds

    comparison = []

    y = df["true_age"].to_numpy(float)

    for method, pred in outputs.items():
        df[f"pred_{method}"] = pred

        m = metrics(
            y,
            pred,
        )

        comparison.append({
            "method": method,
            **m,
        })

    comp = pd.DataFrame(
        comparison
    ).sort_values(
        ["MAE", "RMSE"],
        ascending=[True, True],
    ).reset_index(drop=True)

    raw_v11_mae = float(
        comp.loc[
            comp["method"] == "raw_v11",
            "MAE",
        ].iloc[0]
    )

    best_method = str(
        comp.iloc[0]["method"]
    )

    best_mae = float(
        comp.iloc[0]["MAE"]
    )

    print()
    print("5-FOLD OOF COMPARISON")
    print(
        comp.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("OOF subgroup diagnostics")
    print("-" * 112)

    diag_frames = []

    for method in [
        "raw_v11",
        "raw_run138",
        best_method,
    ]:
        pred_col = f"pred_{method}"
        sub = subgroup_metrics(
            df,
            pred_col,
        )
        sub.insert(
            0,
            "method",
            method,
        )
        diag_frames.append(sub)

    diagnostics = pd.concat(
        diag_frames,
        ignore_index=True,
    )

    print(
        diagnostics.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # Fit final config on the full validation set
    # only if selected method beats V11 RAW
    # --------------------------------------------------------
    improved = (
        best_mae < raw_v11_mae - 1e-6
        and best_method != "raw_v11"
    )

    if improved:
        full_config = fit_full_config(
            best_method,
            df,
            args.seed,
        )
    else:
        full_config = {
            "method": "raw_v11",
        }

    payload = {
        "used": bool(improved),
        "selected_method": (
            best_method
            if improved
            else "raw_v11"
        ),
        "selection_basis": "5-fold validation-only OOF MAE",
        "v11_raw_oof_mae": raw_v11_mae,
        "best_oof_mae": best_mae,
        "oof_improvement_months": raw_v11_mae - best_mae,
        "v11_checkpoint": str(v11_ckpt),
        "v11_checkpoint_val_mae": checkpoint.get("val_mae"),
        "run138_validation_predictions": str(run138_path),
        "n_validation": int(len(df)),
        "config": full_config,
        "enterprise_used": False,
        "test_used": False,
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    df.to_csv(
        out_dir / "validation_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    comp.to_csv(
        out_dir / "oof_method_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    diagnostics.to_csv(
        out_dir / "oof_subgroup_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with open(
        out_dir / "stacker_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(
        out_dir / "fold_configs.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            fold_configs,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 112)
    print("RESULT")
    print("=" * 112)
    print(
        "V11 RAW OOF MAE :",
        f"{raw_v11_mae:.4f}",
    )
    print(
        "Best OOF method :",
        best_method,
    )
    print(
        "Best OOF MAE    :",
        f"{best_mae:.4f}",
    )
    print(
        "Improvement     :",
        f"{raw_v11_mae - best_mae:+.4f} months",
    )
    print(
        "Final selection :",
        payload["selected_method"],
    )

    if improved:
        print(
            "판정            : V11 RAW보다 OOF 개선 -> 결합 후보 유지"
        )
    else:
        print(
            "판정            : V11 RAW 개선 실패 -> 결합 버림"
        )

    print()
    print(
        "Config:",
        out_dir / "stacker_config.json",
    )
    print(
        "※ Enterprise/Test GT는 사용하지 않았습니다."
    )
    print("=" * 112)


if __name__ == "__main__":
    main()