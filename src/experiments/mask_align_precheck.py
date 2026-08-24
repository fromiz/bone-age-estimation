# -*- coding: utf-8 -*-
"""
마스크 기반 정렬 사전 진단  (crop_yolo_seg 구조 전용)

이 스크립트는 아무것도 바꾸지 않습니다. 읽고 재고 그림만 그립니다.
v15(마스크 전처리)를 짜기 전에 아래 5가지를 확정하는 것이 목적입니다.

  1. 데이터 구조   csv 컬럼명 / 이미지-마스크 쌍 매칭률
  2. 마스크 품질   이진성 · 면적비 · 연결성분 · 구멍
  3. 손목 잘림     마스크가 프레임 경계에 닿는 비율   <- 4~8세 문제의 후보 원인
  4. 각도 추정     4가지 방법을 동시에 재고 상호 일치도를 봅니다
                   어느 방법이 안정적인지는 데이터를 봐야 정해집니다
  5. 정렬 영향     회전 후 잘림량 · 엄지 좌우 분포 · 손끝 방향 판정률

사용법
    스크립트 상단 [설정] 블록의 BASE_DIR / SEG_DIR 만 자기 환경에 맞추고 실행합니다.
        python mask_align_precheck.py
    일회성으로 다른 경로를 보려면 인자로 덮어쓸 수 있습니다.
        python mask_align_precheck.py --root "D:/other/crop_yolo_seg" --n 100

산출물 (--out 폴더, 기본 ./precheck_out)
    precheck_summary.txt      수치 요약 (그대로 복사해서 보내주시면 됩니다)
    precheck_angles.csv       샘플별 각도 원자료
    qc_angle_methods.png      각도 4방법 비교 시트 (원본 위에 축을 그림)
    qc_aligned.png            정렬 전/후 비교 시트
    qc_hist.png               각도 분포 히스토그램 · 방법간 산점도
"""
import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("[중단] opencv 가 필요합니다:  pip install opencv-python")
try:
    import pandas as pd
except ImportError:
    sys.exit("[중단] pandas 가 필요합니다:  pip install pandas")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("[중단] matplotlib 가 필요합니다:  pip install matplotlib")

# ════════════════════════════════════════════════════════════════════
# [설정]  여기만 고치면 됩니다.  (v11~v14 와 동일한 방식)
# ════════════════════════════════════════════════════════════════════
BASE_DIR   = Path(r"G:\Project\sinra_cho")
SEG_DIR    = BASE_DIR / "crop_yolo_seg"        # images/ masks/ csv/ 가 있는 폴더
OUT_DIR    = BASE_DIR / "precheck_out"         # 산출물 저장 위치

N_SAMPLE   = 400        # split 당 표본 수. 전수 검사하려면 100000 처럼 크게
SEED       = 42

# 폴더 이름이 다르면 여기서 맞추세요
SPLIT_DIRS = {"train": "train", "valid": "validation", "test": "test"}
IMAGE_SUB  = "images"   # 각 split 아래 이미지 폴더명
MASK_SUB   = "masks"    # 각 split 아래 마스크 폴더명
CSV_SUB    = "csv"      # train.csv / validation.csv / test.csv 가 있는 폴더
META_CSV   = "preprocessing_metadata.csv"      # SEG_DIR 기준 상대경로

# 각도 측정 파라미터
FOREARM_BAND  = (0.05, 0.32)   # 전완축을 잴 때 손 아래에서부터의 세로 비율 구간
WRIST_BAND    = 0.18           # 손목절단선을 잴 때 하단 비율
ALIGN_ITERS   = 2              # 정렬 잔차 측정 시 반복 보정 횟수
# ════════════════════════════════════════════════════════════════════

SPLITS = list(SPLIT_DIRS.values())
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ────────────────────────────────────────────────────────────── I/O
def imread_kr(path, flags=cv2.IMREAD_GRAYSCALE):
    """한글 경로 안전 로더. cv2.imread 는 Windows 한글 경로에서 조용히 실패합니다."""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(buf, flags)
        return img
    except Exception:
        return None


def index_dir(d: Path) -> dict:
    """{stem: path} 인덱스. 확장자가 달라도 매칭되게 stem 으로 잡습니다."""
    out = {}
    if not d.exists():
        return out
    for p in d.iterdir():
        if p.suffix.lower() in IMG_EXT:
            out[p.stem] = p
    return out


# ─────────────────────────────────────────────────── 마스크 기본 측정
def mask_stats(m):
    """이진 마스크의 품질 지표. m 은 0/255 uint8 을 가정하되 아니면 이진화합니다."""
    uniq = np.unique(m)
    binary = len(uniq) <= 2
    bw = (m > 127).astype(np.uint8)
    H, W = bw.shape
    area = int(bw.sum())
    if area < 100:
        return None

    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    comps = int(n_lab - 1)
    big = int(stats[1:, cv2.CC_STAT_AREA].max()) if comps else 0

    # 구멍: 최대 성분을 채운 것과의 차이
    keep = (lab == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))).astype(np.uint8)
    cont, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(keep)
    cv2.drawContours(filled, cont, -1, 1, -1)
    holes = int(filled.sum() - keep.sum())

    ys, xs = np.nonzero(keep)
    return {
        "binary": binary, "n_unique": int(len(uniq)),
        "area_frac": area / (H * W),
        "components": comps,
        "main_frac": big / max(area, 1),
        "hole_frac": holes / max(int(filled.sum()), 1),
        "touch_top": bool((keep[0] > 0).any()),
        "touch_bottom": bool((keep[-1] > 0).any()),
        "touch_left": bool((keep[:, 0] > 0).any()),
        "touch_right": bool((keep[:, -1] > 0).any()),
        "bottom_touch_frac": float((keep[-1] > 0).mean()),
        "bbox_h": int(ys.max() - ys.min() + 1), "bbox_w": int(xs.max() - xs.min() + 1),
        "H": H, "W": W,
        "keep": keep,
    }


# ─────────────────────────────────────────── 각도 추정 4가지
def ang_pca(keep):
    """전체 마스크 주축. 반환: 세로(위쪽)를 0 으로 하는 각도(도). 손가락 벌어짐에 민감."""
    ys, xs = np.nonzero(keep)
    p = np.stack([xs, ys], 1).astype(np.float64)
    p -= p.mean(0)
    _, vec = np.linalg.eigh(np.cov(p.T))
    v = vec[:, -1]
    a = np.degrees(np.arctan2(v[0], -v[1]))      # 위쪽(-y)이 0
    return float((a + 90) % 180 - 90)


def ang_minrect(keep):
    """최소외접 회전사각형. 손 전체 실루엣 기준. 엄지가 벌어지면 흔들립니다."""
    cont, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cont:
        return np.nan
    c = max(cont, key=cv2.contourArea)
    (_, _), (w, h), a = cv2.minAreaRect(c)
    if w > h:                                     # 긴 변이 세로가 되도록 정규화
        a += 90.0
    return float((a + 90) % 180 - 90)


def ang_forearm(keep, lo=FOREARM_BAND[0], hi=FOREARM_BAND[1]):
    """손목쪽 두 밴드의 무게중심을 잇는 축. 손가락 벌어짐과 무관해 안정적.

    lo~hi 는 손 아래에서부터의 세로 비율입니다. 하단 5~32% 구간을 두 개로 나눠
    아래 밴드 중심 -> 위 밴드 중심 벡터를 전완 방향으로 봅니다.
    """
    ys, xs = np.nonzero(keep)
    y0, y1 = int(ys.min()), int(ys.max())
    h = max(1, y1 - y0)
    mid = (lo + hi) / 2.0
    b_lo = keep[y1 - int(hi * h):y1 - int(mid * h) + 1]
    b_hi = keep[y1 - int(mid * h):y1 - int(lo * h) + 1]
    if b_lo.sum() < 50 or b_hi.sum() < 50:
        return np.nan
    cols = np.arange(keep.shape[1], dtype=np.float64)
    cx_lo = np.average(cols, weights=b_lo.sum(0).astype(np.float64) + 1e-9)
    cx_hi = np.average(cols, weights=b_hi.sum(0).astype(np.float64) + 1e-9)
    dy = (int(hi * h) - int(lo * h)) / 2.0
    return float(np.degrees(np.arctan2(cx_lo - cx_hi, max(dy, 1.0))))


def ang_wrist_edge(keep, band=WRIST_BAND, min_pts=25):
    """손목 절단선(마스크 하단 경계)에 직선을 맞춘 각도. 사용자가 그린 파란선.

    [중요] 프레임 하단에 닿은 열은 제외합니다. 그 부분은 해부학적 경계가 아니라
    크롭이 잘라낸 수평 직선이므로, 포함하면 각도가 0 쪽으로 끌려갑니다.
    유효 점이 너무 적으면 NaN 을 반환합니다(억지로 맞추지 않습니다).
    """
    H, W = keep.shape
    ys, xs = np.nonzero(keep)
    y0, y1 = int(ys.min()), int(ys.max())
    h = max(1, y1 - y0)
    ytop = y1 - int(band * h)

    pts_x, pts_y = [], []
    for c in range(W):
        col = np.nonzero(keep[:, c])[0]
        if col.size == 0:
            continue
        b = int(col.max())
        if b < ytop:
            continue
        if b >= H - 2:                            # 프레임 하단에 닿음 -> 크롭 절단선
            continue
        pts_x.append(c); pts_y.append(b)
    if len(pts_x) < min_pts:
        return np.nan, len(pts_x)

    x = np.asarray(pts_x, float); y = np.asarray(pts_y, float)
    # 이상치에 강하도록 1회 재적합
    s, b0 = np.polyfit(x, y, 1)
    r = np.abs(y - (s * x + b0))
    k = r < max(3.0, 2.5 * r.std())
    if k.sum() >= min_pts:
        s, b0 = np.polyfit(x[k], y[k], 1)
    # 손목선은 손 장축에 수직이므로 기울기 부호가 나머지 3개와 반대로 나옵니다.
    # 네 방법을 같은 축(양수=장축이 시계방향으로 기울어짐)으로 맞추기 위해 부호를 뒤집습니다.
    return float(-np.degrees(np.arctan(s))), int(k.sum())


# ─────────────────────────────────────────── 방향(상하 / 좌우) 판정
def finger_side_up(keep):
    """손가락이 위쪽인지 판정. 수직 단면의 연결 조각 수로 봅니다.

    손가락 구간은 손가락 사이 틈 때문에 한 행에 여러 조각이 생기고,
    손목 구간은 하나로 이어집니다. 조각이 많은 쪽이 손가락입니다.
    반환 True = 손가락이 위 (정상), False = 뒤집힘, None = 판정 불가
    """
    ys, _ = np.nonzero(keep)
    y0, y1 = int(ys.min()), int(ys.max())
    h = max(1, y1 - y0)

    def runs(band):
        cnt = []
        for r in band:
            d = np.diff(np.concatenate(([0], (r > 0).astype(np.int8), [0])))
            cnt.append(int((d == 1).sum()))
        return float(np.mean(cnt)) if cnt else 0.0

    top = runs(keep[y0:y0 + int(0.25 * h)])
    bot = runs(keep[y1 - int(0.25 * h):y1])
    if abs(top - bot) < 0.35:
        return None, top, bot
    return (top > bot), top, bot


def thumb_side(keep):
    """엄지 방향. +1 오른쪽 / -1 왼쪽 / 0 판정불가.
    손가락부 수평중심 대비 손바닥부의 좌우 최대 도달거리를 비교합니다."""
    ys, _ = np.nonzero(keep)
    y0, y1 = int(ys.min()), int(ys.max())
    h = max(1, y1 - y0)
    fing = keep[y0:y0 + int(0.30 * h)]
    palm = keep[y0 + int(0.32 * h):y0 + int(0.75 * h)]
    if fing.sum() < 50 or palm.sum() < 50:
        return 0
    cols = np.arange(keep.shape[1], dtype=np.float64)
    cf = float(np.average(cols, weights=fing.sum(0).astype(np.float64) + 1e-9))
    nz = np.nonzero(palm.sum(0))[0]
    d = (float(nz.max()) - cf) - (cf - float(nz.min()))
    return int(np.sign(d)) if abs(d) > 0.03 * keep.shape[1] else 0


def align_iter(keep, method=ang_forearm, iters=ALIGN_ITERS):
    """측정 -> 회전을 반복해 잔차를 줄입니다.

    한 번의 회전으로 각도가 정확히 0 이 되지는 않습니다. 마스크가 프레임 안에서
    돌면서 일부가 잘리고 밴드 위치도 달라져 측정값이 비선형으로 움직이기 때문입니다.
    2회면 대개 잔차가 1도 아래로 들어옵니다.
    반환: (정렬된 마스크, 누적 회전각, 잔차각)
    """
    k = keep.copy()
    total = 0.0
    for _ in range(max(1, iters)):
        a = method(k)
        if not np.isfinite(a):
            break
        H, W = k.shape
        M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), a, 1.0)
        k = (cv2.warpAffine(k * 255, M, (W, H), flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 127).astype(np.uint8)
        total += a
        if abs(a) < 0.3:
            break
    res = method(k)
    return k, total, (res if np.isfinite(res) else np.nan)


def rotate_keep(img, keep, deg):
    """이미지와 마스크를 같은 각도로 회전(중심 기준, 원본 크기 유지)."""
    H, W = keep.shape
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), deg, 1.0)
    ri = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rk = cv2.warpAffine(keep * 255, M, (W, H), flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return ri, (rk > 127).astype(np.uint8)


# ────────────────────────────────────────────────────────── 메인
def main():
    # 기본값은 위 [설정] 블록. 필요하면 커맨드라인으로 덮어쓸 수 있습니다.
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(SEG_DIR), help="기본값: 설정 블록의 SEG_DIR")
    ap.add_argument("--n", type=int, default=N_SAMPLE, help="기본값: 설정 블록의 N_SAMPLE")
    ap.add_argument("--out", default=str(OUT_DIR), help="기본값: 설정 블록의 OUT_DIR")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        sys.exit(f"[중단] 경로 없음: {root}\n"
                 f"        스크립트 상단 [설정] 블록의 BASE_DIR / SEG_DIR 을 고치세요.")
    miss = [sp for sp in SPLITS if not (root / sp / IMAGE_SUB).exists()]
    if len(miss) == len(SPLITS):
        sys.exit(f"[중단] {root} 안에서 split 폴더를 찾지 못했습니다: {miss}\n"
                 f"        기대 구조: <SEG_DIR>/{{{','.join(SPLITS)}}}/{{{IMAGE_SUB},{MASK_SUB}}}\n"
                 f"        폴더명이 다르면 [설정] 블록의 SPLIT_DIRS / IMAGE_SUB / MASK_SUB 를 고치세요.")
    if miss:
        print(f"[경고] 다음 split 은 건너뜁니다(폴더 없음): {miss}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    L = []

    def log(s=""):
        print(s); L.append(str(s))

    log("=" * 72)
    log(" 마스크 정렬 사전 진단")
    log("=" * 72)
    log(f"  SEG_DIR   {root}")
    log(f"  OUT_DIR   {out}")
    log(f"  표본      split 당 최대 {args.n} | seed {args.seed}")
    log(f"  폴더규칙  split={SPLITS} · images='{IMAGE_SUB}' · masks='{MASK_SUB}'")
    log(f"  각도설정  전완밴드 {FOREARM_BAND} · 손목밴드 {WRIST_BAND} · 정렬반복 {ALIGN_ITERS}")

    # ── 1. 구조 ──────────────────────────────────────────────
    log("\n[1] 데이터 구조")
    csv_dir = root / CSV_SUB
    for name in [f"{sp}.csv" for sp in SPLITS]:
        p = csv_dir / name
        if not p.exists():
            log(f"  {name:<16} 없음")
            continue
        try:
            df = pd.read_csv(p, encoding="utf-8-sig", nrows=5)
            log(f"  {name:<16} 컬럼 {list(df.columns)}")
            log(f"  {'':16} 예시 {df.iloc[0].to_dict()}")
        except Exception as e:
            log(f"  {name:<16} 읽기 실패: {e}")

    meta = root / META_CSV
    if meta.exists():
        try:
            md = pd.read_csv(meta, encoding="utf-8-sig", nrows=3)
            log(f"  preprocessing_metadata.csv 컬럼 {list(md.columns)}")
            log(f"  {'':26} 예시 {md.iloc[0].to_dict()}")
        except Exception as e:
            log(f"  preprocessing_metadata.csv 읽기 실패: {e}")

    pairs = {}
    for sp in SPLITS:
        ii = index_dir(root / sp / IMAGE_SUB)
        mm = index_dir(root / sp / MASK_SUB)
        both = sorted(set(ii) & set(mm))
        pairs[sp] = [(ii[k], mm[k], k) for k in both]
        log(f"  {sp:<12} images {len(ii):>6} | masks {len(mm):>6} | 쌍 {len(both):>6}"
            f" | 이미지만 {len(set(ii)-set(mm)):>4} | 마스크만 {len(set(mm)-set(ii)):>4}")
    if not any(pairs.values()):
        sys.exit("[중단] 이미지-마스크 쌍을 찾지 못했습니다. --root 경로를 확인하세요.")

    # ── 2~5. 표본 측정 ───────────────────────────────────────
    rng = np.random.RandomState(args.seed)
    rows, gallery = [], []
    for sp, lst in pairs.items():
        if not lst:
            continue
        idx = rng.choice(len(lst), min(args.n, len(lst)), replace=False)
        for j, i in enumerate(idx):
            ip, mp, stem = lst[i]
            img = imread_kr(ip); msk = imread_kr(mp)
            if img is None or msk is None:
                rows.append({"split": sp, "id": stem, "load_fail": 1}); continue
            if msk.shape[:2] != img.shape[:2]:
                msk = cv2.resize(msk, (img.shape[1], img.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
            st = mask_stats(msk)
            if st is None:
                rows.append({"split": sp, "id": stem, "empty_mask": 1}); continue
            keep = st.pop("keep")

            a_pca = ang_pca(keep)
            a_rect = ang_minrect(keep)
            a_fore = ang_forearm(keep)
            a_wr, n_wr = ang_wrist_edge(keep)
            up, r_top, r_bot = finger_side_up(keep)

            r = {"split": sp, "id": stem, "load_fail": 0, "empty_mask": 0}
            r.update(st)
            r.update({"ang_pca": a_pca, "ang_minrect": a_rect, "ang_forearm": a_fore,
                      "ang_wrist": a_wr, "wrist_pts": n_wr,
                      "finger_up": (np.nan if up is None else int(up)),
                      "runs_top": r_top, "runs_bot": r_bot,
                      "thumb": thumb_side(keep)})

            # 정렬 후 잘림 예측 (전완축 기준)
            use = a_fore if np.isfinite(a_fore) else a_pca
            th = np.deg2rad(abs(use))
            bh, bw = st["bbox_h"], st["bbox_w"]
            r["need_h"] = bh * np.cos(th) + bw * np.sin(th)
            r["need_w"] = bh * np.sin(th) + bw * np.cos(th)
            r["clip_h"] = max(0.0, r["need_h"] - st["H"])
            r["clip_w"] = max(0.0, r["need_w"] - st["W"])
            try:
                _, tot, res = align_iter(keep, ang_forearm, iters=ALIGN_ITERS)
                r["align_total"] = tot; r["align_resid"] = res
            except Exception:
                r["align_total"] = np.nan; r["align_resid"] = np.nan
            rows.append(r)

            if sp == SPLIT_DIRS["train"] and len(gallery) < 6:
                gallery.append((stem, img, keep, a_pca, a_rect, a_fore, a_wr, use))

    df = pd.DataFrame(rows)
    df.to_csv(out / "precheck_angles.csv", index=False, encoding="utf-8-sig")
    ok = df[(df.get("load_fail", 0) == 0) & (df.get("empty_mask", 0) == 0)].copy()
    log(f"\n  측정 표본 {len(ok)} / 로드실패 {int(df.get('load_fail', 0).sum())}"
        f" / 빈마스크 {int(df.get('empty_mask', 0).sum())}")

    # ── 2. 마스크 품질 ───────────────────────────────────────
    log("\n[2] 마스크 품질")
    log(f"  이진(값 2개 이하) 비율     {100*ok['binary'].mean():.1f}%")
    log(f"  면적비          중앙 {ok['area_frac'].median():.3f}  "
        f"5% {ok['area_frac'].quantile(.05):.3f}  95% {ok['area_frac'].quantile(.95):.3f}")
    log(f"  연결성분 2개 이상          {100*(ok['components'] > 1).mean():.1f}%")
    log(f"  최대성분 비중   중앙 {ok['main_frac'].median():.4f}  "
        f"5% {ok['main_frac'].quantile(.05):.4f}")
    log(f"  구멍 비율       중앙 {ok['hole_frac'].median():.4f}  "
        f"95% {ok['hole_frac'].quantile(.95):.4f}")
    bad = ok[(ok.main_frac < 0.95) | (ok.hole_frac > 0.02) | (ok.area_frac < 0.05)]
    log(f"  품질 의심 표본             {len(bad)} 건 ({100*len(bad)/max(len(ok),1):.1f}%)")
    if len(bad):
        log(f"    예시 id: {list(bad['id'].head(8))}")

    # ── 3. 손목 잘림 ─────────────────────────────────────────
    log("\n[3] 프레임 경계 접촉  <- 손목(수근골) 잘림 여부")
    for k, lab in [("touch_bottom", "하단"), ("touch_top", "상단"),
                   ("touch_left", "좌"), ("touch_right", "우")]:
        log(f"  {lab} 접촉 {100*ok[k].mean():5.1f}%")
    log(f"  하단 접촉 폭(마지막 행에서 손 픽셀 비율)  중앙 {ok['bottom_touch_frac'].median():.3f}"
        f"  95% {ok['bottom_touch_frac'].quantile(.95):.3f}")
    heavy = ok[ok.bottom_touch_frac > 0.30]
    log(f"  하단이 30% 이상 잘린 표본  {len(heavy)} 건 ({100*len(heavy)/max(len(ok),1):.1f}%)")
    log("  * 이 비율이 높으면 크롭 단계에서 손목이 날아간 것이고,")
    log("    4~8세 구간이 어떤 처방에도 반응하지 않은 이유일 수 있습니다.")

    # ── 4. 각도 ─────────────────────────────────────────────
    log("\n[4] 각도 추정 4가지  (부호: 양수 = 시계방향으로 기울어짐)")
    cols = ["ang_pca", "ang_minrect", "ang_forearm", "ang_wrist"]
    names = {"ang_pca": "PCA 주축", "ang_minrect": "최소외접사각",
             "ang_forearm": "전완축(밴드중심)", "ang_wrist": "손목절단선"}
    for c in cols:
        v = ok[c].dropna()
        if v.empty:
            log(f"  {names[c]:<18} 산출 실패"); continue
        log(f"  {names[c]:<18} n={len(v):>4}  중앙 {v.median():+6.2f}°  "
            f"|각도| 중앙 {v.abs().median():5.2f}°  "
            f"5~95% [{v.quantile(.05):+6.2f}, {v.quantile(.95):+6.2f}]  "
            f"|각도|>10° {100*(v.abs()>10).mean():4.1f}%")
    log(f"  손목절단선 유효점 부족(NaN)  {100*ok['ang_wrist'].isna().mean():.1f}%")

    log("\n  방법간 상관 (높을수록 서로 같은 것을 재고 있음)")
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = ok[cols[i]], ok[cols[j]]
            k = a.notna() & b.notna()
            if k.sum() > 10:
                r = float(np.corrcoef(a[k], b[k])[0, 1])
                d = float((a[k] - b[k]).abs().median())
                log(f"    {names[cols[i]]:<18} vs {names[cols[j]]:<18} "
                    f"corr {r:+.3f}  중앙 차이 {d:5.2f}°")

    # ── 5. 방향 판정 · 정렬 영향 ─────────────────────────────
    log("\n[5] 방향 판정 · 정렬 영향")
    fu = ok["finger_up"]
    log(f"  손끝 위 판정  정상 {100*(fu==1).mean():.1f}%  뒤집힘 {100*(fu==0).mean():.1f}%  "
        f"판정불가 {100*fu.isna().mean():.1f}%")
    tb = ok["thumb"]
    log(f"  엄지 오른쪽 {100*(tb==1).mean():.1f}%  왼쪽 {100*(tb==-1).mean():.1f}%  "
        f"판정불가 {100*(tb==0).mean():.1f}%")
    rs = ok["align_resid"].dropna().abs()
    if len(rs):
        log(f"  정렬 잔차(2회 보정 후 |각도|)  중앙 {rs.median():.2f}°  "
            f"95% {rs.quantile(.95):.2f}°  1° 이내 {100*(rs<1).mean():.1f}%")
        log("  * 잔차 중앙이 1° 아래면 정렬이 신뢰할 만합니다. 3° 이상이면 방법을 바꿔야 합니다.")
    log(f"  정렬 후 예상 잘림  세로>0 인 표본 {100*(ok['clip_h']>1).mean():.1f}%  "
        f"(중앙 {ok['clip_h'].median():.1f}px)")
    log(f"                     가로>0 인 표본 {100*(ok['clip_w']>1).mean():.1f}%  "
        f"(중앙 {ok['clip_w'].median():.1f}px)")
    log("  * 원본 크롭에서 회전 후 타이트 bbox 를 다시 잡으면 이 잘림은 사라집니다.")
    log("    수치가 크다면 회전을 letterbox 이전 단계에 두는 것이 필수라는 뜻입니다.")

    # ── 그림 ────────────────────────────────────────────────
    try:
        v = [ok[c].dropna() for c in cols]
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
        for c, lab in zip(cols, [names[c] for c in cols]):
            s = ok[c].dropna()
            if len(s):
                ax[0].hist(s, bins=60, range=(-45, 45), histtype="step", label=lab, lw=1.6)
        ax[0].set_title("angle distribution (deg)"); ax[0].legend(fontsize=8)
        ax[0].axvline(0, color="k", lw=.8)
        k = ok["ang_forearm"].notna() & ok["ang_wrist"].notna()
        ax[1].scatter(ok.loc[k, "ang_forearm"], ok.loc[k, "ang_wrist"], s=6, alpha=.4)
        lim = 45; ax[1].plot([-lim, lim], [-lim, lim], "r--", lw=.8)
        ax[1].set_xlabel("forearm axis"); ax[1].set_ylabel("wrist edge")
        ax[1].set_title("forearm vs wrist-edge")
        ax[2].hist(ok["bottom_touch_frac"], bins=50)
        ax[2].set_title("bottom frame touch fraction")
        plt.tight_layout(); plt.savefig(out / "qc_hist.png", dpi=110); plt.close()

        if gallery:
            n = len(gallery)
            fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 8.4))
            axes = np.atleast_2d(axes)
            for c, (stem, img, keep, ap_, ar_, af_, aw_, use) in enumerate(gallery):
                H, W = keep.shape
                vis = cv2.cvtColor(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                                   .astype(np.uint8), cv2.COLOR_GRAY2BGR)
                vis[keep > 0] = (0.6 * vis[keep > 0] + 0.4 * np.array([0, 90, 0])).astype(np.uint8)
                cy, cx = H / 2, W / 2
                for a_, col in [(ap_, (255, 0, 0)), (ar_, (0, 165, 255)),
                                (af_, (0, 0, 255)), (aw_, (255, 0, 255))]:
                    if a_ is None or not np.isfinite(a_):
                        continue
                    t = np.deg2rad(a_)
                    dx, dy = np.sin(t) * H * .45, -np.cos(t) * H * .45
                    cv2.line(vis, (int(cx - dx), int(cy - dy)),
                             (int(cx + dx), int(cy + dy)), col, 3)
                axes[0, c].imshow(vis[:, :, ::-1]); axes[0, c].axis("off")
                axes[0, c].set_title(f"{stem}\npca{ap_:+.1f} rect{ar_:+.1f}\n"
                                     f"fore{af_:+.1f} wrist{aw_:+.1f}", fontsize=7)
                ri, rk = rotate_keep(img, keep, use)
                rv = cv2.cvtColor(cv2.normalize(ri, None, 0, 255, cv2.NORM_MINMAX)
                                  .astype(np.uint8), cv2.COLOR_GRAY2BGR)
                rv[rk > 0] = (0.6 * rv[rk > 0] + 0.4 * np.array([0, 90, 0])).astype(np.uint8)
                cv2.line(rv, (0, int(cy)), (W, int(cy)), (0, 0, 255), 1)
                axes[1, c].imshow(rv[:, :, ::-1]); axes[1, c].axis("off")
            axes[0, 0].set_ylabel("before"); axes[1, 0].set_ylabel("after")
            plt.tight_layout(); plt.savefig(out / "qc_angle_methods.png", dpi=110); plt.close()
            log(f"\n  QC 시트: {out/'qc_angle_methods.png'}  (윗줄 원본+4개 축, 아랫줄 정렬 후)")
            log("    파랑=PCA 주황=최소외접 빨강=전완축 자홍=손목절단선")
    except Exception:
        log("\n  [경고] 그림 생성 실패:")
        log(traceback.format_exc(limit=2))

    log("\n" + "=" * 72)
    log(" 판단 기준")
    log("  - [3] 하단 30% 이상 잘림이 10% 넘으면 -> 크롭 재생성이 정렬보다 우선입니다.")
    log("  - [4] 전완축 vs 손목절단선 corr 이 0.8 이상이면 둘 중 아무거나 써도 됩니다.")
    log("        낮으면 손목절단선이 크롭 인공물에 오염된 것이므로 전완축을 쓰세요.")
    log("  - [4] |각도| 중앙이 3° 미만이면 이미 정렬돼 있어 이득이 작습니다.")
    log("        8° 이상이면 제거할 변동이 실제로 크다는 뜻입니다.")
    log("  - [5] 손끝 위 판정 정상이 97% 미만이면 방향 판정 로직을 보강해야 합니다.")
    log("=" * 72)

    (out / "precheck_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print(f"\n[저장] {out/'precheck_summary.txt'}")


if __name__ == "__main__":
    main()
