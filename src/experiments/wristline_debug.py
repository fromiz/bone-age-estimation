# -*- coding: utf-8 -*-
"""
손목 절단선 검출 후보 진단  [읽기 전용 - 아무것도 저장하지 않습니다]

왜 필요한가
    make_wristline_rot.py 의 검출이 실패했습니다.
        검출각 중앙 -32.25도 / |각도|>30도 가 55.2% / 부호 음수 80.1%
    원인은 '경계 아래가 밝으면 절단선' 이라는 판별이 실제 마스크에서
    거의 모든 열을 통과시켜, V 자 경계 전체에 직선을 맞춰버린 것입니다.
    (검출 열 수 중앙값 395개 = 사실상 경계 전부)

    이번에는 실제 마스크를 먼저 눈으로 보고 방법을 고릅니다.
    이 스크립트는 데이터셋을 만들지 않습니다. 그림 한 장과 표만 냅니다.

무엇을 보여주나
    표본 이미지마다 후보 4가지를 각각 다른 색으로 그립니다.
        빨강   M1 전완 좌우 모서리의 평균 축        (축을 수직으로)
        초록   M2 마스크 하단 경계의 최장 직선구간   (그 선을 수평으로)
        파랑   M3 최하단 지점 주변 국소 접선        (그 선을 수평으로)
        노랑   M4 손 전체 PCA 주축                 (축을 수직으로)
    아래 줄에는 각 방법으로 회전시킨 결과를 나란히 보여줍니다.

    어느 색이 원하시는 선인지 알려주시면 그 방법으로 본 스크립트를 고칩니다.

사용법
    python wristline_debug.py
    python wristline_debug.py --n 16 --split train --seed 7
"""
import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("[중단] opencv 필요:  pip install opencv-python")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("[중단] matplotlib 필요:  pip install matplotlib")

# ════════════════════════════════════════════════════════════════════
# [설정]
# ════════════════════════════════════════════════════════════════════
BASE_DIR  = Path(r"G:\Project\sinra_cho")
SRC_DIR   = BASE_DIR / "crop_yolo_seg"
OUT_DIR   = BASE_DIR / "precheck_out"

N_SAMPLE  = 12
SPLIT     = "train"
SEED      = 28
IMAGE_SUB, MASK_SUB = "images", "masks"

FOREARM_BAND = (0.06, 0.26)   # M1: 손 세로길이의 하단 몇 % 구간을 전완으로 볼지
LOCAL_SPAN   = 0.35           # M3: 최하단 지점 좌우로 마스크 폭의 몇 %를 볼지
# ════════════════════════════════════════════════════════════════════

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def imread_u(p, flags=cv2.IMREAD_GRAYSCALE):
    try:
        return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), flags)
    except Exception:
        return None


def main_component(mask):
    bw = (mask > 127).astype(np.uint8)
    if bw.sum() < 200:
        return None
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    if n <= 1:
        return None
    return (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


# ── 후보 1 : 전완 좌우 모서리의 평균 축 ──────────────────────────────
def m1_forearm_axis(kp):
    """하단 밴드에서 좌/우 경계에 각각 직선을 맞추고 두 기울기를 평균냅니다.

    전완은 폭이 거의 일정한 띠라서 양쪽 모서리가 서로 평행합니다.
    무게중심이 아니라 '양쪽 모서리' 를 쓰는 이유는, 엄지두덩이 한쪽에만
    붙어 있어 무게중심이 그쪽으로 끌려가기 때문입니다.
    반환: 축 각도(도). 양수 = 위쪽이 오른쪽으로 기울어짐
    """
    ys = np.nonzero(kp.sum(1))[0]
    if ys.size < 20:
        return np.nan
    y0, y1 = int(ys.min()), int(ys.max())
    h = max(1, y1 - y0)
    rows, L, R = [], [], []
    for y in range(y1 - int(FOREARM_BAND[1] * h), y1 - int(FOREARM_BAND[0] * h) + 1):
        nz = np.nonzero(kp[y])[0]
        if nz.size < 20:
            continue
        rows.append(y); L.append(nz.min()); R.append(nz.max())
    if len(rows) < 30:
        return np.nan
    r = np.asarray(rows, float)
    sl = np.polyfit(r, np.asarray(L, float), 1)[0]      # x = sl*y + b
    sr = np.polyfit(r, np.asarray(R, float), 1)[0]
    return float(np.degrees(np.arctan((sl + sr) / 2.0)))


# ── 후보 2 : 하단 경계의 최장 직선구간 ───────────────────────────────
def m2_longest_straight(kp, tol=3.0, iters=2500, seed=0):
    """하단 경계점 중 '직선에 잘 맞는 가장 긴 구간' 을 찾습니다.
    절단선은 곧고 길지만 해부학적 테이퍼는 휘어서 짧게 끊깁니다.
    반환: (선의 각도, x0, x1) 또는 (nan, 0, 0)
    """
    h_, w_ = kp.shape
    ys = np.nonzero(kp.sum(1))[0]
    y0, y1 = int(ys.min()), int(ys.max())
    hh = max(1, y1 - y0)
    yt = y1 - int(0.30 * hh)
    P = []
    for c in range(w_):
        col = np.nonzero(kp[:, c])[0]
        if col.size == 0:
            continue
        b = int(col.max())
        if b < yt or b >= h_ - 3:
            continue
        P.append((c, b))
    if len(P) < 60:
        return np.nan, 0, 0
    P = np.asarray(P, float)
    rng = np.random.RandomState(seed)
    best = None
    for _ in range(iters):
        i, j = rng.randint(0, len(P), 2)
        dx, dy = P[j, 0] - P[i, 0], P[j, 1] - P[i, 1]
        if abs(dx) < 50:
            continue
        n = np.hypot(dx, dy)
        d = np.abs((P[:, 0] - P[i, 0]) * dy - (P[:, 1] - P[i, 1]) * dx) / n
        inl = d < tol
        if inl.sum() < 50:
            continue
        span = P[inl, 0].max() - P[inl, 0].min()
        if best is None or span > best[0]:
            best = (span, inl)
    if best is None:
        return np.nan, 0, 0
    inl = best[1]
    s = np.polyfit(P[inl, 0], P[inl, 1], 1)[0]
    return float(np.degrees(np.arctan(s))), float(P[inl, 0].min()), float(P[inl, 0].max())


# ── 후보 3 : 최하단 지점 주변 국소 접선 ──────────────────────────────
def m3_local_tangent(kp):
    """마스크가 가장 아래로 내려온 지점 주변만 보고 접선을 구합니다.
    절단선이 있다면 그 최하단부가 곧 절단선입니다.
    반환: (각도, x0, x1)
    """
    h_, w_ = kp.shape
    bot = np.full(w_, -1)
    for c in range(w_):
        col = np.nonzero(kp[:, c])[0]
        if col.size:
            bot[c] = col.max()
    valid = np.nonzero(bot >= 0)[0]
    if valid.size < 60:
        return np.nan, 0, 0
    cmax = int(valid[np.argmax(bot[valid])])
    half = int(LOCAL_SPAN * 0.5 * (valid.max() - valid.min()))
    x0, x1 = max(valid.min(), cmax - half), min(valid.max(), cmax + half)
    xs = np.arange(x0, x1 + 1)
    xs = xs[bot[xs] >= 0]
    if xs.size < 40:
        return np.nan, 0, 0
    s = np.polyfit(xs.astype(float), bot[xs].astype(float), 1)[0]
    return float(np.degrees(np.arctan(s))), float(xs.min()), float(xs.max())


# ── 후보 4 : 손 전체 PCA ─────────────────────────────────────────────
def m4_pca(kp):
    ys, xs = np.nonzero(kp)
    p = np.stack([xs, ys], 1).astype(float)
    p -= p.mean(0)
    _, v = np.linalg.eigh(np.cov(p.T))
    w = v[:, -1]
    a = np.degrees(np.arctan2(w[0], -w[1]))
    return float((a + 90) % 180 - 90)


def rot(img, kp, deg):
    h, w = kp.shape
    r = np.deg2rad(abs(deg))
    nw = int(h * np.sin(r) + w * np.cos(r)); nh = int(h * np.cos(r) + w * np.sin(r))
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    M[0, 2] += nw / 2.0 - w / 2.0; M[1, 2] += nh / 2.0 - h / 2.0
    return (cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=0),
            (cv2.warpAffine(kp * 255, M, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0) > 127
             ).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC_DIR))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--n", type=int, default=N_SAMPLE)
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    src = Path(args.src); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    di, dm = src / args.split / IMAGE_SUB, src / args.split / MASK_SUB
    if not di.exists():
        sys.exit(f"[중단] 경로 없음: {di}")
    files = sorted(p for p in di.iterdir() if p.suffix.lower() in IMG_EXT)
    rng = np.random.RandomState(args.seed)
    pick = [files[i] for i in rng.choice(len(files), min(args.n, len(files)), replace=False)]

    print("=" * 78)
    print(" 손목 절단선 검출 후보 진단  (읽기 전용)")
    print("=" * 78)
    print(f"  {di}  에서 {len(pick)}장")
    print(f"  {'id':>8} {'M1 전완축':>10} {'M2 최장직선':>12} {'M3 국소접선':>12} {'M4 PCA':>9}")

    rows, cards = [], []
    for p in pick:
        img = imread_u(p)
        mp = next((dm / f"{p.stem}{e}" for e in IMG_EXT if (dm / f"{p.stem}{e}").exists()), None)
        msk = imread_u(mp) if mp else None
        if img is None or msk is None:
            print(f"  {p.stem:>8}  로드 실패"); continue
        if msk.shape[:2] != img.shape[:2]:
            msk = cv2.resize(msk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        kp = main_component(msk)
        if kp is None:
            print(f"  {p.stem:>8}  마스크 비어있음"); continue

        a1 = m1_forearm_axis(kp)
        a2, x2a, x2b = m2_longest_straight(kp)
        a3, x3a, x3b = m3_local_tangent(kp)
        a4 = m4_pca(kp)
        print(f"  {p.stem:>8} {a1:>10.1f} {a2:>12.1f} {a3:>12.1f} {a4:>9.1f}")
        rows.append((p.stem, a1, a2, a3, a4))
        cards.append((p.stem, img, kp, a1, (a2, x2a, x2b), (a3, x3a, x3b), a4))

    if not cards:
        sys.exit("[중단] 표본을 하나도 읽지 못했습니다.")

    A = np.array([[r[1], r[2], r[3], r[4]] for r in rows], float)
    print("\n  방법별 |각도| 중앙값:  M1 %.1f  M2 %.1f  M3 %.1f  M4 %.1f"
          % tuple(np.nanmedian(np.abs(A), axis=0)))
    print("  * M1/M4 는 '축', M2/M3 는 '선' 입니다. 축은 수직으로, 선은 수평으로 맞춥니다.")

    # ── 그림 ─────────────────────────────────────────────────────
    n = len(cards)
    fig, ax = plt.subplots(5, n, figsize=(2.3 * n, 13.5), squeeze=False)
    COL = {"M1": (0, 0, 255), "M2": (0, 200, 0), "M3": (255, 120, 0), "M4": (0, 220, 220)}

    def base(img, kp):
        v = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        im = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
        im[kp > 0] = (0.72 * im[kp > 0] + 0.28 * np.array([0, 90, 0])).astype(np.uint8)
        return im

    for c, (stem, img, kp, a1, m2, m3, a4) in enumerate(cards):
        im = base(img, kp)
        h, w = kp.shape
        ys = np.nonzero(kp.sum(1))[0]; y0, y1 = int(ys.min()), int(ys.max())
        cy = (y0 + y1) / 2.0; cx = w / 2.0
        # 축 계열(M1, M4): 세로 선으로 표시
        for nm, a in (("M1", a1), ("M4", a4)):
            if not np.isfinite(a):
                continue
            t = np.deg2rad(a); dx, dy = np.sin(t) * h * .42, -np.cos(t) * h * .42
            cv2.line(im, (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy)), COL[nm], 4)
        # 선 계열(M2, M3): 검출 구간에 가로 선으로 표시
        for nm, (a, xa, xb) in (("M2", m2), ("M3", m3)):
            if not np.isfinite(a) or xb <= xa:
                continue
            t = np.tan(np.deg2rad(a)); mid = y1 - 0.06 * (y1 - y0)
            cv2.line(im, (int(xa), int(mid - t * (xb - xa) / 2)),
                     (int(xb), int(mid + t * (xb - xa) / 2)), COL[nm], 5)
        ax[0, c].imshow(im[:, :, ::-1]); ax[0, c].axis("off")
        ax[0, c].set_title("%s\nM1 %.1f  M2 %.1f\nM3 %.1f  M4 %.1f"
                           % (stem, a1, m2[0], m3[0], a4), fontsize=7)
        # 각 방법으로 회전한 결과
        for r, (nm, d) in enumerate([("M1", -a1 if np.isfinite(a1) else 0.0),
                                     ("M2", m2[0] if np.isfinite(m2[0]) else 0.0),
                                     ("M3", m3[0] if np.isfinite(m3[0]) else 0.0),
                                     ("M4", -a4 if np.isfinite(a4) else 0.0)], start=1):
            ri, rk = rot(img, kp, d)
            v = base(ri, rk)
            cv2.line(v, (v.shape[1] // 2, 0), (v.shape[1] // 2, v.shape[0]), (0, 0, 255), 2)
            ax[r, c].imshow(v[:, :, ::-1]); ax[r, c].axis("off")
            if c == 0:
                ax[r, c].set_title(nm, fontsize=10, loc="left")
    fig.suptitle("Row1: candidate lines on original   Rows2-5: result after rotating by M1/M2/M3/M4",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    pth = out / "qc_wristline_debug.png"
    plt.savefig(pth, dpi=105); plt.close()
    print(f"\n  저장: {pth}")
    print("  1행 = 원본 위 후보선 (파랑 M1 / 초록 M2 / 주황 M3 / 하늘 M4)")
    print("  2~5행 = 각 방법으로 회전시킨 결과. 빨간 세로선과 비교하세요.")
    print("  어느 방법이 원하시는 결과인지 알려주시면 그 방법으로 확정하겠습니다.")


if __name__ == "__main__":
    main()
