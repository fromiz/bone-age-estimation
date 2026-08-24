# -*- coding: utf-8 -*-
"""
마스크를 뭉개서 각도만 뽑고, 그 각도로 원본 이미지를 돌린다.

    crop_yolo_seg  ->  crop_wrist_align        (새 경로. 기존 출력과 섞이지 않음)

── 설계 ─────────────────────────────────────────────────────────────
마스크는 각도를 뽑는 '도구' 일 뿐이므로 마음껏 뭉개도 됩니다. 크게 평활한
마스크에서 손목선 각도를 재고, 그 각도를 **원본 이미지와 원본 마스크**에
적용합니다. 뭉갠 마스크는 저장되지 않으므로 울퉁불퉁함은 각도 측정에만
영향을 주고 결과물에는 남지 않습니다.

각도는 서로 독립인 두 방법으로 재고, 둘이 일치할 때만 회전합니다.
    E1  하단 윤곽의 최장 직선분      = 손목 절단면 자체
    E2  전완 중심선 기울기의 수직    = 손목선은 전완축에 직교
표본 4396 기준 (평활 sigma 0.03H):
    E1 23.80도 · E2 23.91도 · 불일치 0.11도

불일치가 AGREE_TOL 을 넘으면 마스크가 신뢰할 수 없다는 뜻이므로 회전을
포기하고 LOW_CONF 로 기록합니다. 조용히 틀린 각도로 도는 것보다 낫습니다.

── 실행 ─────────────────────────────────────────────────────────────
  1) 검출 검증   python make_wrist_aligned.py --limit 40 --diag
                 -> crop_wrist_align/qc/qc_estimators.png
                 1행에 E1(파랑) E2(초록) 두 선이 검출된 각도 그대로 그려집니다.
                 두 선이 나란하면 신뢰, 벌어지면 그 이미지는 회전 제외 대상.
  2) 소량 생성   python make_wrist_aligned.py --limit 40 --overwrite
  3) 전량        python make_wrist_aligned.py --overwrite
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("[중단] opencv 필요:  pip install opencv-python")
try:
    import pandas as pd
except ImportError:
    sys.exit("[중단] pandas 필요:  pip install pandas")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("[중단] matplotlib 필요:  pip install matplotlib")

# ════════════════════════════════════════════════════════════════════
# [설정]
# ════════════════════════════════════════════════════════════════════
BASE_DIR = Path(r"G:\Project\sinra_cho")
SRC_DIR = BASE_DIR / "crop_yolo_seg"
DST_DIR = BASE_DIR / "crop_wrist_align"        # 새 경로

SPLITS = ["train", "validation", "test"]
IMAGE_SUB, MASK_SUB, CSV_SUB = "images", "masks", "csv"

# ── 마스크 평활 (각도 측정 전용. 저장 안 됨) ─────────────────────────
SM_CLOSE = 0.06        # 닫힘 커널 (마스크 세로길이 대비). 패인 곳을 메움
SM_OPEN = 0.03         # 열림 커널. 튀어나온 혹 제거
SM_SIGMA = 0.030       # 윤곽 좌표 가우시안 평활 표준편차 (세로길이 대비)

# ── E1  하단 최장 직선분 ─────────────────────────────────────────────
E1_BAND = 0.16         # 최하단에서 위로 이 비율까지가 후보 구간
E1_EPS = 0.010         # 다각형 근사 강도 (둘레 대비)
E1_MINLEN = 0.08       # 선분 최소 길이 (세로길이 대비)
E1_MAXTILT = 60.0      # 이보다 기울면 손 옆선으로 보고 제외
E1_TOL = 4.0           # 재적합 시 후보선 주변 허용 화소

# ── E2  전완 중심선 ──────────────────────────────────────────────────
E2_LO, E2_HI = 0.70, 0.92   # 중심선을 재는 세로 구간 (마스크 상단 기준 비율)

# ── 합의 판정 ────────────────────────────────────────────────────────
AGREE_TOL = 8.0        # |E1-E2| 가 이 값 이하일 때만 회전
FALLBACK = "skip"      # 불일치 시: "skip"(회전 안 함) | "e1" | "e2"
ANGLE_MAX = 45.0       # 이 각도를 넘으면 측정 실패로 보고 회전 안 함
ROT_ITERS = 2          # 회전 후 재측정·재회전 반복

# ── 출력 가공 ────────────────────────────────────────────────────────
MASK_OUT = True        # 마스크 밖을 0 으로 (흰 띠·판독 마커 제거)
MASK_FEATHER = 2       # 마스크를 이 화소만큼 침식
CUT_AT_LINE = False    # 손목선 아래를 잘라낼지. 기본 끔 (--cut 으로 켜기)
KEEP_BELOW = 0.00      # 손목선 아래로 남길 길이 (세로길이 대비)
MIN_KEEP = 0.55        # 잘라도 원래 높이의 이 비율은 남겨야 함

MARGIN_FRAC = 0.03
MIN_SIDE = 128
PNG_LEVEL = 3
QC_N = 10
# ════════════════════════════════════════════════════════════════════

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ────────────────────────────────────────────────────────── I/O ─────
def imread_u(p, flags=cv2.IMREAD_GRAYSCALE):
    """한글 경로 안전 로더."""
    try:
        return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), flags)
    except Exception:
        return None


def imwrite_u(p, img, params=None):
    """한글 경로 안전 저장."""
    try:
        ok, buf = cv2.imencode(Path(p).suffix, img, params or [])
        if not ok:
            return False
        buf.tofile(str(p))
        return True
    except Exception:
        return False


def main_component(mask):
    bw = (mask > 127).astype(np.uint8)
    if bw.sum() < 200:
        return None
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    if n <= 1:
        return None
    return (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


# ────────────────────────────────────────── 마스크 평활 ───────────
def smooth_mask(kp):
    """각도 측정 전용 실루엣. 크게 뭉개도 되는 이유는 저장하지 않기 때문."""
    ys = np.nonzero(kp.sum(1))[0]
    if ys.size < 40:
        return kp
    h = max(1, int(ys.max() - ys.min()))

    def ker(f):
        k = max(3, int(f * h) | 1)
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    m = cv2.morphologyEx(kp, cv2.MORPH_CLOSE, ker(SM_CLOSE))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker(SM_OPEN))
    if m.sum() < 0.3 * kp.sum():          # 과도한 형태연산은 되돌림
        m = kp.copy()

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return m
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    sigma = SM_SIGMA * h
    if len(c) > 30 and sigma >= 1.0:
        r = int(3 * sigma)
        if r < len(c) // 2:
            g = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
            g /= g.sum()
            pad = np.vstack([c[-r:], c, c[:r]])          # 순환 경계
            cs = np.stack([np.convolve(pad[:, i], g, mode="valid") for i in (0, 1)], 1)
            out = np.zeros_like(m)
            cv2.fillPoly(out, [cs.astype(np.int32)], 1)
            if out.sum() > 0.3 * kp.sum():
                return out
    return m


# ──────────────────────────────────────────── 각도 추정기 ─────────
def widest_run(row):
    idx = np.nonzero(row)[0]
    if idx.size == 0:
        return None
    br = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    b = max(br, key=len)
    return int(b[0]), int(b[-1])


def est_bottom_edge(kp):
    """E1: 하단 윤곽의 최장 직선분. 반환 dict(p,q,angle,n) 또는 None."""
    ys = np.nonzero(kp.sum(1))[0]
    if ys.size < 40:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    h = max(1, y1 - y0)

    cnts, _ = cv2.findContours(kp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if len(c) < 20:
        return None
    cf = c.reshape(-1, 1, 2).astype(np.float32)
    ap = cv2.approxPolyDP(cf, E1_EPS * cv2.arcLength(cf, True), True)
    ap = ap.reshape(-1, 2).astype(np.float64)
    if len(ap) < 3:
        return None

    ylim = y1 - E1_BAND * h
    best = None
    for i in range(len(ap)):
        p, q = ap[i], ap[(i + 1) % len(ap)]
        if p[1] < ylim or q[1] < ylim:
            continue
        d = q - p
        L = float(np.hypot(d[0], d[1]))
        if L < E1_MINLEN * h:
            continue
        a = float((np.degrees(np.arctan2(d[1], d[0])) + 90) % 180 - 90)
        if abs(a) > E1_MAXTILT:
            continue
        if best is None or L > best[0]:
            best = (L, p.copy(), q.copy(), a)
    if best is None:
        return None

    L, p, q, a = best
    d = (q - p) / max(L, 1e-9)
    nv = np.array([-d[1], d[0]])
    t = (c - p) @ d
    sel = (np.abs((c - p) @ nv) < E1_TOL) & (t > -0.15 * L) & (t < 1.15 * L)
    P = c[sel]
    npts = int(len(P))
    if npts >= 20:
        Q = P - P.mean(0)
        _, v = np.linalg.eigh(np.cov(Q.T))
        w = v[:, -1]
        a2 = float((np.degrees(np.arctan2(w[1], w[0])) + 90) % 180 - 90)
        if abs(a2 - a) < 15.0:
            a = a2
            d = np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))])
        tt = (P - p) @ d
        p = p + d * float(tt.min())
        q = p + d * float(tt.max() - tt.min())
    return {"p": (float(p[0]), float(p[1])), "q": (float(q[0]), float(q[1])),
            "angle": a, "n": npts}


def est_forearm_axis(kp):
    """E2: 전완 중심선의 수직. 손목선은 전완축에 직교하므로 등가각을 준다."""
    ys = np.nonzero(kp.sum(1))[0]
    if ys.size < 60:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    h = max(1, y1 - y0)
    W = kp.shape[1]

    pts = []
    for y in range(int(y0 + E2_LO * h), min(y1, int(y0 + E2_HI * h)) + 1):
        r = widest_run(kp[y])
        if r is None or r[0] <= 1 or r[1] >= W - 2:      # 화면 밖으로 잘린 행 제외
            continue
        pts.append((y, 0.5 * (r[0] + r[1])))
    if len(pts) < 20:
        return None
    P = np.array(pts, dtype=np.float64)

    k = max(1, len(P) // 60)
    S = P[::k]
    sl = [(S[j, 1] - S[i, 1]) / (S[j, 0] - S[i, 0])
          for i in range(len(S)) for j in range(i + 1, len(S)) if S[j, 0] != S[i, 0]]
    if not sl:
        return None
    tilt = float(np.degrees(np.arctan(np.median(sl))))   # 세로 대비 중심선 기울기
    a = -tilt                                            # 직교 관계 -> 손목선 각도

    cx = float(np.median(P[:, 1]))
    cy = float(y1 - 0.03 * h)
    d = np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))])
    L = 0.30 * h
    return {"p": (cx - d[0] * L, cy - d[1] * L), "q": (cx + d[0] * L, cy + d[1] * L),
            "angle": a, "n": int(len(P))}


def measure(kp_raw):
    """평활 마스크에서 두 추정기를 돌리고 합의 판정.

    반환 (angle 또는 None, info dict)
    """
    sm = smooth_mask(kp_raw)
    e1 = est_bottom_edge(sm)
    e2 = est_forearm_axis(sm)
    info = {"e1": e1, "e2": e2, "smooth": sm,
            "a1": None if e1 is None else e1["angle"],
            "a2": None if e2 is None else e2["angle"],
            "diff": None, "conf": "NONE"}

    if e1 is None and e2 is None:
        info["conf"] = "NO_ESTIMATE"
        return None, info
    if e1 is None or e2 is None:
        info["conf"] = "SINGLE"                          # 한쪽만 나오면 회전 보류
        return None, info

    diff = abs(e1["angle"] - e2["angle"])
    info["diff"] = float(diff)
    if diff <= AGREE_TOL:
        info["conf"] = "HIGH"
        a = 0.5 * (e1["angle"] + e2["angle"])
    else:
        info["conf"] = "LOW"
        if FALLBACK == "e1":
            a = e1["angle"]
        elif FALLBACK == "e2":
            a = e2["angle"]
        else:
            return None, info
    if abs(a) > ANGLE_MAX:
        info["conf"] = "ANGLE_REJECT"
        return None, info
    return float(a), info


# ────────────────────────────────────────────────── 회전/가공 ─────
def finger_up(kp):
    """손끝이 위쪽인지. 수직 단면의 연결 조각 수로 판정."""
    ys, _ = np.nonzero(kp)
    y0, y1 = int(ys.min()), int(ys.max())
    h = max(1, y1 - y0)

    def runs(band):
        if band.size == 0:
            return 0.0
        d = np.diff(np.pad((band > 0).astype(np.int8), ((0, 0), (1, 1))), axis=1)
        return float((d == 1).sum(1).mean())

    t = runs(kp[y0:y0 + int(0.25 * h)])
    b = runs(kp[max(y0, y1 - int(0.25 * h)):y1])
    if abs(t - b) < 0.35:
        return None
    return bool(t > b)


def rot_pair(img, kp, deg, expand=True):
    """원본 이미지와 원본 마스크를 같은 각도로 회전. 양수 = 반시계."""
    h, w = kp.shape
    if expand:
        r = np.deg2rad(abs(deg))
        nw = int(h * np.sin(r) + w * np.cos(r))
        nh = int(h * np.cos(r) + w * np.sin(r))
    else:
        nw, nh = w, h
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    M[0, 2] += nw / 2.0 - w / 2.0
    M[1, 2] += nh / 2.0 - h / 2.0
    ri = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rk = cv2.warpAffine(kp * 255, M, (nw, nh), flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return ri, (rk > 127).astype(np.uint8)


def align(img, kp):
    """180도 뒤집기 + 합의 각도로 회전. 회전은 항상 원본 화소에 적용한다.

    반환 (img, kp, info0, info1, total, flipped, status)
    """
    flipped = False
    if finger_up(kp) is False:
        img, kp = rot_pair(img, kp, 180.0, expand=False)
        flipped = True

    a, info0 = measure(kp)
    if a is None:
        return img, kp, info0, info0, 0.0, flipped, info0["conf"]

    total = 0.0
    info = info0
    for _ in range(max(1, ROT_ITERS)):
        if abs(a) < 0.15 or abs(total + a) > ANGLE_MAX:
            break
        img, kp = rot_pair(img, kp, a, expand=True)
        total += a
        a2, info = measure(kp)
        if a2 is None:
            break
        a = a2
    return img, kp, info0, info, total, flipped, "OK"


def finish(img, kp, info):
    """배경 제거 + (선택) 손목선 절단 + 크롭. 반환 (img, kp, y_cut)"""
    if MASK_OUT:
        m = kp
        if MASK_FEATHER > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * MASK_FEATHER + 1, 2 * MASK_FEATHER + 1))
            m = cv2.erode(kp, k)
            if m.sum() < 0.5 * kp.sum():
                m = kp
        img = np.where(m > 0, img, 0).astype(np.uint8)
        kp = m

    ys = np.nonzero(kp.sum(1))[0]
    if ys.size < 40:
        return None, None, None
    y0, y1 = int(ys.min()), int(ys.max())
    h = max(1, y1 - y0)

    y_cut = None
    if CUT_AT_LINE and info is not None and info.get("e1") is not None:
        e = info["e1"]
        yl = 0.5 * (e["p"][1] + e["q"][1])
        y_cut = int(min(y1, yl + KEEP_BELOW * h))
        if y_cut <= y0 + MIN_KEEP * h:
            y_cut = None
    if y_cut is not None:
        kp = kp.copy(); kp[y_cut + 1:] = 0
        img = img.copy(); img[y_cut + 1:] = 0

    ys, xs = np.nonzero(kp)
    if ys.size < 100:
        return None, None, None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    pad = int(MARGIN_FRAC * min(y1 - y0 + 1, x1 - x0 + 1))
    hh, ww = kp.shape
    y0 = max(0, y0 - pad); x0 = max(0, x0 - pad); x1 = min(ww - 1, x1 + pad)
    y1 = min(hh - 1, y1 if y_cut is not None else y1 + pad)
    if y1 - y0 < MIN_SIDE or x1 - x0 < MIN_SIDE:
        return None, None, None
    return img[y0:y1 + 1, x0:x1 + 1], kp[y0:y1 + 1, x0:x1 + 1], y_cut


# ─────────────────────────────────────────────────────── QC ───────
def draw_line(vv, line, color, thick=3):
    if line is None:
        return
    p = np.array(line["p"], dtype=np.float64)
    q = np.array(line["q"], dtype=np.float64)
    d = q - p
    L = float(max(np.hypot(d[0], d[1]), 1e-6))
    d = d / L
    a = (p - d * 0.7 * L).astype(int)
    b = (q + d * 0.7 * L).astype(int)
    cv2.line(vv, tuple(a), tuple(b), color, 2, cv2.LINE_AA)
    cv2.line(vv, tuple(p.astype(int)), tuple(q.astype(int)), color, thick, cv2.LINE_AA)


def overlay(im, kp, info=None, show_smooth=True):
    """원본 위에 평활 윤곽(노랑) · E1(파랑) · E2(초록)을 검출된 각도 그대로."""
    v = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vv = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
    if kp is not None:
        vv[kp > 0] = (0.78 * vv[kp > 0] + 0.22 * np.array([0, 90, 0])).astype(np.uint8)
    cv2.line(vv, (vv.shape[1] // 2, 0), (vv.shape[1] // 2, vv.shape[0]), (0, 0, 255), 1)
    if info is not None:
        if show_smooth and info.get("smooth") is not None:
            cs, _ = cv2.findContours(info["smooth"], cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vv, cs, -1, (0, 220, 220), 2)      # 평활 윤곽 = 노랑
        draw_line(vv, info.get("e1"), (255, 60, 0), 4)          # E1 = 파랑
        draw_line(vv, info.get("e2"), (60, 220, 60), 3)         # E2 = 초록
    return vv[:, :, ::-1]


def save_sheet(rows, titles, path, col_titles):
    n = len(rows[0])
    fig, ax = plt.subplots(len(rows), n, figsize=(2.5 * n, 4.0 * len(rows)),
                           squeeze=False)
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            if im is None:
                ax[r, c].axis("off")
                continue
            ax[r, c].imshow(im)
            ax[r, c].axis("off")
            if c == 0:
                ax[r, c].text(0.0, -0.03, titles[r], transform=ax[r, c].transAxes,
                              fontsize=10, ha="left", va="top")
    for c in range(min(n, len(col_titles))):
        ax[0, c].set_title(col_titles[c], fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


# ────────────────────────────────────────────────────── main ──────
def main():
    global CUT_AT_LINE, FALLBACK
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC_DIR))
    ap.add_argument("--dst", default=str(DST_DIR))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="split 당 최대 처리 수")
    ap.add_argument("--diag", action="store_true",
                    help="추정기 검증만. 저장하지 않고 QC 시트만 생성")
    ap.add_argument("--cut", action="store_true", help="손목선 아래 절단 켜기")
    ap.add_argument("--fallback", default=FALLBACK, choices=["skip", "e1", "e2"],
                    help="두 추정기 불일치 시 동작")
    args = ap.parse_args()
    if args.cut:
        CUT_AT_LINE = True
    FALLBACK = args.fallback

    src, dst = Path(args.src), Path(args.dst)
    if not src.exists():
        sys.exit(f"[중단] 입력 경로 없음: {src}\n"
                 f"        [설정] 의 BASE_DIR / SRC_DIR 을 고치세요.")
    if str(dst) == str(src):
        sys.exit("[중단] 출력 경로가 입력과 같습니다.")
    (dst / "qc").mkdir(parents=True, exist_ok=True)

    if not args.diag and (src / CSV_SUB).exists():
        (dst / CSV_SUB).mkdir(exist_ok=True)
        for p in (src / CSV_SUB).glob("*.csv"):
            (dst / CSV_SUB / p.name).write_bytes(p.read_bytes())

    print("=" * 74)
    print(" 평활 마스크에서 각도 측정 -> 원본 회전" +
          ("   [진단 모드: 저장 안 함]" if args.diag else ""))
    print("=" * 74)
    print(f"  입력   {src}")
    print(f"  출력   {dst}")
    print(f"  평활   close {SM_CLOSE} · open {SM_OPEN} · 윤곽 sigma {SM_SIGMA}")
    print(f"  합의   |E1-E2| <= {AGREE_TOL:.1f}도 · 불일치 시 {FALLBACK}")
    print(f"  가공   배경제거 {MASK_OUT} · 절단 {'ON' if CUT_AT_LINE else 'OFF'}")
    if args.limit:
        print(f"  [미리보기] split 당 {args.limit}장")

    recs, gal = [], []
    t_start = time.time()
    prm = [int(cv2.IMWRITE_PNG_COMPRESSION), PNG_LEVEL]

    for sp in SPLITS:
        sdi, sdm = src / sp / IMAGE_SUB, src / sp / MASK_SUB
        if not sdi.exists():
            print(f"  [건너뜀] {sp}")
            continue
        if not args.diag:
            (dst / sp / IMAGE_SUB).mkdir(parents=True, exist_ok=True)
            (dst / sp / MASK_SUB).mkdir(parents=True, exist_ok=True)

        files = sorted(p for p in sdi.glob("*") if p.suffix.lower() in IMG_EXT)
        if args.limit:
            files = files[:args.limit]
        step = max(1, len(files) // max(1, QC_N))
        print(f"\n  [{sp}] {len(files):,}장")
        t0, done, fail = time.time(), 0, 0

        for i, ip in enumerate(files):
            stem = ip.stem
            op = dst / sp / IMAGE_SUB / f"{stem}.png"
            if (not args.diag) and op.exists() and not args.overwrite:
                done += 1
                continue

            mp = next((sdm / f"{stem}{e}" for e in IMG_EXT
                       if (sdm / f"{stem}{e}").exists()), None)
            rec = {"split": sp, "id": stem, "status": "OK"}
            img = imread_u(ip)
            msk = imread_u(mp) if mp else None
            if img is None or msk is None:
                rec["status"] = "LOAD_FAIL"; recs.append(rec); fail += 1; continue
            if msk.shape[:2] != img.shape[:2]:
                msk = cv2.resize(msk, (img.shape[1], img.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
            kp = main_component(msk)
            if kp is None:
                rec["status"] = "MASK_EMPTY"; recs.append(rec); fail += 1; continue

            rec["src_h"], rec["src_w"] = int(img.shape[0]), int(img.shape[1])
            img0, kp0 = img.copy(), kp.copy()

            ai, ak, i0, i1, total, flipped, st = align(img, kp)
            rec.update({"rot_total": total, "flipped180": int(flipped),
                        "conf": i0["conf"], "angle_e1": i0["a1"],
                        "angle_e2": i0["a2"], "agree_diff": i0["diff"],
                        "resid_e1": None if i1 is None else i1["a1"],
                        "resid_e2": None if i1 is None else i1["a2"]})

            take_qc = (sp == "train" and len(gal) < QC_N and i % step == 0)

            if args.diag:
                recs.append(rec)
                done += 1
                if take_qc:
                    gal.append((stem, img0, kp0, i0, ai, ak, i1, None, None, total,
                                i0["conf"]))
                continue

            if st != "OK":
                # 회전은 못 했지만 배경제거·크롭은 그대로 수행해 데이터 수를 유지
                rec["status"] = st
            out, okm, y_cut = finish(ai.copy(), ak.copy(), i1)
            if out is None:
                rec["status"] = "TOO_SMALL"; recs.append(rec); fail += 1; continue
            if not imwrite_u(op, out, prm) or \
               not imwrite_u(dst / sp / MASK_SUB / f"{stem}.png", okm * 255, prm):
                rec["status"] = "WRITE_FAIL"; recs.append(rec); fail += 1; continue

            rec.update({"cut_row": -1 if y_cut is None else int(y_cut),
                        "out_h": int(out.shape[0]), "out_w": int(out.shape[1]),
                        "out_ar": float(out.shape[0] / out.shape[1]),
                        "area": float(okm.mean())})
            recs.append(rec)
            done += 1
            if take_qc:
                gal.append((stem, img0, kp0, i0, ai, ak, i1, out, okm, total, i0["conf"]))
            if (i + 1) % 1000 == 0:
                el = time.time() - t0
                print(f"    {i+1:>6,}/{len(files):,}  {el:5.0f}초 "
                      f"({(i+1)/max(el,1e-9):.0f}장/초)")
        print(f"    완료 {done:,} / 실패 {fail:,}  ({time.time()-t0:.0f}초)")

    df = pd.DataFrame(recs)
    if not args.diag and len(df):
        df.to_csv(dst / "wrist_metadata.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 74)
    print(" 요약")
    print("=" * 74)
    if len(df):
        print("[신뢰도]")
        print(df["conf"].value_counts().to_string())
        n_hi = int((df["conf"] == "HIGH").sum())
        print(f"  -> 회전 적용 가능 {n_hi:,} / {len(df):,} ({100*n_hi/max(len(df),1):.1f}%)")
        print("\n[상태]")
        print(df["status"].value_counts().to_string())
        for c, lab in [("angle_e1", "E1 각도 |.|"), ("angle_e2", "E2 각도 |.|"),
                       ("agree_diff", "불일치 |E1-E2|"), ("rot_total", "적용 회전각 |.|"),
                       ("resid_e1", "회전 후 E1 잔차 |.|")]:
            if c not in df.columns:
                continue
            v = pd.to_numeric(df[c], errors="coerce").dropna().abs()
            if not len(v):
                continue
            print(f"  {lab:<18} 중앙 {v.median():6.2f}  90% {v.quantile(.90):6.2f}  "
                  f"최대 {v.max():6.2f}")
    print(f"\n  총 소요 {time.time()-t_start:.0f}초")

    if gal:
        try:
            cols = [f"{g[0]}  rot {g[9]:+.1f}\n{g[10]}"
                    + ("" if g[3]["diff"] is None else f"  d={g[3]['diff']:.1f}")
                    for g in gal]
            r1 = [overlay(g[1], g[2], g[3]) for g in gal]
            r2 = [overlay(g[4], g[5], g[6]) for g in gal]
            if args.diag:
                out_p = dst / "qc" / "qc_estimators.png"
                save_sheet([r1, r2],
                           ["before | E1 blue  E2 green  smoothed outline yellow",
                            "after rotation | both lines must be flat"], out_p, cols)
            else:
                r3 = [overlay(g[7], g[8]) if g[7] is not None else None for g in gal]
                out_p = dst / "qc" / "qc_wrist.png"
                save_sheet([r1, r2, r3],
                           ["before | E1 blue  E2 green  smoothed outline yellow",
                            "after rotation", "output"], out_p, cols)
            print(f"  QC 시트: {out_p}")
            print("  1행에서 파랑·초록이 나란한지 보세요. 벌어진 이미지가 LOW 입니다.")
        except Exception as e:
            print(f"  [경고] QC 시트 실패: {e}")

    if not args.diag:
        info = {"src": str(src), "sm_close": SM_CLOSE, "sm_open": SM_OPEN,
                "sm_sigma": SM_SIGMA, "agree_tol": AGREE_TOL, "fallback": FALLBACK,
                "angle_max": ANGLE_MAX, "mask_out": MASK_OUT,
                "cut_at_line": CUT_AT_LINE,
                "n_ok": int((df.status == "OK").sum()) if len(df) else 0,
                "n_total": int(len(df))}
        (dst / "_DONE_wrist.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n  완료 표식 저장.")


if __name__ == "__main__":
    main()
