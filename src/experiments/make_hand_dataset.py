# -*- coding: utf-8 -*-
"""
손목 끝단 직선을 찾아 수평으로 세우는 정렬 파이프라인  (출력 데이터셋 1개)

    crop_yolo_seg  ->  crop_hand_align

정의
    손목선 : 마스크 하단 윤곽에서 '전완이 끝나면서 생긴 직선 절단면'.
             윤곽선을 다각형 근사한 뒤, 하단 밴드 안에 있는 선분 중
             가장 긴 것을 고르고, 그 선 주변 윤곽점으로 다시 최소제곱
             적합(TLS)하여 각도를 확정합니다.
    회전   : 그 선이 수평이 되도록 (= 세로축과 수직) 이미지를 돌립니다.
    절단   : 회전 후 그 선을 따라 아래를 잘라냅니다.  KEEP_BELOW 로 여유 조절.

중요
    먼저 --diag 로 '검출된 손목선'을 원본 위에 그려서 눈으로 확인하세요.
    선이 손목 끝단에 제대로 얹히지 않으면 회전은 볼 필요도 없습니다.
    --diag 는 아무것도 저장하지 않고 QC 시트만 만듭니다.

실행
    1) 검출 검증      python make_hand_dataset.py --limit 40 --diag
                      -> crop_hand_align/qc/qc_wristline.png
    2) 소량 생성      python make_hand_dataset.py --limit 40 --overwrite
                      -> crop_hand_align/qc/qc_hand.png
    3) 전량           python make_hand_dataset.py --overwrite
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
DST_DIR = BASE_DIR / "crop_hand_align"      # 출력은 이 하나뿐입니다

SPLITS = ["train", "validation", "test"]
IMAGE_SUB, MASK_SUB, CSV_SUB = "images", "masks", "csv"

# ── 손목선 검출 ──────────────────────────────────────────────────────
EDGE_BAND = 0.16      # 마스크 최하단에서 위로 이 비율(세로길이 대비)까지가 후보 구간
EDGE_EPS = 0.010      # 윤곽 다각형 근사 강도 (둘레 대비). 작을수록 잘게 쪼갬
EDGE_MINLEN = 0.08    # 선분 최소 길이 (세로길이 대비)
EDGE_MAXTILT = 60.0   # 이보다 더 기운 선분은 손목선 후보에서 제외 (손 옆선 배제)
REFIT_TOL = 4.0       # 후보선 주변 이 화소 안의 윤곽점으로 재적합
ROT_ITERS = 2         # 회전 후 재검출·재회전 반복
ANGLE_MAX = 45.0      # 이 각도를 넘으면 검출 실패로 보고 회전하지 않음

# ── 배경 제거 ────────────────────────────────────────────────────────
MASK_OUT = True
MASK_FEATHER = 2

# ── 절단 ─────────────────────────────────────────────────────────────
CUT_AT_LINE = True    # 정렬 후 손목선 아래를 잘라낼지
KEEP_BELOW = 0.00     # 손목선 아래로 남길 길이 (세로길이 대비). 0 = 선에서 정확히 절단
MIN_KEEP = 0.55       # 잘라도 원래 마스크 높이의 이 비율은 남겨야 함 (안전장치)

# ── 크롭 / 저장 ──────────────────────────────────────────────────────
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


# ───────────────────────────────────────────── 손목선 검출 ─────────
def outer_contour(kp):
    cnts, _ = cv2.findContours(kp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)


def wrist_line(kp):
    """마스크 하단의 손목 절단면 직선을 찾습니다.

    반환 dict(p=(x,y), q=(x,y), angle=도, length=화소, npts=재적합 점수)
    실패하면 None.
    """
    ys = np.nonzero(kp.sum(1))[0]
    if ys.size < 40:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    h = max(1, y1 - y0)

    c = outer_contour(kp)
    if c is None or len(c) < 20:
        return None

    cf = c.reshape(-1, 1, 2).astype(np.float32)
    peri = cv2.arcLength(cf, True)
    ap = cv2.approxPolyDP(cf, EDGE_EPS * peri, True).reshape(-1, 2).astype(np.float64)
    if len(ap) < 3:
        return None

    ylim = y1 - EDGE_BAND * h          # 후보 구간: 최하단에서 위로 EDGE_BAND
    best = None
    n = len(ap)
    for i in range(n):
        p, q = ap[i], ap[(i + 1) % n]
        if p[1] < ylim or q[1] < ylim:          # 두 끝이 모두 하단 밴드 안이어야 함
            continue
        d = q - p
        L = float(np.hypot(d[0], d[1]))
        if L < EDGE_MINLEN * h:
            continue
        a = np.degrees(np.arctan2(d[1], d[0]))
        a = (a + 90) % 180 - 90                 # -90 ~ +90
        if abs(a) > EDGE_MAXTILT:               # 세로에 가까운 손 옆선 배제
            continue
        if best is None or L > best[0]:
            best = (L, p.copy(), q.copy(), float(a))
    if best is None:
        return None

    L, p, q, a = best
    # 후보선 주변 윤곽점으로 재적합 (다각형 근사가 끊은 부분까지 회수)
    d = (q - p) / max(L, 1e-9)
    nv = np.array([-d[1], d[0]])
    t = (c - p) @ d
    s = np.abs((c - p) @ nv)
    sel = (s < REFIT_TOL) & (t > -0.15 * L) & (t < 1.15 * L)
    P = c[sel]
    npts = int(len(P))
    if npts >= 20:
        Q = P - P.mean(0)
        _, v = np.linalg.eigh(np.cov(Q.T))
        w = v[:, -1]
        a2 = np.degrees(np.arctan2(w[1], w[0]))
        a2 = float((a2 + 90) % 180 - 90)
        if abs(a2 - a) < 15.0:                  # 재적합이 튀면 원래 값 유지
            a = a2
            d = np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))])
        tt = (P - p) @ d
        p = p + d * float(tt.min())
        q = p + d * float(tt.max() - tt.min())
        L = float(np.hypot(q[0] - p[0], q[1] - p[1]))

    return {"p": (float(p[0]), float(p[1])), "q": (float(q[0]), float(q[1])),
            "angle": float(a), "length": L, "npts": npts}


# ────────────────────────────────────────────────── 회전/정렬 ──────
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
    """이미지와 마스크를 같은 각도로 회전. 양수 = 반시계."""
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
    """180도 뒤집기 + 손목선 수평 정렬.

    반환 (img, kp, line_before, line_after, total_deg, flipped, status)
    """
    flipped = False
    if finger_up(kp) is False:
        img, kp = rot_pair(img, kp, 180.0, expand=False)
        flipped = True

    ln0 = wrist_line(kp)
    if ln0 is None:
        return img, kp, None, None, 0.0, flipped, "NO_WRISTLINE"
    if abs(ln0["angle"]) > ANGLE_MAX:
        return img, kp, ln0, ln0, 0.0, flipped, "ANGLE_REJECT"

    total = 0.0
    ln = ln0
    for _ in range(max(1, ROT_ITERS)):
        a = ln["angle"]
        if abs(a) < 0.15:
            break
        if abs(total + a) > ANGLE_MAX:
            break
        img, kp = rot_pair(img, kp, a, expand=True)   # 양수 = 반시계 -> 선이 수평
        total += a
        nl = wrist_line(kp)
        if nl is None:
            break
        ln = nl
    return img, kp, ln0, ln, total, flipped, "OK"


# ──────────────────────────────────────────────── 절단 / 크롭 ──────
def finish(img, kp, line):
    """배경 제거 + (손목선 절단) + 크롭.  반환 (img, kp, y_cut)"""
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
    if CUT_AT_LINE and line is not None:
        yl = 0.5 * (line["p"][1] + line["q"][1])          # 정렬 후이므로 수평선
        y_cut = int(min(y1, yl + KEEP_BELOW * h))
        if y_cut <= y0 + MIN_KEEP * h:                    # 과도 절단 방지
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
def overlay(im, kp, line=None, color=(255, 0, 0)):   # BGR (255,0,0) -> 화면에서 파란색
    """미리보기 한 장. line 이 있으면 검출된 각도 그대로 그립니다."""
    v = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vv = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
    if kp is not None:
        vv[kp > 0] = (0.75 * vv[kp > 0] + 0.25 * np.array([0, 90, 0])).astype(np.uint8)
    cv2.line(vv, (vv.shape[1] // 2, 0), (vv.shape[1] // 2, vv.shape[0]), (0, 0, 255), 2)
    if line is not None:
        p = np.array(line["p"], dtype=np.float64)
        q = np.array(line["q"], dtype=np.float64)
        d = q - p
        L = float(max(np.hypot(d[0], d[1]), 1e-6))
        d = d / L
        a = (p - d * 0.6 * L).astype(int)      # 검출 구간을 양쪽으로 연장해 표시
        b = (q + d * 0.6 * L).astype(int)
        cv2.line(vv, tuple(a), tuple(b), color, 2, cv2.LINE_AA)
        cv2.line(vv, tuple(p.astype(int)), tuple(q.astype(int)), color, 5, cv2.LINE_AA)
    return vv[:, :, ::-1]


def save_sheet(rows, titles, path, col_titles):
    n = len(rows[0])
    fig, ax = plt.subplots(len(rows), n, figsize=(2.4 * n, 3.9 * len(rows)),
                           squeeze=False)
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            if im is None:
                ax[r, c].axis("off")
                continue
            ax[r, c].imshow(im)
            ax[r, c].axis("off")
            if c == 0:
                ax[r, c].text(0.0, -0.04, titles[r], transform=ax[r, c].transAxes,
                              fontsize=10, ha="left", va="top")
    for c in range(min(n, len(col_titles))):
        ax[0, c].set_title(col_titles[c], fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


# ────────────────────────────────────────────────────── main ──────
def main():
    global CUT_AT_LINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC_DIR))
    ap.add_argument("--dst", default=str(DST_DIR))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="split 당 최대 처리 수")
    ap.add_argument("--diag", action="store_true",
                    help="손목선 검출만 확인. 저장하지 않고 QC 시트만 생성")
    ap.add_argument("--no-cut", action="store_true", help="절단하지 않음")
    args = ap.parse_args()
    if args.no_cut:
        CUT_AT_LINE = False

    src, dst = Path(args.src), Path(args.dst)
    if not src.exists():
        sys.exit(f"[중단] 입력 경로 없음: {src}\n"
                 f"        [설정] 의 BASE_DIR / SRC_DIR 을 고치세요.")
    (dst / "qc").mkdir(parents=True, exist_ok=True)

    if not args.diag and (src / CSV_SUB).exists():
        (dst / CSV_SUB).mkdir(exist_ok=True)
        for p in (src / CSV_SUB).glob("*.csv"):
            (dst / CSV_SUB / p.name).write_bytes(p.read_bytes())

    print("=" * 72)
    print(" 손목선 검출 -> 수평 정렬" + ("  [진단 모드: 저장 안 함]" if args.diag else ""))
    print("=" * 72)
    print(f"  입력   {src}")
    print(f"  출력   {dst}")
    print(f"  검출   밴드 {EDGE_BAND} · 최소길이 {EDGE_MINLEN} · "
          f"최대기울기 {EDGE_MAXTILT:.0f}도")
    print(f"  절단   {'ON' if CUT_AT_LINE else 'OFF'} (손목선 아래 {KEEP_BELOW:.2f} 유지)")
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

        files = sorted(p for p in sdi.iterdir() if p.suffix.lower() in IMG_EXT)
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

            ai, ak, ln0, ln1, total, flipped, st = align(img, kp)
            rec.update({"rot_total": total, "flipped180": int(flipped), "align": st,
                        "angle_before": None if ln0 is None else ln0["angle"],
                        "angle_after": None if ln1 is None else ln1["angle"],
                        "line_len": None if ln0 is None else ln0["length"],
                        "line_npts": None if ln0 is None else ln0["npts"]})

            take_qc = (sp == "train" and len(gal) < QC_N and i % step == 0)

            if st != "OK":
                rec["status"] = st
                recs.append(rec)
                fail += 1
                if take_qc:
                    gal.append((stem, img0, kp0, ln0, ai, ak, ln1, None, None, total, st))
                continue

            if args.diag:
                recs.append(rec)
                done += 1
                if take_qc:
                    gal.append((stem, img0, kp0, ln0, ai, ak, ln1, None, None, total, st))
                continue

            out, okm, y_cut = finish(ai.copy(), ak.copy(), ln1)
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
                gal.append((stem, img0, kp0, ln0, ai, ak, ln1, out, okm, total, st))
            if (i + 1) % 1000 == 0:
                el = time.time() - t0
                print(f"    {i+1:>6,}/{len(files):,}  {el:5.0f}초 "
                      f"({(i+1)/max(el,1e-9):.0f}장/초)")
        print(f"    완료 {done:,} / 실패 {fail:,}  ({time.time()-t0:.0f}초)")

    df = pd.DataFrame(recs)
    if not args.diag and len(df):
        df.to_csv(dst / "hand_metadata.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 72)
    print(" 요약")
    print("=" * 72)
    if len(df):
        print(df["status"].value_counts().to_string())
        for c, lab in [("angle_before", "손목선 각도 |.|"),
                       ("angle_after", "정렬 후 잔차 |.|"),
                       ("rot_total", "적용 회전각 |.|")]:
            if c not in df.columns:
                continue
            v = pd.to_numeric(df[c], errors="coerce").dropna().abs()
            if not len(v):
                continue
            print(f"  {lab:<16} 중앙 {v.median():6.2f}도  90% {v.quantile(.90):6.2f}  "
                  f"최대 {v.max():6.2f}")
        if "flipped180" in df.columns:
            fl = pd.to_numeric(df.flipped180, errors="coerce").fillna(0).sum()
            print(f"  180도 뒤집기      {int(fl):,} 건")
    print(f"\n  총 소요 {time.time()-t_start:.0f}초")

    # ── QC 시트 ─────────────────────────────────────────────────────
    if gal:
        try:
            names = [g[0] for g in gal]
            cols = [f"{n}  rot {g[9]:+.1f}  {g[10]}" for n, g in zip(names, gal)]
            r1 = [overlay(g[1], g[2], g[3], (255, 0, 0)) for g in gal]   # 원본+검출선
            r2 = [overlay(g[4], g[5], g[6], (255, 160, 0)) for g in gal]   # 정렬+재검출선
            if args.diag:
                out_p = dst / "qc" / "qc_wristline.png"
                save_sheet([r1, r2],
                           ["before + detected wrist line",
                            "after rotation (line must be flat)"], out_p, cols)
            else:
                r3 = [overlay(g[7], g[8]) if g[7] is not None else None for g in gal]
                out_p = dst / "qc" / "qc_hand.png"
                save_sheet([r1, r2, r3],
                           ["before + detected wrist line", "after rotation", "output"],
                           out_p, cols)
            print(f"  QC 시트: {out_p}")
            print("  1행 파란 선이 손목 끝단에 얹혔는지 먼저 확인. "
                  "2행에서 그 선이 수평이면 정렬 성공.")
        except Exception as e:
            print(f"  [경고] QC 시트 실패: {e}")

    if not args.diag:
        info = {"src": str(src), "edge_band": EDGE_BAND, "edge_eps": EDGE_EPS,
                "edge_minlen": EDGE_MINLEN, "edge_maxtilt": EDGE_MAXTILT,
                "rot_iters": ROT_ITERS, "angle_max": ANGLE_MAX,
                "cut_at_line": CUT_AT_LINE, "keep_below": KEEP_BELOW,
                "n_ok": int((df.status == "OK").sum()) if len(df) else 0,
                "n_total": int(len(df))}
        (dst / "_DONE_hand.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n  완료 표식 저장.")


if __name__ == "__main__":
    main()
