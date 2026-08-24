# -*- coding: utf-8 -*-
"""
손목 절단선 정렬  [A안 - 회전만]     crop_yolo_seg -> crop_wristline_rot

무엇을 하나
    마스크 하단에서 '손목 절단선'(첨부 그림의 파란선)을 자동으로 찾아
    그 선이 수평이 되도록 이미지와 마스크를 회전시켜 새 경로에 저장합니다.
    자르지는 않습니다. 아래 흰 띠도 그대로 남습니다.

절단선을 어떻게 찾나
    마스크 하단 경계 중에서 '바로 아래가 밝은' 열만 골라 직선을 맞춥니다.
        손목 절단선 아래  -> 촬영판/차폐물의 밝은 띠가 있음      -> 채택
        엄지-손목 사이    -> 어두운 배경                        -> 제외
    이 구분이 핵심입니다. 마스크 하단 경계는 뒤집힌 V 자라서 양쪽 날개의
    기울기가 정반대(+25도 / -57도)인데, 길이나 각도만으로는 어느 쪽이
    절단선인지 알 수 없습니다. '아래에 밝은 띠가 있는가' 가 유일한 단서입니다.

    검출에 실패하면 회전하지 않고 status 에 기록합니다.
    억지로 각도를 만들어내지 않습니다.

    검증 (id 4396): 사용자가 직접 그은 파란선 +24.6도, 자동 검출 +25.4도

사용법
    아래 [설정] 블록만 맞추고 실행합니다.
        python make_wristline_rot.py
    다시 만들려면
        python make_wristline_rot.py --overwrite
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
# [설정]  여기만 고치면 됩니다.
# ════════════════════════════════════════════════════════════════════
BASE_DIR  = Path(r"G:\Project\sinra_cho")
SRC_DIR   = BASE_DIR / "crop_yolo_seg"           # 입력  (images/ masks/ csv/)
DST_DIR   = BASE_DIR / "crop_wristline_rot"      # 출력  [A안 - 회전만]

SPLITS    = ["train", "validation", "test"]
IMAGE_SUB = "images"
MASK_SUB  = "masks"
CSV_SUB   = "csv"

CUT_BELOW = False        # [A안] False = 자르지 않음.  B안 스크립트에서는 True
BOTTOM_PAD = 6           # (B안 전용) 절단선 아래로 남길 여유 화소

# 절단선 검출
DET_BAND     = 0.35      # 손 세로 길이의 하단 몇 %에서 경계를 찾을지
DET_BRIGHT_P = 85        # 마스크 밖 화소의 이 백분위보다 밝으면 '밝은 띠'로 판정
DET_LOOK     = 12        # 경계 아래 몇 화소를 평균내어 밝기를 볼지
DET_MIN_COLS = 40        # 유효 열이 이보다 적으면 검출 실패로 처리
DET_MAX_DEG  = 45.0      # 이보다 큰 각도가 나오면 오검출로 보고 회전하지 않음

# 크롭
MARGIN_FRAC = 0.04       # 회전 후 마스크 경계에 붙일 여백 (짧은 변 기준)
MIN_SIDE    = 128
SAVE_MASK   = True
PNG_LEVEL   = 3
QC_N        = 10
# ════════════════════════════════════════════════════════════════════

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def imread_u(path, flags=cv2.IMREAD_GRAYSCALE):
    """한글 경로 안전 로더."""
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    except Exception:
        return None


def imwrite_u(path, img, params=None):
    """한글 경로 안전 저장."""
    try:
        ok, buf = cv2.imencode(Path(path).suffix, img, params or [])
        if not ok:
            return False
        buf.tofile(str(path))
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
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return (lab == k).astype(np.uint8)


def finger_up(keep):
    """손끝이 위쪽인지. 수직 단면의 연결 조각 수로 판정.
    손가락 구간은 틈 때문에 한 행에 여러 조각, 손목은 한 덩어리."""
    ys, _ = np.nonzero(keep)
    y0, y1 = int(ys.min()), int(ys.max())
    h = max(1, y1 - y0)

    def runs(band):
        if band.size == 0:
            return 0.0
        d = np.diff(np.pad((band > 0).astype(np.int8), ((0, 0), (1, 1))), axis=1)
        return float((d == 1).sum(1).mean())

    t = runs(keep[y0:y0 + int(0.25 * h)])
    b = runs(keep[max(y0, y1 - int(0.25 * h)):y1])
    if abs(t - b) < 0.35:
        return None
    return bool(t > b)


def detect_cut_line(img, keep):
    """손목 절단선 각도를 찾습니다.

    반환 (각도[도], 유효열수, x0, x1, y_mid) 또는 None
    각도 부호: 양수 = 오른쪽으로 내려감. 이 값만큼 회전시키면 수평이 됩니다.
    """
    h_, w_ = keep.shape
    ys = np.nonzero(keep.sum(1))[0]
    if ys.size < 10:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    hh = max(1, y1 - y0)
    ytop = y1 - int(DET_BAND * hh)

    outside = img[keep == 0]
    if outside.size < 100:
        return None
    thr = float(np.percentile(outside, DET_BRIGHT_P))

    pts = []
    for c in range(w_):
        col = np.nonzero(keep[:, c])[0]
        if col.size == 0:
            continue
        b = int(col.max())
        if b < ytop or b >= h_ - 3:          # 프레임 하단은 크롭 절단선이라 제외
            continue
        seg = img[b + 2:min(h_, b + 2 + DET_LOOK), c]
        if seg.size and float(seg.mean()) > thr:      # 아래가 밝다 -> 절단선
            pts.append((c, b))
    if len(pts) < DET_MIN_COLS:
        return None

    P = np.asarray(pts, dtype=float)
    s, b0 = np.polyfit(P[:, 0], P[:, 1], 1)
    r = np.abs(P[:, 1] - (s * P[:, 0] + b0))
    k = r < max(3.0, 2.5 * r.std())                   # 이상치 1회 제거
    if k.sum() >= DET_MIN_COLS:
        s, b0 = np.polyfit(P[k, 0], P[k, 1], 1)
        P = P[k]
    ang = float(np.degrees(np.arctan(s)))
    if abs(ang) > DET_MAX_DEG:
        return None
    return ang, int(len(P)), float(P[:, 0].min()), float(P[:, 0].max()), float(P[:, 1].mean())


def rot_pair(img, keep, deg, expand=True):
    """이미지와 마스크를 같은 각도로 회전. expand=True 면 잘림 없이 캔버스를 넓힙니다."""
    h, w = keep.shape
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
    rk = cv2.warpAffine(keep * 255, M, (nw, nh), flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return ri, (rk > 127).astype(np.uint8), M


def tight_crop(img, keep, y_cut=None):
    """마스크 경계 + 여백으로 크롭. y_cut 이 주어지면 그 아래를 잘라냅니다."""
    ys, xs = np.nonzero(keep)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    pad = int(MARGIN_FRAC * min(y1 - y0 + 1, x1 - x0 + 1))
    h, w = keep.shape
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    x1 = min(w - 1, x1 + pad)
    if y_cut is None:
        y1 = min(h - 1, y1 + pad)
    else:
        y1 = int(np.clip(y_cut + BOTTOM_PAD, y0 + MIN_SIDE, h - 1))
    return img[y0:y1 + 1, x0:x1 + 1], keep[y0:y1 + 1, x0:x1 + 1], (x0, y0, x1, y1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC_DIR))
    ap.add_argument("--dst", default=str(DST_DIR))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="split 당 최대 처리 수(테스트용)")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not src.exists():
        sys.exit(f"[중단] 입력 경로 없음: {src}\n"
                 f"        [설정] 블록의 BASE_DIR / SRC_DIR 을 고치세요.")
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "qc").mkdir(exist_ok=True)

    mode = "B안 (절단선 아래 잘라내기)" if CUT_BELOW else "A안 (회전만)"
    print("=" * 70)
    print(f" 손목 절단선 정렬  -  {mode}")
    print("=" * 70)
    print(f"  입력 {src}")
    print(f"  출력 {dst}")
    print(f"  검출 band {DET_BAND} · 밝기 {DET_BRIGHT_P}%ile · look {DET_LOOK}px · "
          f"최소열 {DET_MIN_COLS} · 최대각 {DET_MAX_DEG}도")

    if (src / CSV_SUB).exists():
        (dst / CSV_SUB).mkdir(exist_ok=True)
        for p in (src / CSV_SUB).glob("*.csv"):
            (dst / CSV_SUB / p.name).write_bytes(p.read_bytes())
        print(f"  csv {len(list((dst/CSV_SUB).glob('*.csv')))}개 복사")

    recs, gallery = [], []
    t_start = time.time()

    for sp in SPLITS:
        sdi, sdm = src / sp / IMAGE_SUB, src / sp / MASK_SUB
        if not sdi.exists():
            print(f"  [건너뜀] {sp}: {sdi} 없음")
            continue
        ddi, ddm = dst / sp / IMAGE_SUB, dst / sp / MASK_SUB
        ddi.mkdir(parents=True, exist_ok=True)
        if SAVE_MASK:
            ddm.mkdir(parents=True, exist_ok=True)

        files = sorted(p for p in sdi.iterdir() if p.suffix.lower() in IMG_EXT)
        if args.limit:
            files = files[:args.limit]
        step = max(1, len(files) // max(1, QC_N))
        print(f"\n  [{sp}] {len(files):,}장")
        t0, done, fail, nodet = time.time(), 0, 0, 0

        for i, ip in enumerate(files):
            stem = ip.stem
            op = ddi / f"{stem}.png"
            if op.exists() and not args.overwrite:
                done += 1
                continue

            mp = next((sdm / f"{stem}{e}" for e in IMG_EXT if (sdm / f"{stem}{e}").exists()), None)
            rec = {"split": sp, "id": stem, "status": "OK"}
            img = imread_u(ip)
            msk = imread_u(mp) if mp else None
            if img is None:
                rec["status"] = "IMG_LOAD_FAIL"; recs.append(rec); fail += 1; continue
            if msk is None:
                rec["status"] = "MASK_MISSING"; recs.append(rec); fail += 1; continue
            if msk.shape[:2] != img.shape[:2]:
                msk = cv2.resize(msk, (img.shape[1], img.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
            keep = main_component(msk)
            if keep is None:
                rec["status"] = "MASK_EMPTY"; recs.append(rec); fail += 1; continue

            rec["src_h"], rec["src_w"] = int(img.shape[0]), int(img.shape[1])
            img0, keep0 = img.copy(), keep.copy()

            # 손끝이 아래를 향하면 먼저 180도 뒤집습니다
            up = finger_up(keep)
            rec["flipped180"] = int(up is False)
            if up is False:
                img, keep, _ = rot_pair(img, keep, 180.0, expand=False)

            det = detect_cut_line(img, keep)
            if det is None:
                rec["status"] = "NO_CUTLINE"
                rec["angle"] = np.nan
                nodet += 1
                ri, rk = img, keep                    # 회전하지 않고 그대로 진행
                y_cut = None
            else:
                ang, ncol, x0d, x1d, ymid = det
                rec.update({"angle": ang, "det_cols": ncol,
                            "det_x0": x0d, "det_x1": x1d})
                ri, rk, M = rot_pair(img, keep, ang, expand=True)
                if CUT_BELOW:
                    #  검출선의 양 끝점을 회전 후 좌표계로 옮겨 절단 y 를 구합니다.
                    #  회전 후에는 이 선이 수평이므로 두 끝점의 y 가 거의 같습니다.
                    t = np.tan(np.deg2rad(ang))
                    half = (x1d - x0d) / 2.0
                    p = np.float32([[x0d, ymid - t * half],
                                    [x1d, ymid + t * half]]).reshape(-1, 1, 2)
                    q = cv2.transform(p, M).reshape(-1, 2)
                    y_cut = float(q[:, 1].mean())
                else:
                    y_cut = None

            ci, ck, bbox = tight_crop(ri, rk, y_cut if CUT_BELOW else None)
            if min(ci.shape[:2]) < MIN_SIDE or ck.sum() < 500:
                rec["status"] = "TOO_SMALL"; recs.append(rec); fail += 1; continue

            rec.update({"out_h": int(ci.shape[0]), "out_w": int(ci.shape[1]),
                        "out_ar": float(ci.shape[0] / ci.shape[1]),
                        "area_after": float(ck.mean()),
                        "bbox": "%d,%d,%d,%d" % bbox})

            prm = [int(cv2.IMWRITE_PNG_COMPRESSION), PNG_LEVEL]
            if not imwrite_u(op, ci, prm):
                rec["status"] = "IMG_WRITE_FAIL"; recs.append(rec); fail += 1; continue
            if SAVE_MASK and not imwrite_u(ddm / f"{stem}.png", ck * 255, prm):
                rec["status"] = "MASK_WRITE_FAIL"; recs.append(rec); fail += 1; continue

            recs.append(rec); done += 1
            if sp == "train" and len(gallery) < QC_N and i % step == 0:
                gallery.append((stem, img0, keep0, ci, ck, det))
            if (i + 1) % 1000 == 0:
                el = time.time() - t0
                print(f"    {i+1:>6,}/{len(files):,}  {el:5.0f}초 ({(i+1)/max(el,1e-9):.0f}장/초)")
        print(f"    완료 {done:,} / 실패 {fail:,} / 절단선 미검출 {nodet:,} "
              f"({time.time()-t0:.0f}초)")

    df = pd.DataFrame(recs)
    df.to_csv(dst / "wristline_metadata.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print(" 요약")
    print("=" * 70)
    print(df["status"].value_counts().to_string())
    ok = df[df.status == "OK"]
    a = pd.to_numeric(ok.get("angle"), errors="coerce").dropna()
    if len(a):
        print(f"\n  검출 각도  n={len(a):,}  중앙 {a.median():+.2f}도  "
              f"|각도| 중앙 {a.abs().median():.2f}도")
        print(f"             5~95% [{a.quantile(.05):+.2f}, {a.quantile(.95):+.2f}]  "
              f"최대 |{a.abs().max():.2f}|")
        for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 30), (30, 46)]:
            n = int(((a.abs() >= lo) & (a.abs() < hi)).sum())
            print(f"             |각도| {lo:>2}~{hi:<2}도 : {n:>6,} ({100*n/len(a):5.1f}%)")
    nod = int((df.status == "NO_CUTLINE").sum())
    print(f"\n  절단선 미검출 {nod:,} ({100*nod/max(len(df),1):.2f}%)  <- 회전 없이 저장됨")
    if len(ok):
        print(f"  출력 종횡비 H/W  중앙 {ok.out_ar.median():.3f}  "
              f"5~95% [{ok.out_ar.quantile(.05):.3f}, {ok.out_ar.quantile(.95):.3f}]")
    print(f"\n  메타 저장: {dst/'wristline_metadata.csv'}")
    print(f"  총 소요 {time.time()-t_start:.0f}초")

    if gallery:
        try:
            n = len(gallery)
            fig, ax = plt.subplots(2, n, figsize=(2.5 * n, 8.8), squeeze=False)
            for c, (stem, im0, k0, im1, k1, det) in enumerate(gallery):
                v0 = cv2.cvtColor(cv2.normalize(im0, None, 0, 255, cv2.NORM_MINMAX)
                                  .astype(np.uint8), cv2.COLOR_GRAY2BGR)
                v0[k0 > 0] = (0.7 * v0[k0 > 0] + 0.3 * np.array([0, 90, 0])).astype(np.uint8)
                if det:
                    ang, _, x0d, x1d, ymid = det
                    t = np.tan(np.deg2rad(ang))
                    cv2.line(v0, (int(x0d), int(ymid - t * (x1d - x0d) / 2)),
                             (int(x1d), int(ymid + t * (x1d - x0d) / 2)), (255, 120, 0), 5)
                v1 = cv2.cvtColor(cv2.normalize(im1, None, 0, 255, cv2.NORM_MINMAX)
                                  .astype(np.uint8), cv2.COLOR_GRAY2BGR)
                v1[k1 > 0] = (0.7 * v1[k1 > 0] + 0.3 * np.array([0, 90, 0])).astype(np.uint8)
                cv2.line(v1, (v1.shape[1] // 2, 0), (v1.shape[1] // 2, v1.shape[0]), (0, 0, 255), 2)
                ax[0, c].imshow(v0[:, :, ::-1]); ax[0, c].axis("off")
                ax[1, c].imshow(v1[:, :, ::-1]); ax[1, c].axis("off")
                ax[0, c].set_title(f"{stem}\n{('%+.1f deg' % det[0]) if det else 'no cutline'}",
                                   fontsize=8)
            plt.tight_layout()
            p = dst / "qc" / "qc_wristline.png"
            plt.savefig(p, dpi=110); plt.close()
            print(f"  QC 시트: {p}")
            print("  * 윗줄 주황선이 검출된 절단선입니다. 그 선이 흰 띠의 위 모서리를")
            print("    따라가는지, 아랫줄에서 수평이 됐는지 확인하세요.")
        except Exception as e:
            print(f"  [경고] QC 시트 실패: {e}")

    (dst / "_DONE_wristline.json").write_text(json.dumps({
        "src": str(src), "mode": "cut" if CUT_BELOW else "rotate_only",
        "det": {"band": DET_BAND, "bright_p": DET_BRIGHT_P, "look": DET_LOOK,
                "min_cols": DET_MIN_COLS, "max_deg": DET_MAX_DEG},
        "margin_frac": MARGIN_FRAC, "bottom_pad": BOTTOM_PAD,
        "n_ok": int((df.status == "OK").sum()), "n_total": int(len(df)),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  완료 표식: {dst/'_DONE_wristline.json'}")


if __name__ == "__main__":
    main()
