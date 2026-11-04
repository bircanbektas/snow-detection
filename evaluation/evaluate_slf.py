"""Evaluate avalanche_monitor.analyze on consecutive same-view SLF webcam pairs.

Usage: python evaluate_slf.py <prepared folder> <output folder>
(run prepare_slf.py first)

For each pair, "new avalanche" ground truth = avalanches annotated in the later
image that are not annotated in the earlier one.
"""
import contextlib
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # project folder
from avalanche_monitor import analyze      # noqa: E402

data = Path(sys.argv[1])                   # prepared folder
out = Path(sys.argv[2])
out.mkdir(exist_ok=True)
NEW_MIN_PCT = 0.2                          # new avalanche area needed to call a pair "positive"
orb = cv2.ORB_create(5000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)


def homography(ref, mov):
    k1, d1 = orb.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None)
    k2, d2 = orb.detectAndCompute(cv2.cvtColor(mov, cv2.COLOR_BGR2GRAY), None)
    if d1 is None or d2 is None:
        return None, 0
    good = [p[0] for p in bf.knnMatch(d2, d1, k=2)
            if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 15:
        return None, 0
    src = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return H, (int(inl.sum()) if H is not None else 0)


def parse_time(t):
    return datetime.strptime(t, "%Y%m%d%H%M")


idx = json.load(open(data / "index.json"))
times = {it["key"]: it["time"] for it in idx["items"]}
rows = []
for view in idx["views"]:
    for a, b in zip(view, view[1:]):
        img_a, img_b = cv2.imread(str(data / f"{a}.jpg")), cv2.imread(str(data / f"{b}.jpg"))
        H, inliers = homography(img_a, img_b)
        gap_h = (parse_time(times[b]) - parse_time(times[a])).total_seconds() / 3600
        if H is None or inliers < 60:
            rows.append({"pair": f"{a} -> {b}", "skip": f"not the same view ({inliers} matches)"})
            continue
        h, w = img_a.shape[:2]
        mask_a = cv2.imread(str(data / f"{a}_mask.png"), 0)
        mask_b = cv2.warpPerspective(cv2.imread(str(data / f"{b}_mask.png"), 0), H, (w, h),
                                     flags=cv2.INTER_NEAREST)
        old = cv2.dilate(mask_a, np.ones((15, 15), np.uint8))
        new_gt = ((mask_b > 0) & (old == 0)).astype(np.uint8) * 255
        new_gt = cv2.morphologyEx(new_gt, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        new_pct = np.count_nonzero(new_gt) / new_gt.size * 100

        with contextlib.redirect_stdout(io.StringIO()):
            r = analyze(img_a, img_b, roi=None, roi_file=None)
        pred = cv2.bitwise_or(r["change_mask"], r["motion_mask"])
        hit = np.count_nonzero((pred > 0) & (new_gt > 0))
        near_gt = cv2.dilate(new_gt, np.ones((31, 31), np.uint8))
        recall = hit / max(np.count_nonzero(new_gt), 1) * 100
        precision = (np.count_nonzero((pred > 0) & (near_gt > 0))
                     / max(np.count_nonzero(pred), 1) * 100)

        row = {"pair": f"{a} -> {b}", "gap_h": round(gap_h, 1), "inliers": inliers,
               "new_gt_pct": round(new_pct, 2), "positive": bool(new_pct >= NEW_MIN_PCT),
               "risk": r["risk_level"], "largest": round(r["largest_region"], 2),
               "moving": round(r["motion"]["moving_area"], 2),
               "downslope": r["motion"]["is_downslope"],
               "recall": round(recall, 1) if new_pct >= NEW_MIN_PCT else None,
               "precision": round(precision, 1) if np.count_nonzero(pred) else None}
        rows.append(row)

        # Visual: result panel + ground-truth outline of new avalanches
        panel = r["result_image"].copy()
        cnts, _ = cv2.findContours(new_gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        oy, ox = 100 + h, w        # bottom-right panel origin
        for c in cnts:
            cv2.drawContours(panel, [c + [ox, oy]], -1, (0, 255, 0), 2)
        cv2.imwrite(str(out / f"{a}__{b}.jpg"), cv2.resize(panel, None, fx=0.5, fy=0.5))

json.dump(rows, open(out / "results.json", "w"), indent=1)
for r in rows:
    if "skip" in r:
        print(f"{r['pair']:<48} SKIP {r['skip']}")
    else:
        print(f"{r['pair']:<48} gap={r['gap_h']:>7}h new={r['new_gt_pct']:>5}% "
              f"{'POS' if r['positive'] else 'neg'}  risk={r['risk']:<6} largest={r['largest']:>5}% "
              f"moving={r['moving']:>5}% down={r['downslope']!s:<5} "
              f"recall={r['recall']} precision={r['precision']}")

done = [r for r in rows if "skip" not in r]
for label, sel in (("all pairs", done), ("short gap (<= 6 h)", [r for r in done if r["gap_h"] <= 6])):
    pos = [r for r in sel if r["positive"]]
    neg = [r for r in sel if not r["positive"]]
    print(f"\n[{label}] detected {sum(r['risk'] != 'LOW' for r in pos)}/{len(pos)} positive pairs, "
          f"false alarms on {sum(r['risk'] != 'LOW' for r in neg)}/{len(neg)} negative pairs")
    if pos:
        print(f"  mean recall {np.mean([r['recall'] for r in pos]):.1f}% of new avalanche pixels")
