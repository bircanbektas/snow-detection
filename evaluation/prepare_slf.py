"""Collect unique SLF images, union their avalanche masks, group by camera view.

Dataset: Hafner, E. D. (2024). Data interactive avalanche segmentation. EnviDat.
doi:10.16904/envidat.526 (CC BY-SA 4.0). Download ias_data.zip and unzip the
Train, Vali, Test and WebNew folders.

Usage: python prepare_slf.py <unzipped data folder> <prepared folder>
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

root = Path(sys.argv[1])          # .../slf_data/data
out = Path(sys.argv[2])           # .../slf_data/prepared
out.mkdir(exist_ok=True)
WIDTH = 1200

# 1) unique images: name without the trailing _<avalanche index>
groups = defaultdict(lambda: {"image": None, "masks": []})
for split in ("Train", "Vali", "Test", "WebNew"):
    for img in (root / split / "images").glob("*"):
        key = re.sub(r"_\d+$", "", img.stem)
        groups[key]["image"] = img
        mask = root / split / "masks" / (img.stem + ".png")
        if mask.exists():
            groups[key]["masks"].append(mask)


def timestamp(key):
    digits = re.sub(r"\D", "", key.replace("cam2", ""))
    return digits[:12]  # YYYYMMDDHHMM


items = []
for key, g in sorted(groups.items()):
    img = cv2.imread(str(g["image"]))
    if img is None:
        continue
    h, w = img.shape[:2]
    scale = WIDTH / w
    small = cv2.resize(img, (WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
    union = np.zeros(small.shape[:2], np.uint8)
    for m in g["masks"]:
        mk = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
        if mk is None:
            continue
        mk = cv2.resize(mk, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        union[mk > 0] = 255
    cv2.imwrite(str(out / f"{key}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(out / f"{key}_mask.png"), union)
    items.append({"key": key, "time": timestamp(key), "orig_size": [w, h],
                  "size": [small.shape[1], small.shape[0]], "n_avalanches": len(g["masks"]),
                  "avalanche_pct": float(np.count_nonzero(union) / union.size * 100)})
print("unique images:", len(items))

# 2) group by camera view: same aspect ratio + many RANSAC inlier matches
orb = cv2.ORB_create(3000)
feats = {}
for it in items:
    g = cv2.imread(str(out / f"{it['key']}.jpg"), cv2.IMREAD_GRAYSCALE)
    feats[it["key"]] = orb.detectAndCompute(g, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)


def same_view(a, b):
    (k1, d1), (k2, d2) = feats[a["key"]], feats[b["key"]]
    if d1 is None or d2 is None or a["size"] != b["size"]:
        return 0
    good = [p[0] for p in bf.knnMatch(d1, d2, k=2)
            if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 30:
        return 0
    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(inl.sum()) if H is not None else 0


parent = list(range(len(items)))


def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


for i in range(len(items)):
    for j in range(i + 1, len(items)):
        if find(i) != find(j) and same_view(items[i], items[j]) >= 60:
            parent[find(i)] = find(j)

views = defaultdict(list)
for i, it in enumerate(items):
    views[find(i)].append(it)
view_list = [sorted(v, key=lambda x: x["time"]) for v in views.values() if len(v) > 1]
view_list.sort(key=len, reverse=True)
for vi, v in enumerate(view_list):
    for it in v:
        it["view"] = vi
    print(f"view {vi}: {len(v)} images, {v[0]['time']} .. {v[-1]['time']}, size {v[0]['size']}")

json.dump({"items": items, "views": [[it["key"] for it in v] for v in view_list]},
          open(out / "index.json", "w"), indent=1)
