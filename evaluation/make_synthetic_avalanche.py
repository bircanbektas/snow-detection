"""Synthetic test: paint a rough, shadowed avalanche tongue onto an image.

Usage: python make_synthetic_avalanche.py mountain_2.png
Writes avalanche_small.png and avalanche_big.png to the current folder.
"""
import sys

import cv2
import numpy as np

img = cv2.imread(sys.argv[1])
h, w = img.shape[:2]
rng = np.random.default_rng(0)

for name, scale in (("small", 0.5), ("big", 1.6)):
    # Tongue-shaped debris path running downslope
    x0, y0 = int(w * 0.30), int(h * 0.18)
    pts = np.array([[x0, y0], [x0 + int(80 * scale), y0],
                    [x0 + int(260 * scale), y0 + int(330 * scale)],
                    [x0 + int(60 * scale), y0 + int(360 * scale)]], np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    mask = cv2.GaussianBlur(mask, (31, 31), 0).astype(np.float32) / 255

    # Rough, partly shadowed debris surface
    noise = cv2.GaussianBlur(rng.normal(0.65, 0.12, (h, w)).astype(np.float32), (7, 7), 0)
    factor = 1 - mask * (1 - noise)
    out = np.clip(img.astype(np.float32) * factor[..., None], 0, 255).astype(np.uint8)
    cv2.imwrite(f"avalanche_{name}.png", out)
