"""Synthetic test: a textured snow slab inside a polygon shifts by (dx, dy)."""
import sys

import cv2
import numpy as np

img = cv2.imread(sys.argv[1])
h, w = img.shape[:2]
rng = np.random.default_rng(1)

# Give the slab some visible texture (cracks / clumps) so motion can be tracked
texture = cv2.GaussianBlur(rng.normal(0, 25, (h, w)).astype(np.float32), (9, 9), 0)
base = np.clip(img.astype(np.float32) + texture[..., None], 0, 255).astype(np.uint8)

slab = np.zeros((h, w), np.uint8)
cv2.fillPoly(slab, [np.array([[420, 200], [620, 200], [640, 520], [440, 520]], np.int32)], 255)

before = img.copy()
before[slab > 0] = base[slab > 0]
cv2.imwrite("slide_before.png", before)

for name, (dx, dy) in (("down", (0, 12)), ("up", (0, -12))):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    moved = cv2.warpAffine(base, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    moved_mask = cv2.warpAffine(slab, M, (w, h))
    after = img.copy()
    after[moved_mask > 0] = moved[moved_mask > 0]
    cv2.imwrite(f"slide_{name}.png", after)
