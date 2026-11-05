"""
Deep-learning avalanche detection.

A U-Net (ResNet-34 encoder) outlines avalanches in each photo; avalanches in the later
photo that were not there in the earlier one are reported as new. The model was trained
in deep_learning/avalanche_segmentation.ipynb and runs with OpenCV (no PyTorch needed).
The model file (~98 MB) is downloaded from the GitHub release on first use.
"""
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from avalanche_monitor import align_images

MODEL_URL = ("https://github.com/bircanbektas/snow-detection/releases/download/"
             "model-v1/avalanche_unet.onnx")
MODEL_PATH = Path(__file__).parent / "models" / "avalanche_unet.onnx"

INPUT_W, INPUT_H = 1216, 928   # fixed input canvas of the exported model
WIDTH = 1200                   # working width, as in training
THRESHOLD = 0.7                # chosen on the UIBK test split
NEW_MIN_PCT = 0.2              # new avalanche area (% of image) that raises an alarm

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_model():
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(exist_ok=True)
        tmp = MODEL_PATH.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(MODEL_PATH)
    return cv2.dnn.readNetFromONNX(str(MODEL_PATH))


def avalanche_probability(net, image):
    """Avalanche probability per pixel (average of the image and its mirror)."""
    h, w = image.shape[:2]
    scale = min(INPUT_W / w, INPUT_H / h)
    small = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    padded = cv2.copyMakeBorder(small, 0, INPUT_H - sh, 0, INPUT_W - sw, cv2.BORDER_REFLECT)

    probs = []
    for flip in (False, True):
        x = padded[:, ::-1] if flip else padded
        blob = ((x[..., ::-1].astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]
        net.setInput(np.ascontiguousarray(blob))
        p = 1 / (1 + np.exp(-net.forward()[0, 0]))
        probs.append(p[:, ::-1] if flip else p)
    prob = (probs[0] + probs[1]) / 2
    return cv2.resize(prob[:sh, :sw], (w, h))


def new_regions(later, earlier):
    """Pixels marked in the later mask that were not marked (nearby) in the earlier one."""
    old = cv2.dilate(earlier, np.ones((15, 15), np.uint8))
    new = ((later > 0) & (old == 0)).astype(np.uint8) * 255
    return cv2.morphologyEx(new, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def outline(image, mask, color, fill=0.0):
    if fill:
        overlay = image.copy()
        overlay[mask > 0] = color
        image = cv2.addWeighted(overlay, fill, image, 1 - fill, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, 3)
    return image


def detect_new_avalanches(net, img1, img2, roi_mask=None):
    """Compare two photos of the same slope; returns masks, areas and a result picture."""
    scale = WIDTH / img1.shape[1]
    img1 = cv2.resize(img1, (WIDTH, round(img1.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    aligned, valid = align_images(img1, img2)
    monitored = valid if roi_mask is None else cv2.bitwise_and(
        valid, cv2.resize(roi_mask, valid.shape[1::-1], interpolation=cv2.INTER_NEAREST))

    earlier = (avalanche_probability(net, img1) > THRESHOLD).astype(np.uint8) * 255
    later = (avalanche_probability(net, aligned) > THRESHOLD).astype(np.uint8) * 255
    earlier = cv2.bitwise_and(earlier, monitored)
    later = cv2.bitwise_and(later, monitored)
    new = new_regions(later, earlier)

    area = lambda m: np.count_nonzero(m) / max(np.count_nonzero(monitored), 1) * 100
    new_pct = area(new)

    yellow, red = (0, 220, 255), (0, 0, 255)
    left = outline(img1.copy(), earlier, yellow)
    right = outline(outline(aligned.copy(), cv2.bitwise_and(later, cv2.bitwise_not(new)), yellow),
                    new, red, fill=0.35)
    if roi_mask is not None:
        for img in (left, right):
            outline(img, monitored, (255, 255, 255))
    gap = np.full((left.shape[0], 10, 3), 255, np.uint8)

    return {
        "is_new": new_pct >= NEW_MIN_PCT,
        "new_pct": new_pct,
        "earlier_pct": area(earlier),
        "later_pct": area(later),
        "new_mask": new,
        "result_image": np.hstack([left, gap, right]),
    }
