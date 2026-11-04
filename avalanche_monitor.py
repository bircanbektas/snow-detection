import argparse
import os

import cv2
import numpy as np


# ==================== SETTINGS ====================

TEXTURE_WINDOW = 11          # Window size (pixels) for local roughness / correlation
ROUGHNESS_THRESHOLD = 0.7    # log ratio of roughness counted as "surface became rough"
CORRELATION_THRESHOLD = 0.3  # Local correlation below this = surface pattern changed
MIN_REGION_AREA_RATIO = 0.002  # Ignore changed regions smaller than 0.2% of the ROI
MIN_GOOD_MATCHES = 15        # Minimum feature matches needed for alignment

# Risk thresholds (percent of the monitored area)
# An avalanche is one large connected region, so the largest region matters most.
RISK_HIGH_REGION = 5.0
RISK_HIGH_CHANGE = 10.0
RISK_HIGH_SNOW = 35.0
RISK_MEDIUM_REGION = 1.5
RISK_MEDIUM_CHANGE = 4.0
RISK_MEDIUM_SNOW = 20.0

# Motion direction
DOWNSLOPE_ANGLE = 90         # Fall line in the image: 0 = right, 90 = down, 180 = left
DOWNSLOPE_TOLERANCE = 45     # Degrees around the fall line still counted as downslope
MIN_FLOW_MAGNITUDE = 1.5     # Pixels; smaller flow vectors are treated as noise
MIN_DOWNSLOPE_RATIO = 0.6    # Share of moving pixels that must point downslope
MIN_ELONGATION = 2.0         # Length / width ratio of an avalanche-like tongue
MAX_FB_ERROR = 1.0           # Pixels; forward-backward flow mismatch allowed
MIN_TEXTURE_STD = 4.0        # Local contrast needed to trust the flow
MIN_MOVING_AREA = 1.0        # Percent of the ROI that must move before the flow direction counts


# ==================== PREPROCESSING ====================

def align_images(img_ref, img_moving):
    """
    Align the second image onto the first one.
    A fixed mountain camera still shakes slightly (wind, snow load, zoom).
    Without alignment, these small shifts appear as false "movement".

    Uses ORB keypoints + RANSAC homography.
    Returns the aligned image and a mask of valid (overlapping) pixels.
    """
    h, w = img_ref.shape[:2]
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_mov = cv2.cvtColor(img_moving, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray_ref, None)
    kp2, des2 = orb.detectAndCompute(gray_mov, None)

    homography = None
    good_matches = []
    if des1 is not None and des2 is not None:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        # Lowe's ratio test keeps only distinctive matches
        for pair in matcher.knnMatch(des2, des1, k=2):
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                good_matches.append(pair[0])

    if len(good_matches) >= MIN_GOOD_MATCHES:
        src = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        homography, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if homography is not None:
            print(f"Alignment: {int(inliers.sum())}/{len(good_matches)} matches used")

    if homography is None:
        # Fallback: plain resize (old behaviour)
        print("WARNING: Alignment failed, falling back to simple resize")
        aligned = cv2.resize(img_moving, (w, h))
        return aligned, np.full((h, w), 255, np.uint8)

    aligned = cv2.warpPerspective(img_moving, homography, (w, h))

    # Pixels outside the warped image are black and must not count as change
    ones = np.full(img_moving.shape[:2], 255, np.uint8)
    valid_mask = cv2.warpPerspective(ones, homography, (w, h))
    valid_mask = cv2.erode(valid_mask, np.ones((15, 15), np.uint8))

    return aligned, valid_mask


def normalize_lighting(gray):
    """
    Fog and changing daylight make the whole image brighter or darker.
    CLAHE (local histogram equalization) brings out the slope details
    so that the comparison is less sensitive to these global changes.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def match_brightness(gray, reference, mask):
    """
    Shift and scale the brightness of 'gray' so that its mean/std
    inside the mask equal those of 'reference'.
    """
    m_ref, s_ref = cv2.meanStdDev(reference, mask=mask)
    m_img, s_img = cv2.meanStdDev(gray, mask=mask)
    scale = s_ref[0][0] / max(s_img[0][0], 1e-6)
    adjusted = (gray.astype(np.float32) - m_img[0][0]) * scale + m_ref[0][0]
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def load_roi_mask(roi_arg, roi_file, shape):
    """
    Region of interest = the slope to monitor.
    Sky, buildings and lift poles should be excluded from the analysis.

    - roi_file: black/white image, white = analyse (same size as image 1)
    - roi_arg:  "x,y,w,h" rectangle
    - none:     the whole image is analysed
    """
    h, w = shape[:2]
    if roi_file and os.path.exists(roi_file):
        mask = cv2.imread(roi_file, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        print(f"ROI: mask file '{roi_file}'")
        return mask

    mask = np.zeros((h, w), np.uint8)
    if roi_arg:
        x, y, rw, rh = [int(v) for v in roi_arg.split(",")]
        mask[y:y + rh, x:x + rw] = 255
        print(f"ROI: rectangle x={x}, y={y}, w={rw}, h={rh}")
    else:
        mask[:] = 255
        print("ROI: whole image (use --roi or roi_mask.png to restrict)")
    return mask


# ==================== ANALYSIS ====================

def calculate_snow_density(image, mask):
    """
    Calculate snow density (percentage of snow pixels inside the mask).
    Snow is bright AND colourless (low saturation).
    A fixed threshold (e.g. 200) fails in fog, where everything is bright,
    so the brightness threshold is chosen automatically with Otsu's method.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation, value = hsv[:, :, 1], hsv[:, :, 2]

    pixels = value[mask > 0]
    otsu_value, _ = cv2.threshold(pixels.reshape(-1, 1), 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    snow_mask = ((value > otsu_value) & (saturation < 60)).astype(np.uint8) * 255
    snow_mask = cv2.bitwise_and(snow_mask, mask)

    snow_percentage = cv2.countNonZero(snow_mask) / max(cv2.countNonZero(mask), 1) * 100
    return snow_percentage, snow_mask


def detect_sky(image):
    """
    Sky and clouds change all the time and must not count as snow change.
    Sky = clearly blue area connected to the top border of the image.
    (Fog and white clouds cannot be told apart from snow by colour, so they are kept.)
    """
    b, g, r = [c.astype(np.int16) for c in cv2.split(image)]
    _, std = local_stats(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 15)
    # Smooth as well as blue: forests and blue shadows on snow have texture
    candidate = ((b > r + 25) & (b >= g) & (std < 6)).astype(np.uint8) * 255
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    # Only regions touching the top edge are sky (blue shadows on snow are not)
    count, labels = cv2.connectedComponents(candidate)
    top_labels = set(np.unique(labels[0, :])) - {0}
    sky = np.isin(labels, list(top_labels)).astype(np.uint8) * 255
    if cv2.countNonZero(sky) > 0.5 * sky.size:
        return np.zeros_like(sky)  # implausible, do not trust it
    return cv2.dilate(sky, np.ones((15, 15), np.uint8))


def local_stats(gray, ksize):
    """Local mean and standard deviation in a ksize x ksize window."""
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (ksize, ksize))
    var = np.maximum(cv2.blur(g * g, (ksize, ksize)) - mean ** 2, 0)
    return mean, np.sqrt(var)


def detect_changes(gray1, gray2, mask):
    """
    Illumination-invariant change detection on aligned images.

    Plain brightness differencing fails on real webcams: moving shadows,
    clouds and day-to-day light changes look like "change" everywhere.
    Two measures that a shadow does not change are used instead:
    - roughness (local std / mean): a shadow darkens a surface but keeps its
      relative contrast; avalanche debris turns smooth snow into a rough surface
    - local correlation (structure term of SSIM): brightness scaling keeps the
      pattern, a real surface change destroys it

    Small isolated regions (noise, falling snowflakes, moving people)
    are removed; an avalanche is a large connected region.
    """
    k = TEXTURE_WINDOW
    m1, s1 = local_stats(gray1, k)
    m2, s2 = local_stats(gray2, k)
    rough1, rough2 = s1 / (m1 + 10), s2 / (m2 + 10)
    rough_increase = np.log((rough2 + 0.01) / (rough1 + 0.01))

    g1, g2 = gray1.astype(np.float32), gray2.astype(np.float32)
    corr = (cv2.blur(g1 * g2, (k, k)) - m1 * m2) / (s1 * s2 + 1e-3)
    textured = (s1 > MIN_TEXTURE_STD * 1.5) & (s2 > MIN_TEXTURE_STD * 1.5)

    changed = (rough_increase > ROUGHNESS_THRESHOLD) | (textured & (corr < CORRELATION_THRESHOLD))
    thresh_diff = (changed & (mask > 0)).astype(np.uint8) * 255
    thresh_diff = cv2.morphologyEx(thresh_diff, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    thresh_diff = cv2.morphologyEx(thresh_diff, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    # Change score for the visualization (0 = same surface, 255 = strongly changed)
    score = np.maximum(rough_increase / (2 * ROUGHNESS_THRESHOLD),
                       np.where(textured, (1 - corr) / 2, 0))
    difference = np.clip(score * 255, 0, 255).astype(np.uint8)
    difference = cv2.bitwise_and(difference, mask)

    # Keep only large connected regions
    roi_area = max(cv2.countNonZero(mask), 1)
    min_area = roi_area * MIN_REGION_AREA_RATIO
    count, labels, stats, _ = cv2.connectedComponentsWithStats(thresh_diff)
    filtered = np.zeros_like(thresh_diff)
    regions = []
    largest_area = 0
    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            filtered[labels == i] = 255
            regions.append(stats[i, :4])
            largest_area = max(largest_area, area)

    change_percentage = cv2.countNonZero(filtered) / roi_area * 100
    largest_percentage = largest_area / roi_area * 100
    return difference, filtered, change_percentage, largest_percentage, regions


# ==================== MOTION DIRECTION ====================

def analyze_motion(gray1, gray2, change_mask, roi_mask, downslope_deg=DOWNSLOPE_ANGLE):
    """
    An avalanche moves DOWNSLOPE. Two independent clues are checked:

    1) Optical flow (Farneback): which parts of the snow surface moved, and where to?
       This finds a sliding snow slab even when its brightness hardly changes,
       so it works as a second detector next to image differencing.
    2) Shape of the largest changed region: avalanche paths are long, narrow
       tongues stretched along the fall line.

    downslope_deg is the fall-line direction in image coordinates
    (0 = right, 90 = down, 180 = left). Default: straight down in the image.
    """
    d = np.deg2rad(downslope_deg)
    down = np.array([np.cos(d), np.sin(d)])
    cos_tol = np.cos(np.deg2rad(DOWNSLOPE_TOLERANCE))

    result = {"flow": None, "motion_mask": np.zeros_like(roi_mask), "moving_area": 0.0,
              "flow_magnitude": 0.0, "flow_angle": None, "downslope_ratio": 0.0,
              "elongation": 0.0, "shape_angle": None,
              "flow_downslope": False, "shape_downslope": False, "is_downslope": False}

    # 1) Dense optical flow over the monitored area
    flow_params = dict(pyr_scale=0.5, levels=4, winsize=25, iterations=5,
                       poly_n=7, poly_sigma=1.5, flags=0)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, **flow_params)
    back = cv2.calcOpticalFlowFarneback(gray2, gray1, None, **flow_params)
    result["flow"] = flow

    # Reliability checks:
    # - forward-backward: moving forward then back must return to the start point
    # - texture: flat foggy areas have no pattern to track, so their flow is random
    h, w = gray1.shape
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    back_at_target = cv2.remap(back, grid_x + flow[..., 0], grid_y + flow[..., 1],
                               cv2.INTER_LINEAR)
    fb_error = np.linalg.norm(flow + back_at_target, axis=2)
    gray_f = gray1.astype(np.float32)
    local_std = np.sqrt(np.maximum(cv2.blur(gray_f ** 2, (15, 15))
                                   - cv2.blur(gray_f, (15, 15)) ** 2, 0))
    reliable = (fb_error < MAX_FB_ERROR) & (local_std > MIN_TEXTURE_STD)

    magnitude = np.linalg.norm(flow, axis=2)
    motion_mask = ((magnitude > MIN_FLOW_MAGNITUDE) & reliable
                   & (roi_mask > 0)).astype(np.uint8) * 255
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))

    # Keep only large coherently moving regions (fog and noise give small specks)
    roi_area = max(cv2.countNonZero(roi_mask), 1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(motion_mask)
    keep = [i for i in range(1, count)
            if stats[i, cv2.CC_STAT_AREA] >= roi_area * MIN_REGION_AREA_RATIO]
    motion_mask = np.isin(labels, keep).astype(np.uint8) * 255
    result["motion_mask"] = motion_mask
    result["moving_area"] = cv2.countNonZero(motion_mask) / roi_area * 100

    moving = flow[motion_mask > 0]
    if len(moving) > 0:
        unit = moving / np.linalg.norm(moving, axis=1, keepdims=True)
        result["downslope_ratio"] = float(np.mean(unit @ down > cos_tol))
        mean_vec = moving.mean(axis=0)
        result["flow_magnitude"] = float(np.linalg.norm(mean_vec))
        result["flow_angle"] = float(np.rad2deg(np.arctan2(mean_vec[1], mean_vec[0])) % 360)
        result["flow_downslope"] = bool(result["downslope_ratio"] > MIN_DOWNSLOPE_RATIO
                                        and result["moving_area"] >= MIN_MOVING_AREA)

    # 2) Shape of the largest changed region (principal axis from image moments)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(change_mask)
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = cv2.moments((labels == largest).astype(np.uint8), binaryImage=True)
        mu20, mu02, mu11 = m["mu20"] / m["m00"], m["mu02"] / m["m00"], m["mu11"] / m["m00"]
        spread = np.sqrt(4 * mu11 ** 2 + (mu20 - mu02) ** 2)
        major, minor = (mu20 + mu02 + spread) / 2, (mu20 + mu02 - spread) / 2
        result["elongation"] = float(np.sqrt(major / max(minor, 1e-6)))
        axis_angle = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
        result["shape_angle"] = float(np.rad2deg(axis_angle) % 180)
        # The axis has no direction, so compare with |cos|
        axis = np.array([np.cos(axis_angle), np.sin(axis_angle)])
        # Thin strips along edges are common noise; only a sizeable region counts
        large_enough = stats[largest, cv2.CC_STAT_AREA] / roi_area * 100 >= MIN_MOVING_AREA
        result["shape_downslope"] = bool(large_enough and result["elongation"] > MIN_ELONGATION
                                         and abs(axis @ down) > cos_tol)

    result["is_downslope"] = bool(result["flow_downslope"] or result["shape_downslope"])
    return result


# ==================== RISK ====================

RISK_LEVELS = ["LOW", "MEDIUM", "HIGH"]
RISK_MESSAGES = {
    "LOW": ("Normal conditions - No immediate danger", (0, 255, 0)),
    "MEDIUM": ("CAUTION - Notable changes detected", (0, 165, 255)),
    "HIGH": ("!!! AVALANCHE WARNING - Significant movement detected !!!", (0, 0, 255)),
}


def assess_risk(largest_percentage, change_percentage, snow_change_percentage, downslope):
    if (largest_percentage > RISK_HIGH_REGION or change_percentage > RISK_HIGH_CHANGE
            or snow_change_percentage > RISK_HIGH_SNOW):
        level = 2
    elif (largest_percentage > RISK_MEDIUM_REGION or change_percentage > RISK_MEDIUM_CHANGE
            or snow_change_percentage > RISK_MEDIUM_SNOW):
        level = 1
    else:
        level = 0

    # Downslope movement of a changed region raises the risk by one level
    if downslope:
        level = min(level + 1, 2)

    risk_level = RISK_LEVELS[level]
    message, color = RISK_MESSAGES[risk_level]
    return risk_level, message, color


# ==================== VISUALIZATION ====================

def draw_motion(image, motion, change_mask, downslope_deg):
    """Outline moving regions, draw flow arrows and the expected fall line."""
    contours, _ = cv2.findContours(motion["motion_mask"], cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (255, 255, 0), 2)

    flow = motion["flow"]
    show_mask = cv2.bitwise_or(change_mask, motion["motion_mask"])
    if flow is not None:
        step = 30
        for y in range(step // 2, image.shape[0], step):
            for x in range(step // 2, image.shape[1], step):
                if show_mask[y, x] == 0:
                    continue
                fx, fy = flow[y, x]
                if np.hypot(fx, fy) < MIN_FLOW_MAGNITUDE:
                    continue
                end = (int(x + fx * 3), int(y + fy * 3))
                cv2.arrowedLine(image, (x, y), end, (255, 255, 0), 2, tipLength=0.4)

    # Expected downslope direction (legend in the corner)
    d = np.deg2rad(downslope_deg)
    origin = (image.shape[1] - 80, 90)
    tip = (int(origin[0] + 50 * np.cos(d)), int(origin[1] + 50 * np.sin(d)))
    cv2.arrowedLine(image, origin, tip, (255, 255, 255), 3, tipLength=0.3)
    cv2.putText(image, "downslope", (origin[0] - 70, origin[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def build_result_image(img1, img2_aligned, difference, change_mask, roi_mask, regions,
                       risk_level, alert_color, change_percentage, largest_percentage,
                       snow_change, motion, downslope_deg):
    height, width = img1.shape[:2]

    # Changes drawn in red over the second image
    overlay = img2_aligned.copy()
    overlay[change_mask > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(img2_aligned, 0.6, overlay, 0.4, 0)
    for x, y, w, h in regions:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
    draw_motion(overlay, motion, change_mask, downslope_deg)

    # Show the ROI border on the first image
    img1_roi = img1.copy()
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img1_roi, contours, -1, (255, 255, 0), 2)

    diff_colored = cv2.applyColorMap(cv2.normalize(difference, None, 0, 255, cv2.NORM_MINMAX),
                                     cv2.COLORMAP_JET)

    top = np.hstack([img1_roi, img2_aligned])
    bottom = np.hstack([diff_colored, overlay])
    result_display = np.vstack([top, bottom])

    labels = [("Image 1 (Earlier) + ROI", (20, 40)),
              ("Image 2 (Aligned)", (width + 20, 40)),
              ("Difference Map", (20, height + 40)),
              ("Detected Changes + Motion", (width + 20, height + 40))]
    for text, pos in labels:
        cv2.putText(result_display, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
        cv2.putText(result_display, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    direction = "DOWNSLOPE" if motion["is_downslope"] else "not downslope"
    banner = np.zeros((100, result_display.shape[1], 3), dtype=np.uint8)
    banner[:] = alert_color
    cv2.putText(banner, f"RISK LEVEL: {risk_level}", (50, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(banner, f"Change: {change_percentage:.1f}% | Largest region: "
                        f"{largest_percentage:.1f}% | Snow: {snow_change:+.1f}% | "
                        f"Motion: {direction}",
                (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return np.vstack([banner, result_display])


# ==================== MAIN PIPELINE ====================

def analyze(img1, img2, roi=None, roi_file=None, roi_mask=None,
            downslope_deg=DOWNSLOPE_ANGLE):
    """
    Compare two images taken at different times (e.g., 30 minutes apart)
    and detect changes that might indicate avalanche risk.

    img1, img2: BGR images (numpy arrays)
    roi_mask:   optional black/white mask array (white = monitored area)
    Returns a dictionary with all measurements and the result image.
    """
    # 1) Align image 2 onto image 1 (removes camera shake)
    img2_aligned, valid_mask = align_images(img1, img2)

    # 2) Region of interest (only the slope), limited to overlapping pixels
    if roi_mask is not None:
        roi_mask = cv2.resize(roi_mask, (img1.shape[1], img1.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        _, roi_mask = cv2.threshold(roi_mask, 127, 255, cv2.THRESH_BINARY)
    else:
        roi_mask = load_roi_mask(roi, roi_file, img1.shape)
    roi_mask = cv2.bitwise_and(roi_mask, valid_mask)

    # Sky and clouds are never part of the slope
    sky = cv2.bitwise_or(detect_sky(img1), detect_sky(img2_aligned))
    roi_mask = cv2.bitwise_and(roi_mask, cv2.bitwise_not(sky))

    # 3) Snow coverage
    snow_density1, _ = calculate_snow_density(img1, roi_mask)
    snow_density2, _ = calculate_snow_density(img2_aligned, roi_mask)
    snow_change = snow_density2 - snow_density1

    # 4) Illumination-invariant change detection (on the original brightness)
    raw1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    raw2 = cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2GRAY)
    difference, change_mask, change_percentage, largest_percentage, regions = \
        detect_changes(raw1, raw2, roi_mask)

    # Optical flow works better on fog / light normalized images
    gray1 = normalize_lighting(raw1)
    gray2 = match_brightness(normalize_lighting(raw2), gray1, roi_mask)

    # 5) Motion direction
    motion = analyze_motion(gray1, gray2, change_mask, roi_mask, downslope_deg)

    # 6) Risk assessment (an area sliding downslope counts like a changed region)
    sliding_area = motion["moving_area"] if motion["flow_downslope"] else 0.0
    risk_level, warning_message, alert_color = assess_risk(
        max(largest_percentage, sliding_area), change_percentage,
        abs(snow_change), motion["is_downslope"])

    result_image = build_result_image(img1, img2_aligned, difference, change_mask, roi_mask,
                                      regions, risk_level, alert_color, change_percentage,
                                      largest_percentage, snow_change, motion, downslope_deg)

    return {"risk_level": risk_level, "warning_message": warning_message,
            "snow_density1": snow_density1, "snow_density2": snow_density2,
            "snow_change": snow_change, "change_percentage": change_percentage,
            "largest_region": largest_percentage, "regions": len(regions),
            "motion": {k: v for k, v in motion.items() if k not in ("flow", "motion_mask")},
            "change_mask": change_mask, "motion_mask": motion["motion_mask"],
            "result_image": result_image}


def detect_avalanche_risk(img1_path, img2_path, roi=None, roi_file=None,
                          downslope_deg=DOWNSLOPE_ANGLE,
                          output_path='avalanche_analysis_result.jpg', show=True):
    """Command line version: reads files, prints a report, saves the result image."""
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None:
        print(f"ERROR: Could not load first image: {img1_path}")
        return None
    if img2 is None:
        print(f"ERROR: Could not load second image: {img2_path}")
        return None

    print("\n" + "=" * 60)
    print("AVALANCHE MONITORING SYSTEM - ANALYSIS STARTED")
    print("=" * 60 + "\n")

    r = analyze(img1, img2, roi=roi, roi_file=roi_file, downslope_deg=downslope_deg)
    m = r["motion"]

    print(f"\nImage 1 - Snow Coverage: {r['snow_density1']:.2f}%")
    print(f"Image 2 - Snow Coverage: {r['snow_density2']:.2f}%")
    print(f"\nSnow Coverage Change: {r['snow_change']:+.2f}%")
    print(f"Visual Change Detected: {r['change_percentage']:.2f}% ({r['regions']} region(s))")
    print(f"Largest Changed Region: {r['largest_region']:.2f}%")

    print(f"\nMOTION DIRECTION (downslope = {downslope_deg:.0f} deg):")
    if m["flow_angle"] is not None:
        print(f"  Optical flow: {m['moving_area']:.2f}% of area moving, "
              f"{m['flow_magnitude']:.1f} px towards {m['flow_angle']:.0f} deg, "
              f"{m['downslope_ratio'] * 100:.0f}% of moving pixels downslope")
    else:
        print("  Optical flow: no significant texture movement")
    if m["shape_angle"] is not None:
        print(f"  Region shape: elongation {m['elongation']:.1f}, axis {m['shape_angle']:.0f} deg")
    print(f"  Downslope movement: {'YES' if m['is_downslope'] else 'NO'}")

    print("\n" + "-" * 60)
    print("RISK ASSESSMENT:")
    print("-" * 60)
    print(f"Risk Level: {r['risk_level']}")
    print(f"Status: {r['warning_message']}")
    print("-" * 60)

    cv2.imwrite(output_path, r["result_image"])
    print(f"\nResult saved to: {output_path}")

    if show:
        cv2.imshow('Avalanche Monitoring - Analysis Results', r["result_image"])
        print("\nPress any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED")
    print("=" * 60 + "\n")

    return r


# ==================== MAIN PROGRAM ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avalanche Early Warning System")
    parser.add_argument("image1", nargs="?", default="mountain_1.png", help="Earlier image")
    parser.add_argument("image2", nargs="?", default="mountain_2.png", help="Later image")
    parser.add_argument("--roi", help="Slope rectangle to monitor: x,y,w,h")
    parser.add_argument("--roi-mask", default="roi_mask.png",
                        help="Black/white mask image, white = monitor (default: roi_mask.png)")
    parser.add_argument("--downslope", type=float, default=DOWNSLOPE_ANGLE,
                        help="Fall-line direction in the image, degrees "
                             "(0 = right, 90 = down, 180 = left). Default: 90")
    parser.add_argument("--output", default="avalanche_analysis_result.jpg")
    parser.add_argument("--no-show", action="store_true", help="Do not open a window")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("AVALANCHE EARLY WARNING SYSTEM")
    print("Snow Movement Detection Using Image Comparison")
    print("=" * 60)

    for path in (args.image1, args.image2):
        if not os.path.exists(path):
            print(f"\nERROR: Image not found: {path}")
            print("Please place your images in the same folder as this script.")
            raise SystemExit(1)

    detect_avalanche_risk(args.image1, args.image2, roi=args.roi, roi_file=args.roi_mask,
                          downslope_deg=args.downslope, output_path=args.output,
                          show=not args.no_show)
