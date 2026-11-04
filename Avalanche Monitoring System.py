import argparse
import os

import cv2
import numpy as np


# ==================== SETTINGS ====================

DIFF_THRESHOLD = 30          # Minimum brightness difference counted as change
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


def detect_changes(gray1, gray2, mask):
    """
    Image differencing on aligned + normalized images.
    Small isolated regions (noise, falling snowflakes, moving people)
    are removed; an avalanche is a large connected region.
    """
    gray1_blur = cv2.GaussianBlur(gray1, (21, 21), 0)
    gray2_blur = cv2.GaussianBlur(gray2, (21, 21), 0)

    difference = cv2.absdiff(gray1_blur, gray2_blur)
    difference = cv2.bitwise_and(difference, mask)

    _, thresh_diff = cv2.threshold(difference, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    thresh_diff = cv2.morphologyEx(thresh_diff, cv2.MORPH_OPEN, kernel)
    thresh_diff = cv2.dilate(thresh_diff, kernel, iterations=2)

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


def assess_risk(largest_percentage, change_percentage, snow_change_percentage):
    if (largest_percentage > RISK_HIGH_REGION or change_percentage > RISK_HIGH_CHANGE
            or snow_change_percentage > RISK_HIGH_SNOW):
        return ("HIGH", "!!! AVALANCHE WARNING - Significant movement detected !!!",
                (0, 0, 255))
    if (largest_percentage > RISK_MEDIUM_REGION or change_percentage > RISK_MEDIUM_CHANGE
            or snow_change_percentage > RISK_MEDIUM_SNOW):
        return "MEDIUM", "CAUTION - Notable changes detected", (0, 165, 255)
    return "LOW", "Normal conditions - No immediate danger", (0, 255, 0)


# ==================== VISUALIZATION ====================

def build_result_image(img1, img2_aligned, difference, change_mask, roi_mask,
                       regions, risk_level, alert_color, change_percentage,
                       largest_percentage, snow_change):
    height, width = img1.shape[:2]

    # Changes drawn in red over the second image
    overlay = img2_aligned.copy()
    overlay[change_mask > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(img2_aligned, 0.6, overlay, 0.4, 0)
    for x, y, w, h in regions:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)

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
              ("Detected Changes", (width + 20, height + 40))]
    for text, pos in labels:
        cv2.putText(result_display, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
        cv2.putText(result_display, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    banner = np.zeros((100, result_display.shape[1], 3), dtype=np.uint8)
    banner[:] = alert_color
    cv2.putText(banner, f"RISK LEVEL: {risk_level}", (50, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(banner, f"Change: {change_percentage:.1f}% | Largest region: "
                        f"{largest_percentage:.1f}% | Snow: {snow_change:+.1f}%",
                (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return np.vstack([banner, result_display])


# ==================== MAIN PIPELINE ====================

def detect_avalanche_risk(img1_path, img2_path, roi=None, roi_file=None,
                          output_path='avalanche_analysis_result.jpg', show=True):
    """
    Compare two images taken at different times (e.g., 30 minutes apart)
    Detect significant changes that might indicate avalanche risk
    """
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

    # 1) Align image 2 onto image 1 (removes camera shake)
    img2_aligned, valid_mask = align_images(img1, img2)

    # 2) Region of interest (only the slope), limited to overlapping pixels
    roi_mask = load_roi_mask(roi, roi_file, img1.shape)
    roi_mask = cv2.bitwise_and(roi_mask, valid_mask)

    # 3) Snow coverage
    snow_density1, _ = calculate_snow_density(img1, roi_mask)
    snow_density2, _ = calculate_snow_density(img2_aligned, roi_mask)
    snow_change = snow_density2 - snow_density1

    print(f"\nImage 1 - Snow Coverage: {snow_density1:.2f}%")
    print(f"Image 2 - Snow Coverage: {snow_density2:.2f}%")
    print(f"\nSnow Coverage Change: {snow_change:+.2f}%")

    # 4) Lighting / fog normalization, then change detection
    gray1 = normalize_lighting(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
    gray2 = normalize_lighting(cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2GRAY))
    gray2 = match_brightness(gray2, gray1, roi_mask)

    difference, change_mask, change_percentage, largest_percentage, regions = \
        detect_changes(gray1, gray2, roi_mask)
    print(f"Visual Change Detected: {change_percentage:.2f}% ({len(regions)} region(s))")
    print(f"Largest Changed Region: {largest_percentage:.2f}%")

    # 5) Risk assessment
    risk_level, warning_message, alert_color = assess_risk(largest_percentage, change_percentage,
                                                           abs(snow_change))

    print("\n" + "-" * 60)
    print("RISK ASSESSMENT:")
    print("-" * 60)
    print(f"Risk Level: {risk_level}")
    print(f"Status: {warning_message}")
    print("-" * 60)

    final_display = build_result_image(img1, img2_aligned, difference, change_mask, roi_mask,
                                       regions, risk_level, alert_color,
                                       change_percentage, largest_percentage, snow_change)

    cv2.imwrite(output_path, final_display)
    print(f"\nResult saved to: {output_path}")

    if show:
        cv2.imshow('Avalanche Monitoring - Analysis Results', final_display)
        print("\nPress any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED")
    print("=" * 60 + "\n")

    return {"risk_level": risk_level, "change_percentage": change_percentage,
            "largest_region": largest_percentage, "snow_change": snow_change,
            "regions": len(regions)}


# ==================== MAIN PROGRAM ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avalanche Early Warning System")
    parser.add_argument("image1", nargs="?", default="mountain_1.png", help="Earlier image")
    parser.add_argument("image2", nargs="?", default="mountain_2.png", help="Later image")
    parser.add_argument("--roi", help="Slope rectangle to monitor: x,y,w,h")
    parser.add_argument("--roi-mask", default="roi_mask.png",
                        help="Black/white mask image, white = monitor (default: roi_mask.png)")
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
                          output_path=args.output, show=not args.no_show)
