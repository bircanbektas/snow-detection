"""Web interface for the avalanche early warning system.

Run with:  streamlit run app.py
"""
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from avalanche_monitor import analyze

HERE = Path(__file__).parent
SAMPLES = (HERE / "mountain_1.png", HERE / "mountain_2.png")


@st.cache_data(max_entries=20, show_spinner=False)
def run_analysis(bytes1, bytes2, roi_percent, downslope_deg):
    img1 = cv2.imdecode(np.frombuffer(bytes1, np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(bytes2, np.uint8), cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        return None

    roi = None
    if roi_percent is not None:
        (left, right), (top, bottom) = roi_percent
        h, w = img1.shape[:2]
        x, y = int(w * left / 100), int(h * top / 100)
        roi = f"{x},{y},{int(w * right / 100) - x},{int(h * bottom / 100) - y}"

    return analyze(img1, img2, roi=roi, roi_file=None, downslope_deg=downslope_deg)


st.set_page_config(page_title="Avalanche early warning", page_icon=":material/landslide:",
                   layout="wide")

st.title(":material/landslide: Avalanche early warning")
st.caption("Compares two photos from a fixed mountain camera taken some time apart "
           "and estimates the avalanche risk from snow movement.")

# ---------- Settings ----------
with st.sidebar:
    st.header("Settings")
    downslope = st.slider("Downslope direction (degrees)", 0, 180, 90, step=5,
                          help="Fall line of the slope in the image: "
                               "0 = right, 90 = straight down, 180 = left.")

    restrict = st.toggle("Monitor only part of the image", value=False,
                         help="Exclude sky, buildings and lift poles from the analysis.")
    roi_percent = None
    if restrict:
        horizontal = st.slider("Horizontal range (%)", 0, 100, (0, 100))
        vertical = st.slider("Vertical range (%)", 0, 100, (20, 100))
        roi_percent = (horizontal, vertical)

# ---------- Input images ----------
source = st.segmented_control("Images", ["Sample images", "Upload my own"],
                              default="Sample images")

bytes1 = bytes2 = None
if source == "Upload my own":
    col1, col2 = st.columns(2)
    file1 = col1.file_uploader("Earlier image", type=["png", "jpg", "jpeg"])
    file2 = col2.file_uploader("Later image", type=["png", "jpg", "jpeg"])
    if file1 and file2:
        bytes1, bytes2 = file1.getvalue(), file2.getvalue()
    else:
        st.info("Upload two photos of the same slope, taken from the same camera.",
                icon=":material/upload:")
else:
    bytes1, bytes2 = SAMPLES[0].read_bytes(), SAMPLES[1].read_bytes()

if bytes1 is None:
    st.stop()

# ---------- Analysis ----------
with st.spinner("Aligning images and detecting movement..."):
    result = run_analysis(bytes1, bytes2, roi_percent, downslope)

if result is None:
    st.error("One of the files could not be read as an image.", icon=":material/error:")
    st.stop()

motion = result["motion"]
risk = result["risk_level"]
alert = {"LOW": st.success, "MEDIUM": st.warning, "HIGH": st.error}[risk]
icon = {"LOW": ":material/check_circle:", "MEDIUM": ":material/warning:",
        "HIGH": ":material/crisis_alert:"}[risk]
alert(result["warning_message"], title=f"Risk level: {risk}", icon=icon)

with st.container(horizontal=True):
    st.metric("Largest changed region", f"{result['largest_region']:.1f} %", border=True)
    st.metric("Total change", f"{result['change_percentage']:.1f} %", border=True)
    st.metric("Moving area", f"{motion['moving_area']:.1f} %", border=True)
    st.metric("Snow coverage change", f"{result['snow_change']:+.1f} %", border=True)
    st.metric("Movement direction", "Downslope" if motion["is_downslope"] else "Not downslope",
              border=True)

st.image(result["result_image"], channels="BGR", width="stretch",
         caption="Top: earlier image with monitored area, later image after alignment. "
                 "Bottom: difference map, detected changes (red) and moving areas (cyan arrows).",
         alt=f"Analysis panels for the two images, risk level {risk}")

with st.expander("How does it work?"):
    st.markdown(
        "1. **Alignment** – the later photo is warped onto the earlier one, "
        "so camera shake is not mistaken for movement.\n"
        "2. **Sky removal** – clear blue sky is excluded automatically.\n"
        "3. **Change detection** – looks for changes in surface texture (roughness and "
        "pattern) instead of brightness, so moving shadows are ignored; "
        "only large regions are kept.\n"
        "4. **Motion** – optical flow finds snow that slid, even without a colour change; "
        "the shape check finds long tongues along the fall line.\n"
        "5. **Risk** – based on the largest changed or sliding region; "
        "downslope movement raises the risk by one level."
    )
