"""Web interface for the avalanche early warning system.

Run with:  streamlit run app.py
"""
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

import ai_detector
from avalanche_monitor import analyze

HERE = Path(__file__).parent
SAMPLES = {
    "Real webcam (SLF Davos)": (HERE / "samples" / "slf_davos_0935.jpg",
                                HERE / "samples" / "slf_davos_1135.jpg"),
    "Synthetic": (HERE / "mountain_1.png", HERE / "mountain_2.png"),
}
AI, CLASSICAL = "AI model (U-Net)", "Classical image processing"


def decode(data):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def roi_mask(shape, roi_percent):
    if roi_percent is None:
        return None
    (left, right), (top, bottom) = roi_percent
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    mask[int(h * top / 100):int(h * bottom / 100), int(w * left / 100):int(w * right / 100)] = 255
    return mask


@st.cache_resource(show_spinner="Downloading the AI model (~100 MB, only the first time)...")
def get_model():
    return ai_detector.load_model()


@st.cache_data(max_entries=20, show_spinner=False)
def run_classical(bytes1, bytes2, roi_percent, downslope_deg):
    img1, img2 = decode(bytes1), decode(bytes2)
    if img1 is None or img2 is None:
        return None
    return analyze(img1, img2, roi_mask=roi_mask(img1.shape, roi_percent),
                   downslope_deg=downslope_deg)


@st.cache_data(max_entries=20, show_spinner=False)
def run_ai(bytes1, bytes2, roi_percent):
    img1, img2 = decode(bytes1), decode(bytes2)
    if img1 is None or img2 is None:
        return None
    return ai_detector.detect_new_avalanches(get_model(), img1, img2,
                                             roi_mask(img1.shape, roi_percent))


st.set_page_config(page_title="Avalanche early warning", page_icon=":material/landslide:",
                   layout="wide")

st.title(":material/landslide: Avalanche early warning")
st.caption("Compares two photos from a fixed mountain camera taken some time apart "
           "and detects new avalanches.")

# ---------- Settings ----------
with st.sidebar:
    st.header("Settings")
    method = st.radio("Detection method", [AI, CLASSICAL],
                      help="The AI model was trained on expert-annotated avalanche photos and "
                           "raises far fewer false alarms on real webcams.")
    if method == CLASSICAL:
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
source = st.segmented_control("Images", [*SAMPLES, "Upload my own"],
                              default="Real webcam (SLF Davos)")

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
elif source in SAMPLES:
    bytes1, bytes2 = (p.read_bytes() for p in SAMPLES[source])
    if source == "Real webcam (SLF Davos)":
        st.caption("Dischma valley webcam, 19 March 2021, 09:35 and 11:35. "
                   "Source: [SLF / EnviDat](https://doi.org/10.16904/envidat.526), CC BY-SA 4.0. "
                   "This pair was part of the final model's training data.")

if bytes1 is None:
    st.stop()

# ---------- AI model ----------
if method == AI:
    with st.spinner("Aligning images and looking for avalanches..."):
        try:
            result = run_ai(bytes1, bytes2, roi_percent)
        except Exception as e:
            st.error(f"The AI model could not be loaded or run: {e}", icon=":material/error:")
            st.stop()
    if result is None:
        st.error("One of the files could not be read as an image.", icon=":material/error:")
        st.stop()

    if result["is_new"]:
        st.error(f"New avalanche deposits cover {result['new_pct']:.1f} % of the monitored area.",
                 title="New avalanche detected", icon=":material/crisis_alert:")
    else:
        st.success("No new avalanche between the two photos.", title="No new avalanche",
                   icon=":material/check_circle:")

    with st.container(horizontal=True):
        st.metric("New avalanche area", f"{result['new_pct']:.1f} %", border=True)
        st.metric("Avalanches in earlier photo", f"{result['earlier_pct']:.1f} %", border=True)
        st.metric("Avalanches in later photo", f"{result['later_pct']:.1f} %", border=True)

    st.image(result["result_image"], channels="BGR", width="stretch",
             caption="Left: earlier photo. Right: later photo after alignment. "
                     "Yellow: avalanches already visible earlier, red: new avalanches.",
             alt="Earlier and later photo with avalanche outlines")

    with st.expander("How does it work?"):
        st.markdown(
            "1. **Alignment** – the later photo is warped onto the earlier one, "
            "so camera shake is not mistaken for change.\n"
            "2. **Avalanche segmentation** – a U-Net neural network outlines avalanche "
            "deposits in each photo. It was pre-trained on ~3,600 annotated field photos "
            "(UIBK) and fine-tuned on SLF webcam images.\n"
            "3. **New avalanches** – outlines in the later photo that were not there in the "
            "earlier one. An area of at least 0.2 % of the image raises the alarm.\n\n"
            "On 34 real webcam pairs it found 8 of 13 new avalanches with 5 false alarms; "
            "the classical method had 17 false alarms. Details in the README."
        )
    st.stop()

# ---------- Classical ----------
with st.spinner("Aligning images and detecting movement..."):
    result = run_classical(bytes1, bytes2, roi_percent, downslope)

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
