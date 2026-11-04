# Snow Movement Detection for Avalanche Early Warning

**Author:** Bircan Bektaş  
**Course:** Image Processing – Term Project  

## 📋 Project Description
This project detects snow movement and potential avalanche risk by comparing two images taken at different time intervals (e.g., 30 minutes apart).  
Unlike video-based systems, this approach works with periodic images from a fixed camera, making it suitable for real-world mountain monitoring.

## 🎯 Objectives
- Compare snow-covered mountain images
- Detect visual and snow density changes
- Detect whether snow moved downslope
- Assess avalanche risk level (LOW / MEDIUM / HIGH)
- Generate visual analysis results, from the command line or a web interface

## 🛠️ Technologies Used
- Python 3  
- OpenCV  
- NumPy  
- Streamlit (web interface)

## 🚀 Usage
Install the requirements:
```
pip install -r requirements.txt
```

### Web interface
```
streamlit run app.py
```
Use the sample images or upload your own two photos. The sidebar sets the downslope
direction and the part of the image to monitor.

### Command line
```
python avalanche_monitor.py                         # mountain_1.png vs mountain_2.png
python avalanche_monitor.py earlier.jpg later.jpg
python avalanche_monitor.py --roi 0,150,1194,550    # monitor only this rectangle
python avalanche_monitor.py --downslope 60          # fall line points down-right
```
- Instead of `--roi`, a black/white `roi_mask.png` (white = monitored area) can be placed
  next to the script.
- `--downslope` is the fall-line direction in the image: 0 = right, 90 = down (default),
  180 = left.
- The result is saved to `avalanche_analysis_result.jpg` (`--no-show` skips the window).

## 📊 Method Overview
1. **Image alignment** – ORB keypoints + RANSAC homography align the second image
   onto the first, so camera shake is not mistaken for snow movement.
2. **Region of interest** – only the monitored slope is analysed.
3. **Lighting / fog normalization** – CLAHE and brightness matching reduce the effect
   of fog and changing daylight.
4. **Snow detection** – bright and low-saturation pixels; the brightness threshold is
   chosen automatically with Otsu's method instead of a fixed value.
5. **Change detection** – image differencing, morphological cleaning and removal of
   small regions (noise, snowflakes, people).
6. **Motion direction** – two independent clues:
   - *Optical flow* (Farneback) finds snow that slid even when its brightness hardly
     changed. Flow is trusted only where the surface has texture and the
     forward-backward check agrees, which filters out drifting fog.
   - *Region shape*: avalanche paths are long tongues along the fall line, measured
     with image moments (elongation and main axis angle).
7. **Risk classification** – based on the largest changed (or downslope-sliding)
   region, total changed area and snow coverage change:

   | Level  | Largest region | Total change | Snow change |
   |--------|----------------|--------------|-------------|
   | HIGH   | > 5 %          | > 10 %       | > 35 %      |
   | MEDIUM | > 1.5 %        | > 4 %        | > 20 %      |

   Downslope movement raises the risk by one level.

## 🧪 Test Results
| Test case                                  | Detected by           | Direction     | Risk   |
|--------------------------------------------|-----------------------|---------------|--------|
| Real images (camera shifted, fog)          | –                     | –             | LOW    |
| Synthetic small avalanche tongue (2 %)     | Differencing + shape  | Downslope     | HIGH   |
| Synthetic large avalanche tongue (16 %)    | Differencing + shape  | Downslope     | HIGH   |
| Synthetic snow slab sliding down 12 px     | Optical flow          | Downslope     | HIGH   |
| Synthetic snow slab moving up 12 px        | Optical flow          | Not downslope | LOW    |

- Before alignment, the camera shift alone produced 2.2 % false change.
- The sliding slab is invisible to plain differencing (0 % change); only optical flow finds it.
- The synthetic tests are artificial; thresholds should be tuned on real avalanche images.

## 📂 Project Structure
```
snow-detection/
├── avalanche_monitor.py           # analysis pipeline + command line
├── app.py                         # Streamlit web interface
├── requirements.txt
├── mountain_1.png
├── mountain_2.png
├── avalanche_analysis_result.jpg
└── README.md
```

## 📝 Note
This project is developed for educational purposes. Real avalanche monitoring systems require additional sensors and environmental data.
