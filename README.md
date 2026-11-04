# Snow Movement Detection for Avalanche Early Warning

**Author:** Bircan Bektaş  
**Course:** Image Processing – Term Project  

## 📋 Project Description
This project detects snow movement and potential avalanche risk by comparing two images taken at different time intervals (e.g., 30 minutes apart).  
Unlike video-based systems, this approach works with periodic images from a fixed camera, making it suitable for real-world mountain monitoring.

## 🎯 Objectives
- Compare snow-covered mountain images
- Detect visual and snow density changes
- Assess avalanche risk level (LOW / MEDIUM / HIGH)
- Generate visual analysis results

## 🛠️ Technologies Used
- Python 3  
- OpenCV  
- NumPy  

## 🚀 Usage
1. Install the requirements:
   ```
   pip install -r requirements.txt
   ```
2. Run the program (defaults to `mountain_1.png` and `mountain_2.png`):
   ```
   python "Avalanche Monitoring System.py"
   python "Avalanche Monitoring System.py" earlier.jpg later.jpg
   ```
3. Optional: monitor only the slope (exclude sky, buildings, poles):
   ```
   python "Avalanche Monitoring System.py" --roi 0,150,1194,550
   ```
   or put a black/white `roi_mask.png` (white = monitored area) next to the script.
4. View the generated result image: `avalanche_analysis_result.jpg`
   (use `--no-show` to skip the preview window)

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
6. **Risk classification** – based on the largest connected changed region, total
   changed area and snow coverage change:

   | Level  | Largest region | Total change | Snow change |
   |--------|----------------|--------------|-------------|
   | HIGH   | > 5 %          | > 10 %       | > 35 %      |
   | MEDIUM | > 1.5 %        | > 4 %        | > 20 %      |

## 🧪 Test Results
| Test case                              | Largest region | Risk   |
|----------------------------------------|----------------|--------|
| Real images (camera shifted, fog)      | 0.0 %          | LOW    |
| Synthetic small avalanche              | 2.1 %          | MEDIUM |
| Synthetic large avalanche              | 15.9 %         | HIGH   |

Before alignment, the camera shift alone produced 2.2 % false change.

## 📂 Project Structure
snow-detection2/
├── Avalanche Monitoring System.py
├── requirements.txt
├── mountain_1.png
├── mountain_2.png
├── avalanche_analysis_result.jpg
└── README.md
## 📝 Note
This project is developed for educational purposes. Real avalanche monitoring systems require additional sensors and environmental data.