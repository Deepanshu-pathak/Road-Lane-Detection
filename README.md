# Road Lane Line Detection System using OpenCV

This project implements a **real-time road lane line detection system** using Python and OpenCV. It processes either a single image or video frames to detect lane lines, highlighting them on the road in green.

## Features

* Canny edge detection
* Gaussian blur for noise reduction
* Region of Interest (ROI) masking
* Hough Transform for line detection
* Slope-based line averaging for stable detection
* Overlay of detected lane lines on the original frame
* Real-time video processing

---

## Project Structure

```bash
lane-line-detection/
├── test_image.jpg         # (optional) Sample image for single-frame testing
├── test2.mp4              # Input video file for real-time lane detection
├── lanes.py               # Main Python script
├── README.md              # Project documentation
```

---

## Requirements

* Python 3.x
* OpenCV
* NumPy

Install dependencies:

```bash
pip install opencv-python numpy
```

---

## How It Works

### 1. **Canny Edge Detection**

* Converts image to grayscale
* Applies Gaussian Blur to reduce noise
* Uses `cv2.Canny()` to detect edges

### 2. **Region of Interest (ROI)**

* A polygonal mask is defined to cover only the road ahead
* Everything outside this triangle is ignored

### 3. **Hough Line Transform**

* Uses `cv2.HoughLinesP()` to detect lines within the ROI
* Parameters like `minLineLength` and `maxLineGap` control detection sensitivity

### 4. **Slope-Based Line Averaging**

* Separates left and right lane lines based on slope
* Averages multiple detected lines for each side to form smooth lanes

### 5. **Overlay Detected Lines**

* Detected lanes are drawn using `cv2.line()`
* Combined with original frame using `cv2.addWeighted()`

### 6. **Video Capture Loop**

* Reads video frames using `cv2.VideoCapture()`
* Applies entire pipeline to each frame
* Breaks when 'q' is pressed

---

## Important Functions

### `canny_method(img)`

Returns a Canny edge-detected version of the input image after grayscale conversion and Gaussian blur.

### `region_of_interest(img)`

Applies a triangular mask to keep only the region where lane lines are expected.

### `average_slope_intercept(img, lines)`

Calculates and averages slope and intercepts for left and right lane lines.

### `display_lines(img, lines)`

Draws the detected average lane lines on a black image for visualization.

---

## How to Run

### Run on Video:

```bash
python lanes.py
```

### Optional: Run on Image

Uncomment the image-related block in the script to test on a single image:

```python
image = cv2.imread('test_image.jpg')
... # follow the image block in the code
```

---

## Output

* Lane lines will be displayed in green on the output video frame.
* Press `q` to exit the video window.

---

## License

This project is for educational and research purposes only.
