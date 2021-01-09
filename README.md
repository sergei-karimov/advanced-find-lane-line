# Advanced Find Lane Line project

![result image](result_images/result.jpg)
1. **result image**

## Introduction
> This is Advanced lane finding project of Udacity's Self-Driving Car Engineering Nanodegree. We already completed lane finding project in the first project. In that project, we could find lane lines and made robust algorighm for shadow and some of occlusion. It might be enough in the straight highway. But there are many curve lines in the road and that's why we need to detect curve lanes. In this project we'll find lane lines more specifically with computer vision.
## Environment
> MacOS Big Sur, Python 3.8.5, OpenCV 4.4.0.46
## Files
- [main.py](main.py) to do main work
- [image_processing.py](image_processing.py) to process the single image
- [video_processing.py](video_processing.py) to process the fideo file
## The goal / steps of this project are the following:
- Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
- Apply a distortion correction to raw images.
- Use color transforms, gradients, etc., to create a thresholded binary image.
- Apply a perspective transform to rectify a binary image ("birds-eye view").
- Detect lane pixels and fit to find the lane boundary.
- Determine the curvature of the lane and vehicle position with respect to center.
- Warp the detected lane boundaries back onto the original image.
- Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## Pipeline
- Load image
- Calibrate camera
- Perspective transform
- Convert given image to HSL image
- Isolate yellow and white color from HSL image
- Convert image to grayscale
- Select region of interest
- Trace region of interest and discard all other lines identified by our previous step
- Apply a perspective transform to rectify binary image
- Find lanes
- Draw segments of roads and road info

### 1. Load image
### 2. Calibrate camera
### 3. Perspective transform
Perspective Transform is a feature that is very useful if you want to align the image properly . It transform the image in a straight manner after Perspective Transformation is applied to it.
We can assume the road is a flat plane. We take 4 points of straight of lane lines and put to respective transform function. As a result we get Bird's eye view.
### 4. Convert given image to HSL image
### 5. Isolate yellow and white color from HSL image
### 6. Convert image to grayscale
### 7. Select region of interest
### 8. Trace region of interest and discard all other lines identified by our previous step
### 9. Apply a perspective transform to rectify binary image
### 10. Find lanes
### 11. Draw segments of roads and road info
