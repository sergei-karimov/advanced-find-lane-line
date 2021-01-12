# Advanced Find Lane Line project

![result image](result_images/result.jpg)
1. *result image*

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
Load image from disk.

![given image](result_images/given_image.jpg)
2. *given image*

### 2. Calibrate camera
Have the camera matrix and distortion coefficients been computed correctly and checked on one of the calibration images as a test?
The code for this step is contained in:
```python
    def camera_calibration(cls, directory_name):
        calibration_pics_loc = os.listdir(directory_name)
        calibration_images = []

        for i in calibration_pics_loc:
            i = f'{directory_name}/{i}'
            image = cv2.imread(i)
            calibration_images.append(image)

        # Prepare object points
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays for later storing object points and image points
        objpoints = []
        imgpoints = []

        for image in calibration_images:

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                cv2.drawChessboardCorners(image, (9, 6), corners, ret)

        # Get undistortion info and undistort
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        return mtx, dist
```
This script loads calibration images of chessboards taken at different angles. Each image is grayscaled and sent into cv2.drawChessboardCorners. The resulting "objpoints" are the (x, y, z) coordinates of the chessboard corners in the world.

Finally the corner points are sent to cv2.calibrateCamera to get resulting image points and object points. This dictionary is then saved for reuse in undistorted other images in the pipeline.

### 3. Perspective transform
Perspective Transform is a feature that is very useful if you want to align the image properly . It transforms the image in a straight manner after Perspective Transformation is applied to it.
We can assume the road is a flat plane. We take 4 points of straight of lane lines and put to respective transform function. As a result we get Bird's eye view.

![undistorted image](result_images/undistorted_image.jpg)
3. *undistorted image*

### 4. Convert given image to HSL image
How does it look like when images are converted from RGB to HSL color space?

![HLS image](result_images/hls_image.jpg)
4. *HLS image*

### 5. Isolate yellow and white color from HSL image
Let’s build a filter to select those white and yellow lines. I want to select particular range of each channels (Hue, Saturation and Light).

Both the white and yellow lines are clearly recognizable.

![isolated image](result_images/isolated_image.jpg)
5. *Isolated image*

### 6. Convert image to grayscale
### 7. Select region of interest
### 8. Trace region of interest and discard all other lines identified by our previous step
### 9. Apply a perspective transform to rectify binary image
### 10. Find lanes
### 11. Draw segments of roads and road info
