import numpy as np
import cv2
import matplotlib.image as mpimg
import glob

images = glob.glob('camera_cal/calibration*.jpg')

objpoints = []
imgpoints = []


def get_calibration():
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    for image in images:

        given_image = mpimg.imread(image)
        grayed_image = cv2.cvtColor(given_image, cv2.COLOR_BGR2GRAY)

        return_value, corners = cv2.findChessboardCorners(grayed_image, (9, 6), None)
        if return_value:
            imgpoints.append(corners)
            objpoints.append(objp)
        else:
            continue

    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, grayed_image.shape[::-1], None, None)
    return mtx, dist
