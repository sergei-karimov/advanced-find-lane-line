import cv2
import numpy as np
from calibration import get_calibration
from combine_functions import gradient_combine, hls_combine, comb_result
from lines import Line, warp_image, find_LR_lines, draw_lane
from image_show import imageShow

mtx, dist = get_calibration()
left_line = Line()
right_line = Line()

th_sobelx, th_sobely, th_mag, th_dir = (35, 100), (30, 255), (30, 255), (0.7, 1.3)
th_h, th_l, th_s = (10, 100), (0, 60), (85, 255)


def image_processing(file_name):
    given_image = cv2.imread(file_name)
    undistorded_image = cv2.undistort(given_image, mtx, dist, None, mtx)
    undistorded_image = cv2.resize(undistorded_image, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA)
    rows, cols = undistorded_image.shape[:2]

    gradient = gradient_combine(undistorded_image, th_sobelx, th_sobely, th_mag, th_dir)
    hls = hls_combine(undistorded_image, th_h, th_l, th_s)
    combined_result = comb_result(gradient, hls)
    imageShow(combined_result)

    c_rows, c_cols = combined_result.shape[:2]
    s_LTop2, s_RTop2 = [c_cols / 2 - 24, 5], [c_cols / 2 + 24, 5]
    s_LBot2, s_RBot2 = [110, c_rows], [c_cols - 110, c_rows]

    src = np.float32([s_LBot2, s_LTop2, s_RTop2, s_RBot2])
    dst = np.float32([(170, 720), (170, 0), (550, 0), (550, 720)])

    warp_img, M, Minv = warp_image(combined_result, src, dst, (720, 720))
    searching_img = find_LR_lines(warp_img, left_line, right_line)
    w_comb_result, w_color_result = draw_lane(searching_img, left_line, right_line)
    imageShow(w_comb_result)

    # Draw lines
    color_result = cv2.warpPerspective(w_color_result, Minv, (c_cols, c_rows))
    combine_result = np.zeros_like(undistorded_image)
    combine_result[220:rows - 12, 0:cols] = color_result

    # Combine result and given image
    result = cv2.addWeighted(undistorded_image, 1, combine_result, 0.3, 0)
    imageShow(result)

