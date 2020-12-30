import cv2
import numpy as np

from calibration import get_calibration
from lines import Line, find_LR_lines, draw_lane
from image_show import imageShow

mtx, dist = get_calibration()
left_line = Line()
right_line = Line()

threshold_x, threshold_y, th_mag, th_dir = (35, 100), (30, 255), (30, 255), (0.7, 1.3)
th_h, th_l, th_s = (10, 100), (0, 60), (85, 255)


def dir_sobel(gray_image, kernel_size, threshold):
    abs_x = np.absolute(cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=kernel_size))
    abs_y = np.absolute(cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=kernel_size))

    direction = np.arctan2(abs_x, abs_y)

    binary = np.zeros_like(direction)
    binary[(direction >= threshold[0]) & (direction <= threshold[1])] = 1

    return binary


def combined_sobels(x, y, magnitude, gray_image, kernel_size=3, angle_threshold=(0, np.pi / 2)):
    direction = dir_sobel(gray_image, kernel_size=kernel_size, threshold=angle_threshold)

    combined = np.zeros_like(direction)
    combined[(x == 1) | ((y == 1) & (magnitude == 1) & (direction == 1))] = 1

    return combined


def magnitude_sobel(undistorted_gray_image, kernel_size, threshold):
    sx = cv2.Sobel(undistorted_gray_image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sy = cv2.Sobel(undistorted_gray_image, cv2.CV_64F, 0, 1, ksize=kernel_size)

    sxy = np.sqrt(np.square(sx) + np.square(sy))
    scaled_sxy = np.uint8(255 * sxy / np.max(sxy))

    result = np.zeros_like(scaled_sxy)
    result[(scaled_sxy >= threshold[0]) & (scaled_sxy <= threshold[1])] = 1

    return result


def abs_sobel(undistorted_gray_image, x_dir, kernel_size, threshold):
    sobel = cv2.Sobel(
        undistorted_gray_image,
        cv2.CV_64F,
        1,
        0,
        ksize=kernel_size
    ) if x_dir else cv2.Sobel(undistorted_gray_image, cv2.CV_64F, 0, 1, ksize=kernel_size)
    sobel_abs = np.absolute(sobel)
    sobel_scaled = np.uint8(255 * sobel / np.max(sobel_abs))

    gradient_mask = np.zeros_like(sobel_scaled)
    gradient_mask[(threshold[0] <= sobel_scaled) & (sobel_scaled <= threshold[1])] = 1
    return gradient_mask


def compute_hls_white_yellow_binary(undistorted_image):
    hls_image = cv2.cvtColor(undistorted_image, cv2.COLOR_RGB2HLS)

    img_hls_yellow_bin = np.zeros_like(hls_image[:, :, 0])
    img_hls_yellow_bin[((hls_image[:, :, 0] >= 15) & (hls_image[:, :, 0] <= 35))
                       & ((hls_image[:, :, 1] >= 30) & (hls_image[:, :, 1] <= 204))
                       & ((hls_image[:, :, 2] >= 115) & (hls_image[:, :, 2] <= 255))
                       ] = 1

    img_hls_white_bin = np.zeros_like(hls_image[:, :, 0])
    img_hls_white_bin[((hls_image[:, :, 0] >= 0) & (hls_image[:, :, 0] <= 255))
                      & ((hls_image[:, :, 1] >= 200) & (hls_image[:, :, 1] <= 255))
                      & ((hls_image[:, :, 2] >= 0) & (hls_image[:, :, 2] <= 255))
                      ] = 1

    img_hls_white_yellow_bin = np.zeros_like(hls_image[:, :, 0])
    img_hls_white_yellow_bin[(img_hls_yellow_bin == 1) | (img_hls_white_bin == 1)] = 1

    return img_hls_white_yellow_bin


def region_of_interest(image, vertices):
    mask = np.zeros_like(image)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, [vertices], ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def perspective_transform(image, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    Mi = cv2.getPerspectiveTransform(dst, src)
    image_size = (image.shape[1], image.shape[0])
    warped = cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR)

    return warped, M, Mi


def image_processing(file_name):
    given_image = cv2.imread(file_name)
    undistorted_image = cv2.undistort(given_image, mtx, dist, None, mtx)
    # undistorted_image = cv2.resize(undistorted_image, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA)
    undistorted_gray_image = cv2.cvtColor(undistorted_image, cv2.COLOR_RGB2GRAY)
    magnitude = magnitude_sobel(undistorted_gray_image, kernel_size=15, threshold=(80, 200))
    x = abs_sobel(undistorted_gray_image, x_dir=True, kernel_size=15, threshold=(20, 120))
    y = abs_sobel(undistorted_gray_image, x_dir=False, kernel_size=15, threshold=(20, 120))
    undistorted_yellow_white_hls_image = compute_hls_white_yellow_binary(undistorted_image)

    imageShow(undistorted_yellow_white_hls_image)

    sobel_combined = combined_sobels(
        x,
        y,
        magnitude,
        undistorted_gray_image,
        kernel_size=15,
        angle_threshold=(np.pi / 4, np.pi / 2)
    )
    color_binary = np.dstack(
        (
            np.zeros_like(sobel_combined),
            sobel_combined,
            undistorted_yellow_white_hls_image
        )
    ) * 255
    color_binary = color_binary.astype(np.uint8)
    imageShow(color_binary)

    combined_binary = np.zeros_like(undistorted_yellow_white_hls_image)
    combined_binary[(sobel_combined == 1) | (undistorted_yellow_white_hls_image == 1)] = 1
    imageShow(combined_binary, is_gray=True)

    height_image, width_image = combined_binary.shape
    poi = np.array([
        [width_image * 0.01, height_image * 0.99],
        [width_image * 0.46, height_image * 0.6],
        [width_image * 0.57, height_image * 0.6],
        [width_image * 0.99, height_image * 0.99]], np.int32)
    masked_image = region_of_interest(combined_binary, poi)
    imageShow(masked_image)

    src = poi.astype(np.float32)
    dst = np.array([[200, height_image], [200, 0], [1000, 0], [1000, height_image]], np.float32)
    transformed_image, M, Mi = perspective_transform(masked_image, src, dst)
    imageShow(transformed_image)

    searching_img = find_LR_lines(transformed_image, left_line, right_line)
    w_comb_result, w_color_result = draw_lane(searching_img, left_line, right_line)
    imageShow(w_comb_result)

    # Draw lines
    color_result = cv2.warpPerspective(w_color_result, Mi, (width_image, height_image))
    combine_result = np.zeros_like(undistorted_image)
    imageShow(color_result)

    combine_result[0:height_image, 0:width_image] = color_result

    # Combine result and given image
    result = cv2.addWeighted(undistorted_image, 1, combine_result, 0.3, 0)
    imageShow(result)
