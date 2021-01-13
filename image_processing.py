import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import linalg


class ImageProcessing(object):
    acc = 0
    frame = None
    mtx = None
    dist = None

    @classmethod
    def invoke(cls, file_name, debug=False):
        if type(file_name) is str:
            # load image from disk
            cls.frame = cls.from_disk(file_name)
            cls.show(cls.frame, "Given image")
        else:
            cls.acc += 1
            print(f"Frame number: {cls.acc}")
            if cls.acc in [59]:
                cls.save_image(file_name, f"frame_n_{cls.acc}")
            cls.frame = file_name

        # Calibrate
        if cls.mtx is None:
            cls.mtx, cls.dist = cls.camera_calibration('camera_cal')

        # Perspective transformation
        undistorted_image = cls.undistort_image(cls.frame)
        if debug:
            cls.show(undistorted_image, 'Undistorted image')
            cls.save_image(undistorted_image, 'undistorted_image')

        # convert given image to HSL image
        hls_image = cls.to_hls(undistorted_image)
        if debug:
            cls.show(hls_image, "HLS image")
            cls.save_image(hls_image, "hls_image")

        # isolate yellow and white color from HSL image
        white_mask = cls.get_isolated_white_mask(hls_image)
        yellow_mask = cls.get_isolated_yellow_mask(hls_image)
        isolated_image = cls.isolate_white_and_yellow(cls.frame, white_mask, yellow_mask)

        if debug:
            cls.show(isolated_image, "Isolated image")
            cls.save_image(isolated_image, "isolated_image")

        # convert image to grayscale
        gray_image = cls.to_gray_scale(isolated_image)
        if debug:
            cls.show(gray_image, "GrayScale image", True)
            cls.save_image(gray_image, "grayScale_image", True)

        # get image size
        height_image, width_image, _ = cls.frame.shape

        # Region of interest
        roi = np.array(
            [
                [width_image * 0.01, height_image],
                [width_image * 0.465, height_image * 0.608],
                [width_image * 0.55, height_image * 0.608],
                [width_image * 0.99, height_image]
            ],
            np.int32
        )

        copy = cls.frame.copy()
        tmp = cls.draw_roi(copy, roi)
        if debug:
            cls.show(tmp, "TMP")
            cls.save_image(tmp, "roi_image")

        # Trace Region of Interest and discard all other lines identified by our previous step
        selected_image = cls.get_selected_image(gray_image, roi)
        if debug:
            cls.show(selected_image, "Selected image", True)
            cls.save_image(selected_image, 'selected_image', True)

        # Apply a perspective transform to rectify binary image
        warped_image, M, Minv = cls.to_warped_image(selected_image)
        if debug:
            cls.show(warped_image, "Warped image", True)
            cls.save_image(warped_image, "warped_image", True)

        # Find lanes
        left_lane, right_lane, center, left_curve_rad, right_curve_rad, out_image = cls.find_lanes(warped_image)
        if debug:
            cls.show(out_image, 'Out image', True)
            cls.save_image(out_image, 'out_image', True)

        if left_lane is None or right_lane is None:
            return cls.frame

        # Draw segments of roads
        result = cls.draw_segment_of_road(cls.frame, Minv, left_lane, right_lane, center, left_curve_rad, right_curve_rad)
        if debug:
            cls.show(result, "Result image")
            cls.save_image(result, "result_image")

        return result

    @classmethod
    def save_image(cls, result, file_name, is_gray=False):
        if not is_gray:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'result_images/{file_name}.jpg', result)
        return result

    @classmethod
    def show(cls, image, title, is_gray=False):
        if is_gray:
            plt.imshow(image, cmap="gray")
        else:
            plt.imshow(image)
        plt.title(title)
        plt.show()

    @classmethod
    def to_hls(cls, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        return image

    @classmethod
    def get_isolated_white_mask(cls, image):
        # # isolate white color
        l_channel = image[:, :, 1]

        # Sobel x
        abs_sobelx = np.absolute(cv2.Sobel(l_channel, cv2.CV_64F, 1, 0))
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        mask = np.zeros_like(scaled_sobel)
        mask[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1

        return mask

    @classmethod
    def to_gray_scale(cls, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return gray_image

    @classmethod
    def from_disk(cls, file_name, to_rgb=True):
        if to_rgb:
            image = cv2.imread(file_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.imread(file_name)
        return image

    @classmethod
    def get_isolated_yellow_mask(cls, image):
        s_channel = image[:, :, 2]
        mask = np.zeros_like(s_channel)
        mask[(s_channel >= 170) & (s_channel <= 255)] = 1

        return mask

    @classmethod
    def isolate_white_and_yellow(cls, image, white_mask, yellow_mask):
        mask = white_mask | yellow_mask
        isolated_image = cv2.bitwise_and(image, image, mask=mask)
        return isolated_image

    @classmethod
    def get_selected_image(cls, image, vertices):
        mask = np.zeros_like(image)
        if len(image.shape) > 2:
            channel_count = image.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        cv2.fillPoly(mask, [vertices], ignore_mask_color)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

    @classmethod
    def to_binary_image(cls, image):
        def thresh_it(threshed_image):
            low = 10
            high = 160
            binary = np.zeros_like(threshed_image)
            binary[(threshed_image >= low) & (threshed_image <= high)] = 1
            return binary

        def get_magnitude(to_magnitude_image):
            x = 1
            y = 0

            sobel = cv2.Sobel(to_magnitude_image, cv2.CV_64F, x, y, ksize=3)
            abs_sobel = np.absolute(sobel)
            scaled = np.uint8(255.0 * abs_sobel / np.max(abs_sobel))
            return thresh_it(scaled)

        def get_direction(to_direction_image):
            x = 0
            y = 1

            sobel = cv2.Sobel(to_direction_image, cv2.CV_64F, x, y, ksize=3)
            abs_sobel = np.absolute(sobel)
            scaled = np.uint8(255.0 * abs_sobel / np.max(abs_sobel))
            return thresh_it(scaled)

        magnitude = get_magnitude(image)
        direction = get_direction(image)
        binary_image = np.zeros_like(magnitude)
        binary_image[((magnitude == 1) & (direction == 1))] = 1
        return binary_image

    @classmethod
    def to_warped_image(cls, image):
        def get_transform_matrix(transformed_image):
            image_size = (transformed_image.shape[1], transformed_image.shape[0])
            width_image = image_size[0]
            height_image = image_size[1]
            src = np.float32(
                [
                    [301, 720],
                    [616, 454],
                    [735, 454],
                    [1244, 720]
                ]
            )
            offset = 301
            dst = np.float32(
                [
                    [offset, image_size[1]],  # xl:yb
                    [offset, 0],  # xl:yt
                    [image_size[0] - offset, 0],  # xr:yt
                    [image_size[0] - offset, image_size[1]],  # xr:yb
                ]
            )

            M = cv2.getPerspectiveTransform(src, dst)
            Minv = cv2.getPerspectiveTransform(dst, src)

            return M, Minv

        img_size = (image.shape[1], image.shape[0])
        M, Minv = get_transform_matrix(image)
        warped_image = cv2.warpPerspective(image, M, img_size)

        return warped_image, M, Minv

    @classmethod
    def draw_roi(cls, image, vertices):
        color = [255, 0, 0]
        w = 2
        cv2.line(image, (vertices[0][0], vertices[0][1]), (vertices[1][0], vertices[1][1]), color, w)
        cv2.line(image, (vertices[1][0], vertices[1][1]), (vertices[2][0], vertices[2][1]), color, w)
        cv2.line(image, (vertices[2][0], vertices[2][1]), (vertices[3][0], vertices[3][1]), color, w)
        cv2.line(image, (vertices[3][0], vertices[3][1]), (vertices[0][0], vertices[0][1]), color, w)

        return image

    @classmethod
    def find_lanes(cls, image):
        count_of_windows = 9
        margin = 110
        minimum_founded_pixels = 50

        # Define conversions in x and y from pixels space to meters
        y_px = 30 / 720  # meters per pixel in y dimension
        x_px = 3.7 / 700  # meters per pixel in x dimension

        histogram = np.sum(image[image.shape[0] // 2:, :], axis=0)
        out_image = np.dstack((image, image, image)) * 255
        midpoint = np.int(histogram.shape[0] / 2)
        left_base = np.argmax(histogram[:midpoint - 150])
        right_base = np.argmax(histogram[midpoint + 150:]) + midpoint + 150
        window_height = np.int(image.shape[0] / count_of_windows)
        non_zero = image.nonzero()
        non_zero_y = np.array(non_zero[0])
        non_zero_x = np.array(non_zero[1])
        current_left = left_base
        current_right = right_base

        left_lane_idx = []
        right_lane_idx = []

        for idx in range(count_of_windows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = image.shape[0] - (idx + 1) * window_height
            win_y_high = image.shape[0] - idx * window_height
            win_xleft_low = current_left - margin
            win_xleft_high = current_left + margin
            win_xright_low = current_right - margin
            win_xright_high = current_right + margin

            # Identify the nonzero pixels in x and y within the window
            good_left_idx = ((non_zero_y >= win_y_low) & (non_zero_y < win_y_high) & (non_zero_x >= win_xleft_low) & (
                    non_zero_x < win_xleft_high)).nonzero()[0]
            good_right_idx = ((non_zero_y >= win_y_low) & (non_zero_y < win_y_high) & (non_zero_x >= win_xright_low) & (
                    non_zero_x < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_idx.append(good_left_idx)
            right_lane_idx.append(good_right_idx)

            # If you found > minimum_founded_pixels pixels, recenter next window on their mean position
            if len(good_left_idx) > minimum_founded_pixels:
                current_left = np.int(np.mean(non_zero_x[good_left_idx]))
            if len(good_right_idx) > minimum_founded_pixels:
                current_right = np.int(np.mean(non_zero_x[good_right_idx]))

        # Concatenate the arrays of indices
        left_lane_idx = np.concatenate(left_lane_idx)
        right_lane_idx = np.concatenate(right_lane_idx)

        # Extract left and right line pixel positions
        left_x = non_zero_x[left_lane_idx]
        left_y = non_zero_y[left_lane_idx]
        right_x = non_zero_x[right_lane_idx]
        right_y = non_zero_y[right_lane_idx]

        # Fit a second order polynomial to each
        try:
            left_fit = np.polyfit(left_y, left_x, 2)
            right_fit = np.polyfit(right_y, right_x, 2)
            center = abs(640 - ((left_base + right_base) / 2))
            left_curve_rad = ((1 + (2 * left_fit[0] * np.max(left_y) + left_fit[1]) ** 2) ** 1.5) \
                             / np.absolute(2 * left_fit[0])
            right_curve_rad = ((1 + (2 * right_fit[0] * np.max(left_y) + right_fit[1]) ** 2) ** 1.5) \
                              / np.absolute(2 * right_fit[0])

            return left_fit, right_fit, center, left_curve_rad, right_curve_rad, out_image
        except:
            img = cv2.cvtColor(cls.frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f'frame_n_{cls.acc}.jpg', img)

        return None, None, None, None, None, None

    @classmethod
    def draw_segment_of_road(cls, image, Minv, left_lane, right_lane, center, left_curve_rad, right_curve_rad):
        y = image.shape[0]
        plot_y = np.linspace(0, y - 1, y)
        color_warp = np.zeros_like(image).astype(np.uint8)

        # Calculate points.
        left_x = left_lane[0] * plot_y ** 2 + left_lane[1] * plot_y + left_lane[2]
        right_x = right_lane[0] * plot_y ** 2 + right_lane[1] * plot_y + right_lane[2]

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_x, plot_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_x, plot_y])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        warped_image = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
        combined_image = cv2.addWeighted(image, 1, warped_image, 0.3, 0)

        # cv2.putText(combined_image, f'frame: {cls.acc}', (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)
        if center < 640:
            cv2.putText(
                combined_image,
                f'Vehicle is {center * 3.7 / 700:.2f} m left of center',
                (200, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (209, 80, 0, 255),
                3
            )
        else:
            cv2.putText(
                combined_image,
                f'Vehicle is {center * 3.7 / 700:.2f} m right of center',
                (200, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (209, 80, 0, 255),
            )
        cv2.putText(
            combined_image,
            f'Radius of curvature is {int((left_curve_rad + right_curve_rad) / 2)}m right of center',
            (200, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (209, 80, 0, 255),
            3
        )

        return combined_image

    @classmethod
    def hough_lines(cls, image):
        lines = cv2.HoughLinesP(image, 1, np.pi / 180, 20, lines=np.array([]), minLineLength=10, maxLineGap=50)
        return lines

    @classmethod
    def draw_lane_lines(cls, image, lines):

        def extrapolate_lines(height, width, left_k, left_b, right_k, right_b):

            y1 = int(height * 0.99)
            y2 = int(height * 0.6)

            try:
                left_x1 = int((y1 - left_b) / left_k)
                left_x2 = int((y2 - left_b) / left_k)

                right_x1 = int((y1 - right_b) / right_k)
                right_x2 = int((y2 - right_b) / right_k)
            except:
                print(
                    f'EXCEPTION in frame number {cls.acc}: {height}, {width}, {left_k}, {left_b}, {right_k}, {right_b}'
                )
                img = cv2.cvtColor(cls.frame, cv2.COLOR_BGR2RGB)
                cv2.imwrite(f'frame_n_{cls.acc}.jpg', img)

                y1 = 0
                y2 = 0
                left_x1 = int(0)
                left_x2 = int(0)

                right_x1 = int(0)
                right_x2 = int(0)

            left_line = [(left_x1, y1), (left_x2, y2)]
            right_line = [(right_x1, y1), (right_x2, y2)]

            return left_line, right_line

        color = [255, 0, 0]
        thickness = 13

        param1 = []
        param = []
        param3 = []
        param2 = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 == x1:
                    continue  # ignore a vertical line
                if y2 == y1:
                    continue  # ignore a horizontal line

                A = np.array([[x1, 1], [x2, 1]])
                c = np.array([y1, y2])

                # to find 'K' and 'b'
                k, b = linalg.solve(A, c)
                if np.abs(k) < 0.5:  # sets a some threshold
                    continue

                # length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
                if k < 0:
                    param1.append(b)
                    param.append(k)
                elif k > 0:
                    param3.append(b)
                    param2.append(k)
        height, width = image.shape
        left_line, right_line = extrapolate_lines(
            height,
            width,
            np.median(param),
            np.median(param1),
            np.median(param2),
            np.median(param3)
        )

        cv2.line(image, left_line[0], left_line[1], color, thickness)
        cv2.line(image, right_line[0], right_line[1], color, thickness)

        return image

    @classmethod
    def undistort_image(cls, image):
        return cv2.undistort(image, cls.mtx, cls.dist, None, cls.mtx)

    @classmethod
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
