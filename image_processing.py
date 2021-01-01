import cv2
import numpy as np
import matplotlib.pyplot as plt


class ImageProcessing(object):
    @classmethod
    def invoke(cls, file_name):
        # load image from disk
        image = cls.from_disk(file_name)
        cls.show(image, "Given image")

        # convert given image to HSL image
        hls_image = cls.to_hls(image)
        cls.show(hls_image, "HLS image")

        # isolate yellow and white color from HSL image
        white_mask = cls.get_isolated_white_mask(hls_image)
        yellow_mask = cls.get_isolated_yellow_mask(hls_image)
        isolated_image = cls.isolate_white_and_yellow(image, white_mask, yellow_mask)
        cls.show(isolated_image, "Isolated image")

        # convert image to grayscale
        gray_image = cls.to_gray_scale(isolated_image)
        cls.show(gray_image, "GrayScale image", True)

        # apply Gaussian Blur to smoothen edges
        blurred_image = cls.apply_gaussian_blur(gray_image)
        cls.show(blurred_image, "Blurred image", True)

        # apply Canny Edge Detection
        edged_image = cls.apply_canny_edge(blurred_image)
        cls.show(edged_image, "Edged image", True)

        # get image size
        height_image, width_image, _ = image.shape

        # Region of interest
        roi = np.array(
            [
                [width_image * 0.01, height_image],
                [width_image * 0.43, height_image * 0.66],
                [width_image * 0.61, height_image * 0.66],
                [width_image * 0.99, height_image]
            ],
            np.int32
        )

        copy = image.copy()
        tmp = cls.draw_roi(copy, roi)
        cls.show(tmp, "TMP")

        # Trace Region of Interest and discard all other lines identified by our previous step
        selected_image = cls.get_selected_image(edged_image, roi)
        cls.show(selected_image, "ROI image", True)

        # Apply a perspective transform to rectify binary image
        warped_image = cls.to_warped_image(selected_image)
        cls.show(warped_image, "Warped image", True)

    @classmethod
    def show(cls, image, title, is_gray=False):
        if is_gray:
            plt.imshow(image, cmap="Greys_r")
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
        # isolate white color
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
        # isolate yellow color
        s_channel = image[:, :, 2]
        # Sobel x
        abs_sobelx = np.absolute(cv2.Sobel(s_channel, cv2.CV_64F, 1, 0))
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Threshold color channel
        mask = np.zeros_like(s_channel)
        mask[(s_channel >= 170) & (s_channel <= 255)] = 1

        return mask

    @classmethod
    def isolate_white_and_yellow(cls, image, white_mask, yellow_mask):
        mask = white_mask | yellow_mask
        isolated_image = cv2.bitwise_and(image, image, mask=mask)
        return isolated_image

    @classmethod
    def apply_gaussian_blur(cls, image):
        kernel_size = 15
        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return blurred_image

    @classmethod
    def apply_canny_edge(cls, image):
        low_threshold = 50
        high_threshold = 150
        edged_image = cv2.Canny(image, low_threshold, high_threshold)
        return edged_image

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
                    [width_image * 0.01, height_image],
                    [width_image * 0.43, height_image * 0.66],
                    [width_image * 0.61, height_image * 0.66],
                    [width_image * 0.99, height_image]
                ]
            )
            offset = width_image * 0.01
            dst = np.float32(
                [
                    [offset, image_size[1]],                    # xl:yb
                    [offset, 0],                                # xl:yt
                    [image_size[0] - offset, 0],                # xr:yt
                    [image_size[0] - offset, image_size[1]],    # xr:yb
                ]
            )

            M = cv2.getPerspectiveTransform(src, dst)
            Minv = cv2.getPerspectiveTransform(dst, src)

            return M, Minv

        img_size = (image.shape[1], image.shape[0])
        M, Minv = get_transform_matrix(image)
        warped_image = cv2.warpPerspective(image, M, img_size)

        return warped_image

    @classmethod
    def draw_roi(cls, image, vertices):
        color = [255, 0, 0]
        w = 2
        cv2.line(image, (vertices[0][0], vertices[0][1]), (vertices[1][0], vertices[1][1]), color, w)
        cv2.line(image, (vertices[1][0], vertices[1][1]), (vertices[2][0], vertices[2][1]), color, w)
        cv2.line(image, (vertices[2][0], vertices[2][1]), (vertices[3][0], vertices[3][1]), color, w)
        cv2.line(image, (vertices[3][0], vertices[3][1]), (vertices[0][0], vertices[0][1]), color, w)

        return image
