from image_processing import ImageProcessing
from video_processing import video_processing


def main(image_name, is_image):
    if is_image:
        processing = ImageProcessing()
        processing.invoke(image_name, True)
    else:
        video_processing(image_name)


if __name__ == '__main__':
    # file_name = 'test_images/straight_lines1.jpg'
    # main(image_name=file_name, is_image=True)
    # file_name = 'test_images/straight_lines2.jpg'
    # main(image_name=file_name, is_image=True)
    # file_name = 'test_images/test1.jpg'
    # main(image_name=file_name, is_image=True)
    # file_name = 'test_images/test2.jpg'
    # main(image_name=file_name, is_image=True)
    # file_name = 'test_images/test3.jpg'
    # main(image_name=file_name, is_image=True)
    # file_name = 'test_images/test4.jpg'
    # main(image_name=file_name, is_image=True)
    # file_name = 'test_images/test5.jpg'
    # main(image_name=file_name, is_image=True)
    # file_name = 'test_images/test6.jpg'
    # main(image_name=file_name, is_image=True)
    # file_name = 'project_video.mp4'
    # main(file_name, is_image=False)
    # file_name = 'challenge_video.mp4'
    # main(file_name, is_image=False)
    file_name = 'harder_challenge_video.mp4'
    main(file_name, is_image=False)
