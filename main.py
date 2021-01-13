import argparse

from image_processing import ImageProcessing
from video_processing import video_processing


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s ARGUMENTS FILE_NAME",
        description="Processing image or video file to find lane lines."
    )

    parser.add_argument("-i", "--image", type=str)
    parser.add_argument("-v", "--video", type=str)

    return parser


def main(file_name, is_image):
    if is_image:
        processing = ImageProcessing()
        processing.invoke(file_name, True)
    else:
        video_processing(file_name)


if __name__ == '__main__':
    parser = init_argparse()
    args = parser.parse_args()
    file_name = ""
    is_image = False

    if args.image:
        file_name = args.image
        is_image = True
    elif args.video:
        file_name = args.video
        is_image = False
        print(f"video name: {args.video}")
    else:
        print(parser.usage)

    try:
        main(file_name=file_name, is_image=is_image)
    except Exception as ex:
        print(f"File {file_name} not found. ({ex})")

    # file_name = 'result_images/frame_n_557.jpg'
    # main(image_name=file_name, is_image=True)
    # file_name = 'result_images/frame_n_785.jpg'
    # main(image_name=file_name, is_image=True)
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
    # file_name = 'harder_challenge_video.mp4'
    # main(file_name, is_image=False)
