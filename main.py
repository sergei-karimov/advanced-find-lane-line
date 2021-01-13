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
