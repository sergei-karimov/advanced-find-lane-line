from image_processing import image_processing
from video_processing import video_processing


def main(file_name, type):
    if type == 'image':
        image_processing(file_name)
    elif type == 'video':
        video_processing(file_name)
    else:
        print('Type must be image or video')


if __name__ == '__main__':
    file_name = 'test_images/straight_lines1.jpg'
    main(file_name, 'image')
