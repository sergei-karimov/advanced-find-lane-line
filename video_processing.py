from moviepy.video.io.VideoFileClip import VideoFileClip
from image_processing import ImageProcessing


def video_processing(file_name):
    video_output = "output_videos/" + file_name[:-4] + "_result.mp4"

    image_processing = ImageProcessing.invoke

    clip1 = VideoFileClip(file_name)
    white_clip = clip1.fl_image(image_processing)
    white_clip.write_videofile(video_output, audio=False)