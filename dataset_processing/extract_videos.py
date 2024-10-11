from pathlib import Path

from moviepy.editor import VideoFileClip
from PIL import Image

input_data_dir = Path("data/generative_rendering_results")
output_data_dir = Path("data/generative_rendering_results_processed")


class Cropper:

    def __init__(self):
        self.square_w = 320

    def crop_vid_window(self, vid: VideoFileClip, i):
        y_offset = self.square_w * i
        return vid.crop(
            x1=(self.square_w * i),
            y1=0,
            x2=(self.square_w * (i + 1)),
            y2=0 + self.square_w,
        )


cropper = Cropper()

scenes = ["ball", "fox", "girl", "rumba", "silly_dance"]

for scene in scenes:

    # read video
    input_video_path = input_data_dir / "results" / f"{scene}.mp4"
    video = VideoFileClip(str(input_video_path))

    # make output dir
    scene_vids_dir = output_data_dir / scene
    scene_vids_dir.mkdir(exist_ok=True)

    # save uv video
    uv_cropped = cropper.crop_vid_window(video, 0)
    uv_cropped.write_videofile(str(scene_vids_dir / "uv.mp4"))

    # for each output, save it
    for i in range(1, 5):
        cropped = cropper.crop_vid_window(video, i)
        cropped.write_videofile(str(scene_vids_dir / f"out_{i}.mp4"))
