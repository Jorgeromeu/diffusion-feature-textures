from pathlib import Path

from moviepy.editor import VideoFileClip

input_data_dir = Path("data/generative_rendering_results")
output_data_dir = Path("data/generative_rendering_results_processed")

input_video_path = input_data_dir / "combined_captioned.mp4"


class Cropper:
    def __init__(self):
        self.x_offsets = [0, 410, 772, 1188, 1554]
        self.y_offsets = [92, 502, 916]
        self.square_w = 362

    def crop_vid_window(self, vid: VideoFileClip, x_i, y_i):
        x = self.x_offsets[x_i]
        y = self.y_offsets[y_i]
        return vid.crop(x1=x, y1=y, x2=x + self.square_w, y2=y + self.square_w)


video = VideoFileClip(str(input_video_path))
cropper = Cropper()

scenes = ["sneaker", "house", "bed"]

for s_i, scene in enumerate(scenes):
    # Create a directory for the scene
    scene_vids_dir = output_data_dir / scene
    scene_vids_dir.mkdir(exist_ok=True)

    # get uv video
    uv_cropped = cropper.crop_vid_window(video, 0, s_i)

    # write uv video
    uv_cropped.write_videofile(str(scene_vids_dir / "uv.mp4"))

    for i in range(1, 4):
        cropped = cropper.crop_vid_window(video, i + 1, s_i)
        cropped_path = scene_vids_dir / f"out_{i}.mp4"
        cropped.write_videofile(str(cropped_path))
