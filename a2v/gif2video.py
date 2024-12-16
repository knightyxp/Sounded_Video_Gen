import imageio
import os
from glob import glob

root_folder = '/home/yazhou/disk1/projects/edit/AnimateDiff/samples/1-ToonYou-noLora-2023-08-03T20-30-50/sample'

gif_paths = sorted(glob(os.path.join(root_folder, '*.gif')))
for gif_path in gif_paths:
    gif_name = os.path.basename(gif_path).split('.')[0]
    out_path = os.path.join(root_folder, f'{gif_name}.mp4')
    # read gif, convert to mp4
    # def gif2video(path, out_path):
    with imageio.get_reader(gif_path) as reader:
        frames = [frame for frame in reader]
        print(len(frames))
        # imageio.mimsave(out_path, frames, fps=8)

    video_writer = imageio.get_writer(out_path, fps=8)
    for frame in frames:
        video_writer.append_data(frame)
    video_writer.close()


