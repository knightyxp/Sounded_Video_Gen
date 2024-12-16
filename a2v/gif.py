import imageio
import numpy as np



path = '/home/yazhou/disk1/projects/edit/AnimateDiff/samples/5-RealisticVision-2023-07-25T21-48-52/sample/0-b&w-photo-of-42-y.o-man-in-black-clothes,-bald,.gif'
with imageio.get_reader(path) as reader:
    frames1 = np.array([frame for frame in reader])
    print(frames1.shape)

