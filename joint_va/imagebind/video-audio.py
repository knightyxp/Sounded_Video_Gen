from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# text_list=["A dog.", "A car", "A bird"]
# image_paths=[".assets/dog_image.jpg", ".assets/car_image.jpg", ".assets/bird_image.jpg"]
# audio_paths=[".assets/dog_audio.wav", ".assets/car_audio.wav", ".assets/bird_audio.wav"]

# audio_paths = ['/home/yazhou/disk1/projects/edit/dataset/landscape/train/splashing_water/JmhWMRN_DT0#3408#3418.wav', 
#                 '/home/yazhou/disk1/projects/edit/dataset/landscape/train/splashing_water/bUntI90An-U#2540#2550.wav',
#                 '/home/yazhou/disk1/projects/edit/dataset/landscape/train/splashing_water/U22mj35rYZM#2997#3007.wav']

# video_paths = ['/home/yazhou/disk1/projects/edit/dataset/landscape/train/splashing_water/JmhWMRN_DT0#3408#3418.mp4', 
#                 '/home/yazhou/disk1/projects/edit/dataset/landscape/train/splashing_water/bUntI90An-U#2540#2550.mp4',
#                 '/home/yazhou/disk1/projects/edit/dataset/landscape/train/splashing_water/U22mj35rYZM#2997#3007.mp4']

# audio_paths = ['/home/yazhou/disk1/projects/edit/dataset/landscape/train/splashing_water_processed/JmhWMRN_DT0#3408#3418_clip.wav', 
#                 '/home/yazhou/disk1/projects/edit/dataset/landscape/train/splashing_water_processed/bUntI90An-U#2540#2550_clip.wav',
#                 '/home/yazhou/disk1/projects/edit/dataset/landscape/train/fire_crackling_processed/_deFeO_gnwc#178#188_clip.wav']

# video_paths = ['/home/yazhou/disk1/projects/edit/dataset/landscape/train/splashing_water_processed/JmhWMRN_DT0#3408#3418_clip.mp4', 
#                 '/home/yazhou/disk1/projects/edit/dataset/landscape/train/splashing_water_processed/bUntI90An-U#2540#2550_clip.mp4',
#                 '/home/yazhou/disk1/projects/edit/dataset/landscape/train/fire_crackling_processed/_deFeO_gnwc#178#188_clip.mp4']


audio_paths = ['/home/yazhou/disk1/projects/edit/dataset/landscape/train/own/qiaogu.wav', 
                '/home/yazhou/disk1/projects/edit/dataset/landscape/train/own/jiazigu.wav']

video_paths = ['/home/yazhou/disk1/projects/edit/dataset/landscape/train/own/qiaogu.mp4', 
                '/home/yazhou/disk1/projects/edit/dataset/landscape/train/own/jiazigu.mp4']


device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

# Load data
inputs = {
    # ModalityType.TEXT: data.load_and_transform_text(text_list, device),
    ModalityType.VISION: data.load_and_transform_video_data(video_paths, device),
    ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
}

with torch.no_grad():
    embeddings = model(inputs)

print(type(embeddings))
print(embeddings.keys())
print(embeddings[ModalityType.VISION].shape)
print(embeddings[ModalityType.AUDIO].shape)

# print(
#     "Vision x Text: ",
#     torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1),
# )
# print(
#     "Audio x Text: ",
#     torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1),
# )
print(
    "Vision x Audio: ",
    torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=-1),
)

# Vision x Audio:  tensor([[0.0766, 0.2165, 0.7069],
        # [0.0264, 0.2742, 0.6994],
        # [0.0457, 0.3233, 0.6309]], device='cuda:0')
