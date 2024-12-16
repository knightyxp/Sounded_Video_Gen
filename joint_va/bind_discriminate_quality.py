from imagebind_data import load_and_transform_audio_data_from_waveform, load_and_transform_video_data_from_tensor_real, load_and_transform_video_data, load_and_transform_audio_data, load_and_transform_text
from imagebind.imagebind.models import imagebind_model
from imagebind.imagebind.models.imagebind_model import ModalityType

import torch.nn.functional as F
import os 

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# define bind model 
device = "cuda"
bind_model = imagebind_model.imagebind_huge(pretrained=True)

bind_model.eval()
bind_model.to(device)
# bind_model.to(torch.float16)


for p in bind_model.parameters():
    p.requires_grad = False


root_path = '/home/yazhou/disk1/projects/edit/joint-generation/samples/5-RealisticVision-abspath-2023-10-26T22-32-13'
# compute the score of the video and audio 
# video_before_path = os.path.join(root_path, '0_step25_org_a boring bear is standing behind a bar table. High Definition._25_org.mp4') 
# audio_before_path = os.path.join(root_path, '0_step25_org_a boring bear is standing behind a bar table. High Definition..wav') 
# video_after_path = os.path.join(root_path, '0_step25_lr0.1_a boring bear is standing behind a bar table. High Definition..mp4') 
# audio_after_path = os.path.join(root_path, '0_step25_lr0.1_a boring bear is standing behind a bar table. High Definition..wav') 

# video_before_path = os.path.join(root_path, '0_step25_org_A car made of sushi._25_org.mp4') 
# audio_before_path = os.path.join(root_path, '0_step25_org_A car made of sushi..wav') 
# video_after_path = os.path.join(root_path, '0_step25_lr0.1_A car made of sushi..mp4') 
# audio_after_path = os.path.join(root_path, '0_step25_lr0.1_A car made of sushi..wav') 

# video_before_path = os.path.join(root_path, '0_step25_org_A yellow tiger with lightning around it_25_org.mp4') 
# audio_before_path = os.path.join(root_path, '0_step25_org_A yellow tiger with lightning around it.wav') 
# video_after_path = os.path.join(root_path, '0_step25_lr0.1_A yellow tiger with lightning around it.mp4') 
# audio_after_path = os.path.join(root_path, '0_step25_lr0.1_A yellow tiger with lightning around it.wav') 

# video_before_path = os.path.join(root_path, '0_step25_org_A giant spaceship is landing on mars in the sunset. High Definition._25_org_combine.mp4') 
# audio_before_path = os.path.join(root_path, '0_step25_org_A giant spaceship is landing on mars in the sunset. High Definition..wav') 
# video_after_path = os.path.join(root_path, '0_step25_lr0.1_A giant spaceship is landing on mars in the sunset. High Definition.-combine.mp4') 
# audio_after_path = os.path.join(root_path, '0_step25_lr0.1_A giant spaceship is landing on mars in the sunset. High Definition..wav') 

# prompt = 'A giant spaceship is landing on mars in the sunset. High Definition.'


# video_before_path = os.path.join(root_path, '0_step25_org_Drone flythrough of a tropical jungle covered in snow. High Definition._25_org_combine.mp4') 
# audio_before_path = os.path.join(root_path, '0_step25_org_Drone flythrough of a tropical jungle covered in snow. High Definition..wav') 
# video_after_path = os.path.join(root_path, '0_step25_lr0.1_Drone flythrough of a tropical jungle covered in snow. High Definition.-combine.mp4') 
# audio_after_path = os.path.join(root_path, '0_step25_lr0.1_Drone flythrough of a tropical jungle covered in snow. High Definition..wav') 

# prompt = 'Drone flythrough of a tropical jungle covered in snow. High Definition.'


# video_before_path = os.path.join(root_path, '0_step25_org_Face of happy mature woman smiling._25_org_combine.mp4') 
# audio_before_path = os.path.join(root_path, '0_step25_org_Face of happy mature woman smiling..wav') 
# video_after_path = os.path.join(root_path, '0_step25_lr0.1_Face of happy mature woman smiling.-combine.mp4') 
# audio_after_path = os.path.join(root_path, '0_step25_lr0.1_Face of happy mature woman smiling..wav') 

# prompt = 'Face of happy mature woman smiling.'


video_before_path = os.path.join(root_path, '0_step25_org_combine_A bear with sunglasses making smoothies in a kitchen.mp4') 
audio_before_path = os.path.join(root_path, '0_step25_org_A bear with sunglasses making smoothies in a kitchen.wav') 
video_after_path = os.path.join(root_path, '0_step25_lr0.1_A bear with sunglasses making smoothies in a kitchen-combine.mp4') 
audio_after_path = os.path.join(root_path, '0_step25_lr0.1_A bear with sunglasses making smoothies in a kitchen.wav') 

prompt = 'A bear with sunglasses making smoothies in a kitchen'


# load vidoe and audio 
clip_duration = 1
clips_per_video = 1

video_before = load_and_transform_video_data([video_before_path], device=device, clip_duration=clip_duration, clips_per_video=clips_per_video, n_samples_per_clip=2) 
video_after = load_and_transform_video_data([video_after_path],device=device, clip_duration=clip_duration, clips_per_video=clips_per_video, n_samples_per_clip=2)

audio_before = load_and_transform_audio_data([audio_before_path],device=device, target_length=204, clip_duration=clip_duration, clips_per_video=clips_per_video)
audio_after = load_and_transform_audio_data([audio_after_path],device=device, target_length=204, clip_duration=clip_duration, clips_per_video=clips_per_video)


inputs_before = {
    ModalityType.TEXT: load_and_transform_text([prompt], device),
    ModalityType.VISION: video_before,
    ModalityType.AUDIO: audio_before,
}

embeddings_before = bind_model(inputs_before)

similarity_before = F.cosine_similarity(embeddings_before[ModalityType.VISION], embeddings_before[ModalityType.AUDIO])

text_audio_similarity_before = F.cosine_similarity(embeddings_before[ModalityType.TEXT], embeddings_before[ModalityType.AUDIO])
text_vision_similarity_before = F.cosine_similarity(embeddings_before[ModalityType.TEXT], embeddings_before[ModalityType.VISION])

inputs_after = {
    ModalityType.TEXT: load_and_transform_text([prompt], device),
    ModalityType.VISION: video_after,
    ModalityType.AUDIO: audio_after,
}


embeddings_after = bind_model(inputs_after)
print(embeddings_after[ModalityType.VISION].shape, embeddings_after[ModalityType.AUDIO].shape, embeddings_after[ModalityType.TEXT].shape)

similarity_after = F.cosine_similarity(embeddings_after[ModalityType.VISION], embeddings_after[ModalityType.AUDIO])
text_audio_similarity_after = F.cosine_similarity(embeddings_after[ModalityType.TEXT], embeddings_after[ModalityType.AUDIO])
text_vision_similarity_after = F.cosine_similarity(embeddings_after[ModalityType.TEXT], embeddings_after[ModalityType.VISION])



print("vision audio before: ", similarity_before)
print("text vision before: ", text_vision_similarity_before)
print("text audio before: ", text_audio_similarity_before)
print("vision audio after: ", similarity_after)
print("text vision after: ", text_vision_similarity_after)
print("text audio after: ", text_audio_similarity_after)


# 