import os
from pipeline.pipeline_image2sound_video import Audio_Video_LDMPipeline
import torch
import soundfile as sf
from accelerate.utils import set_seed
from audioldm.models.unet import UNet2DConditionModel
from moviepy.editor import VideoFileClip, AudioFileClip
from glob import glob
import argparse
import math
import random
from utils.utils import instantiate_from_config
from omegaconf import OmegaConf
from collections import OrderedDict
from diffusers.models import  AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.utils import export_to_video,load_image

parser = argparse.ArgumentParser()
parser.add_argument("--optimize_text", action='store_true', default=False)
parser.add_argument("--double_loss", action='store_true', default=False)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=500)
parser.add_argument("--init_latents", action='store_true', default=False)
parser.add_argument("--seed", type=int, default=30) 

args = parser.parse_args()

weight_dtype = torch.float32

# repo_id = "cvssp/audioldm-m-full"
local_model_path = './ckpt/audioldm-m-full'
audio_unet = UNet2DConditionModel.from_pretrained(local_model_path, subfolder='unet').to('cuda:0',dtype=weight_dtype)

##########load video model#############
video_weight_dtype = torch.float16
video_transformer = CogVideoXTransformer3DModel.from_pretrained('./ckpt/CogVideoX-5b-I2V', subfolder='transformer').to('cuda:1',dtype=video_weight_dtype)
video_vae = AutoencoderKLCogVideoX.from_pretrained('./ckpt/CogVideoX-5b-I2V', subfolder='vae').to('cuda:1',dtype=video_weight_dtype)
video_text_tokenizer = T5Tokenizer.from_pretrained('./ckpt/CogVideoX-5b-I2V', subfolder='tokenizer')
video_text_encoder = T5EncoderModel.from_pretrained('./ckpt/CogVideoX-5b-I2V', subfolder='text_encoder').to('cuda:1',dtype=video_weight_dtype)
video_scheduler = CogVideoXDDIMScheduler.from_pretrained('./ckpt/CogVideoX-5b-I2V', subfolder='scheduler')


print('finish load video model')
##########load video model#############

pipe = Audio_Video_LDMPipeline.from_pretrained(local_model_path, 
        video_vae=video_vae, video_text_encoder=video_text_encoder,
        video_text_tokenizer=video_text_tokenizer,transformer=video_transformer,
        video_scheduler=video_scheduler, torch_dtype=torch.float32)
pipe = pipe.to("cuda")

inf_steps = 50
lr = 0
num_optimization_steps = 1
audio_length = 5
clip_duration = 2
clips_per_video = 1
#vp['audio_length']
cur_seed = 45
optimization_starting_point = 0.2

cur_out_dir = f"samples/i2v_play_violin_long_prompt"
os.makedirs(cur_out_dir, exist_ok=True)

set_seed(cur_seed)
generator = torch.Generator(device='cuda')

generator.manual_seed(cur_seed)

# prompt = "A man is playing an accordion"
prompt = "A man is playing the violin, performing a gentle classical tune with clear and melodic notes, steady rhythm, and a harmonious tone"
negative_prompt = 'blur, haze, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation'
ngpu=torch.cuda.device_count()
print(f"num_gpus={ngpu}")

# pipe.video_vae.enable_gradient_checkpointing()

image_dir = './samples/play_violin/00000.jpg'
image = load_image(image_dir)

if len(prompt) > 100:
    prompt_to_save = prompt[:100]
else:
    prompt_to_save = prompt

video, audio = pipe.bind_forward_triple_loss(image=image,prompt=prompt,latents=None, num_inference_steps=inf_steps, audio_length_in_s= audio_length, generator=generator, 
                learning_rate=lr, clip_duration=clip_duration, 
                clips_per_video=clips_per_video, num_optimization_steps= num_optimization_steps, return_dict=False)

export_to_video(video,f"{cur_out_dir}/output_video.mp4", fps=10)
sf.write(f"{cur_out_dir}/output_audio.wav", audio, samplerate=16000)

audio = AudioFileClip(f"{cur_out_dir}/output_audio.wav")
video = VideoFileClip(f"{cur_out_dir}/output_video.mp4")
video = video.set_audio(audio)
video.write_videofile(
    f"{cur_out_dir}/ouput_video_joint.mp4",
    codec="libx264",
)

