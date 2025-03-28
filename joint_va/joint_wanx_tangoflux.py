import os
from pipeline.pipeline_wanx_tangoflux import Audio_Video_LDMPipeline
import torch
import soundfile as sf
from accelerate.utils import set_seed
from TangoFlux.tangoflux.model import TangoFlux
import json
from moviepy.editor import VideoFileClip, AudioFileClip
from glob import glob
import argparse
import math
import random
from utils.utils import instantiate_from_config
from omegaconf import OmegaConf
from collections import OrderedDict
from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video,load_image
from safetensors.torch import load_file
from diffusers import AutoencoderOobleck, AutoencoderKLWan
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
import torchaudio

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=500)
parser.add_argument("--init_latents", action='store_true', default=False)
parser.add_argument("--seed", type=int, default=30) 

args = parser.parse_args()

weight_dtype = torch.float32

########load audio model #############

audio_vae = AutoencoderOobleck()

tango_flux_path = 'ckpt/TangoFlux'

vae_weights = load_file("{}/vae.safetensors".format(tango_flux_path))
audio_vae.load_state_dict(vae_weights)
audio_vae = audio_vae.to('cuda:0', dtype=weight_dtype)

tangoflux_model_config = 'tangoflux_config.yaml'
tangoflux_weights = load_file("{}/tangoflux.safetensors".format(tango_flux_path))
with open("{}/config.json".format(tango_flux_path), "r") as f:
    audio_config = json.load(f)

TangoFlux = TangoFlux(audio_config)
TangoFlux.load_state_dict(tangoflux_weights , strict=False)


## split Tango Flux components
audio_transformer = TangoFlux.transformer.to('cuda:0', dtype=weight_dtype)
audio_text_encoder =  TangoFlux.text_encoder.to('cuda:0', dtype=weight_dtype)
audio_tokenizer =  TangoFlux.tokenizer
audio_scheduler =  TangoFlux.noise_scheduler
audio_fc = TangoFlux.fc.to('cuda:0', dtype=weight_dtype)
audio_duration_emebdder = TangoFlux.duration_emebdder.to('cuda:0', dtype=weight_dtype)

## split Tango Flux components

########load audio model #############



##########load video model#############
wanx_path = './ckpt/Wan2.1-T2V-1.3B-Diffusers'
video_vae = AutoencoderKLWan.from_pretrained(wanx_path, subfolder="vae", torch_dtype=torch.float32)
print('finish load video model')

##########load video model#############

pipe = Audio_Video_LDMPipeline.from_pretrained(wanx_path, vae=video_vae,
        audio_vae=audio_vae, audio_text_encoder=audio_text_encoder, 
        audio_tokenizer = audio_tokenizer, audio_transformer=audio_transformer,
        audio_scheduler = audio_scheduler, audio_fc = audio_fc, audio_duration_emebdder = audio_duration_emebdder, torch_dtype=torch.bfloat16,)
#flow_shift = 5.0  # 5.0 for 720P, 3.0 for 480P
#pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)


# pipe = pipe.to("cuda")

inf_steps = 50
lr = 0
num_optimization_steps = 1
audio_length = 8
clip_duration = 2
clips_per_video = 1
#vp['audio_length']
cur_seed = 45
optimization_starting_point = 0

cur_out_dir = f"samples/wanx+tangoflux/t2v_play_violin_long_prompt_optimize_gradient"
os.makedirs(cur_out_dir, exist_ok=True)

set_seed(cur_seed)
generator = torch.Generator(device='cuda')

generator.manual_seed(cur_seed)

#prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
#"A white man shoots a black rifle. He wears a black cap, a green T-shirt, and beige pants."
#"A man is playing the violin"
prompt = "A man is playing the violin, performing a gentle classical tune with clear and melodic notes, steady rhythm, and a harmonious tone"
ngpu=torch.cuda.device_count()
print(f"num_gpus={ngpu}")

# pipe.video_vae.enable_gradient_checkpointing()

# image_dir = './samples/play_violin/00000.jpg'
# image = load_image(image_dir)

if len(prompt) > 100:
    prompt_to_save = prompt[:100]
else:
    prompt_to_save = prompt

video, audio = pipe.bind_forward_triple_loss(prompt=prompt,latents=None, num_inference_steps=inf_steps, audio_duration= audio_length, generator=generator, 
                learning_rate=lr, clip_duration=clip_duration, 
                clips_per_video=clips_per_video, num_optimization_steps= num_optimization_steps, optimization_starting_point = optimization_starting_point,
                return_dict=False)

export_to_video(video,f"{cur_out_dir}/output_video.mp4", fps=10)
torchaudio.save(f"{cur_out_dir}/output_audio.wav", audio, sample_rate=44100)

audio = AudioFileClip(f"{cur_out_dir}/output_audio.wav")
video = VideoFileClip(f"{cur_out_dir}/output_video.mp4")
video = video.set_audio(audio)
video.write_videofile(
    f"{cur_out_dir}/ouput_video_joint.mp4",
    codec="libx264",
)

