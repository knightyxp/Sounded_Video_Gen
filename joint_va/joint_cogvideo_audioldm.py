import sys 
import os
from tqdm import tqdm
from audioldm.pipelines.pipeline_audioldm import AudioLDMPipeline

import torch
import soundfile as sf
from accelerate.utils import set_seed
from audioldm.models.unet import UNet2DConditionModel
from moviepy.editor import VideoFileClip, AudioFileClip


import argparse
import datetime
from pathlib import Path
from omegaconf import OmegaConf
import diffusers
from diffusers import AutoencoderKL, DDIMScheduler


from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


from diffusers.utils.import_utils import is_xformers_available
from safetensors import safe_open

from diffusers.models import  AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.utils import export_to_video
from cogvideox.pipeline_cogvideox import CogVideoXPipeline

from imagebind_data import load_and_transform_audio_data_from_waveform, load_and_transform_video_data_from_tensor_real
from imagebind.imagebind.models import imagebind_model
from imagebind.imagebind.models.imagebind_model import ModalityType

import torch.nn.functional as F

# import pdb; pdb.set_trace()
bind_device = torch.device("cuda:0")
audio_device = torch.device("cuda:0")
video_device = torch.device("cuda:1")

#  ------ define audioldm model ------
my_model_path = './ckpt/audioldm-m-full'
unet = UNet2DConditionModel.from_pretrained(my_model_path, subfolder='unet').to(audio_device)
audio_pipe = AudioLDMPipeline.from_pretrained(my_model_path, unet=unet)
audio_pipe = audio_pipe.to(audio_device)

print('yes we come to here')

# ------ define animatediff model ------
parser = argparse.ArgumentParser()

parser.add_argument("--L", type=int, default=16 )
parser.add_argument("--W", type=int, default=512)
parser.add_argument("--H", type=int, default=512)

cog_video_args = parser.parse_args()
#=============load video model==============#

cogvideo_path = './ckpt/CogVideoX-2b'
# weight_dtype = torch.float16
# video_transformer = CogVideoXTransformer3DModel.from_pretrained(cogvideo_path, subfolder='transformer').to('cuda:0',dtype=weight_dtype)
# video_vae = AutoencoderKLCogVideoX.from_pretrained(cogvideo_path, subfolder='vae').to('cuda:0',dtype=weight_dtype)
# video_text_tokenizer = T5Tokenizer.from_pretrained(cogvideo_path, subfolder='tokenizer')
# video_text_encoder = T5EncoderModel.from_pretrained(cogvideo_path, subfolder='text_encoder').to('cuda:0',dtype=weight_dtype)
# video_scheduler = CogVideoXDDIMScheduler.from_pretrained(cogvideo_path, subfolder='scheduler')

# video_pipe = CogVideoXPipeline(vae=video_vae,text_encoder=video_text_encoder,transformer=video_transformer,
#                                 tokenizer=video_text_tokenizer,scheduler=video_scheduler).to(video_device)


video_pipe = CogVideoXPipeline.from_pretrained(cogvideo_path, torch_dtype=torch.float).to(video_device)
# video_pipe.scheduler = CogVideoXDDIMScheduler.from_config(video_pipe.scheduler.config, timestep_spacing="trailing")
# video_pipe.enable_model_cpu_offload()
video_pipe.vae.enable_tiling()
print('finish load video model')
#=============load video model==============#


time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
savedir = f"samples/divide_pipe-{time_str}-video"
os.makedirs(savedir)

samples = []

#============ define and load imagebind model =================#
bind_model = imagebind_model.imagebind_huge(pretrained=False)
state_dict = torch.load("imagebind/.checkpoints/imagebind_huge.pth", map_location=bind_device)
bind_model.load_state_dict(state_dict)
bind_model.eval()
bind_model.to(bind_device)
for p in bind_model.parameters():
    p.requires_grad = False
#============ define and load imagebind model =================#

random_seed = 45
torch.manual_seed(random_seed)

# prompt = 'water splashing the rocks in slow motion'
prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."
video_guidance_scale = 6.0

num_inference_steps = 50
learning_rate = 0
clip_duration = 5
clips_per_video = 1

audio_step_args, audio_latents = audio_pipe.bind_prepare(
        prompt = prompt,
        audio_length_in_s = 2.0,
        num_inference_steps = num_inference_steps,
        guidance_scale = 2.5
    )


video_step_args, video_latents = video_pipe.bind_prepare(
        prompt = prompt,
        video_length = cog_video_args.L,
        height = cog_video_args.H,
        width = cog_video_args.W,
        num_inference_steps = num_inference_steps,
        guidance_scale = video_guidance_scale,
        device = video_device,
    )

print('audio_latents: ', audio_latents.device, audio_latents.dtype, 'video_latents: ', video_latents.device,video_latents.dtype) # [1, 8, 50, 16], [1, 4, 16, 64, 64]

# audio_latents.to(torch.float16)
# video_latents.to(torch.float16)

# video_step_args["latents_dtype"] = torch.float16
# audio_step_args["latents_dtype"] = torch.float16
# call denoising step and then optimize step


for cur_step in tqdm(range(num_inference_steps), desc="Inference Steps"):
    with torch.autograd.set_detect_anomaly(True):
        # denoising step
        xt_audio, noise_pred_audio = audio_pipe.bind_step(**audio_step_args, cur_step=cur_step, latents=audio_latents)
        xt_video, noise_pred_video = video_pipe.bind_step(**video_step_args, cur_step=cur_step, latents=video_latents, guidance_scale=video_guidance_scale)
        
        # print('xt_video',xt_video.shape,xt_video.dtype)

        print('xt_video',xt_video.device,xt_video.shape,xt_video.dtype)   


        xt_audio_temp = xt_audio.detach()
        xt_video_temp = xt_video.detach()

        xt_audio_temp.requires_grad = True
        xt_video_temp.requires_grad = True
        with torch.no_grad():
            x0_audio, x0_audio_waveform = audio_pipe.xt2x0(latents_temp=xt_audio_temp, timesteps=audio_step_args["timesteps"], cur_step=cur_step, noise_pred=noise_pred_audio)
            x0_video = video_pipe.xt2x0(latents_temp=xt_video_temp, timesteps=video_step_args["timesteps"], cur_step=cur_step, noise_pred=noise_pred_video)

        # optimize step 
        # define imagebind model 
        # gradient backward 
        # update latents 

        print('x0_audio_waveform shape: ',x0_audio_waveform.dtype, x0_audio_waveform.shape) # [1, 32032]
        print('x0 video shape', x0_video.dtype,x0_video.shape)

        x0_imagebind_audio_input = load_and_transform_audio_data_from_waveform(x0_audio_waveform, org_sample_rate=16000, 
                                                    device=bind_device, target_length=204, clip_duration=clip_duration, clips_per_video=clips_per_video)
                                
        #print('x0_video shape: ', x0_video.shape) # [1, 3, 4, 224, 224]
        x0_imegebind_video_input = load_and_transform_video_data_from_tensor_real(x0_video, bind_device, clip_duration=clip_duration, clips_per_video=clips_per_video, n_samples_per_clip=2)

        print(x0_imegebind_video_input.dtype, x0_imagebind_audio_input.dtype)
        # exit()

        inputs = {
            ModalityType.VISION: x0_imegebind_video_input,
            ModalityType.AUDIO: x0_imagebind_audio_input,
        }

        embeddings = bind_model(inputs)

        bind_loss = 1 - F.cosine_similarity(embeddings[ModalityType.VISION], embeddings[ModalityType.AUDIO])

        print('bind_loss_vision_audio',bind_loss)
        
        optimizer_video = torch.optim.Adam([xt_audio_temp], lr=learning_rate) 
        optimizer_audio = torch.optim.Adam([xt_video_temp], lr=learning_rate) 


        bind_loss.backward() 

        optimizer_video.step()
        optimizer_video.zero_grad()

        optimizer_audio.step()
        optimizer_audio.zero_grad()

        audio_latents = xt_audio_temp.detach()
        video_latents = xt_video_temp.detach()


audio = audio_pipe.bind_finish(original_waveform_length = audio_step_args["original_waveform_length"], 
                                latents=audio_latents).audios[0]
video = video_pipe.bind_finish(latents=video_latents).videos

print('audio.shape', audio.shape) # (32000,)
print('video.shape', video.shape) # (1, 3, 4, 224, 224)

# save audio and video
# out_path = 'temp'


sf.write(f"{savedir}/ours_{prompt}_{sample_idx:02d}.wav", audio, samplerate=16000)
save_videos_grid(video, f"{savedir}/ours_{prompt}_{sample_idx:02d}.gif")

audio = AudioFileClip(f"{savedir}/ours_{prompt}_{sample_idx:02d}.wav")
video = VideoFileClip(f"{savedir}/ours_{prompt}_{sample_idx:02d}.mp4")
video = video.set_audio(audio)
video.write_videofile(
    f"{savedir}/ours_{prompt}_{sample_idx:02d}-combine.mp4",
    codec="libx264",
    audio_codec="aac"
)


# sf.write(rf"{savedir}/sample/{sample_idx}-{prompt}.wav", audio, samplerate=16000)


# save_videos_grid(video, f"{savedir}/{sample_idx}-{prompt}.gif")


