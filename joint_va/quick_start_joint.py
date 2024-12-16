import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys 
# sys.path.append('/home/yazhou/disk1/projects/edit/AudioLDM')
sys.path.append('/home/xianyang/Data/code/AnimateDiff')

print('0')
# import pdb; pdb.set_trace()
from tqdm import tqdm
from audioldm.pipelines.pipeline_audioldm import AudioLDMPipeline
# from diffusers import AudioLDMPipeline
import torch
import soundfile as sf
from accelerate.utils import set_seed
from audioldm.models.unet import UNet2DConditionModel
from moviepy.editor import VideoFileClip, AudioFileClip

print('1')

import inspect
import argparse
import datetime
from pathlib import Path
from omegaconf import OmegaConf
import diffusers
from diffusers import AutoencoderKL, DDIMScheduler

print('2')

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora
from diffusers.utils.import_utils import is_xformers_available
from safetensors import safe_open

print('3')


from imagebind_data import load_and_transform_audio_data_from_waveform, load_and_transform_video_data_from_tensor_real
from imagebind.imagebind.models import imagebind_model
from imagebind.imagebind.models.imagebind_model import ModalityType

import torch.nn.functional as F

# import pdb; pdb.set_trace()
#  ------ define audioldm model ------
my_model_path = '/home/xianyang/Data/code/Seeing-and-Hearing/v2a/ckpt/audioldm-m-full'
# unet = UNet2DConditionModel.from_pretrained(my_model_path, subfolder='unet', torch_dtype=torch.float16).to('cuda')
# audio_pipe = AudioLDMPipeline.from_pretrained(my_model_path, unet=unet, torch_dtype=torch.float16)
unet = UNet2DConditionModel.from_pretrained(my_model_path, subfolder='unet').to('cuda')
audio_pipe = AudioLDMPipeline.from_pretrained(my_model_path, unet=unet)
audio_pipe = audio_pipe.to("cuda")

print('yes we come to here')

# ------ define animatediff model ------
parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_model_path", type=str, default="models/StableDiffusion/stable-diffusion-v1-5",)
parser.add_argument("--inference_config",      type=str, default="configs/inference/inference.yaml")    
parser.add_argument("--config",                type=str, required=True)

parser.add_argument("--L", type=int, default=16 )
parser.add_argument("--W", type=int, default=512)
parser.add_argument("--H", type=int, default=512)

animatediff_args = parser.parse_args()

*_, func_args = inspect.getargvalues(inspect.currentframe())
func_args = dict(func_args)

time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
savedir = f"samples/{Path(animatediff_args.config).stem}-{time_str}-frames16_512"
os.makedirs(savedir)
inference_config = OmegaConf.load(animatediff_args.inference_config)

config  = OmegaConf.load(animatediff_args.config) # 1-ToonYou.yaml 
samples = []

sample_idx = 0
for model_idx, (config_key, model_config) in enumerate(list(config.items())):
    
    motion_modules = model_config.motion_module
    motion_modules = [motion_modules] if isinstance(motion_modules, str) else list(motion_modules)
    for motion_module in motion_modules:
    
        ### >>> create validation pipeline >>> ###
        # tokenizer    = CLIPTokenizer.from_pretrained(animatediff_args.pretrained_model_path, subfolder="tokenizer", torch_dtype=torch.float16)
        # text_encoder = CLIPTextModel.from_pretrained(animatediff_args.pretrained_model_path, subfolder="text_encoder", torch_dtype=torch.float16)
        # vae          = AutoencoderKL.from_pretrained(animatediff_args.pretrained_model_path, subfolder="vae", torch_dtype=torch.float16)            
        # unet         = UNet3DConditionModel.from_pretrained_2d(animatediff_args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))      

        tokenizer    = CLIPTokenizer.from_pretrained(animatediff_args.pretrained_model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(animatediff_args.pretrained_model_path, subfolder="text_encoder")
        vae          = AutoencoderKL.from_pretrained(animatediff_args.pretrained_model_path, subfolder="vae")            
        unet         = UNet3DConditionModel.from_pretrained_2d(animatediff_args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))      

        # unet.to(torch.float16)
        # if is_xformers_available(): unet.enable_xformers_memory_efficient_attention()
        # else: assert False

        video_pipe = AnimationPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
        ).to("cuda")


        # text_embeddings_ = video_pipe._encode_prompt('water splashing the rocks in slow motion', 'cuda', 1, True, '')
        # print('text_embeddings_:', text_embeddings_.dtype)
        # exit()

        # video_pipe.to(torch_dtype=torch.float16)

        # 1. unet ckpt
        # 1.1 motion module
        motion_module_state_dict = torch.load(motion_module, map_location="cpu")
        if "global_step" in motion_module_state_dict: func_args.update({"global_step": motion_module_state_dict["global_step"]})
        missing, unexpected = video_pipe.unet.load_state_dict(motion_module_state_dict, strict=False)
        assert len(unexpected) == 0

        # 1.2 T2I
        if model_config.path != "":
            if model_config.path.endswith(".ckpt"):
                state_dict = torch.load(model_config.path)
                video_pipe.unet.load_state_dict(state_dict)
                
            elif model_config.path.endswith(".safetensors"):
                state_dict = {}
                with safe_open(model_config.path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
                        
                is_lora = all("lora" in k for k in state_dict.keys())
                # print('state_dict.keys()', state_dict.keys())
                print(f"Is LoRA: {is_lora}") # False
                if not is_lora:
                    base_state_dict = state_dict
                else:
                    base_state_dict = {}
                    with safe_open(model_config.base, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            base_state_dict[key] = f.get_tensor(key)                
                
                # vae
                converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_state_dict, video_pipe.vae.config)
                def convert_ldm_vae_checkpoint2(ckpt):
                    ckpt_new={}
                    for k, v in ckpt.items():
                        if "key" in k:
                            k = k.replace("key", "to_k")
                        if "query" in k:
                            k = k.replace("query", "to_q")
                        if "value" in k:
                            k = k.replace("value", "to_v")
                        if "proj_attn" in k:
                            k = k.replace("proj_attn", "to_out.0")
                        ckpt_new[k] = v
                    return ckpt_new
                converted_vae_checkpoint = convert_ldm_vae_checkpoint2(converted_vae_checkpoint)
                # converted_vae_checkpoint = {k.replace("vae.", ""): v for k, v in converted_vae_checkpoint.items()}
                video_pipe.vae.load_state_dict(converted_vae_checkpoint)
                # unet
                converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_state_dict, video_pipe.unet.config)
                video_pipe.unet.load_state_dict(converted_unet_checkpoint, strict=False)
                # text_model
                video_pipe.text_encoder = convert_ldm_clip_checkpoint(base_state_dict)

                # video_pipe.enable_xformers_memory_efficient_attention()
                
                # import pdb
                # pdb.set_trace()
                if is_lora:
                    video_pipe = convert_lora(video_pipe, state_dict, alpha=model_config.lora_alpha)

        # video_pipe.text_encoder.to(torch.float16)
        # video_pipe.unet.to(torch.float16)
        # video_pipe.vae.to(torch.float16)

        
        video_pipe.to("cuda")


device = "cuda"
# define imagebind model
bind_model = imagebind_model.imagebind_huge(pretrained=False)

state_dict = torch.load("/home/xianyang/Data/code/Seeing-and-Hearing/v2a/imagebind/.checkpoints/imagebind_huge.pth", map_location=device)
bind_model.load_state_dict(state_dict)

bind_model.eval()
bind_model.to(device)
# bind_model.to(torch.float16)


for p in bind_model.parameters():
    p.requires_grad = False


random_seed = 10499524853910852697
torch.manual_seed(random_seed)

# prompt = 'water splashing the rocks in slow motion'
prompt = 'photo of coastline, rocks, storm weather, wind, waves, lightning, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3'
negative_prompt = 'blur, haze, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation'
# call pipeline prepare
# original_waveform_length, device, do_classifier_free_guidance, prompt_embeds, timesteps, latents, latents_dtype, extra_step_kwargs

num_inference_steps = 25
learning_rate = 0
clip_duration = 1
clips_per_video = 1

audio_step_args, audio_latents = audio_pipe.bind_prepare(
        prompt = prompt,
        audio_length_in_s = 2.0,
        num_inference_steps = num_inference_steps,
        guidance_scale = 2.5
    )
# device, do_classifier_free_guidance, text_embeddings, timesteps, latents, latents_dtype, extra_step_kwargs
# video_step_args, video_latents = video_pipe.bind_prepare(prompt = prompt,
#         audio_length_in_s = 2.0,
#         num_inference_steps = 50,
#         guidance_scale = 2.5)
video_step_args, video_latents = video_pipe.bind_prepare(
        prompt = prompt,
        negative_prompt = negative_prompt,
        video_length = animatediff_args.L,
        height = animatediff_args.H,
        width = animatediff_args.W,
        num_inference_steps = num_inference_steps,
        guidance_scale = 7.5
    )

print('audio_latents: ', audio_latents.shape, 'video_latents: ', video_latents.shape) # [1, 8, 50, 16], [1, 4, 16, 64, 64]


# audio_latents.to(torch.float16)
# video_latents.to(torch.float16)

# video_step_args["latents_dtype"] = torch.float16
# audio_step_args["latents_dtype"] = torch.float16
# call denoising step and then optimize step




for cur_step in tqdm(range(num_inference_steps), desc="Inference Steps"):
    with torch.autograd.set_detect_anomaly(True):
        # denoising step
        xt_audio, noise_pred_audio = audio_pipe.bind_step(**audio_step_args, cur_step=cur_step, latents=audio_latents)
        xt_video, noise_pred_video = video_pipe.bind_step(**video_step_args, cur_step=cur_step, latents=video_latents)

        xt_audio_temp = xt_audio.detach()
        xt_video_temp = xt_video.detach()

        xt_audio_temp.requires_grad = True
        xt_video_temp.requires_grad = True

        x0_audio, x0_audio_waveform = audio_pipe.xt2x0(latents_temp=xt_audio_temp, timesteps=audio_step_args["timesteps"], cur_step=cur_step, noise_pred=noise_pred_audio)
        x0_video = video_pipe.xt2x0(latents_temp=xt_video_temp, timesteps=video_step_args["timesteps"], cur_step=cur_step, noise_pred=noise_pred_video)

        # optimize step 
        # define imagebind model 
        # gradient backward 
        # update latents 

        #print('x0_audio_waveform shape: ',x0_audio_waveform.dtype, x0_audio_waveform.shape) # [1, 32032]

        x0_imagebind_audio_input = load_and_transform_audio_data_from_waveform(x0_audio_waveform, org_sample_rate=16000, 
                                                    device=device, target_length=204, clip_duration=clip_duration, clips_per_video=clips_per_video)
                                
        #print('x0_video shape: ', x0_video.shape) # [1, 3, 4, 224, 224]
        x0_imegebind_video_input = load_and_transform_video_data_from_tensor_real(x0_video, device, clip_duration=clip_duration, clips_per_video=clips_per_video, n_samples_per_clip=2)

        #print(x0_imegebind_video_input.dtype, x0_imagebind_audio_input.dtype)
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


