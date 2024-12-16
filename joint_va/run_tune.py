import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import sys 
sys.path.append('/home/yxingag/llmsvgen/yazhou/sear/joint/AudioLDM')
sys.path.append('/home/yxingag/llmsvgen/yazhou/sear/joint/AnimateDiff')

from AudioLDM.audioldm.pipelines.pipeline_audioldm import AudioLDMPipeline
# from diffusers import AudioLDMPipeline
import torch
import soundfile as sf
from accelerate.utils import set_seed
from AudioLDM.audioldm.models.unet import UNet2DConditionModel
from moviepy.editor import VideoFileClip, AudioFileClip

import inspect
import argparse
import datetime
from pathlib import Path
from omegaconf import OmegaConf
import diffusers
from diffusers import AutoencoderKL, DDIMScheduler

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from AnimateDiff.animatediff.models.unet import UNet3DConditionModel
from AnimateDiff.animatediff.pipelines.pipeline_animation import AnimationPipeline
from AnimateDiff.animatediff.utils.util import save_videos_grid
from AnimateDiff.animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from AnimateDiff.animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora
from diffusers.utils.import_utils import is_xformers_available
from safetensors import safe_open

from imagebind_data import load_and_transform_audio_data_from_waveform, load_and_transform_video_data_from_tensor_real, load_and_transform_text
from imagebind.imagebind.models import imagebind_model
from imagebind.imagebind.models.imagebind_model import ModalityType

import torch.nn.functional as F
from glob import glob 



# import pdb; pdb.set_trace()
#  ------ define audioldm model ------
my_model_path = '/home/yxingag/llmsvgen/yazhou/sear/joint/AudioLDM/ckpt/audioldm-m-full'
# unet = UNet2DConditionModel.from_pretrained(my_model_path, subfolder='unet', torch_dtype=torch.float16).to('cuda')
# audio_pipe = AudioLDMPipeline.from_pretrained(my_model_path, unet=unet, torch_dtype=torch.float16)
unet = UNet2DConditionModel.from_pretrained(my_model_path, subfolder='unet').to('cuda:0')
audio_pipe = AudioLDMPipeline.from_pretrained(my_model_path, unet=unet)
audio_pipe = audio_pipe.to("cuda:0")


# ------ define animatediff model ------
parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_model_path", type=str, default="models/StableDiffusion/stable-diffusion-v1-5",)
parser.add_argument("--inference_config",      type=str, default="configs/inference/inference.yaml")    
parser.add_argument("--config",                type=str, required=True)
parser.add_argument("--eval_set",      type=str, default="/home/yazhou/disk1/projects/edit/others/blip2/key_frames_landscape_200_sample")    
parser.add_argument("--video_prompt_txt",      type=str, default="1021_video_audio_eval.txt")    
parser.add_argument("--audio_prompt_txt",      type=str, default="1021_video_audio_eval.txt")    

parser.add_argument("--L", type=int, default=16 )
parser.add_argument("--W", type=int, default=512)
parser.add_argument("--H", type=int, default=512)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=500)
parser.add_argument("--n_sample_per_prompt", type=int, default=1)
parser.add_argument("--optimize_text", action='store_true') 


animatediff_args = parser.parse_args()

print('arguments: ', animatediff_args)

video_prompt_list = []
with open(animatediff_args.video_prompt_txt, 'r') as f_open:
    for line in f_open:
        video_prompt_list.append(line.strip())

video_prompt_list = video_prompt_list[animatediff_args.start:animatediff_args.end] 

audio_prompt_list = []
with open(animatediff_args.audio_prompt_txt, 'r') as f_open:
    for line in f_open:
        audio_prompt_list.append(line.strip())

audio_prompt_list = audio_prompt_list[animatediff_args.start:animatediff_args.end] 

print('all promots: ', video_prompt_list, audio_prompt_list) 

*_, func_args = inspect.getargvalues(inspect.currentframe())
func_args = dict(func_args)

time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
savedir = f"samples/{Path(animatediff_args.config).stem}-{time_str}"
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
        if is_xformers_available(): unet.enable_xformers_memory_efficient_attention()
        else: assert False

        video_pipe = AnimationPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
        ).to("cuda:1")


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

        
        video_pipe.to("cuda:1")


device = "cuda:0"
# define imagebind model
bind_model = imagebind_model.imagebind_huge(pretrained=True)

bind_model.eval()
bind_model.to(device)
# bind_model.to(torch.float16)


for p in bind_model.parameters():
    p.requires_grad = False


random_seed = 10499524853910852697
torch.manual_seed(random_seed)

# prompt = 'water splashing the rocks in slow motion'
# prompt = 'photo of coastline, rocks, storm weather, wind, waves, lightning, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3'
negative_prompt = 'blur, haze, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation'
# call pipeline prepare

# num_inference_steps = [25, 35, 40, 50]
num_inference_steps = [50] # TODO 
# num_inference_steps = [{'video': 25, 'audio': 30}] # TODO 
# learning_rate = [0.1, 0.01]
# learning_rate = [0.1] # TODO 
learning_rate = [{'video': 0.05, 'audio': 0.05}]

clip_duration = 1
clips_per_video = 1
savedir = savedir + f'_inferstep{num_inference_steps[0]}_videolr{learning_rate[0]['video']}_audiolr{learning_rate[0]['audio']}'
os.makedirs(savedir, exist_ok=True)

count = 0
for video_prompt, audio_prompt in zip(video_prompt_list, audio_prompt_list):
    for n_step in num_inference_steps:
        num_warmup_step_audio = n_step * 0.2

        # generate org video and audio 
        generator = torch.Generator(device='cuda')
        for sample_id in range(animatediff_args.n_sample_per_prompt):
            audio_step_args, audio_latents = audio_pipe.bind_prepare(
                    prompt = audio_prompt,
                    audio_length_in_s = 2.0,
                    num_inference_steps = n_step,
                    guidance_scale = 2.5
                )
            # device, do_classifier_free_guidance, text_embeddings, timesteps, latents, latents_dtype, extra_step_kwargs
            video_step_args, video_latents = video_pipe.bind_prepare(
                    prompt = video_prompt,
                    negative_prompt = negative_prompt,
                    video_length = animatediff_args.L,
                    height = animatediff_args.H,
                    width = animatediff_args.W,
                    num_inference_steps = n_step,
                    guidance_scale = 7.5
                )

            audio_org = audio_pipe(audio_prompt, num_inference_steps=n_step, audio_length_in_s=2.0, generator=generator, latents=audio_latents).audios[0]

            video_org = video_pipe(
                video_prompt,
                negative_prompt     = negative_prompt,
                num_inference_steps = n_step,
                guidance_scale      = 7.5,
                width               = animatediff_args.W,
                height              = animatediff_args.H,
                video_length        = animatediff_args.L,
                latents             = video_latents
            ).videos

            if len(audio_prompt) > 100:
                prompt_to_save = audio_prompt[:100]
            else:
                prompt_to_save = audio_prompt
            
            sf.write(f"{savedir}/{count:04d}_{prompt_to_save}_{sample_id:02d}_org.wav", audio_org, samplerate=16000) 
            save_videos_grid(video_org, f"{savedir}/{count:04d}_{prompt_to_save}_{sample_id:02d}_org.gif") 

            # save video + audio 
            audio = AudioFileClip(f"{savedir}/{count:04d}_{prompt_to_save}_{sample_id:02d}_org.wav")
            video = VideoFileClip(f"{savedir}/{count:04d}_{prompt_to_save}_{sample_id:02d}_org.mp4")
            video = video.set_audio(audio)
            video.write_videofile(f"{savedir}/{count:04d}_{prompt_to_save}_{sample_id:02d}_org_combine.mp4", codec="libx264")                


            for lr in learning_rate:
                # original_waveform_length, device, do_classifier_free_guidance, prompt_embeds, timesteps, latents, latents_dtype, extra_step_kwargs
                # audio_step_args, audio_latents = audio_pipe.bind_prepare(
                #         prompt = prompt,
                #         audio_length_in_s = 2.0,
                #         num_inference_steps = n_step,
                #         guidance_scale = 2.5
                #     )
                # # device, do_classifier_free_guidance, text_embeddings, timesteps, latents, latents_dtype, extra_step_kwargs
                # video_step_args, video_latents = video_pipe.bind_prepare(
                #         prompt = prompt,
                #         negative_prompt = negative_prompt,
                #         video_length = animatediff_args.L,
                #         height = animatediff_args.H,
                #         width = animatediff_args.W,
                #         num_inference_steps = n_step,
                #         guidance_scale = 7.5
                #     )

                for cur_step in range(n_step):
                    with torch.autograd.set_detect_anomaly(True):
                        prompt_embeds_video = video_step_args['text_embeddings']
                        prompt_embeds_video = prompt_embeds_video.detach()
                        
                        # denoising step
                        xt_audio, noise_pred_audio = audio_pipe.bind_step(**audio_step_args, cur_step=cur_step, latents=audio_latents)
                        xt_video, noise_pred_video = video_pipe.bind_step(**video_step_args, cur_step=cur_step, latents=video_latents)

                        xt_audio_temp = xt_audio.detach()
                        xt_video_temp = xt_video.detach()
                        
                        if cur_step >= num_warmup_step_audio:
                            xt_audio_temp.requires_grad = True
                        
                        if animatediff_args.optimize_text:
                            prompt_embeds_video.requires_grad = True
                            xt_video_temp.requires_grad = False
                        else:
                            xt_video_temp.requires_grad = True
                            prompt_embeds_video.requires_grad = False

                        x0_audio, x0_audio_waveform = audio_pipe.xt2x0(latents_temp=xt_audio_temp, timesteps=audio_step_args["timesteps"], cur_step=cur_step, noise_pred=noise_pred_audio)
                        x0_video = video_pipe.xt2x0(latents_temp=xt_video_temp, timesteps=video_step_args["timesteps"], cur_step=cur_step, noise_pred=noise_pred_video)

                        # optimize step 
                        # define imagebind model 
                        # gradient backward 
                        # update latents 
                        x0_imagebind_audio_input = load_and_transform_audio_data_from_waveform(x0_audio_waveform, org_sample_rate=16000, 
                                                                    device=device, target_length=204, clip_duration=clip_duration, clips_per_video=clips_per_video)
                                                
                        x0_imegebind_video_input = load_and_transform_video_data_from_tensor_real(x0_video, device, clip_duration=clip_duration, clips_per_video=clips_per_video, n_samples_per_clip=2)

                        inputs = {
                            ModalityType.VISION: x0_imegebind_video_input,
                            ModalityType.AUDIO: x0_imagebind_audio_input,
                            ModalityType.TEXT: load_and_transform_text([audio_prompt, video_prompt], device),
                        }

                        embeddings = bind_model(inputs)

                        print('shape: ', embeddings[ModalityType.TEXT].shape)
                        # exit()

                        bind_loss_vision_audio = 1 - F.cosine_similarity(embeddings[ModalityType.VISION], embeddings[ModalityType.AUDIO])
                        bind_loss_text_audio = 1 - F.cosine_similarity(embeddings[ModalityType.TEXT][:1], embeddings[ModalityType.AUDIO])
                        bind_loss_text_vision = 1 - F.cosine_similarity(embeddings[ModalityType.TEXT][1:], embeddings[ModalityType.VISION])

                        bind_loss = bind_loss_vision_audio + bind_loss_text_audio + bind_loss_text_vision
                        
                        if animatediff_args.optimize_text:
                            optimizer_video = torch.optim.Adam([prompt_embeds_video], lr=lr['video']) 
                        else:
                            optimizer_video = torch.optim.Adam([xt_video_temp], lr=lr['video']) 
                        optimizer_audio = torch.optim.Adam([xt_audio_temp], lr=lr['audio']) 

                        bind_loss.backward() 

                        optimizer_video.step()
                        optimizer_video.zero_grad()

                        if cur_step >= num_warmup_step_audio:
                            optimizer_audio.step()
                            optimizer_audio.zero_grad()

                        audio_latents = xt_audio_temp.detach()
                        video_latents = xt_video_temp.detach() 
                        
                        if animatediff_args.optimize_text:
                            video_step_args['text_embeddings'] = prompt_embeds_video.detach()

                audio = audio_pipe.bind_finish(original_waveform_length = audio_step_args["original_waveform_length"], 
                                                latents=audio_latents).audios[0]
                video = video_pipe.bind_finish(latents=video_latents).videos

                print('audio.shape', audio.shape) # (32000,)
                print('video.shape', video.shape) # (1, 3, 4, 224, 224)

                # save audio and video
                # out_path = 'temp'

                # {sample_idx}_step{n_step}_org_{prompt}.wav
                # sf.write(f"{savedir}/{sample_idx}_step{n_step}_lr{lr}_{prompt}.wav", audio, samplerate=16000)
                # save_videos_grid(video, f"{savedir}/{sample_idx}_step{n_step}_lr{lr}_{prompt}.gif")

                # audio = AudioFileClip(f"{savedir}/{sample_idx}_step{n_step}_lr{lr}_{prompt}.wav")
                # video = VideoFileClip(f"{savedir}/{sample_idx}_step{n_step}_lr{lr}_{prompt}.mp4")
                # video = video.set_audio(audio)
                # video.write_videofile(f"{savedir}/{sample_idx}_step{n_step}_lr{lr}_{prompt}-combine.mp4", codec="libx264")

                sf.write(f"{savedir}/{count:04d}_{prompt_to_save}_{sample_id:02d}_ours.wav", audio, samplerate=16000)
                save_videos_grid(video, f"{savedir}/{count:04d}_{prompt_to_save}_{sample_id:02d}_ours.gif")

                audio = AudioFileClip(f"{savedir}/{count:04d}_{prompt_to_save}_{sample_id:02d}_ours.wav")
                video = VideoFileClip(f"{savedir}/{count:04d}_{prompt_to_save}_{sample_id:02d}_ours.mp4")
                video = video.set_audio(audio)
                video.write_videofile(f"{savedir}/{count:04d}_{prompt_to_save}_{sample_id:02d}_ours_combine.mp4", codec="libx264")

    count += 1 