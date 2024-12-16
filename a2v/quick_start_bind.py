import argparse
import datetime
import inspect
import os
from omegaconf import OmegaConf

import torch

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora
from diffusers.utils.import_utils import is_xformers_available

from einops import rearrange, repeat

import csv, pdb, glob
from safetensors import safe_open
import math
from pathlib import Path

from moviepy.editor import VideoFileClip, AudioFileClip

def main(args):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)
    
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"results/{Path(args.config).stem}-{time_str}"
    if args.save_suffix != "": savedir += "-" + args.save_suffix
    os.makedirs(savedir)

    # load configs
    config  = OmegaConf.load(args.config) # 1-ToonYou.yaml 
    inference_config = OmegaConf.load(args.inference_config)

    # load audios and prompts 
    audio_paths = sorted(glob.glob(os.path.join(args.audio_root, "*.wav")))
    if args.prompt_root is None:
        prompt_txt_paths = sorted(glob.glob(os.path.join(args.audio_root, "*.txt")))
    else:
        prompt_txt_paths = sorted(glob.glob(os.path.join(args.prompt_root, "*.txt")))
    alls = len(prompt_txt_paths)

    if args.reverse_order:
        audio_paths = audio_paths[::-1]
        prompt_txt_paths = prompt_txt_paths[::-1]
    if args.end_idx is not None:
        audio_paths = audio_paths[:args.end_idx]
        prompt_txt_paths = prompt_txt_paths[:args.end_idx]
    if args.start_idx is not None:
        audio_paths = audio_paths[args.start_idx:]
        prompt_txt_paths = prompt_txt_paths[args.start_idx:]
    print(f'all samples={alls}, start_idx={args.start_idx}, end_idx={args.end_idx}, actual samples={len(prompt_txt_paths)}')

    for model_idx, (config_key, model_config) in enumerate(list(config.items())):
        
        motion_modules = model_config.motion_module
        motion_modules = [motion_modules] if isinstance(motion_modules, str) else list(motion_modules)
        for motion_module in motion_modules:
            
            # init pipeline
            tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
            vae          = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")      
            unet         = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))
            
            if is_xformers_available(): 
                # enable xformers in unet
                unet.enable_xformers_memory_efficient_attention()
            else: 
                assert False
            
            pipeline = AnimationPipeline(
                vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
            ).to("cuda:0")
            ngpu=torch.cuda.device_count()
            print(f"num_gpus={ngpu}")
            pipeline.vae = pipeline.vae.to(f"cuda:{ngpu-1}") # use the last gpu
            
            if args.gradient_checkpointing:
                unet.enable_gradient_checkpointing()

            # unet load ckpt
            # load motion module
            sd = torch.load(motion_module, map_location="cpu")
            if "global_step" in sd: func_args.update({"global_step": sd["global_step"]})
            missing, unexpected = pipeline.unet.load_state_dict(sd, strict=False)
            assert len(unexpected) == 0
            
            # 1.2 T2I
            if model_config.path != "":
                if model_config.path.endswith(".ckpt"):
                    state_dict = torch.load(model_config.path)
                    pipeline.unet.load_state_dict(state_dict)
                elif model_config.path.endswith(".safetensors"):
                    state_dict = {}
                    with safe_open(model_config.path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            state_dict[key] = f.get_tensor(key)
                            
                    is_lora = all("lora" in k for k in state_dict.keys())
                    print(f"Is LoRA: {is_lora}") # False

                    if not is_lora:
                        base_state_dict = state_dict
                    else:
                        base_state_dict = {}
                        with safe_open(model_config.base, framework="pt", device="cpu") as f:
                            for key in f.keys():
                                base_state_dict[key] = f.get_tensor(key)                
                    
                    # vae
                    converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_state_dict, pipeline.vae.config)
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

                    pipeline.vae.load_state_dict(converted_vae_checkpoint)
                    # unet
                    converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_state_dict, pipeline.unet.config)
                    pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
                    # text_model
                    pipeline.text_encoder = convert_ldm_clip_checkpoint(base_state_dict)
                    
                    # import pdb
                    # pdb.set_trace()
                    if is_lora:
                        pipeline = convert_lora(pipeline, state_dict, alpha=model_config.lora_alpha)

            # pipeline.to("cuda:0")

    # start inference
    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    negative_prompt = 'blur, haze, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation'

    # set sampling args
    num_inference_steps = args.num_inference_steps
    learning_rate = args.learning_rate #[0.1, 0.01]
    clip_duration = 1
    clips_per_video = 1

    sample_idx = 0
    for audio_path, prompt_txt_path in zip(audio_paths, prompt_txt_paths):
        audio_org = AudioFileClip(audio_path)
        sample_idx += 1
        with open(prompt_txt_path, "r") as f:
            prompt = f.read().strip()
        
        latents = pipeline.prepare_latents(
            1,
            4,
            args.L,
            args.H,
            args.W,
            dtype=torch.float32, 
            device=torch.device("cuda:0"), 
            generator=None,
        )
        for n_step in num_inference_steps:
            if args.skip_original:
                pass
            else:
                print(f"---- Original Forward ----------------")
                # direct sampling via A2T2V
                video_org = pipeline(
                    prompt,
                    negative_prompt     = negative_prompt,
                    num_inference_steps = n_step,
                    guidance_scale      = args.guidance_scale,
                    width               = args.W,
                    height              = args.H,
                    video_length        = args.L,
                    latents             = latents,
                ).videos

                # save video + audio
                # audio = AudioFileClip(f"{savedir}/{sample_idx}_step{n_step}_org_{prompt}.wav")
                # save_videos_grid(video_org, f"{savedir}/{sample_idx}_step{n_step}_lr{lr}_{prompt}.gif")
                ori_dir = os.path.join(savedir, "ori")
                os.makedirs(ori_dir, exist_ok=True)
                fn = f"{sample_idx:04d}_{prompt}_step{n_step}_ori"
                # TODO audio may be longer than video!!
                save_videos_grid(video_org, f"{ori_dir}/{fn}.gif") 
                video = VideoFileClip(f"{ori_dir}/{fn}.mp4")
                video = video.set_audio(audio_org)
                video.write_videofile(f"{ori_dir}/{fn}-withaudio.mp4", codec="libx264")
            # sampling via Imagebind guidance
            for lr in learning_rate:
                print(f"---- ImageBind Guided Forward, lr = {lr} --------")
                if args.optim_cond_embd:
                    if args.using_diff_foley:
                        video = pipeline.bind_forward_optim_cond_embd_diff_foley(
                            prompt,
                            negative_prompt     = negative_prompt,
                            num_inference_steps = n_step,
                            guidance_scale      = args.guidance_scale,
                            width               = args.W,
                            height              = args.H,
                            video_length        = args.L,
                            learning_rate       = lr,
                            clip_duration       = clip_duration,
                            clips_per_video     = clips_per_video,
                            num_optimization_steps = args.num_optimization_steps,
                            audio_paths         = [audio_path],
                            latents             = latents,
                            optimization_starting_point = args.optimization_starting_point,
                            optimization_ending_point   = args.optimization_ending_point,
                            with_text_loss      = args.with_text_loss,
                            optim_mode          = args.optim_mode,
                            all_frames_loss     = args.all_frames_loss,
                            bind_device         = args.bind_device,
                            norm_embed          = args.norm_embed,
                            n_samples_per_clip  = args.n_samples_per_clip,
                        ).videos
                    else:
                        video = pipeline.bind_forward_optim_cond_embd(
                            prompt,
                            negative_prompt     = negative_prompt,
                            num_inference_steps = n_step,
                            guidance_scale      = args.guidance_scale,
                            width               = args.W,
                            height              = args.H,
                            video_length        = args.L,
                            learning_rate       = lr,
                            clip_duration       = clip_duration,
                            clips_per_video     = clips_per_video,
                            num_optimization_steps = args.num_optimization_steps,
                            audio_paths         = [audio_path],
                            latents             = latents,
                            optimization_starting_point = args.optimization_starting_point,
                            optimization_ending_point   = args.optimization_ending_point,
                            with_text_loss      = args.with_text_loss,
                            optim_mode          = args.optim_mode,
                            all_frames_loss     = args.all_frames_loss,
                            bind_device         = args.bind_device,
                            norm_embed          = args.norm_embed,
                            n_samples_per_clip  = args.n_samples_per_clip,
                        ).videos

                else:
                    if args.using_diff_foley:
                        raise NotImplementedError
                    video = pipeline.bind_forward(
                        prompt,
                        negative_prompt     = negative_prompt,
                        num_inference_steps = n_step,
                        guidance_scale      = args.guidance_scale,
                        width               = args.W,
                        height              = args.H,
                        video_length        = args.L,
                        learning_rate       = lr,
                        clip_duration       = clip_duration,
                        clips_per_video     = clips_per_video,
                        num_optimization_steps = 1,
                        audio_paths         = [audio_path],
                        latents             = latents,
                        optimization_starting_point = args.optimization_starting_point,
                        optimization_ending_point   = args.optimization_ending_point,
                        with_text_loss      = args.with_text_loss,
                        all_frames_loss     = args.all_frames_loss,
                        bind_device         = args.bind_device,
                        n_samples_per_clip  = args.n_samples_per_clip,
                    ).videos
                ours_dir = os.path.join(savedir, "with_bind")
                os.makedirs(ours_dir, exist_ok=True)
                fn = f"{sample_idx:04d}_{prompt}_step{n_step}_lr{lr}"

                save_videos_grid(video, f"{ours_dir}/{fn}.gif")
                video = VideoFileClip(f"{ours_dir}/{fn}.mp4")
                video = video.set_audio(audio_org)
                video.write_videofile(f"{ours_dir}/{fn}-withaudio.mp4", codec="libx264")
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default="models/StableDiffusion/stable-diffusion-v1-5",)
    parser.add_argument("--inference_config",      type=str, default="configs/inference/inference.yaml")    
    parser.add_argument("--config",                type=str, required=True)
    parser.add_argument("--audio_root", type=str, default="/home/yazhou/disk1/projects/edit/dataset/vggsound_500_sample_audios",)
    parser.add_argument("--prompt_root", type=str, default=None,)
    
    parser.add_argument("--L", type=int, default=16 )
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)
    
    parser.add_argument("--random_seed", type=int, default=10499524853910852697)
    parser.add_argument("--learning_rate", type=float, default=[0.1, 0.01], nargs="+")
    parser.add_argument("--num_inference_steps", type=int, default=[25, 35, 40, 50], nargs="+")
    
    parser.add_argument("--optimization_starting_point", type=float, default=0.2)
    parser.add_argument("--optimization_ending_point", type=float, default=1)
    parser.add_argument("--num_optimization_steps",      type=int, default=1)
    parser.add_argument("--save_suffix",                 type=str)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    
    parser.add_argument("--optim_cond_embd", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--with_text_loss", action="store_true")
    parser.add_argument("--skip_original", action="store_true")
    parser.add_argument("--all_frames_loss", action="store_true")
    parser.add_argument("--reverse_order", action="store_true")
    parser.add_argument("--optim_mode",    type=str, default="both")
    parser.add_argument("--bind_device",    type=str, default="cuda:1")
    parser.add_argument("--norm_embed", action="store_true")
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--using_diff_foley", action="store_true")
    parser.add_argument("--n_samples_per_clip", type=int, default=2)

    args = parser.parse_args()
    main(args)

