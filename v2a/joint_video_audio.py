import os
from audioldm.pipelines.pipeline_av_joint import Audio_Video_LDMPipeline
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
from diffusers.utils import export_to_video

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"



parser = argparse.ArgumentParser()
parser.add_argument("--eval_set_root", type=str, default="eval-set/generative")
parser.add_argument("--out_root", type=str, default="results-bind")
parser.add_argument("--prompt_root", type=str, default="results-bind")
parser.add_argument("--optimize_text", action='store_true', default=False)
parser.add_argument("--double_loss", action='store_true', default=False)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=500)
parser.add_argument("--init_latents", action='store_true', default=False)
parser.add_argument("--seed", type=int, default=30) 

args = parser.parse_args()

weight_dtype = torch.float16

# repo_id = "cvssp/audioldm-m-full"
local_model_path = 'ckpt/audioldm-m-full'
audio_unet = UNet2DConditionModel.from_pretrained(local_model_path, subfolder='unet').to('cuda:0',dtype=weight_dtype)

##########load video model#############

video_transformer = CogVideoXTransformer3DModel.from_pretrained('./ckpt/CogVideoX-2b', subfolder='transformer').to('cuda:1',dtype=weight_dtype)
video_vae = AutoencoderKLCogVideoX.from_pretrained('./ckpt/CogVideoX-5b-I2V', subfolder='vae').to('cuda:1',dtype=weight_dtype)
video_text_tokenizer = T5Tokenizer.from_pretrained('./ckpt/CogVideoX-5b-I2V', subfolder='tokenizer')
video_text_encoder = T5EncoderModel.from_pretrained('./ckpt/CogVideoX-5b-I2V', subfolder='text_encoder').to('cuda:1',dtype=weight_dtype)
video_scheduler = CogVideoXDDIMScheduler.from_pretrained('./ckpt/CogVideoX-5b-I2V', subfolder='scheduler')


print('finish load video model')
##########load video model#############

pipe = Audio_Video_LDMPipeline.from_pretrained(local_model_path, 
        video_vae=video_vae, video_text_encoder=video_text_encoder,
        video_text_tokenizer=video_text_tokenizer,transformer=video_transformer,
        video_scheduler=video_scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")



os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
# torch.use_deterministic_algorithms(True)
torch.use_deterministic_algorithms(True, warn_only=True)

# Enable CUDNN deterministic mode
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False

out_dir = args.out_root

config_seed_dict = {
    '0jZtLuEdjrk_000110':30,
    '0OriTE8vb6s_000150':77,
    '0VHVqjGXmBM_000030':30,
    '1EtApg0Hgyw_000075':30,
    '1PgwxYCi-qE_000220':45,
    'AvTGh7DiLI_000052':56,
    'imD3yh_zKg_000052':30,
    'jy_M41E9Xo_000379':56,
    'L_--bn4bys_000008':30
}

def get_video_name_and_prompt_demo(root):
    video_name_and_prompt = []
    txt_root = args.prompt_root
    all_text_files = sorted(glob(f"{txt_root}/*.txt"))

    videos = sorted(glob(f"{root}/*.mp4"))
    for video in videos[args.start:args.end]:
        video_name = video.split('/')[-1].split('.')[0]
        img_path = video.replace(root, './demo/key_frames').replace('.mp4', '_0.jpeg')  # Construct image path
        seed = config_seed_dict[video_name]
        txt_path = f"{txt_root}/{video_name}_0.txt"
        if not os.path.exists(txt_path):
            continue
        with open(txt_path, 'r') as f:
            prompt = f.readline().strip()
        #print(f"video: {video}, img: {img_path}, prompt: {prompt}")
        try:
            video_length = math.ceil(VideoFileClip(video).duration)
        except UnicodeDecodeError:
            continue
        video_name_and_prompt.append({'video_name': video, 'img':img_path, 'prompt': prompt, 'audio_length': video_length, 'seed':seed})
    
    return video_name_and_prompt


video_name_and_prompt = get_video_name_and_prompt_demo(args.eval_set_root) 


for vp in video_name_and_prompt:
    if vp['video_name'] == './demo/source/1PgwxYCi-qE_000220.mp4':
        video_name = vp['video_name']
        video_folder_name = os.path.dirname(video_name).split('/')[-1]
        video_base_name = 'name_' + video_name.split('/')[-1].split('.')[0]
        prompt = vp['prompt']
        img = vp['img']
        video_paths = [video_name]

        try:
            video = VideoFileClip(video_paths[0])
        except:
            continue
        inf_steps = [30]
        lrs = [0]
        num_optimization_steps = [1]
        clip_duration = 1
        clips_per_video = 1
        #vp['audio_length']
        cur_seed = vp['seed']
        optimization_starting_point = 0.2
        bind_params = [{'clip_duration': 1, 'clips_per_video': 1}]

        cur_out_dir = f"{out_dir}_inf_steps{inf_steps[0]}_lr{lrs[0]}/{video_folder_name}"
        os.makedirs(cur_out_dir, exist_ok=True)

        set_seed(cur_seed)
        generator = torch.Generator(device='cuda')

        generator.manual_seed(cur_seed)


        ngpu=torch.cuda.device_count()
        print(f"num_gpus={ngpu}")
        pipe.vae = pipe.vae.to(f"cuda:{ngpu-1}") # use the last gpu
        pipe.video_vae = pipe.video_vae.to(f"cuda:{ngpu-1}") 
        pipe.video_vae.enable_gradient_checkpointing()
        for bp in bind_params:
            for step in inf_steps:
                try:
                    video = VideoFileClip(video_paths[0])
                except:
                    continue

                if len(prompt) > 100:
                    prompt_to_save = prompt[:100]
                else:
                    prompt_to_save = prompt
                
                for opt_step in num_optimization_steps:
                    for lr in lrs:
                        #pipe.enable_model_cpu_offload()
                        video, audio, _ = pipe.bind_forward_triple_loss(prompt,latents=None, num_inference_steps=step, audio_length_in_s=vp['audio_length'], generator=generator, 
                                        video_paths=video_paths, learning_rate=lr, clip_duration=bp['clip_duration'], 
                                        clips_per_video=bp['clips_per_video'], num_optimization_steps=opt_step, return_dict=False)

                        export_to_video(video,rf"{cur_out_dir}/{video_base_name}_seed{cur_seed}_ouput_video.mp4", fps=10)
                        sf.write(rf"{cur_out_dir}/{video_base_name}_seed{cur_seed}.wav", audio, samplerate=16000)
                        audio = AudioFileClip(rf"{cur_out_dir}/{video_base_name}_seed{cur_seed}.wav")
                        video = VideoFileClip(rf"{cur_out_dir}/{video_base_name}_seed{cur_seed}_ouput_video.mp4")
                        video = video.set_audio(audio)
                        video.write_videofile(rf"{cur_out_dir}/{video_base_name}_seed{cur_seed}_output_joint_av.mp4")

