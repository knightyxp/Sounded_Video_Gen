

# CUDA_VISIBLE_DEVICES=0,1,6 python quick_start_bind.py --config configs/prompts/5-RealisticVision-abspath.yaml \
#                             --pretrained_model_path models/StableDiffusion/stable-diffusion-v1-5 \
#                             --inference_config configs/inference/inference.yaml \
#                             --L 12 \
#                             --W 224 \
#                             --H 224 \
#                             --audio_root /home/yazhou/disk1/projects/edit/dataset/vggsound_500_sample_audios

# CUDA_VISIBLE_DEVICES=0,1 python quick_start_bind.py --config configs/prompts/5-RealisticVision-abspath.yaml \
#                             --pretrained_model_path models/StableDiffusion/stable-diffusion-v1-5 \
#                             --inference_config configs/inference/inference.yaml \
#                             --L 16 \
#                             --W 224 \
#                             --H 224 \
#                             --audio_root /home/yazhou/disk1/projects/edit/dataset/vggsound_500_sample_audios \
#                             --gradient_checkpointing \
#                             --num_inference_steps 25 \
#                             --learning_rate 0.01 \
#                             --optimization_starting_point 0 \
#                             --optim_cond_embd \
#                             --random_seed 1111 --num_optimization_steps 1 \
#                             --save_suffix seed-1111-optimstep-1-with_text_loss-VideoPrompt-norm_embed_s42_e150 \
#                             --with_text_loss \
#                             --prompt_root /home/yazhou/disk1/projects/edit/others/blip2/key_frames_vggsound \
#                             --bind_device cuda:0 --norm_embed --start_idx 42 --end_idx 150 --skip_original 

# CUDA_VISIBLE_DEVICES=0,1 python quick_start_bind.py --config configs/prompts/5-RealisticVision-abspath.yaml \
#                             --pretrained_model_path models/StableDiffusion/stable-diffusion-v1-5 \
#                             --inference_config configs/inference/inference.yaml \
#                             --L 16 \
#                             --W 224 \
#                             --H 224 \
#                             --audio_root /home/yazhou/disk1/projects/edit/dataset/vggsound_500_sample_audios \
#                             --gradient_checkpointing \
#                             --num_inference_steps 25 \
#                             --learning_rate 0.01 \
#                             --optimization_starting_point 0 \
#                             --optim_cond_embd \
#                             --random_seed 1111 --num_optimization_steps 1 \
#                             --save_suffix seed-1111-optimstep-1-lr0.01-with_text_loss-VideoPrompt-norm_embed-FixBug \
#                             --with_text_loss \
#                             --prompt_root /home/yazhou/disk1/projects/edit/others/blip2/key_frames_vggsound \
#                             --bind_device cuda:0 --skip_original --norm_embed 
                            # 
#  --start_idx 0 --end_idx 50 
# --all_frames_loss
# --optim_mode only_cond 
# --reverse_order

# CUDA_VISIBLE_DEVICES=3,4,5 /home/yingqing/anaconda3/envs/video-audio/bin/python quick_start_bind.py \
#                             --config configs/prompts/5-RealisticVision-abspath.yaml \
#                             --pretrained_model_path models/StableDiffusion/stable-diffusion-v1-5 \
#                             --inference_config configs/inference/inference.yaml \
#                             --L 16 \
#                             --W 224 \
#                             --H 224 \
#                             --audio_root /home/yazhou/disk1/projects/edit/dataset/vggsound_500_sample_audios \
#                             --gradient_checkpointing \
#                             --num_inference_steps 25 \
#                             --learning_rate 0.05 \
#                             --optimization_starting_point 0 \
#                             --random_seed 1111 --num_optimization_steps 1 \
#                             --save_suffix seed-1111-optimstep-1-withTextLoss-optim-latent-startpoint0-lr0.05-all_frames_loss \
#                             --skip_original --with_text_loss --all_frames_loss
# CUDA_VISIBLE_DEVICES=0,2 python quick_start_bind.py --config configs/prompts/5-RealisticVision-abspath.yaml \
#                             --pretrained_model_path models/StableDiffusion/stable-diffusion-v1-5 \
#                             --inference_config configs/inference/inference.yaml \
#                             --L 16 \
#                             --W 224 \
#                             --H 224 \
#                             --audio_root /home/yazhou/disk1/projects/edit/dataset/vggsound_500_sample_audios \
#                             --gradient_checkpointing \
#                             --num_inference_steps 25 \
#                             --learning_rate 0.01 \
#                             --optimization_starting_point 0 \
#                             --optimization_ending_point 1 \
#                             --random_seed 1111 --num_optimization_steps 1 \
#                             --save_suffix 500SAMPLES-seed-1111-optimstep-1-start1-end1-lr0.01-with_text_loss-VideoPrompt-FixBug-cfg5-nsamplesperclip3-reverse_order \
#                             --optim_cond_embd \
#                             --with_text_loss \
#                             --prompt_root /home/yazhou/disk1/projects/edit/others/blip2/key_frames_vggsound \
#                             --norm_embed --guidance_scale 5 --n_samples_per_clip 3 --bind_device cuda:0 --reverse_order
                            
# another lora
CUDA_VISIBLE_DEVICES=5,6 python quick_start_bind.py --config configs/prompts/1-ToonYou-hyq.yaml \
                            --pretrained_model_path models/StableDiffusion/stable-diffusion-v1-5 \
                            --inference_config configs/inference/inference.yaml \
                            --L 16 \
                            --W 224 \
                            --H 224 \
                            --audio_root /home/yazhou/disk1/projects/edit/dataset/vggsound_500_sample_audios \
                            --gradient_checkpointing \
                            --num_inference_steps 25 \
                            --learning_rate 0.01 \
                            --optimization_starting_point 0 \
                            --optimization_ending_point 1 \
                            --random_seed 1111 --num_optimization_steps 1 \
                            --save_suffix 500SAMPLES-seed-1111-optimstep-1-start1-end1-lr0.01-with_text_loss-VideoPrompt-FixBug-cfg5-nsamplesperclip3-LORA-AutoPrompt \
                            --optim_cond_embd \
                            --with_text_loss \
                            --norm_embed --guidance_scale 7.5 --n_samples_per_clip 3 --bind_device cuda:0
                            # --prompt_root /home/yazhou/disk1/projects/edit/others/blip2/key_frames_vggsound \

                            # --all_frames_loss --skip_original

# CUDA_VISIBLE_DEVICES=0,1 python quick_start_bind.py --config configs/prompts/5-RealisticVision-abspath.yaml \
#                             --pretrained_model_path models/StableDiffusion/stable-diffusion-v1-5 \
#                             --inference_config configs/inference/inference.yaml \
#                             --L 16 \
#                             --W 224 \
#                             --H 224 \
#                             --audio_root /home/yazhou/disk1/projects/edit/dataset/vggsound_500_sample_audios_4s \
#                             --gradient_checkpointing \
#                             --num_inference_steps 25 \
#                             --learning_rate 0.01 \
#                             --optimization_starting_point 0 \
#                             --random_seed 1111 --num_optimization_steps 1 \
#                             --save_suffix seed-1111-optimstep-1-lr0.01-with_text_loss-VideoPrompt-OnlyDifffloey-OptimLatent \
#                             --with_text_loss \
#                             --prompt_root /home/yazhou/disk1/projects/edit/others/blip2/key_frames_vggsound \
#                             --bind_device cuda:0 --using_diff_foley --skip_original
                            # --optim_cond_embd \
                            # --norm_embed