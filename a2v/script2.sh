

# CUDA_VISIBLE_DEVICES=0,1,6 python quick_start_bind.py --config configs/prompts/5-RealisticVision-abspath.yaml \
#                             --pretrained_model_path models/StableDiffusion/stable-diffusion-v1-5 \
#                             --inference_config configs/inference/inference.yaml \
#                             --L 12 \
#                             --W 224 \
#                             --H 224 \
#                             --audio_root /home/yazhou/disk1/projects/edit/dataset/vggsound_500_sample_audios

# CUDA_VISIBLE_DEVICES=2,3 python quick_start_bind.py --config configs/prompts/5-RealisticVision-abspath.yaml \
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
#                             --save_suffix seed-1111-optimstep-1-with_text_loss-VideoPrompt-reverse_order \
#                             --with_text_loss \
#                             --prompt_root /home/yazhou/disk1/projects/edit/others/blip2/key_frames_vggsound \
#                             --bind_device cuda:0 --reverse_order

# CUDA_VISIBLE_DEVICES=2,3 python quick_start_bind.py --config configs/prompts/5-RealisticVision-abspath.yaml \
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
#                             --optim_cond_embd \
#                             --random_seed 1111 --num_optimization_steps 1 \
#                             --save_suffix seed-1111-optimstep-1-with_text_loss-VideoPrompt-reverse_order \
#                             --with_text_loss \
#                             --prompt_root /home/yazhou/disk1/projects/edit/others/blip2/key_frames_vggsound \
#                             --bind_device cuda:0 --reverse_order

# CUDA_VISIBLE_DEVICES=3,5 python quick_start_bind.py --config configs/prompts/5-RealisticVision-abspath.yaml \
#                             --pretrained_model_path models/StableDiffusion/stable-diffusion-v1-5 \
#                             --inference_config configs/inference/inference.yaml \
#                             --L 16 \
#                             --W 224 \
#                             --H 224 \
#                             --audio_root /disk1/yingqing/disk2/dataset/vggsound_3k_1122_audio \
#                             --gradient_checkpointing \
#                             --num_inference_steps 25 \
#                             --learning_rate 0.01 \
#                             --optimization_starting_point 0 \
#                             --optim_cond_embd \
#                             --random_seed 1111 --num_optimization_steps 1 \
#                             --save_suffix seed-1111-optimstep-1-with_text_loss-VideoPrompt-reverse_order-3k-reverse_order \
#                             --with_text_loss \
#                             --prompt_root /home/yazhou/disk1/projects/edit/others/blip2/key_frames_VGGSound_3k \
#                             --bind_device cuda:0 --reverse_order

CUDA_VISIBLE_DEVICES=6,7 python quick_start_bind.py --config configs/prompts/5-RealisticVision-abspath.yaml \
                            --pretrained_model_path models/StableDiffusion/stable-diffusion-v1-5 \
                            --inference_config configs/inference/inference.yaml \
                            --L 16 \
                            --W 224 \
                            --H 224 \
                            --audio_root /disk1/yingqing/disk2/dataset/vggsound_3k_1122_audio \
                            --gradient_checkpointing \
                            --num_inference_steps 25 \
                            --learning_rate 0.01 \
                            --optimization_starting_point 0 \
                            --optim_cond_embd \
                            --random_seed 1111 --num_optimization_steps 1 \
                            --save_suffix seed-1111-optimstep-1-with_text_loss-VideoPrompt-reverse_order-3k-reverse_order \
                            --with_text_loss \
                            --prompt_root /home/yazhou/disk1/projects/edit/others/blip2/key_frames_VGGSound_3k \
                            --bind_device cuda:0 --reverse_order \
                            --start_idx 0 \
                            --end_idx 1500

# 

# --skip_original 
# --all_frames_loss
# --optim_mode only_cond 

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