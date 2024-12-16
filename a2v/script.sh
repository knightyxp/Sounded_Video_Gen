

# CUDA_VISIBLE_DEVICES=0,1,6 python quick_start_bind.py --config configs/prompts/5-RealisticVision-abspath.yaml \
#                             --pretrained_model_path models/StableDiffusion/stable-diffusion-v1-5 \
#                             --inference_config configs/inference/inference.yaml \
#                             --L 12 \
#                             --W 224 \
#                             --H 224 \
#                             --audio_root /home/yazhou/disk1/projects/edit/dataset/vggsound_500_sample_audios

CUDA_VISIBLE_DEVICES=0,1,2 python quick_start_bind.py --config configs/prompts/5-RealisticVision-abspath.yaml \
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
                            --optim_cond_embd \
                            --random_seed 1111 --num_optimization_steps 1 \
                            --save_suffix seed-1111-optimstep-1-with_text_loss \
                            --with_text_loss 
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