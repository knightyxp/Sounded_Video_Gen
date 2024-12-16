

CUDA_VISIBLE_DEVICES=1,3,4 python quick_start_bind.py --config configs/prompts/5-RealisticVision-abspath.yaml \
                            --pretrained_model_path models/StableDiffusion/stable-diffusion-v1-5 \
                            --inference_config configs/inference/inference.yaml \
                            --L 16 \
                            --W 224 \
                            --H 224 \
                            --audio_root /disk1/yingqing/disk2/code/AnimateDiff-Audio2Video/data/eval_10_videos \
                            --gradient_checkpointing \
                            --num_inference_steps 25 \
                            --learning_rate 0.01 \
                            --optimization_starting_point 0 \
                            --optimization_ending_point 1 \
                            --random_seed 1111 --num_optimization_steps 1 \
                            --save_suffix seed-1111-optimstep-1-start1-end1-lr0.01-with_text_loss-VideoPrompt-FixBug-10EvalVideo-cfg5-nsamplesperclip2-all_frames_loss \
                            --optim_cond_embd \
                            --with_text_loss \
                            --prompt_root /disk1/yingqing/disk2/code/AnimateDiff-Audio2Video/data/prompt-extract-from-video \
                            --norm_embed --guidance_scale 5 --n_samples_per_clip 2 \
                            --all_frames_loss --skip_original
                            # 
                            # --bind_device cuda:0 \
                            
#  --start_idx 0 --end_idx 50 
# --all_frames_loss
# --optim_mode only_cond 
# --reverse_order
                            # 
