


CUDA_VISIBLE_DEVICES=0,1 python run.py --config /home/yxingag/sear/joint_clean/AnimateDiff/configs/prompts/5-RealisticVision-abspath.yaml \
                            --pretrained_model_path /home/yxingag/sear/joint_clean/AnimateDiff/models/StableDiffusion/stable-diffusion-v1-5 \
                            --inference_config /home/yxingag/sear/joint_cleans/AnimateDiff/configs/inference/inference.yaml \
                            --L 12 \
                            --W 224 \
                            --H 224 \
                            --n_sample_per_prompt 30 \
                            --start 0 \
                            --end 50 

