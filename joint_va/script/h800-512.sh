


# CUDA_VISIBLE_DEVICES=$1 python quick_start_joint_distributed_triple_loss_rebuttal_tune.py --config /home/yxingag/llmsvgen/yazhou/sear/joint/AnimateDiff/configs/prompts/$2.yaml \
#                             --pretrained_model_path /home/yxingag/llmsvgen/yazhou/sear/joint/AnimateDiff/models/StableDiffusion/stable-diffusion-v1-5 \
#                             --inference_config /home/yxingag/llmsvgen/yazhou/sear/joint/AnimateDiff/configs/inference/inference.yaml \
#                             --L 14 \
#                             --W 480 \
#                             --H 480 \
#                             --n_sample_per_prompt 3 \
#                             --start 0 \
#                             --end 50 \
#                             --video_prompt_txt animatediff_prompts_$2-video.txt \
#                             --audio_prompt_txt animatediff_prompts_$2-audio.txt \
#                             $3


CUDA_VISIBLE_DEVICES=$1 python quick_start_joint_distributed_triple_loss_rebuttal_tune.py --config /home/yxingag/llmsvgen/yazhou/sear/joint/AnimateDiff/configs/prompts/$2.yaml \
                            --pretrained_model_path /home/yxingag/llmsvgen/yazhou/sear/joint/AnimateDiff/models/StableDiffusion/stable-diffusion-v1-5 \
                            --inference_config /home/yxingag/llmsvgen/yazhou/sear/joint/AnimateDiff/configs/inference/inference.yaml \
                            --L 14 \
                            --W 480 \
                            --H 480 \
                            --n_sample_per_prompt 3 \
                            --start 0 \
                            --end 50 \
                            --video_prompt_txt animatediff_prompts_$2.txt \
                            --audio_prompt_txt animatediff_prompts_$2.txt \
                            $3

