CUDA_VISIBLE_DEVICES=0,1 python quick_start_joint_distributed_triple_loss_final_rebuttal.py --config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/prompts/5-RealisticVision-abspath.yaml \
                            --pretrained_model_path /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/models/StableDiffusion/stable-diffusion-v1-5 \
                            --inference_config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/inference/inference.yaml \
                            --L 12 \
                            --W 224 \
                            --H 224 \
                            --n_sample_per_prompt 30 \
                            --seed 10499524853910852697 \
                            --start 0 \
                            --end 25 


CUDA_VISIBLE_DEVICES=2,3 python quick_start_joint_distributed_triple_loss_final_rebuttal.py --config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/prompts/5-RealisticVision-abspath.yaml \
                            --pretrained_model_path /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/models/StableDiffusion/stable-diffusion-v1-5 \
                            --inference_config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/inference/inference.yaml \
                            --L 12 \
                            --W 224 \
                            --H 224 \
                            --n_sample_per_prompt 30 \
                            --seed 10499524853910852697 \
                            --start 25 \
                            --end 50 


CUDA_VISIBLE_DEVICES=4,5 python quick_start_joint_distributed_triple_loss_final_rebuttal.py --config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/prompts/5-RealisticVision-abspath.yaml \
                            --pretrained_model_path /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/models/StableDiffusion/stable-diffusion-v1-5 \
                            --inference_config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/inference/inference.yaml \
                            --L 12 \
                            --W 224 \
                            --H 224 \
                            --n_sample_per_prompt 30 \
                            --seed 10499524853910852697 \
                            --start 50 \
                            --end 75 


CUDA_VISIBLE_DEVICES=7,6 python quick_start_joint_distributed_triple_loss_final_rebuttal.py --config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/prompts/5-RealisticVision-abspath.yaml \
                            --pretrained_model_path /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/models/StableDiffusion/stable-diffusion-v1-5 \
                            --inference_config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/inference/inference.yaml \
                            --L 12 \
                            --W 224 \
                            --H 224 \
                            --n_sample_per_prompt 30 \
                            --seed 10499524853910852697 \
                            --start 75 \
                            --end 100 

# -- with correct script 
CUDA_VISIBLE_DEVICES=0,1 python quick_start_joint_distributed_triple_loss_rebuttal.py --config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/prompts/5-RealisticVision-abspath.yaml \
                            --pretrained_model_path /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/models/StableDiffusion/stable-diffusion-v1-5 \
                            --inference_config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/inference/inference.yaml \
                            --L 12 \
                            --W 224 \
                            --H 224 \
                            --n_sample_per_prompt 30 \
                            --start 0 \
                            --end 50 

CUDA_VISIBLE_DEVICES=4,5 python quick_start_joint_distributed_triple_loss_rebuttal.py --config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/prompts/5-RealisticVision-abspath.yaml \
                            --pretrained_model_path /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/models/StableDiffusion/stable-diffusion-v1-5 \
                            --inference_config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/inference/inference.yaml \
                            --L 12 \
                            --W 224 \
                            --H 224 \
                            --n_sample_per_prompt 30 \
                            --start 50 \
                            --end 100 

