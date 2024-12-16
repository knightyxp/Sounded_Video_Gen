python quick_start_joint.py --config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/prompts/5-RealisticVision-abspath.yaml \
                            --pretrained_model_path /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/models/StableDiffusion/stable-diffusion-v1-5 \
                            --inference_config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/inference/inference.yaml \
                            --L 6 \
                            --W 384 \
                            --H 384
                            


CUDA_VISIBLE_DEVICES=2,3,5 python quick_start_joint_distributed.py --config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/prompts/5-RealisticVision-abspath.yaml \
                            --pretrained_model_path /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/models/StableDiffusion/stable-diffusion-v1-5 \
                            --inference_config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/inference/inference.yaml \
                            --L 16 \
                            --W 224 \
                            --H 224


CUDA_VISIBLE_DEVICES=0,2,3 python quick_start_joint_distributed_triple_loss_consistency.py --config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/prompts/5-RealisticVision-abspath.yaml \
                            --pretrained_model_path /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/models/StableDiffusion/stable-diffusion-v1-5 \
                            --inference_config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/inference/inference.yaml \
                            --L 12 \
                            --W 224 \
                            --H 224

CUDA_VISIBLE_DEVICES=0,4,1 python quick_start_joint_distributed_triple_loss_consistency.py --config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/prompts/5-RealisticVision-abspath.yaml \
                            --pretrained_model_path /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/models/StableDiffusion/stable-diffusion-v1-5 \
                            --inference_config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/inference/inference.yaml \
                            --L 12 \
                            --W 224 \
                            --H 224


CUDA_VISIBLE_DEVICES=0,1,2 python quick_start_joint_distributed_triple_loss_audioldm2.py --config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/prompts/5-RealisticVision-abspath.yaml \
                            --pretrained_model_path /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/models/StableDiffusion/stable-diffusion-v1-5 \
                            --inference_config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/inference/inference.yaml \
                            --L 12 \
                            --W 224 \
                            --H 224


CUDA_VISIBLE_DEVICES=2,5,7 python quick_start_joint_distributed_triple_loss.py --config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/prompts/5-RealisticVision-abspath.yaml \
                            --pretrained_model_path /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/models/StableDiffusion/stable-diffusion-v1-5 \
                            --inference_config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/inference/inference.yaml \
                            --L 12 \
                            --W 224 \
                            --H 224


CUDA_VISIBLE_DEVICES=6,7 python quick_start_joint_distributed_triple_loss_final.py --config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/prompts/5-RealisticVision-abspath.yaml \
                            --pretrained_model_path /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/models/StableDiffusion/stable-diffusion-v1-5 \
                            --inference_config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/inference/inference.yaml \
                            --eval_set /home/yazhou/disk1/projects/edit/others/blip2/key_frames_landscape_200_sample \
                            --L 16 \
                            --W 224 \
                            --H 224


CUDA_VISIBLE_DEVICES=6,7 python quick_start_joint_distributed_triple_loss_final.py --config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/prompts/5-RealisticVision-abspath.yaml \
                            --pretrained_model_path /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/models/StableDiffusion/stable-diffusion-v1-5 \
                            --inference_config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/inference/inference.yaml \
                            --eval_set /home/yazhou/disk1/projects/edit/others/blip2/key_frames_landscape_200_sample \
                            --L 12 \
                            --W 224 \
                            --H 224

CUDA_VISIBLE_DEVICES=2,3 python quick_start_joint_distributed_triple_loss_final.py --config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/prompts/5-RealisticVision-abspath.yaml \
                            --pretrained_model_path /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/models/StableDiffusion/stable-diffusion-v1-5 \
                            --inference_config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/inference/inference.yaml \
                            --eval_set /home/yazhou/disk1/projects/edit/others/blip2/key_frames_landscape_200_sample \
                            --L 12 \
                            --W 224 \
                            --H 224

CUDA_VISIBLE_DEVICES=4,5 python quick_start_joint_distributed_triple_loss_final.py --config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/prompts/5-RealisticVision-abspath.yaml \
                            --pretrained_model_path /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/models/StableDiffusion/stable-diffusion-v1-5 \
                            --inference_config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/inference/inference.yaml \
                            --eval_set /home/yazhou/disk1/projects/edit/others/blip2/key_frames_landscape_200_sample \
                            --L 12 \
                            --W 224 \
                            --H 224


CUDA_VISIBLE_DEVICES=1,6 python quick_start_joint_distributed_triple_loss_final.py --config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/prompts/5-RealisticVision-abspath.yaml \
                            --pretrained_model_path /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/models/StableDiffusion/stable-diffusion-v1-5 \
                            --inference_config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/inference/inference.yaml \
                            --eval_set /home/yazhou/disk1/projects/edit/others/blip2/key_frames_landscape_200_sample \
                            --L 12 \
                            --W 224 \
                            --H 224


CUDA_VISIBLE_DEVICES=0,1 python quick_start_joint_distributed_triple_loss_final.py --config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/prompts/5-RealisticVision-abspath.yaml \
                            --pretrained_model_path /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/models/StableDiffusion/stable-diffusion-v1-5 \
                            --inference_config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/inference/inference.yaml \
                            --eval_set /home/yazhou/disk1/projects/edit/others/blip2/key_frames_VGGSound_3k \
                            --L 12 \
                            --W 224 \
                            --H 224 \
                            --start 0 \
                            --end 1500

CUDA_VISIBLE_DEVICES=2,6 python quick_start_joint_distributed_triple_loss_final.py --config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/prompts/5-RealisticVision-abspath.yaml \
                            --pretrained_model_path /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/models/StableDiffusion/stable-diffusion-v1-5 \
                            --inference_config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/inference/inference.yaml \
                            --eval_set /home/yazhou/disk1/projects/edit/others/blip2/key_frames_VGGSound_3k \
                            --L 12 \
                            --W 224 \
                            --H 224 \
                            --start 1500 \
                            --end 3000


CUDA_VISIBLE_DEVICES=0,1 python quick_start_joint_distributed_triple_loss_final.py --config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/prompts/5-RealisticVision-abspath.yaml \
                            --pretrained_model_path /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/models/StableDiffusion/stable-diffusion-v1-5 \
                            --inference_config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/inference/inference.yaml \
                            --eval_set /home/yazhou/disk1/projects/edit/others/blip2/key_frames_landscape_200_sample \
                            --L 12 \
                            --W 224 \
                            --H 224 \
                            --n_sample_per_prompt 5 \
                            --seed 10499524853910852697


CUDA_VISIBLE_DEVICES=2,3 python quick_start_joint_distributed_triple_loss_final.py --config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/prompts/5-RealisticVision-abspath.yaml \
                            --pretrained_model_path /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/models/StableDiffusion/stable-diffusion-v1-5 \
                            --inference_config /home/yazhou/disk1/projects/edit/joint-generation/AnimateDiff/configs/inference/inference.yaml \
                            --eval_set /home/yazhou/disk1/projects/edit/others/blip2/key_frames_landscape_200_sample \
                            --L 12 \
                            --W 224 \
                            --H 224 \
                            --n_sample_per_prompt 5 \
                            --seed 10499524853910852700


