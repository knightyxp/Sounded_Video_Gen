
# CUDA_VISIBLE_DEVICES=0
# python quick_start_joint.py --config /home/xianyang/Data/code/AnimateDiff/configs/RealisticVision.yaml \
#                             --pretrained_model_path /home/xianyang/Data/code/FateZero/ckpt/stable-diffusion-v1-5 \
#                             --inference_config ./animatediff/inference.yaml \
#                             --L 16 \
#                             --W 224 \
#                             --H 224



CUDA_VISIBLE_DEVICES=0,1
python joint_cogvideo_audioldm.py  --L 49 \
                            --W 480 \
                            --H 720