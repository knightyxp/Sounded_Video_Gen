# extract key frame 
# python extract_key_frame.py --root ./demo/source --out_root ./demo/key_frames

# # do caption 
# python qwen_caption.py --imgdir ./demo/key_frames


# generate audio 
# CUDA_VISIBLE_DEVICES=0,1 python joint_video_audio.py \
#                     --eval_set_root ./demo/source \
#                     --prompt_root ./demo/key_frames \
#                     --out_root output/demo \
#                     --double_loss \
#                     --start 0 \
#                     --end 15 
                    
                    # --init_latents

# CUDA_VISIBLE_DEVICES=0,1 python joint_image2_sounded_video.py \
#                     --double_loss \
#                     --start 0 \
#                     --end 15 
                    


CUDA_VISIBLE_DEVICES=0,1 python joint_wanx_tangoflux.py \
                    --start 0 \
                    --end 15 