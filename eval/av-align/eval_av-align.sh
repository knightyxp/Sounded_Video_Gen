#!/bin/bash

input_video_dir=/home/xianyang/Data/code/Seeing-and-Hearing/joint_va/samples/panda_guitar_49_frames
input_wav_dir=/home/xianyang/Data/code/Seeing-and-Hearing/joint_va/samples/panda_guitar_49_frames
cache_path="./video_cache.json"

python av-align.py --input_video_dir $input_video_dir --input_wav_dir $input_wav_dir --cache_path $cache_path
