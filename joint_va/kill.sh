
ps -ef | grep pipeline.sh | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep joint_video_audio.py | grep -v grep | awk '{print $2}' | xargs kill -9