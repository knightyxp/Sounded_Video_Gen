import os, glob, subprocess

root = '/disk1/zeyue/project/data/vggsound/vggsound_3k_1122'
videos = sorted(glob.glob(os.path.join(root, '*.mp4')))
out_root = '/disk1/yingqing/disk2/dataset/vggsound_3k_1122_audio'
os.makedirs(out_root, exist_ok=True)
# extract audio and save 
for video_path in videos:
    video_name = os.path.basename(video_path).split('.')[0]
    audio_path = os.path.join(out_root, f'{video_name}.wav')
    subprocess.call(['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', audio_path])


# video="/Volumes/My Passport/movies/1995-07狮子王.mp4"
# out="/Volumes/My Passport/movies/1995-07狮子王-audio.wav"

# # extract audio and save
# subprocess.call(['ffmpeg', '-i', video, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', out])
