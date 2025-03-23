
from moviepy.editor import VideoFileClip, AudioFileClip

cur_work_dir = '/home/xianyang/Data/code/Seeing-and-Hearing/joint_va/samples/i2v_play_violin_long_prompt' 
audio = AudioFileClip(f"{cur_work_dir}/output_audio.wav")
video = VideoFileClip(f"{cur_work_dir}/output_video.mp4")
video = video.set_audio(audio)
# video.write_videofile(
#     f"{cur_work_dir}/output_av.mp4",
#     codec="libx264",
# )
video.write_videofile(
    f"{cur_work_dir}/output_av_acc.mp4",
    codec="libx264",
    audio_codec="aac",
)
