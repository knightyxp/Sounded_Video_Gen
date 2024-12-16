import os 
import decord 



# path = '/home/yazhou/disk1/projects/edit/AnimateDiff/samples/5-RealisticVision-2023-07-25T21-48-52/sample/2-photo-of-coastline,-rocks,-storm-weather,-wind,-waves,-lightning,-8k.mp4'

# vr = decord.VideoReader(path)

# print(len(vr))


import torchaudio
import torch
import torch.nn.functional as F

class MyPipeline(torch.nn.Module):
    def __init__(
        self,
        sample_rate=16000, n_mels=128, n_fft=1024, hop_length=250
    ):
        super().__init__()
        self.waveform_to_mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, 
                        n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.waveform_to_mel(waveform)

        return mel


audio_transform_pipeline = MyPipeline()
# audio_transform_pipeline.to(device=torch.device("cuda"))

import librosa
# audio = torchaudio.load('/home/yazhou/disk1/projects/edit/AudioLDM/generate_samples/temp_folder/-_X6oUUgmJU_000001_new_fps_21.5_truncate_0_4.0.mp4')[0]
audio, _ = librosa.load('/home/yazhou/disk1/projects/edit/dataset/vggsound_500_sample_audios/-__L3T1Yv_4_000000.wav', sr=16000)
print(type(audio), audio.shape)
audio = torch.from_numpy(audio).unsqueeze(0) 
mel = audio_transform_pipeline(audio)
print(mel.shape)



