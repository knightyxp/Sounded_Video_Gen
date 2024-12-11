import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

# Models: "THUDM/CogVideoX-2b" or "THUDM/CogVideoX-5b"
pipe = CogVideoXPipeline.from_pretrained("./ckpt/CogVideoX-2b", torch_dtype=torch.float16).to("cuda")
prompt = (
    "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. "
    "The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other "
    "pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, "
    "casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. "
    "The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical "
    "atmosphere of this unique musical performance."
)
video = pipe(prompt=prompt, guidance_scale=6, num_inference_steps=50).frames[0]
export_to_video(video, "output.mp4", fps=8)