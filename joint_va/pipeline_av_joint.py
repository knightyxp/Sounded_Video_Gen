# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import ClapTextModelWithProjection, RobertaTokenizer, RobertaTokenizerFast, SpeechT5HifiGan
from transformers import T5EncoderModel, T5Tokenizer

# from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models import AutoencoderKL
from audioldm.models.unet import UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import is_accelerate_available, logging, replace_example_docstring
# from diffusers.utils import is_accelerate_available, logging, randn_tensor, replace_example_docstring
from audioldm.pipelines.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import AudioPipelineOutput, DiffusionPipeline
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput

from diffusers.models import  AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.video_processor import VideoProcessor

from imagebind_data import load_and_transform_video_data_from_tensor_real,load_and_transform_audio_data_from_waveform, load_and_transform_video_data, load_and_transform_text, load_and_transform_vision_data, waveform2melspec
from imagebind.imagebind.models import imagebind_model
from imagebind.imagebind.models.imagebind_model import ModalityType

import torchaudio

import soundfile as sf
import os

from utils.utils import instantiate_from_config
from omegaconf import OmegaConf
from diffusers.utils import export_to_video, load_video



def enable_gradient_checkpointing(model):
    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = True


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import AudioLDMPipeline

        >>> pipe = AudioLDMPipeline.from_pretrained("cvssp/audioldm", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> prompt = "A hammer hitting a wooden surface"
        >>> audio = pipe(prompt).audio[0]
        ```
"""

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


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps




class Audio_Video_LDMPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-audio generation using AudioLDM.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode audios to and from latent representations.
        text_encoder ([`ClapTextModelWithProjection`]):
            Frozen text-encoder. AudioLDM uses the text portion of
            [CLAP](https://huggingface.co/docs/transformers/main/model_doc/clap#transformers.ClapTextModelWithProjection),
            specifically the [RoBERTa HSTAT-unfused](https://huggingface.co/laion/clap-htsat-unfused) variant.
        tokenizer ([`PreTrainedTokenizer`]):
            Tokenizer of class
            [RobertaTokenizer](https://huggingface.co/docs/transformers/model_doc/roberta#transformers.RobertaTokenizer).
        unet ([`UNet2DConditionModel`]): U-Net architecture to denoise the encoded audio latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded audio latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        vocoder ([`SpeechT5HifiGan`]):
            Vocoder of class
            [SpeechT5HifiGan](https://huggingface.co/docs/transformers/main/en/model_doc/speecht5#transformers.SpeechT5HifiGan).
    """

    model_cpu_offload_seq = "video_text_encoder->transformer->video_vae"

    def __init__(
        self,
        video_vae: AutoencoderKLCogVideoX,
        video_text_tokenizer: T5Tokenizer,
        video_text_encoder: T5EncoderModel,
        transformer: CogVideoXTransformer3DModel,
        video_scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
        vae: AutoencoderKL,
        text_encoder: ClapTextModelWithProjection,
        tokenizer: Union[RobertaTokenizer, RobertaTokenizerFast],
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        vocoder: SpeechT5HifiGan,
    ):
        super().__init__()

        self.register_modules(
            video_vae = video_vae,
            video_text_tokenizer = video_text_tokenizer, 
            video_text_encoder=video_text_encoder,
            transformer = transformer,
            video_scheduler = video_scheduler,
            vae = vae,
            tokenizer=tokenizer,
            text_encoder = text_encoder,
            unet = unet,
            scheduler = scheduler,
            vocoder = vocoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)


        ## cogvideo vae
        self.video_vae_scale_factor_spatial = (
            2 ** (len(self.video_vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.video_vae_scale_factor_temporal = (
            self.video_vae.config.temporal_compression_ratio if hasattr(self, "vae") and self.vae is not None else 4
        )
        self.video_vae_scaling_factor_image = (
            self.video_vae.config.scaling_factor if hasattr(self, "vae") and self.vae is not None else 0.7
        )

        self.video_processor = VideoProcessor(vae_scale_factor=self.video_vae_scale_factor_spatial)


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and vocoder have their state dicts saved to CPU and then are moved to a `torch.device('meta')
        and loaded to GPU only when their specific submodule has its `forward` method called.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.video_unet, self.text_encoder, self.vae, self.vocoder]:
            cpu_offload(cpu_offloaded_model, device)


    @property
    def video_guidance_scale(self):
        return self._video_guidance_scale

    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._execution_device
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    # Copied from diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline._get_t5_prompt_embeds
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.video_text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.video_text_tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.video_text_tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.video_text_tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.video_text_encoder(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds


    # Copied from diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def _encode_prompt(
        self,
        prompt,
        device,
        num_waveforms_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device (`torch.device`):
                torch device
            num_waveforms_per_prompt (`int`):
                number of waveforms that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the audio generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLAP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask.to(device),
            )
            prompt_embeds = prompt_embeds.text_embeds
            # additional L_2 normalization over each hidden-state
            prompt_embeds = F.normalize(prompt_embeds, dim=-1)

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        (
            bs_embed,
            seq_len,
        ) = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_waveforms_per_prompt)
        prompt_embeds = prompt_embeds.view(bs_embed * num_waveforms_per_prompt, seq_len)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            uncond_input_ids = uncond_input.input_ids.to(device)
            attention_mask = uncond_input.attention_mask.to(device)

            negative_prompt_embeds = self.text_encoder(
                uncond_input_ids,
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds.text_embeds
            # additional L_2 normalization over each hidden-state
            negative_prompt_embeds = F.normalize(negative_prompt_embeds, dim=-1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_waveforms_per_prompt)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_waveforms_per_prompt, seq_len)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds


    def decode_video_latents_bind(self, latents):
        # video_latents: b, c, t, h, w (1,12,16,60,90)

        video_latents = video_latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        video_latents = 1 / self.video_vae_scaling_factor_image * video_latents

        video_latents_device = video_latents.device

        if video_latents_device != self.video_vae.device:
            multi_gpu = True
        else:
            multi_gpu = False

        if multi_gpu:
            video_latents = video_latents.to(self.video_vae.device)

        video = self.video_vae.decode(video_latents).sample
        video = self.video_processor.postprocess_video(video=video,output_type='pt')

        return video

    def decode_video_latents(self, video_latents: torch.Tensor) -> torch.Tensor:
        video_latents = video_latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        video_latents = 1 / self.video_vae_scaling_factor_image * video_latents

        video_latents_device = video_latents.device


        # if video_latents_device != self.video_vae.device:
        #     multi_gpu = True
        # else:
        #     multi_gpu = False

        # if multi_gpu:
        #     video_latents = video_latents.to(self.video_vae.device)

        # print(f'video_latents device: {video_latents.device}, video_latents shape, dtype: {video_latents.shape, video_latents.dtype} ')
        # print(f'self.video_vae device: {self.video_vae.device}')
        # print('multi gpu:', multi_gpu)

        # org entirely decode latents
        frames = self.video_vae.decode(video_latents).sample

        ## slice decode video latents
        # video = [] ##video_latents b,c,f,h,w
        # for frame_idx in range(0, video_latents.shape[2], 2):
        #     latents_slice = video_latents[:, :, frame_idx:frame_idx+2, :, :]
        #     video.append(self.video_vae.decode(latents_slice).sample)
        # video = torch.cat(video, dim=2)
        # print('video',video.shape)

        return frames

    def decode_audio_latents(self, audio_latents):
        audio_latents = 1 / self.vae.config.scaling_factor * audio_latents
        ## for multi card 
        audio_latents_device = audio_latents.device

        mel_spectrogram = self.vae.decode(audio_latents).sample
        return mel_spectrogram

    def mel_spectrogram_to_waveform(self, mel_spectrogram):
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)

        waveform = self.vocoder(mel_spectrogram)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        waveform = waveform.cpu().float()
        return waveform

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        audio_length_in_s,
        vocoder_upsample_factor,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        min_audio_length_in_s = vocoder_upsample_factor * self.vae_scale_factor
        if audio_length_in_s < min_audio_length_in_s:
            raise ValueError(
                f"`audio_length_in_s` has to be a positive value greater than or equal to {min_audio_length_in_s}, but "
                f"is {audio_length_in_s}."
            )

        if self.vocoder.config.model_in_dim % self.vae_scale_factor != 0:
            raise ValueError(
                f"The number of frequency bins in the vocoder's log-mel spectrogram has to be divisible by the "
                f"VAE scale factor, but got {self.vocoder.config.model_in_dim} bins and a scale factor of "
                f"{self.vae_scale_factor}."
            )

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents with width->self.vocoder.config.model_in_dim
    def prepare_latents(self, batch_size, num_channels_latents, height, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            self.vocoder.config.model_in_dim // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (self.video_vae_scale_factor_spatial * self.transformer.config.patch_size)
        grid_width = width // (self.video_vae_scale_factor_spatial * self.transformer.config.patch_size)
        base_size_width = 720 // (self.video_vae_scale_factor_spatial * self.transformer.config.patch_size)
        base_size_height = 480 // (self.video_vae_scale_factor_spatial * self.transformer.config.patch_size)

        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=self.transformer.config.attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
        )

        freqs_cos = freqs_cos.to(device=device)
        freqs_sin = freqs_sin.to(device=device)
        return freqs_cos, freqs_sin


    def prepare_video_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        shape = (
            batch_size,
            (num_frames - 1) // self.video_vae_scale_factor_temporal + 1,
            num_channels_latents,
            height // self.video_vae_scale_factor_spatial,
            width // self.video_vae_scale_factor_spatial,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.video_scheduler.init_noise_sigma
        return latents




    def bind_forward_triple_loss(
        self,
        prompt: Union[str, List[str]] = None,
        audio_length_in_s: Optional[float] = None,
        num_inference_steps: int = 10,
        audio_guidance_scale: float = 2.5,
        video_guidance_scale: float = 6,
        learning_rate: float = 0.1,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_waveforms_per_prompt: Optional[int] = 1,
        clip_duration: float = 2.0,
        clips_per_video: int = 5,
        num_optimization_steps: int = 1,
        optimization_starting_point: float = 0.2,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        output_type: Optional[str] = "np",
    ):
        # 0. Convert audio input length from seconds to spectrogram height
        vocoder_upsample_factor = np.prod(self.vocoder.config.upsample_rates) / self.vocoder.config.sampling_rate

        if audio_length_in_s is None:
            audio_length_in_s = self.unet.config.sample_size * self.vae_scale_factor * vocoder_upsample_factor

        height_audio = int(audio_length_in_s / vocoder_upsample_factor)

        original_waveform_length = int(audio_length_in_s * self.vocoder.config.sampling_rate)
        if height_audio % self.vae_scale_factor != 0:
            height_audio = int(np.ceil(height_audio / self.vae_scale_factor)) * self.vae_scale_factor
            logger.info(
                f"Audio length in seconds {audio_length_in_s} is increased to {height_audio * vocoder_upsample_factor} "
                f"so that it can be handled by the model. It will be cut to {audio_length_in_s} after the "
                f"denoising process."
            )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            audio_length_in_s,
            vocoder_upsample_factor,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )


        ### some setting for cogvideox
        self._video_guidance_scale = video_guidance_scale
        self._video_attention_kwargs = attention_kwargs


        bind_device = torch.device("cuda:0")
        audio_device = torch.device("cuda:0")
        video_device = torch.device("cuda:1")


        ## set audio pipe to cuda 0
        self.unet = self.unet.to(audio_device)
        self.text_encoder = self.text_encoder.to(audio_device)
        self.vae = self.vae.to(audio_device)
        self.vocoder = self.vocoder.to(audio_device)  


        ## set video pipe to cuda 1
        self.video_vae = self.video_vae.to(video_device)
        self.transformer = self.transformer.to(video_device)
        self.video_text_encoder = self.video_text_encoder.to(video_device)

        ##decrease memory cost for decode
        self.video_vae.enable_tiling()
        self.video_vae.enable_slicing()


        # self.video_vae._set_gradient_checkpointing(self.video_vae.encoder, True)
        # self.video_vae._set_gradient_checkpointing(self.video_vae.decoder, True)

        # self.video_vae.decoder.training = True
        # print(self.video_vae.encoder.gradient_checkpointing)  # 应返回 True
        # print(self.video_vae.decoder.gradient_checkpointing)  # 应返回 True


        # from accelerate import cpu_offload
        # vae_device = torch.device(f"cuda:{1}")

        # for cpu_offloaded_model in [self.video_vae]:
        #     if cpu_offloaded_model is not None:
        #         cpu_offload(cpu_offloaded_model, vae_device)


        #set dtype
        latents_dtype = torch.float32
        # self.text_encoder = self.text_encoder.to("cuda:0",dtype=latents_dtype)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]


        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = audio_guidance_scale > 1.0

        # 3. Encode input prompt
        audio_prompt_embeds = self._encode_prompt(
            prompt,
            audio_device,
            num_waveforms_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        num_videos_per_prompt = 1
        print('audio_device',audio_device,'video_device',video_device,"bind_device",bind_device)

        video_prompt_embeds, video_negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=1,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            device=video_device,
            dtype =latents_dtype,
        )

        if do_classifier_free_guidance:
            prompt_embeds_video = torch.cat([video_negative_prompt_embeds, video_prompt_embeds], dim=0)


        # 4. Prepare timesteps
        ### prepare video scheduler timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.video_scheduler, num_inference_steps, video_device)
        self._num_timesteps = len(timesteps)

        ### prepare audio scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=audio_device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables

        num_channels_latents = self.unet.config.in_channels

        audio_latents = self.prepare_latents(
            batch_size * num_waveforms_per_prompt,
            num_channels_latents,
            height_audio,
            latents_dtype,
            audio_device,
            generator,
            latents,
        )

        #prepare video latents
        height = 224
        width = 224
        num_frames = 4
        latent_channels = self.transformer.config.in_channels
        video_latents = self.prepare_video_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            latents_dtype,
            video_device,
            generator=None,
        )
        print('video_latents',video_latents.device,"    video transformer device",self.transformer.device)
        print('video_latents',video_latents.shape,video_latents.dtype)
        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)


        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, video_latents.size(1), video_device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        #image_bind_video_input = load_and_transform_video_data(video_paths, device, clip_duration=clip_duration, clips_per_video=clips_per_video, n_samples_per_clip=2)
        #print('image_bind_video_input',image_bind_video_input.shape)

        bind_model = imagebind_model.imagebind_huge(pretrained=False)

        state_dict = torch.load("/home/xianyang/Data/code/Seeing-and-Hearing/v2a/imagebind/.checkpoints/imagebind_huge.pth", map_location=bind_device)

        bind_model.eval()
        bind_model = bind_model.to(dtype=torch.float32,device=bind_device)

        for p in bind_model.parameters():
            p.requires_grad = False
        


        print('attention_kwargs',attention_kwargs)


        ## cpu off load


        # 7. Denoising loop
        #num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        num_warmup_steps_bind = int(len(timesteps) * optimization_starting_point)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                #===================audio denoising======================#
                # expand the audio latents if we are doing classifier free guidance
                audio_latent_model_input = torch.cat([audio_latents] * 2) if do_classifier_free_guidance else audio_latents
                
                audio_latent_model_input = self.scheduler.scale_model_input(audio_latent_model_input, t)

                # predict the audio noise residual
                with torch.no_grad():
                    audio_noise_pred = self.unet(
                        audio_latent_model_input,
                        t,
                        encoder_hidden_states=None,
                        class_labels=audio_prompt_embeds,
                        cross_attention_kwargs=attention_kwargs,
                    ).sample.to(dtype=latents_dtype)

                # perform guidance
                if do_classifier_free_guidance:
                    audio_noise_pred_uncond, audio_noise_pred_text = audio_noise_pred.chunk(2)
                    audio_noise_pred = audio_noise_pred_uncond + audio_guidance_scale * (audio_noise_pred_text - audio_noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                audio_latents = self.scheduler.step(audio_noise_pred, t, audio_latents, **extra_step_kwargs).prev_sample   


                #===================audio denoising======================#

                #===================video denoising======================#
                video_latent_model_input = torch.cat([video_latents] * 2) if do_classifier_free_guidance else video_latents
                video_latent_model_input = self.video_scheduler.scale_model_input(video_latent_model_input, t)
                #print('video_latent_model_input',video_latent_model_input.shape)
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(video_latent_model_input.shape[0]).to(video_device)
                with torch.no_grad():
                    # predict noise model_output
                    video_noise_pred = self.transformer(
                        hidden_states=video_latent_model_input,
                        encoder_hidden_states=prompt_embeds_video,
                        timestep=timestep,
                        image_rotary_emb=image_rotary_emb,
                        attention_kwargs= self._video_attention_kwargs,  #### set as None in default cogvideo pipe
                        return_dict=False,
                    )[0]
                    #video_noise_pred = video_noise_pred.float()

                if do_classifier_free_guidance:
                    video_noise_pred_uncond, video_noise_pred_text = video_noise_pred.chunk(2)
                    video_noise_pred = video_noise_pred_uncond + self._video_guidance_scale * (video_noise_pred_text - video_noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1

                video_latents = self.video_scheduler.step(video_noise_pred, t, video_latents, **extra_step_kwargs, return_dict=False)[0]

                video_latents = video_latents.to(latents_dtype)

                #===================video denoising======================#

                #===================joint va triangle loss optimization======================#

                for optim_step in range(num_optimization_steps):
                    with torch.autograd.set_detect_anomaly(True):

                        video_latents_temp = video_latents.detach()
                        video_latents_temp.requires_grad = True 

                        audio_latents_temp = audio_latents.detach()
                        if i > num_warmup_steps_bind:
                            audio_latents_temp.requires_grad = True 



                        #================compute audio=================#
                        # 1. compute x0 
                        audio_x0 = 1/(self.scheduler.alphas_cumprod[t] ** 0.5) * (audio_latents_temp - (1-self.scheduler.alphas_cumprod[t])**0.5 * audio_noise_pred)
                    
                        print('audio_x0',audio_x0.shape,audio_x0.dtype,audio_x0.device)
                        # 2. decode x0 with vae decoder 
                        x0_mel_spectrogram = self.decode_audio_latents(audio_x0)

                        if x0_mel_spectrogram.dim() == 4:
                            x0_mel_spectrogram = x0_mel_spectrogram.squeeze(1)

                        # 3. convert mel-spectrogram to waveform
                        x0_waveform = self.vocoder(x0_mel_spectrogram).clone()
                        x0_waveform = x0_waveform.to(torch.float32)

                        # 4. waveform to imagebind mel-spectrogram
                        x0_imagebind_audio_input = load_and_transform_audio_data_from_waveform(x0_waveform, org_sample_rate=self.vocoder.config.sampling_rate, 
                                            device=bind_device, target_length=204, clip_duration=clip_duration, clips_per_video=clips_per_video).to(torch.float32)

                        del x0_waveform, x0_mel_spectrogram, audio_x0
                        #================compute audio=================#

                        #================compute video=================#

                        # 1. compute video x0 
                        video_x0 = 1/(self.video_scheduler.alphas_cumprod[t] ** 0.5) * (video_latents_temp - (1-self.video_scheduler.alphas_cumprod[t])**0.5 * video_noise_pred) 
                        # 2. decode video x0 with video vae decoder 
                        #self.video_vae.disable_gradient_checkpointing()
                        x0_video = self.decode_video_latents(video_x0).to(dtype=torch.float32, device=bind_device)
                        # print('x0_video',x0_video.shape)

                        # # 3. x0_video to frames to export 
                        #video = self.video_processor.postprocess_video(video=video, output_type=output_type)
                        # export_to_video(video,f'output_{t}.mp4',fps=10)
                        # # 4. reload video to image bind video input
                        x0_imagebind_video_input = load_and_transform_video_data_from_tensor_real(x0_video, bind_device, clip_duration=clip_duration, clips_per_video=clips_per_video, n_samples_per_clip=2).to(torch.float32)
                        # image_bind_video_input = load_and_transform_video_data('output_{t}.mp4', device, clip_duration=clip_duration, clips_per_video=clips_per_video, n_samples_per_clip=2)
                        
                        #================compute video=================#

                        print(x0_imagebind_video_input.dtype, x0_imagebind_audio_input.dtype)
                        # compute loss with imagebind 
                        if isinstance(prompt, str):
                            prompt_bind = [prompt]
                        inputs = {
                            ModalityType.VISION: x0_imagebind_video_input,
                            ModalityType.AUDIO: x0_imagebind_audio_input,
                            ModalityType.TEXT: load_and_transform_text(prompt_bind, bind_device)
                        }

                        # with torch.no_grad():
                        embeddings = bind_model(inputs)

                        bind_loss_text_vision = 1 - F.cosine_similarity(embeddings[ModalityType.TEXT], embeddings[ModalityType.VISION])     

                        bind_loss_text_audio = 1 - F.cosine_similarity(embeddings[ModalityType.TEXT], embeddings[ModalityType.AUDIO])

                        bind_loss_vision_audio = 1 - F.cosine_similarity(embeddings[ModalityType.VISION], embeddings[ModalityType.AUDIO])

                        bind_loss =  bind_loss_text_vision + bind_loss_text_audio + bind_loss_vision_audio    
                        #bind_loss =  bind_loss_vision_audio                    
                        print('bind_loss_text_vision',bind_loss_text_vision)
                        print('bind_loss_text_audio',bind_loss_text_audio)
                        print('bind_loss_vision_audio',bind_loss_vision_audio)           

                        optimizer_video = torch.optim.Adam([video_latents_temp], lr=learning_rate) 
                        optimizer_audio = torch.optim.Adam([audio_latents_temp], lr=learning_rate) 

                        bind_loss.backward() 

                        optimizer_video.step()
                        optimizer_video.zero_grad()

                        if i >= num_warmup_steps_bind:

                            optimizer_audio.step()
                            optimizer_audio.zero_grad()

                        print('audio_latents_temp',audio_latents_temp.dtype,audio_latents_temp.device)
                        print('video_latents_temp',video_latents_temp.dtype,video_latents_temp.device)

                        audio_latents = audio_latents_temp.detach()
                        video_latents = video_latents_temp.detach()



                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, video_latents)

        audio_latents.requires_grad = False
        video_latents.requires_grad = False

        # 8. Post-processing
        # decode audio 
        mel_spectrogram = self.decode_audio_latents(audio_latents)

        audio = self.mel_spectrogram_to_waveform(mel_spectrogram) # [1, 128032]
        audio = audio[:, :original_waveform_length][0] # [1, 128000]
        audio = audio.detach().numpy()
        print('output audio',audio.shape,audio.dtype)
        # decode video
        video = self.decode_video_latents(video_latents).detach()
        video = self.video_processor.postprocess_video(video=video, output_type=output_type)[0]
        print('output video',video.shape,video.dtype)
        # Offload all models

        self.maybe_free_model_hooks()
        if not return_dict:
            return (video,audio,)

        return CogVideoXPipelineOutput(frames=video),AudioPipelineOutput(audios=audio)
