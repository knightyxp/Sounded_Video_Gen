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


import html
import ftfy
import regex as re

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers.models.embeddings import get_3d_rotary_pos_embed

from diffusers import AutoencoderOobleck, FluxTransformer2DModel

from diffusers.utils import is_accelerate_available, logging, replace_example_docstring
# from diffusers.utils import is_accelerate_available, logging, randn_tensor, replace_example_docstring
from audioldm.pipelines.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import AudioPipelineOutput, DiffusionPipeline
from TangoFlux.tangoflux.model import TangoFlux

from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput

from diffusers.models import  AutoencoderKLWan, WanTransformer3DModel
from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.video_processor import VideoProcessor
from diffusers.image_processor import PipelineImageInput

from imagebind_data import load_and_transform_video_data_from_tensor_real,load_and_transform_audio_data_from_waveform, load_and_transform_video_data, load_and_transform_text, load_and_transform_vision_data, waveform2melspec
from imagebind.imagebind.models import imagebind_model
from imagebind.imagebind.models.imagebind_model import ModalityType

import torchaudio

import soundfile as sf
import os

from utils.utils import instantiate_from_config
from omegaconf import OmegaConf
from diffusers.utils import export_to_video, load_video

from transformers import T5EncoderModel, T5TokenizerFast


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


class StableAudioPositionalEmbedding(nn.Module):
    """Used for continuous time
    Adapted from Stable Audio Open.
    """

    def __init__(self, dim: int):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        times = times[..., None]
        freqs = times * self.weights[None] * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((times, fouriered), dim=-1)
        return fouriered


class DurationEmbedder(nn.Module):
    """
    A simple linear projection model to map numbers to a latent space.

    Code is adapted from
    https://github.com/Stability-AI/stable-audio-tools

    Args:
        number_embedding_dim (`int`):
            Dimensionality of the number embeddings.
        min_value (`int`):
            The minimum value of the seconds number conditioning modules.
        max_value (`int`):
            The maximum value of the seconds number conditioning modules
        internal_dim (`int`):
            Dimensionality of the intermediate number hidden states.
    """

    def __init__(
        self,
        number_embedding_dim,
        min_value,
        max_value,
        internal_dim: Optional[int] = 256,
    ):
        super().__init__()
        self.time_positional_embedding = nn.Sequential(
            StableAudioPositionalEmbedding(internal_dim),
            nn.Linear(in_features=internal_dim + 1, out_features=number_embedding_dim),
        )

        self.number_embedding_dim = number_embedding_dim
        self.min_value = min_value
        self.max_value = max_value
        self.dtype = torch.float32

    def forward(
        self,
        floats: torch.Tensor,
    ):
        floats = floats.clamp(self.min_value, self.max_value)

        normalized_floats = (floats - self.min_value) / (
            self.max_value - self.min_value
        )

        # Cast floats to same type as embedder
        embedder_dtype = next(self.time_positional_embedding.parameters()).dtype
        normalized_floats = normalized_floats.to(embedder_dtype)

        embedding = self.time_positional_embedding(normalized_floats)
        float_embeds = embedding.view(-1, 1, self.number_embedding_dim)

        return float_embeds



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


class PseudoDecoder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output


class Audio_Video_LDMPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using Wan.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        tokenizer ([`T5Tokenizer`]):
            Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer),
            specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        text_encoder ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        transformer ([`WanTransformer3DModel`]):
            Conditional Transformer to denoise the input latents.
        scheduler ([`UniPCMultistepScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
    """


    model_cpu_offload_seq = "video_text_encoder->transformer->video_vae"

    def __init__(
        self,
        ##org
        vae: AutoencoderKLWan,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        transformer: WanTransformer3DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        #
        audio_vae: AutoencoderOobleck,
        audio_text_encoder: T5EncoderModel,
        audio_tokenizer: T5TokenizerFast,
        audio_transformer: FluxTransformer2DModel,
        audio_scheduler: FlowMatchEulerDiscreteScheduler,
        audio_fc: nn.Module,
        audio_duration_emebdder: nn.Module,
    ):
        super().__init__()

        # self.audio_max_text_seq_len = 64

        # self.audio_fc = nn.Sequential(
        #     nn.Linear(1024, 1024), nn.ReLU()
        # )
        # self.audio_duration_emebdder = DurationEmbedder(
        #     1024, min_value=0, max_value=30,
        # )

        # self.audio_transformer = FluxTransformer2DModel(
        #     in_channels= 64,
        #     num_layers= 6,
        #     num_single_layers = 18,
        #     attention_head_dim = 128,
        #     num_attention_heads= 8,
        #     joint_attention_dim= 1024,
        #     pooled_projection_dim=1024,
        #     guidance_embeds=False,
        # )
        ##audio backbone config init##
        self.audio_seq_len = 645

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            audio_vae = audio_vae,
            audio_text_encoder = audio_text_encoder,
            audio_tokenizer = audio_tokenizer,
            audio_transformer =  audio_transformer,
            audio_scheduler = audio_scheduler,
            audio_fc = audio_fc,
            audio_duration_emebdder = audio_duration_emebdder,
        )


        # video vae
        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)


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

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds


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


    def encode_audio_prompt(
        self,
        prompt: List[str],
        num_samples_per_prompt,
    ):
        device = self.audio_text_encoder.device
        text_inputs  = self.audio_tokenizer(
            prompt,
            max_length=self.audio_tokenizer.model_max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids, attention_mask = text_inputs.input_ids.to(device), text_inputs.attention_mask.to(
            device
        )

        with torch.no_grad():
            prompt_embeds = self.audio_text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )[0]

        prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        attention_mask = attention_mask.repeat_interleave(num_samples_per_prompt, 0)

        # # get unconditional embeddings for classifier free guidance
        uncond_tokens = [""]

        max_length = prompt_embeds.shape[1]
        uncond_input = self.tokenizer(
            uncond_tokens,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        uncond_input_ids = uncond_input.input_ids.to(device)
        uncond_attention_mask = uncond_input.attention_mask.to(device)

        with torch.no_grad():
            negative_prompt_embeds = self.audio_text_encoder(
                input_ids=uncond_input_ids, attention_mask=uncond_attention_mask
            )[0]

        negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(
            num_samples_per_prompt, 0
        )
        uncond_attention_mask = uncond_attention_mask.repeat_interleave(
            num_samples_per_prompt, 0
        )

        # For classifier free guidance, we need to do two forward passes.
        # We concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        prompt_mask = torch.cat([uncond_attention_mask, attention_mask])
        boolean_prompt_mask = (prompt_mask == 1).to(device)

        return prompt_embeds, boolean_prompt_mask

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
        audio_duration,
        callback_steps,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):

        if audio_duration is None:
            raise ValueError(
                f"`audio_duration` has to be a positive value "
            )

        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

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

    def prepare_video_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def encode_duration(self, duration):
        return self.audio_duration_emebdder(duration)


    def decode_video_latents_bind(self, video_latents):
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

        latents = video_latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean

        video = self.vae.decode(latents, return_dict=False)[0]
        return video

    def decode_audio_latents(self, audio_latents, audio_duration):
        for param in self.audio_vae.parameters():
            param.requires_grad = False

        wave = self.audio_vae.decode(audio_latents.transpose(2, 1)).sample[0]
        waveform_end = int(audio_duration * self.audio_vae.config.sampling_rate)
        wave = wave[:, :waveform_end]
        return wave

    def bind_forward_triple_loss(
        self,
        prompt: Union[str, List[str]] = None,
        audio_duration: Optional[float] = None,
        num_inference_steps: int = 10,
        audio_guidance_scale: float = 3.0,
        video_guidance_scale: float = 5.0,
        learning_rate: float = 0.1,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_samples_per_prompt: Optional[int] = 1,
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
        max_sequence_length: int = 512,
    ):

        # device setting
        bind_device = torch.device("cuda:0")
        audio_device = torch.device("cuda:0")
        video_device = torch.device("cuda:1")

        # 0. TangoFLux initial setting

        if not isinstance(prompt, list):
            prompt = [prompt]
        if not isinstance(audio_duration, torch.Tensor):
            audio_duration = torch.tensor([audio_duration], device=audio_device)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            audio_duration,
            callback_steps,
            height,
            width,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        ### some setting for wanx
        self._video_guidance_scale = video_guidance_scale
        self._video_attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        ## set audio pipe to cuda 0
        self.audio_transformer = self.audio_transformer.to(audio_device)
        self.audio_text_encoder = self.audio_text_encoder.to(audio_device)
        self.audio_vae = self.audio_vae.to(audio_device)
        self.audio_fc = self.audio_fc.to(audio_device)  
        self.audio_duration_emebdder= self.audio_duration_emebdder.to(audio_device)  

        ## set video pipe to cuda 1
        self.vae = self.vae.to(video_device)
        self.transformer = self.transformer.to(video_device)
        self.text_encoder = self.text_encoder.to(video_device)


        do_classifier_free_guidance = audio_guidance_scale > 1.0 and video_guidance_scale >1.0

        #set dtype
        latents_dtype = torch.float32
        video_latents_dtype = torch.bfloat16
        
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        ### start of audio backbone tango setting
        ### prepare audio text embedding and duration

        audio_duration_hidden_states = self.encode_duration(audio_duration)

        if do_classifier_free_guidance:
            bsz = 2 * num_samples_per_prompt

            audio_encoder_hidden_states, audio_boolean_encoder_mask = (
                self.encode_audio_prompt(
                    prompt, num_samples_per_prompt=num_samples_per_prompt,
                )
            )
            audio_duration_hidden_states = audio_duration_hidden_states.repeat(bsz, 1, 1)

        ## process audio mask
        audio_mask_expanded = audio_boolean_encoder_mask.unsqueeze(-1).expand_as(
            audio_encoder_hidden_states
        )
        audio_masked_data = torch.where(
            audio_mask_expanded, audio_encoder_hidden_states, torch.tensor(float("nan"))
        )

        audio_pooled = torch.nanmean(audio_masked_data, dim=1)
        audio_pooled_projection = self.audio_fc(audio_pooled)

        audio_encoder_hidden_states = torch.cat([audio_encoder_hidden_states, audio_duration_hidden_states], dim=1)  ## (bs,seq_len,dim)
        audio_encoder_hidden_states = audio_encoder_hidden_states.to(audio_device)

        ## prepare audio latents
        audio_latents = torch.randn(num_samples_per_prompt, self.audio_seq_len, 64).to(audio_device)

        ## prepare tangoflux input
        txt_ids = torch.zeros(bsz, audio_encoder_hidden_states.shape[1], 3).to(audio_device)
        audio_ids = (
            torch.arange(self.audio_seq_len)
            .unsqueeze(0)
            .unsqueeze(-1)
            .repeat(bsz, 1, 3)
            .to(audio_device)
        )

        ### end of audio backbone: tango setting

        print('audio_device',audio_device,'video_device',video_device,"bind_device",bind_device)

        video_prompt_embeds, video_negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=1,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            device=video_device,
            dtype =video_latents_dtype,
        )

        video_prompt_embeds = video_prompt_embeds.to(video_latents_dtype)
        if video_negative_prompt_embeds is not None:
            video_negative_prompt_embeds = video_negative_prompt_embeds.to(video_latents_dtype)


        # 4. Prepare timesteps
        ### prepare audio scheduler timesteps
        self.audio_scheduler.set_timesteps(num_inference_steps, device=audio_device)
        ## video scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=video_device)
        timesteps = self.scheduler.timesteps


        # 5. Prepare latent variables

        num_channels_latents = self.transformer.config.in_channels

        #prepare video latents

        video_latents = self.prepare_video_latents(
            batch_size * num_samples_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            video_latents_dtype ,
            video_device,
            generator,
            latents,
        ).to(video_device)


        print('video_latents',video_latents.device,"video transformer device",self.transformer.device, self.transformer.dtype, 'video vae dtype', self.vae.dtype)
        print('video_latents',video_latents.shape,video_latents.dtype)

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        #image_bind_video_input = load_and_transform_video_data(video_paths, device, clip_duration=clip_duration, clips_per_video=clips_per_video, n_samples_per_clip=2)
        #print('image_bind_video_input',image_bind_video_input.shape)

        bind_model = imagebind_model.imagebind_huge(pretrained=False)

        state_dict = torch.load("imagebind/.checkpoints/imagebind_huge.pth", map_location=bind_device)

        bind_model.eval()
        bind_model = bind_model.to(dtype=torch.float32,device=bind_device)

        for p in bind_model.parameters():
            p.requires_grad = False
        

        print('attention_kwargs',attention_kwargs)

        self.audio_vae.enable_slicing()
        self.vae.eval()
        ## cpu off load


        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        num_warmup_steps_bind = int(len(timesteps) * optimization_starting_point)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                #===================audio denoising======================#
                # expand the audio latents if we are doing classifier free guidance
                audio_latent_model_input = torch.cat([audio_latents] * 2) if do_classifier_free_guidance else audio_latents
                
                # predict the audio noise residual
                with torch.no_grad():
                    audio_noise_pred = self.audio_transformer(
                        hidden_states = audio_latent_model_input,
                        timestep = torch.tensor([t / 1000], device=audio_device),
                        guidance = None,
                        pooled_projections = audio_pooled_projection,
                        encoder_hidden_states=audio_encoder_hidden_states,
                        txt_ids=txt_ids,
                        img_ids=audio_ids,
                        return_dict=False,
                    )[0].to(dtype=latents_dtype)

                # perform guidance
                if do_classifier_free_guidance:
                    audio_noise_pred_uncond, audio_noise_pred_text = audio_noise_pred.chunk(2)
                    audio_noise_pred = audio_noise_pred_uncond + audio_guidance_scale * (audio_noise_pred_text - audio_noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                t_audio_time_step = t.to(audio_device)

                audio_latents = self.audio_scheduler.step(audio_noise_pred, t_audio_time_step, audio_latents).prev_sample   


                #===================audio denoising======================#

                #===================video denoising======================#
                self._current_timestep = t
                timestep = t.expand(video_latents.shape[0])

                video_latent_model_input = video_latents.to(video_latents_dtype)


                #print('video_latent_model_input',video_latent_model_input.shape)
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(video_latent_model_input.shape[0]).to(video_device)
                with torch.no_grad():
                    # predict noise model_output
                    video_noise_pred = self.transformer(
                        hidden_states=video_latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=video_prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0].to(video_device)
                    #video_noise_pred = video_noise_pred.float()

                    if do_classifier_free_guidance:
                        video_noise_uncond = self.transformer(
                            hidden_states=video_latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=video_negative_prompt_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0].to(video_device)

                        video_noise_pred = video_noise_uncond + self._video_guidance_scale * (video_noise_pred - video_noise_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                t = t.to(video_device)
                print(video_noise_pred.device, t.device, video_latents.device)
                
                video_latents = self.scheduler.step(video_noise_pred, t, video_latents, return_dict=False)[0]

                video_latents = video_latents.to(video_latents_dtype)

                #===================video denoising======================#

                #===================joint va triangle loss optimization======================#

                for optim_step in range(num_optimization_steps):
                    with torch.autograd.set_detect_anomaly(True):
                        video_latents_temp = video_latents
                        #video_latents_temp = video_latents.detach()
                        video_latents_temp.requires_grad = True 

                        audio_latents_temp = audio_latents
                        # audio_latents_temp = audio_latents.detach()
                        audio_latents_temp.requires_grad = True 

                        #================compute audio=================#
                        # 1. compute x0 
                        # 使用当前步对应的 sigma
                        sigma_audio = self.audio_scheduler.sigmas[i]  # 这里 i 对应当前时间步
                        # 直接利用闭式公式恢复 x₀（注意：这里假设 audio_noise_pred 是当前步的预测流动）
                        audio_x0 = audio_latents_temp - sigma_audio * audio_noise_pred
                        print('audio_x0', audio_x0.shape, audio_x0.dtype, audio_x0.device)
                        # 2. decode x0 with vae decoder 

                        # x0_wave = self.decode_audio_latents(audio_x0, audio_duration)
                        x0_wave = PseudoDecoder.apply(self.decode_audio_latents(audio_x0, audio_duration))
                        x0_waveform = x0_wave.to(torch.float32)

                        # 4. waveform to imagebind mel-spectrogram
                        x0_imagebind_audio_input = load_and_transform_audio_data_from_waveform(x0_waveform, org_sample_rate=self.audio_vae.config.sampling_rate, 
                                            device=bind_device, target_length=204, clip_duration=clip_duration, clips_per_video=clips_per_video).to(torch.float32)

                        del x0_waveform, audio_x0
                        #================compute audio=================#

                        #================compute video=================#
                        with torch.no_grad():
                            sigma_video = self.scheduler.sigmas[i]  # 当前视频步的 sigma
                            video_x0 = video_latents_temp - sigma_video * video_noise_pred
                            # 2. decode video x0 with video vae decoder 
                            x0_video = PseudoDecoder.apply(self.decode_video_latents(video_x0))
                            #x0_video = self.decode_video_latents(video_x0).to(dtype=torch.float32, device=bind_device)
                            x0_video = (x0_video  / 2 + 0.5).clamp(0, 1)
                        # # 3. x0_video to frames to export 
                        #video = self.video_processor.postprocess_video(video=video, output_type=output_type)
                        # export_to_video(video,f'output_{t}.mp4',fps=10)
                        # # 4. reload video to image bind video input
                        x0_imagebind_video_input = load_and_transform_video_data_from_tensor_real(x0_video, bind_device, clip_duration=clip_duration, clips_per_video=clips_per_video, n_samples_per_clip=2).to(torch.float32) 
                        #================compute video=================#

                        print(x0_imagebind_video_input.dtype, x0_imagebind_audio_input.dtype)

                        # compute loss with imagebind 
                        if isinstance(prompt, str):
                            prompt_bind = [prompt]
                        else:
                            #print('prompt is list',isinstance(prompt,list))
                            prompt_bind = prompt
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
                        print('bind_loss_text_vision',bind_loss_text_vision,bind_loss_text_vision.requires_grad)
                        print('bind_loss_text_audio',bind_loss_text_audio,bind_loss_text_audio.requires_grad)
                        print('bind_loss_vision_audio',bind_loss_vision_audio,bind_loss_vision_audio.requires_grad)           

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

        del video_latents_temp,audio_latents_temp
        # torch.cuda.set_device(1)
        self._current_timestep = None
        torch.cuda.empty_cache()
        audio_latents.requires_grad = False
        video_latents.requires_grad = False

        # 8. Post-processing
        # decode audio 

        audio = self.decode_audio_latents(audio_latents,audio_duration=audio_duration)
        print('output audio',audio.shape,audio.dtype)
        audio = audio.detach().cpu()
        # decode video
        with torch.autograd.set_detect_anomaly(True):
            with torch.no_grad():
                video = self.decode_video_latents(video_latents).detach()
                video = self.video_processor.postprocess_video(video=video, output_type=output_type)[0]
        print('output video',video.shape,video.dtype)
        # Offload all models

        self.maybe_free_model_hooks()
        if not return_dict:
            return (video,audio,)

        return video,audio
