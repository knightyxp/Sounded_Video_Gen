# Adapted from https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/pipelines/pipeline_tuneavideo.py

import inspect
from typing import Callable, List, Optional, Union
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer
from contextlib import nullcontext

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput

from einops import rearrange

from ..models.unet import UNet3DConditionModel
import torch.nn.functional as F
# from .imagebind_data import load_and_transform_video_data_from_tensor 
# from imagebind.imagebind.models.imagebind_model import ModalityType

from .imagebind_data import load_and_transform_audio_data, load_and_transform_text, load_and_transform_video_data_from_tensor_real
from imagebind.imagebind.models import imagebind_model
from imagebind.imagebind.models.imagebind_model import ModalityType

# from DiffFoley.diff_foley.util import instantiate_from_config
# from DiffFoley.inference.demo_util import Extract_CAVP_Features 
import torchaudio

import librosa


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name




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



@dataclass
class AnimationPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class AnimationPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)


    @property
    def _execution_device(self):
        return torch.device('cuda:0')
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        # print(f'_execution_device={self.device}')
        return self.device

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        device = torch.device('cuda:0')
        self.text_encoder = self.text_encoder.to(device)
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None
        try:
            text_embeddings = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
        except:
            import pdb;pdb.set_trace()
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
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

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        # print(f'text_embeddings={text_embeddings.device}')
        return text_embeddings

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # print('latents.shape', latents.shape)
        # video = self.vae.decode(latents).sample
        
        latents_device = latents.device
        if latents_device != self.vae.device:
            multi_gpu = True
        else:
            multi_gpu = False

        if multi_gpu:
            latents = latents.to(self.vae.device)

        video = []
        # for frame_idx in tqdm(range(latents.shape[0])):
        for frame_idx in range(latents.shape[0]):
            video.append(self.vae.decode(latents[frame_idx:frame_idx+1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        if video.requires_grad == True:
            video = video.detach()
        video = video.cpu().float().numpy()
        return video

    def decode_latents_bind(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        latents_device = latents.device
        if latents_device != self.vae.device:
            multi_gpu = True
        else:
            multi_gpu = False

        if multi_gpu:
            latents = latents.to(self.vae.device)

        # print('latents.shape', latents.shape)
        # video = self.vae.decode(latents).sample
        video = []
        # for frame_idx in tqdm(range(latents.shape[0])):
        for frame_idx in range(latents.shape[0]):
            video.append(self.vae.decode(latents[frame_idx:frame_idx+1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        # video = video.cpu().float().numpy()
        return video

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

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = shape
                # shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        video_length: Optional[int],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # Define call parameters
        # batch_size = 1 if isinstance(prompt, str) else len(prompt)
        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
        if negative_prompt is not None:
            negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size 
        text_embeddings = self._encode_prompt(
            prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )
        latents_dtype = latents.dtype
        # print('latent shape: ', latents.shape) # [1, 4, 16, 64, 64]

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                # import pdb;pdb.set_trace()
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample.to(dtype=latents_dtype)
                # noise_pred = []
                # import pdb
                # pdb.set_trace()
                # for batch_idx in range(latent_model_input.shape[0]):
                #     noise_pred_single = self.unet(latent_model_input[batch_idx:batch_idx+1], t, encoder_hidden_states=text_embeddings[batch_idx:batch_idx+1]).sample.to(dtype=latents_dtype)
                #     noise_pred.append(noise_pred_single)
                # noise_pred = torch.cat(noise_pred)

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # Post-processing
        video = self.decode_latents(latents)

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return AnimationPipelineOutput(videos=video)

    def bind_prepare(
        self,
        prompt: Union[str, List[str]],
        video_length: Optional[int],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # Define call parameters
        # batch_size = 1 if isinstance(prompt, str) else len(prompt)
        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
        if negative_prompt is not None:
            negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size 
        text_embeddings = self._encode_prompt(
            prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # print('text_embeddings.dtype: ', text_embeddings.dtype)
        # exit()

        # Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )
        latents_dtype = latents.dtype
        # print('latent shape: ', latents.shape) # [1, 4, 16, 64, 64]

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        out_dict = {
            "device": device,
            "do_classifier_free_guidance": do_classifier_free_guidance,
            "text_embeddings": text_embeddings,
            "timesteps": timesteps,
            # "latents": latents,
            "latents_dtype": latents_dtype,
            "extra_step_kwargs": extra_step_kwargs,
        }

        for p in self.vae.parameters():
            # print('p: ', p)
            p.requires_grad = False

        return out_dict, latents

    @torch.no_grad()
    def bind_step(
        self,
        device,
        do_classifier_free_guidance,
        text_embeddings,
        timesteps,
        latents,
        latents_dtype,
        extra_step_kwargs,
        cur_step,


        # prompt: Union[str, List[str]],
        # video_length: Optional[int],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        # **kwargs,
    ):
        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        num_warmup_steps_bind = int(len(timesteps) * 0.2)
        # with self.progress_bar(total=num_inference_steps) as progress_bar:
        # for i, t in enumerate(timesteps):
        

        t = timesteps[cur_step]

        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample.to(dtype=latents_dtype)

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        # # compute x0
        # latents_temp = latents.detach()
        # # if latents.is_leaf:
        # latents_temp.requires_grad = True

        # optimizer = torch.optim.Adam([latents_temp], lr=learning_rate)

        # if cur_step > num_warmup_steps_bind:
        #     for optim_step in range(1):
        #         with torch.autograd.set_detect_anomaly(True):
        #             # 1. compute x0 
        #             x0 = 1/(self.scheduler.alphas_cumprod[t] ** 0.5) * (latents_temp - (1-self.scheduler.alphas_cumprod[t])**0.5 * noise_pred) 
                    
        #             # 2. decode x0 with vae decoder 
        #             x0_video = self.decode_latents(x0) # TODO check the shape of it 
        #             print('x0_video: ', x0_video.requires_grad, x0_video.shape) # [1, 1, 16000]

        #             # if x0_mel_spectrogram.dim() == 4:
        #             #     x0_mel_spectrogram = x0_mel_spectrogram.squeeze(1)

        #             # 3. convert mel-spectrogram to waveform
        #             # x0_waveform = self.vocoder(x0_mel_spectrogram) # TODO save this [1, 128032]

        #             # print('x0_waveform: ', x0_waveform.requires_grad) # [1, 1, 16000]

        #             # save the intermediate results 
        #             # x0_waveform_save = x0_waveform[:, :original_waveform_length] 
        #             # x0_waveform_save = x0_waveform_save.detach().cpu().numpy()
        #             # print('x0 type: ', type(x0_waveform_save))
        #             # sf.write(f"intermediate/{prompt}_x0_waveform_{i:02d}.wav", x0_waveform_save[0], samplerate=16000)
        #             # sf.write(os.path.join('intermediate', video_name+prompt, f'x0_waveform_{i:02d}_{num_inference_steps}_{learning_rate}.wav'), x0_waveform_save[0], samplerate=16000)


        #             # 4. waveform to imagebind mel-spectrogram
        #             # x0_imagebind_audio_input = load_and_transform_audio_data_from_waveform(x0_waveform, org_sample_rate=self.vocoder.config.sampling_rate, 
        #                                 # device=device, target_length=204, clip_duration=clip_duration, clips_per_video=clips_per_video)
        #             x0_imagebind_video_input = load_and_transform_video_data_from_tensor(x0_video, device, 
        #                         clip_duration=clip_duration, clips_per_video=clips_per_video, 
        #                         n_samples_per_clip=2)
                    
        #             # compute loss with imagebind 
        #             inputs = {
        #                 ModalityType.VISION: x0_imagebind_video_input,
        #                 ModalityType.AUDIO: image_bind_audio_input,
        #             }

        #             print('inputs: ', inputs[ModalityType.VISION].requires_grad, inputs[ModalityType.AUDIO].requires_grad)
        #             print('inputs shape: ', inputs[ModalityType.VISION].shape, inputs[ModalityType.AUDIO].shape)
        #             # with torch.no_grad():
        #             embeddings = bind_model(inputs)
        #             print('embeddings: ', embeddings[ModalityType.VISION].requires_grad, embeddings[ModalityType.AUDIO].requires_grad)

        #             # bind_loss = torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=-1)
        #             # bind_loss = torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T)
        #             bind_loss = 1 - F.cosine_similarity(embeddings[ModalityType.VISION], embeddings[ModalityType.AUDIO])
            
        #             # similarity_recoder.write(str(bind_loss)+'\n')

        #             bind_loss.backward() 
        #             optimizer.step()
        #             optimizer.zero_grad()

        # latents = latents_temp.detach()
        # call the callback, if provided
        if cur_step == len(timesteps) - 1 or ((cur_step + 1) > num_warmup_steps and (cur_step + 1) % self.scheduler.order == 0):
            # progress_bar.update()
            if callback is not None and cur_step % callback_steps == 0:
                callback(cur_step, t, latents)
    
        return latents, noise_pred

    def bind_finish(
        self,
        latents,
        return_dict: bool = True,
        output_type: Optional[str] = "tensor",
    ):
        # Post-processing
        video = self.decode_latents(latents)

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return AnimationPipelineOutput(videos=video)

    def xt2x0(
        self,
        latents_temp,
        noise_pred,
        timesteps,
        cur_step,        
    ):

        t = timesteps[cur_step]

        x0 = 1/(self.scheduler.alphas_cumprod[t] ** 0.5) * (latents_temp - (1-self.scheduler.alphas_cumprod[t])**0.5 * noise_pred) 

        x0_out = self.decode_latents_bind(x0)

        return x0_out


    def bind_forward(
        self,
        prompt: Union[str, List[str]],
        video_length: Optional[int],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        
        learning_rate: float = 0.1,
        clip_duration: float = 2.0,
        clips_per_video: int = 5,
        num_optimization_steps: int = 1,
        use_imagebind_lora: bool = False,
        audio_paths: Union[str, List[str]] = None,

        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        with_text_loss: bool = False,
        optimization_starting_point: float = 0.2,
        optimization_ending_point: float = 1,
        all_frames_loss: bool = False,
        bind_device: str = "cuda:1",
        n_samples_per_clip: int = 2,
        **kwargs,
    ):
        """
        Optimize latent xt based on ImageBind guidance
        """
        # get latent height and width
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # get batch size
        # batch_size = 1 if isinstance(prompt, str) else len(prompt)
        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
        if negative_prompt is not None:
            negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size 
        text_embeddings = self._encode_prompt(
            prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )
        latents_dtype = latents.dtype
        # print('latent shape: ', latents.shape) # [1, 4, 16, 64, 64]

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # define imagebind model
        bind_model = imagebind_model.imagebind_huge(pretrained=True)
        bind_model.eval()
        bind_model.to(bind_device)
        for p in bind_model.parameters():
            p.requires_grad = False
        
        # Get audio data
        image_bind_audio_input = load_and_transform_audio_data(audio_paths, device=bind_device, 
                                                        target_length=204, clip_duration=clip_duration, 
                                                        clips_per_video=clips_per_video)

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        num_warmup_steps_bind = int(len(timesteps) * optimization_starting_point)
        num_end_steps = int(len(timesteps) * optimization_ending_point)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # Denoising loop
            for i, t in enumerate(timesteps):
                
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input, 
                        t, 
                        encoder_hidden_states=text_embeddings
                    ).sample.to(dtype=latents_dtype)
                
                # perform cfg guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # Start ImageBind guidance
                latents_temp = latents.detach()
                latents_temp.requires_grad = True

                # Optimize latents_temp
                optimizer = torch.optim.Adam([latents_temp], lr=learning_rate)
                
                if num_end_steps >= i > num_warmup_steps_bind: # Skip very noisy timesteps
                    for optim_step in range(num_optimization_steps):
                        with torch.autograd.set_detect_anomaly(True):
                            
                            # compute x0 of generated video
                            x0_video = self.xt2x0(latents_temp, timesteps=timesteps, cur_step=i, noise_pred=noise_pred) 
                            bind_loss = self.cal_ImageBind_loss(
                                bind_model, x0_video, image_bind_audio_input, bind_device, clip_duration, clips_per_video, 
                                with_text_loss=with_text_loss, prompt=prompt, all_frames_loss=all_frames_loss,
                                n_samples_per_clip=n_samples_per_clip,
                            )
                            # backpropagate & update latent
                            bind_loss.backward() 
                            optimizer.step()
                            optimizer.zero_grad()

                latents = latents_temp.detach()

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # Post-processing
        video = self.decode_latents(latents)

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return AnimationPipelineOutput(videos=video)
    
    def cal_ImageBind_loss(self, bind_model, x0_video, image_bind_audio_input, bind_device, clip_duration, clips_per_video, 
                           with_text_loss=False, prompt=None, all_frames_loss=False, n_samples_per_clip=2):
        
        def make_bind_input(x0_imegebind_video_input, prompt):
            if with_text_loss:
                if isinstance(prompt, str):
                    prompt = [prompt]
                inputs = {
                    ModalityType.VISION: x0_imegebind_video_input,
                    ModalityType.AUDIO: image_bind_audio_input,
                    ModalityType.TEXT: load_and_transform_text(prompt, bind_device)
                }
            else:
                # forward imagebind 
                inputs = {
                    ModalityType.VISION: x0_imegebind_video_input,
                    ModalityType.AUDIO: image_bind_audio_input,
                }
            return inputs
        
        def cal_loss(embeddings):
            # calculate loss
            if with_text_loss:
                bind_loss_text_audio = 1 - F.cosine_similarity(embeddings[ModalityType.TEXT], embeddings[ModalityType.AUDIO])
                bind_loss_vision_audio = 1 - F.cosine_similarity(embeddings[ModalityType.VISION], embeddings[ModalityType.AUDIO])
                bind_loss = bind_loss_text_audio + bind_loss_vision_audio
            else:
                bind_loss = 1 - F.cosine_similarity(embeddings[ModalityType.VISION], embeddings[ModalityType.AUDIO])
            return bind_loss
        
        if all_frames_loss:
            bind_loss = 0
            for i in range(x0_video.shape[2]-1):
                x0_video_clip = x0_video[:,:, i:i+n_samples_per_clip, :, :]
                # transform the generated video
                x0_imegebind_video_input = load_and_transform_video_data_from_tensor_real(
                    x0_video_clip, bind_device, clip_duration=clip_duration, clips_per_video=clips_per_video, 
                    n_samples_per_clip=n_samples_per_clip
                )
                inputs = make_bind_input(x0_imegebind_video_input, prompt)
                embeddings = bind_model(inputs)
                bind_loss += cal_loss(embeddings)
        else:
            # transform the generated video
            x0_imegebind_video_input = load_and_transform_video_data_from_tensor_real(
                x0_video, bind_device, clip_duration=clip_duration, clips_per_video=clips_per_video, 
                n_samples_per_clip=n_samples_per_clip
            )
            inputs = make_bind_input(x0_imegebind_video_input, prompt)
            embeddings = bind_model(inputs)
            bind_loss = cal_loss(embeddings)
        return bind_loss

    def bind_forward_optim_cond_embd(
        self,
        prompt: Union[str, List[str]],
        video_length: Optional[int],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        
        learning_rate: float = 0.1,
        clip_duration: float = 2.0,
        clips_per_video: int = 5,
        optimization_starting_point: float = 0.2,
        optimization_ending_point: float = 1,
        num_optimization_steps: int = 1,
        use_imagebind_lora: bool = False,
        audio_paths: Union[str, List[str]] = None,

        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        with_text_loss: bool = False,
        optim_mode: str = "both",
        all_frames_loss: bool = False,
        bind_device: str = "cuda:1",
        norm_embed: bool = False,
        n_samples_per_clip: int = 2,
        **kwargs,
    ):
        """
        Optimize cond embedding based on ImageBind guidance
        """
        # get latent height and width
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # get batch size
        # batch_size = 1 if isinstance(prompt, str) else len(prompt)
        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
        if negative_prompt is not None:
            negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size 
        text_embeddings = self._encode_prompt(
            prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
        )
        # detach for optimization
        text_embeddings = text_embeddings.detach()
        embed_scale = text_embeddings.norm().item()

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )
        latents_dtype = latents.dtype
        # print('latent shape: ', latents.shape) # [1, 4, 16, 64, 64]

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # define imagebind model
        bind_model = imagebind_model.imagebind_huge(pretrained=True)
        bind_model.eval()
        bind_model.to(bind_device)
        for p in bind_model.parameters():
            p.requires_grad = False
        
        # Get audio data
        image_bind_audio_input = load_and_transform_audio_data(audio_paths, device=bind_device, 
                                                        target_length=204, clip_duration=clip_duration, 
                                                        clips_per_video=clips_per_video)

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        num_warmup_steps_bind = int(len(timesteps) * optimization_starting_point)
        num_end_steps = int(len(timesteps) * optimization_ending_point)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # Denoising loop
            for i, t in enumerate(timesteps):
                for optim_step in range(num_optimization_steps):
                    if optim_mode == "both":
                        text_embeddings = text_embeddings.detach()
                        text_embeddings.requires_grad = True
                    elif optim_mode == "only_cond":
                        text_embeddings = text_embeddings.detach()
                        uncond_embeddings, cond_embeddings = text_embeddings[0], text_embeddings[1]
                        cond_embeddings.requires_grad = True
                        uncond_embeddings.requires_grad = False
                        text_embeddings = torch.stack([uncond_embeddings, cond_embeddings], dim=0)
                    elif optim_mode == "only_uncond":
                        text_embeddings = text_embeddings.detach()
                        uncond_embeddings, cond_embeddings = text_embeddings[0], text_embeddings[1]
                        cond_embeddings.requires_grad = False
                        uncond_embeddings.requires_grad = True
                        text_embeddings = torch.stack([uncond_embeddings, cond_embeddings], dim=0)
                    else:
                        raise ValueError

                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    latent_model_input = latent_model_input.detach()
                    latent_model_input.requires_grad = False

                    context = torch.no_grad() if i <= num_warmup_steps_bind else nullcontext()

                    # predict the noise
                    with context:
                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings
                                            ).sample.to(dtype=latents_dtype)
                    
                    # perform cfg guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # Optimize text_embeddings
                    if optim_mode == "both":
                        optimizer = torch.optim.Adam([text_embeddings], lr=learning_rate)
                    elif optim_mode == "only_cond":
                        optimizer = torch.optim.Adam([cond_embeddings], lr=learning_rate)
                    elif optim_mode == "only_uncond":
                        optimizer = torch.optim.Adam([uncond_embeddings], lr=learning_rate)
                    else:
                        raise ValueError

                    if num_end_steps >= i > num_warmup_steps_bind: # Skip very noisy timesteps
                        with torch.autograd.set_detect_anomaly(True):
                            # 1. compute x0 
                            x0_video = self.xt2x0(latents, timesteps=timesteps, cur_step=i, noise_pred=noise_pred) 
                            bind_loss = self.cal_ImageBind_loss(
                                bind_model, x0_video, image_bind_audio_input, bind_device, clip_duration, clips_per_video, 
                                with_text_loss=with_text_loss, prompt=prompt, all_frames_loss=all_frames_loss,
                                n_samples_per_clip=n_samples_per_clip,
                            )
                            
                            # backpropagate & update latent
                            bind_loss.backward()
                            if False:
                                if optim_mode == "both":
                                    print('grad of target embeddings: ', text_embeddings.grad.shape)
                                elif optim_mode == "only_cond":
                                    print('grad of target embeddings: ', cond_embeddings.grad.shape)
                                elif optim_mode == "only_uncond":
                                    print('grad of target embeddings: ', uncond_embeddings.grad.shape)
                                else:
                                    raise ValueError

                            optimizer.step()
                            optimizer.zero_grad()
                            if norm_embed:
                                if optim_mode == "both":
                                    text_embeddings = text_embeddings / text_embeddings.norm() * embed_scale
                                else:
                                    raise ValueError


                        if optim_mode == "only_cond" or optim_mode == "only_uncond":
                            text_embeddings = torch.stack([uncond_embeddings, cond_embeddings], dim=0)
                    latents = latents.detach()
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # Post-processing
        video = self.decode_latents(latents)

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return AnimationPipelineOutput(videos=video)

    
    # def cal_DiffFoley_loss(self, x0_video, cavp_audio_feat):
    #     # extract video feature 

    #     return loss

    # def bind_forward_optim_cond_embd_diff_foley(
    #     self,
    #     prompt: Union[str, List[str]],
    #     video_length: Optional[int],
    #     height: Optional[int] = None,
    #     width: Optional[int] = None,
    #     num_inference_steps: int = 50,
    #     guidance_scale: float = 7.5,
        
    #     learning_rate: float = 0.1,
    #     clip_duration: float = 2.0,
    #     clips_per_video: int = 5,
    #     optimization_starting_point: float = 0.2,
    #     optimization_ending_point: float = 1,
    #     num_optimization_steps: int = 1,
    #     use_imagebind_lora: bool = False,
    #     audio_paths: Union[str, List[str]] = None,

    #     negative_prompt: Optional[Union[str, List[str]]] = None,
    #     num_videos_per_prompt: Optional[int] = 1,
    #     eta: float = 0.0,
    #     generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    #     latents: Optional[torch.FloatTensor] = None,
    #     output_type: Optional[str] = "tensor",
    #     return_dict: bool = True,
    #     callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    #     callback_steps: Optional[int] = 1,
    #     with_text_loss: bool = False,
    #     optim_mode: str = "both",
    #     all_frames_loss: bool = False,
    #     bind_device: str = "cuda:1",
    #     norm_embed: bool = False,
    #     n_samples_per_clip: int = 2,
    #     **kwargs,
    # ):
    #     """
    #     Optimize cond embedding based on ImageBind guidance
    #     """
    #     # get latent height and width
    #     height = height or self.unet.config.sample_size * self.vae_scale_factor
    #     width = width or self.unet.config.sample_size * self.vae_scale_factor

    #     # Check inputs. Raise error if not correct
    #     self.check_inputs(prompt, height, width, callback_steps)

    #     # get batch size
    #     # batch_size = 1 if isinstance(prompt, str) else len(prompt)
    #     batch_size = 1
    #     if latents is not None:
    #         batch_size = latents.shape[0]
    #     if isinstance(prompt, list):
    #         batch_size = len(prompt)

    #     device = self._execution_device

    #     # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    #     # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    #     # corresponds to doing no classifier free guidance.
    #     do_classifier_free_guidance = guidance_scale > 1.0

    #     # Encode input prompt
    #     prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
    #     if negative_prompt is not None:
    #         negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size 
    #     text_embeddings = self._encode_prompt(
    #         prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
    #     )
    #     # detach for optimization
    #     text_embeddings = text_embeddings.detach()
    #     embed_scale = text_embeddings.norm().item()

    #     # Prepare timesteps
    #     self.scheduler.set_timesteps(num_inference_steps, device=device)
    #     timesteps = self.scheduler.timesteps

    #     # Prepare latent variables
    #     num_channels_latents = self.unet.in_channels
    #     latents = self.prepare_latents(
    #         batch_size * num_videos_per_prompt,
    #         num_channels_latents,
    #         video_length,
    #         height,
    #         width,
    #         text_embeddings.dtype,
    #         device,
    #         generator,
    #         latents,
    #     )
    #     latents_dtype = latents.dtype
    #     # print('latent shape: ', latents.shape) # [1, 4, 16, 64, 64]

    #     # Prepare extra step kwargs.
    #     extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    #     # define imagebind model
    #     bind_model = imagebind_model.imagebind_huge(pretrained=True)
    #     bind_model.eval()
    #     bind_model.to(bind_device)
    #     for p in bind_model.parameters():
    #         p.requires_grad = False
        
    #     # Get audio data
    #     image_bind_audio_input = load_and_transform_audio_data(audio_paths, device=bind_device, 
    #                                                     target_length=204, clip_duration=clip_duration, 
    #                                                     clips_per_video=clips_per_video)

    #     # define diff foley model 
    #     fps = 4                                                     #  CAVP default FPS=4, Don't change it.
    #     batch_size = 16    
    #     cavp_config_path = "/home/yazhou/disk1/projects/edit/AudioLDM/DiffFoley/inference/config/Stage1_CAVP.yaml"  #  CAVP Config
    #     cavp_ckpt_path = "/home/yazhou/disk1/projects/edit/AudioLDM/DiffFoley/inference/diff_foley_ckpt/cavp_epoch66.ckpt"      #  CAVP Ckpt
    #     extract_cavp = Extract_CAVP_Features(fps=fps, 
    #                     batch_size=batch_size, device=bind_device, 
    #                     config_path=cavp_config_path, ckpt_path=cavp_ckpt_path)

    #     print('bind_device: ', bind_device)
    #     # # extract video feature with cavp:
    #     # start_second = 0
    #     # truncate_second = 4.0
    #     # tmp_path = "./generate_samples/temp_folder" 
    #     # video_feat_cavp, video_path_high_fps = extract_cavp.video_forward(video_paths[0], start_second, truncate_second, tmp_path=tmp_path)
        
    #     # extract audio feature with cavp
    #     # 1. load wav and convert to mel-spectrogram
    #     # 2. extract audio feature with cavp
    #     audio_transform_pipeline = MyPipeline()
    #     audio_transform_pipeline.to(device=torch.device("cuda"))

    #     audio_wav = librosa.load(audio_paths[0], sr=16000)[0]
    #     audio_wav = torch.from_numpy(audio_wav).to(device).unsqueeze(0)
    #     mel = audio_transform_pipeline(audio_wav)
    #     with torch.no_grad():
    #         audio_feat_cavp = extract_cavp.audio_forward(mel)
    #     print('audio_wav shape: ', audio_wav.shape, 'audio_feat_cavp shape: ', audio_feat_cavp.shape)  
            
        
    #     # Denoising loop
    #     num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    #     num_warmup_steps_bind = int(len(timesteps) * optimization_starting_point)
    #     num_end_steps = int(len(timesteps) * optimization_ending_point)

    #     with self.progress_bar(total=num_inference_steps) as progress_bar:
    #         # Denoising loop
    #         for i, t in enumerate(timesteps):
    #             for optim_step in range(num_optimization_steps):
    #                 if optim_mode == "both":
    #                     text_embeddings = text_embeddings.detach()
    #                     text_embeddings.requires_grad = True
    #                 elif optim_mode == "only_cond":
    #                     text_embeddings = text_embeddings.detach()
    #                     uncond_embeddings, cond_embeddings = text_embeddings[0], text_embeddings[1]
    #                     cond_embeddings.requires_grad = True
    #                     uncond_embeddings.requires_grad = False
    #                     text_embeddings = torch.stack([uncond_embeddings, cond_embeddings], dim=0)
    #                 elif optim_mode == "only_uncond":
    #                     text_embeddings = text_embeddings.detach()
    #                     uncond_embeddings, cond_embeddings = text_embeddings[0], text_embeddings[1]
    #                     cond_embeddings.requires_grad = False
    #                     uncond_embeddings.requires_grad = True
    #                     text_embeddings = torch.stack([uncond_embeddings, cond_embeddings], dim=0)
    #                 else:
    #                     raise ValueError

    #                 latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    #                 latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
    #                 latent_model_input = latent_model_input.detach()
    #                 latent_model_input.requires_grad = False

    #                 context = torch.no_grad() if i <= num_warmup_steps_bind else nullcontext()

    #                 # predict the noise
    #                 with context:
    #                     noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings
    #                                         ).sample.to(dtype=latents_dtype)
                    
    #                 # perform cfg guidance
    #                 if do_classifier_free_guidance:
    #                     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    #                     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    #                 # compute the previous noisy sample x_t -> x_t-1
    #                 latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

    #                 # Optimize text_embeddings
    #                 if optim_mode == "both":
    #                     optimizer = torch.optim.Adam([text_embeddings], lr=learning_rate)
    #                 elif optim_mode == "only_cond":
    #                     optimizer = torch.optim.Adam([cond_embeddings], lr=learning_rate)
    #                 elif optim_mode == "only_uncond":
    #                     optimizer = torch.optim.Adam([uncond_embeddings], lr=learning_rate)
    #                 else:
    #                     raise ValueError

    #                 if num_end_steps >= i > num_warmup_steps_bind: # Skip very noisy timesteps
    #                     with torch.autograd.set_detect_anomaly(True):
    #                         # 1. compute x0 
    #                         x0_video = self.xt2x0(latents, timesteps=timesteps, cur_step=i, noise_pred=noise_pred) 
    #                         bind_loss = self.cal_ImageBind_loss(
    #                             bind_model, x0_video, image_bind_audio_input, bind_device, clip_duration, clips_per_video, 
    #                             with_text_loss=with_text_loss, prompt=prompt, all_frames_loss=all_frames_loss,
    #                             n_samples_per_clip=n_samples_per_clip,
    #                         )
    #                         # print('bind_device: ', bind_device, x0_video.device)
    #                         # print('x0_video: ', x0_video.max(), x0_video.min())
    #                         # exit()
    #                         video_feat_cavp = extract_cavp.video_forward_from_tensor(x0_video, bind_device)

    #                         # compute diff foley loss 
    #                         print(video_feat_cavp.shape,audio_feat_cavp.shape)
    #                         foley_loss = 1 - F.cosine_similarity(video_feat_cavp, audio_feat_cavp)

    #                         # loss = bind_loss + foley_loss
    #                         loss = foley_loss

    #                         # backpropagate & update latent
    #                         loss.backward()
    #                         if optim_mode == "both":
    #                             print('grad of target embeddings: ', text_embeddings.grad.shape)
    #                         elif optim_mode == "only_cond":
    #                             print('grad of target embeddings: ', cond_embeddings.grad.shape)
    #                         elif optim_mode == "only_uncond":
    #                             print('grad of target embeddings: ', uncond_embeddings.grad.shape)
    #                         else:
    #                             raise ValueError

    #                         optimizer.step()
    #                         optimizer.zero_grad()
    #                         if norm_embed:
    #                             if optim_mode == "both":
    #                                 text_embeddings = text_embeddings / text_embeddings.norm() * embed_scale
    #                             else:
    #                                 raise ValueError

    #                     if optim_mode == "only_cond" or optim_mode == "only_uncond":
    #                         text_embeddings = torch.stack([uncond_embeddings, cond_embeddings], dim=0)
    #                 latents = latents.detach()
    #             # call the callback, if provided
    #             if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
    #                 progress_bar.update()
    #                 if callback is not None and i % callback_steps == 0:
    #                     callback(i, t, latents)

    #     # Post-processing
    #     video = self.decode_latents(latents)

    #     # Convert to tensor
    #     if output_type == "tensor":
    #         video = torch.from_numpy(video)

    #     if not return_dict:
    #         return video

    #     return AnimationPipelineOutput(videos=video)

