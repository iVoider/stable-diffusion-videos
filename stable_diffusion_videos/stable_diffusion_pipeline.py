import inspect
import warnings
from tqdm.auto import tqdm
from typing import List, Optional, Union

import torch
from diffusers import ModelMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import \
    StableDiffusionSafetyChecker
from diffusers.schedulers import (DDIMScheduler, LMSDiscreteScheduler,
                                  PNDMScheduler)
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

import numpy as np
import torch
from typing import Optional, Tuple, Union
from scipy import integrate
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput

class EulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Implements Algorithm 2 (Euler steps) from Karras et al. (2022).
    for discrete beta schedules. Based on the original k-diffusion implementation by
    Katherine Crowson:
    https://github.com/crowsonkb/k-diffusion/blob/481677d114f6ea445aa009cf5bd7a9cdee909e47/k_diffusion/sampling.py#L51
    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`~ConfigMixin`] also provides general loading and saving functionality via the [`~ConfigMixin.save_config`] and
    [`~ConfigMixin.from_config`] functions.
    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        beta_start (`float`): the starting `beta` value of inference.
        beta_end (`float`): the final `beta` value.
        beta_schedule (`str`):
            the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear` or `scaled_linear`.
        trained_betas (`np.ndarray`, optional):
            option to pass an array of betas directly to the constructor to bypass `beta_start`, `beta_end` etc.
            options to clip the variance used when adding noise to the denoised sample. Choose from `fixed_small`,
            `fixed_small_log`, `fixed_large`, `fixed_large_log`, `learned` or `learned_range`.
        tensor_format (`str`): whether the scheduler expects pytorch or numpy arrays.
    """

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 50,
        beta_start: float = 0.00085, #sensible defaults
        beta_end: float = 0.012,
        beta_schedule: str = "linear",
        trained_betas: Optional[np.ndarray] = None,
        tensor_format: str = "pt",
    ):
        if trained_betas is not None:
            self.betas = np.asarray(trained_betas)
        if beta_schedule == "linear":
            self.betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = np.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=np.float32) ** 2
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)

        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5

        # setable values
        self.num_inference_steps = None
        self.timesteps = np.arange(0, num_train_timesteps)[::-1].copy()
        self.derivatives = []

        self.tensor_format = tensor_format
        self.set_format(tensor_format=tensor_format)

    def set_timesteps(self, num_inference_steps: int):
        """
        Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.
        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
        self.num_inference_steps = num_inference_steps
        self.timesteps = np.linspace(self.config.num_train_timesteps - 1, 0, num_inference_steps, dtype=float)

        low_idx = np.floor(self.timesteps).astype(int)
        high_idx = np.ceil(self.timesteps).astype(int)
        frac = np.mod(self.timesteps, 1.0)
        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = (1 - frac) * sigmas[low_idx] + frac * sigmas[high_idx]
        self.sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)

        self.derivatives = []

        self.set_format(tensor_format=self.tensor_format)

    def step(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],
        timestep: int,
        sample: Union[torch.FloatTensor, np.ndarray],
        s_churn: float = 0.,
        s_tmin:  float = 0.,
        s_tmax: float = float('inf'),
        s_noise:  float = 1.,
        generator=None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).
        Args:
            model_output (`torch.FloatTensor` or `np.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor` or `np.ndarray`):
                current instance of sample being created by diffusion process.
            s_churn (`float`)
            s_tmin  (`float`)
            s_tmax  (`float`)
            s_noise (`float`)
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class
        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.SchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        sigma = self.sigmas[timestep]
        gamma = min(s_churn / (len(self.sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.
        eps = torch.randn(sample.size(), dtype=sample.dtype, layout=sample.layout, device=sample.device, generator=generator) * s_noise
        sigma_hat = sigma * (gamma + 1)
        if gamma > 0:
            sample = sample + eps * (sigma_hat ** 2 - sigma ** 2) ** 0.5
        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        pred_original_sample = sample - sigma_hat * model_output

        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma_hat
        self.derivatives.append(derivative)

        dt = self.sigmas[timestep + 1] - sigma_hat

        prev_sample = sample + derivative * dt

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    def add_noise(
        self,
        original_samples: Union[torch.FloatTensor, np.ndarray],
        noise: Union[torch.FloatTensor, np.ndarray],
        timesteps: Union[torch.IntTensor, np.ndarray],
    ) -> Union[torch.FloatTensor, np.ndarray]:
        if self.tensor_format == "pt":
            timesteps = timesteps.to(self.sigmas.device)
        sigmas = self.match_shape(self.sigmas[timesteps], noise)
        noisy_samples = original_samples + noise * sigmas

        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps

class StableDiffusionPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        scheduler = scheduler.set_format("pt")
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
    
    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                a number is provided, uses as many slices as `attention_head_dim // slice_size`. In this case,
                `attention_head_dim` must be a multiple of `slice_size`.
        """
        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        r"""
        Disable sliced attention computation. If `enable_attention_slicing` was previously invoked, this method will go
        back to computing attention in one step.
        """
        # set slice_size = `None` to disable `attention slicing`
        self.enable_attention_slicing(None)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        text_embeddings: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        **kwargs,
    ):
        if "torch_device" in kwargs:
            device = kwargs.pop("torch_device")
            warnings.warn(
                "`torch_device` is deprecated as an input argument to `__call__` and"
                " will be removed in v0.3.0. Consider using `pipe.to(torch_device)`"
                " instead."
            )

            # Set device as before (to be removed in 0.3.0)
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(device)

        if text_embeddings is None:
            if isinstance(prompt, str):
                batch_size = 1
            elif isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                raise ValueError(
                    f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
                )

            if height % 8 != 0 or width % 8 != 0:
                raise ValueError(
                    "`height` and `width` have to be divisible by 8 but are"
                    f" {height} and {width}."
                )

            # get prompt text embeddings
            text_input = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        else:
            batch_size = text_embeddings.shape[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            # max_length = text_input.input_ids.shape[-1]
            max_length = 77  # self.tokenizer.model_max_length
            uncond_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device)
            )[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # get the initial random noise unless the user supplied it
        latents_shape = (batch_size, self.unet.in_channels, height // 8, width // 8)
        if latents is None:
            latents = torch.randn(
                latents_shape,
                generator=generator,
                device=self.device,
            )
        else:
            if latents.shape != latents_shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected"
                    f" {latents_shape}"
                )
            latents = latents.to(self.device)

        # set timesteps
        accepts_offset = "offset" in set(
            inspect.signature(self.scheduler.set_timesteps).parameters.keys()
        )
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
        if not isinstance(self.scheduler, DDIMScheduler) and not isinstance(self.scheduler, PNDMScheduler):
            latents = latents * self.scheduler.sigmas[0]

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            if not isinstance(self.scheduler, DDIMScheduler) and not isinstance(self.scheduler, PNDMScheduler):
                sigma = self.scheduler.sigmas[i]
                # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            )["sample"]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            if not isinstance(self.scheduler, DDIMScheduler) and not isinstance(self.scheduler, PNDMScheduler):
                latents = self.scheduler.step(
                    noise_pred, i, latents, **extra_step_kwargs
                )["prev_sample"]
            else:
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                )["prev_sample"]

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        safety_cheker_input = self.feature_extractor(
            self.numpy_to_pil(image), return_tensors="pt"
        ).to(self.device)

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return {"sample": image, "nsfw_content_detected": False}

    def embed_text(self, text):
        """Helper to embed some text"""
        with torch.autocast("cuda"):
            text_input = self.tokenizer(
                text,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                embed = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return embed


class NoCheck(ModelMixin):
    """Can be used in place of safety checker. Use responsibly and at your own risk."""
    def __init__(self):
        super().__init__()
        self.register_parameter(name='asdf', param=torch.nn.Parameter(torch.randn(3)))

    def forward(self, images=None, **kwargs):
        return images, [False]
