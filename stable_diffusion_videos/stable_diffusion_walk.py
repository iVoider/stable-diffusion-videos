import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from diffusers.schedulers import (DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler)
from typing import Optional, Tuple, Union
from scipy import integrate
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput
from .stable_diffusion_pipeline import StableDiffusionPipeline

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
        num_train_timesteps: int = 1000,
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
        print(type(self.sigmas), type(timestep))
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

pipeline = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=True,
    torch_dtype=torch.float16,
    revision="fp16",
).to("cuda")

default_scheduler = PNDMScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
)
ddim_scheduler = DDIMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
)
klms_scheduler = LMSDiscreteScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
)

euler_scheduler = EulerDiscreteScheduler(
   beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
)

SCHEDULERS = dict(default=default_scheduler, ddim=ddim_scheduler, klms=klms_scheduler, euler = euler_scheduler)

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """helper function to spherically interpolate two arrays v1 v2"""

    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2

def make_video_ffmpeg(frame_dir, output_file_name='output.mp4', frame_filename="frame%06d.png", fps=30):
    frame_ref_path = str(frame_dir / frame_filename)
    video_path = str(frame_dir / output_file_name)
    subprocess.call(
        f"ffmpeg -r {fps} -i {frame_ref_path} -vcodec libx264 -crf 10 -pix_fmt yuv420p"
        f" {video_path}".split()
    )
    return video_path


def walk(
    prompts=["blueberry spaghetti", "strawberry spaghetti"],
    seeds=[42, 123],
    num_steps=5,
    output_dir="dreams",
    name="berry_good_spaghetti",
    height=512,
    width=512,
    guidance_scale=7.5,
    eta=0.0,
    num_inference_steps=50,
    do_loop=False,
    make_video=False,
    use_lerp_for_text=True,
    scheduler="klms",  # choices: default, ddim, klms
    disable_tqdm=False,
    upsample=False,
    fps=30,
    less_vram=False,
    resume=False,
    batch_size=1,
    frame_filename_ext='.png',
):
    """Generate video frames/a video given a list of prompts and seeds.

    Args:
        prompts (List[str], optional): List of . Defaults to ["blueberry spaghetti", "strawberry spaghetti"].
        seeds (List[int], optional): List of random seeds corresponding to given prompts.
        num_steps (int, optional): Number of steps to walk. Increase this value to 60-200 for good results. Defaults to 5.
        output_dir (str, optional): Root dir where images will be saved. Defaults to "dreams".
        name (str, optional): Sub directory of output_dir to save this run's files. Defaults to "berry_good_spaghetti".
        height (int, optional): Height of image to generate. Defaults to 512.
        width (int, optional): Width of image to generate. Defaults to 512.
        guidance_scale (float, optional): Higher = more adherance to prompt. Lower = let model take the wheel. Defaults to 7.5.
        eta (float, optional): ETA. Defaults to 0.0.
        num_inference_steps (int, optional): Number of diffusion steps. Defaults to 50.
        do_loop (bool, optional): Whether to loop from last prompt back to first. Defaults to False.
        make_video (bool, optional): Whether to make a video or just save the images. Defaults to False.
        use_lerp_for_text (bool, optional): Use LERP instead of SLERP for text embeddings when walking. Defaults to True.
        scheduler (str, optional): Which scheduler to use. Defaults to "klms". Choices are "default", "ddim", "klms".
        disable_tqdm (bool, optional): Whether to turn off the tqdm progress bars. Defaults to False.
        upsample (bool, optional): If True, uses Real-ESRGAN to upsample images 4x. Requires it to be installed
            which you can do by running: `pip install git+https://github.com/xinntao/Real-ESRGAN.git`. Defaults to False.
        fps (int, optional): The frames per second (fps) that you want the video to use. Does nothing if make_video is False. Defaults to 30.
        less_vram (bool, optional): Allow higher resolution output on smaller GPUs. Yields same result at the expense of 10% speed. Defaults to False.
        resume (bool, optional): When set to True, resume from provided '<output_dir>/<name>' path. Useful if your run was terminated
            part of the way through.
        batch_size (int, optional): Number of examples per batch fed to pipeline. Increase this until you
            run out of VRAM. Defaults to 1.
        frame_filename_ext (str, optional): File extension to use when saving/resuming. Update this to
            ".jpg" to save or resume generating jpg images instead. Defaults to ".png".

    Returns:
        str: Path to video file saved if make_video=True, else None.
    """
    if upsample:
        from .upsampling import PipelineRealESRGAN

        upsampling_pipeline = PipelineRealESRGAN.from_pretrained('nateraw/real-esrgan')

    if less_vram:
        pipeline.enable_attention_slicing()

    output_path = Path(output_dir) / name
    output_path.mkdir(exist_ok=True, parents=True)
    prompt_config_path = output_path / 'prompt_config.json'

    if not resume:
        # Write prompt info to file in output dir so we can keep track of what we did
        prompt_config_path.write_text(
            json.dumps(
                dict(
                    prompts=prompts,
                    seeds=seeds,
                    num_steps=num_steps,
                    name=name,
                    guidance_scale=guidance_scale,
                    eta=eta,
                    num_inference_steps=num_inference_steps,
                    do_loop=do_loop,
                    make_video=make_video,
                    use_lerp_for_text=use_lerp_for_text,
                    scheduler=scheduler,
                    upsample=upsample,
                    fps=fps,
                    height=height,
                    width=width,
                ),
                indent=2,
                sort_keys=False,
            )
        )
    else:
        # When resuming, we load all available info from existing prompt config, using kwargs passed in where necessary
        if not prompt_config_path.exists():
            raise FileNotFoundError(f"You specified resume=True, but no prompt config file was found at {prompt_config_path}")

        data = json.load(open(prompt_config_path))
        prompts = data['prompts']
        seeds = data['seeds']
        num_steps = data['num_steps']
        height = data['height'] if 'height' in data else height
        width = data['width'] if 'width' in data else width
        guidance_scale = data['guidance_scale']
        eta = data['eta']
        num_inference_steps = data['num_inference_steps']
        do_loop = data['do_loop']
        make_video = data['make_video']
        use_lerp_for_text = data['use_lerp_for_text']
        scheduler = data['scheduler']
        disable_tqdm=disable_tqdm
        upsample = data['upsample'] if 'upsample' in data else upsample
        fps = data['fps'] if 'fps' in data else fps

        resume_step = int(sorted(output_path.glob(f"frame*{frame_filename_ext}"))[-1].stem[5:])
        print(f"\nResuming {output_path} from step {resume_step}...")


    if upsample:
        from .upsampling import PipelineRealESRGAN

        upsampling_pipeline = PipelineRealESRGAN.from_pretrained('nateraw/real-esrgan')

    pipeline.set_progress_bar_config(disable=disable_tqdm)
    pipeline.scheduler = SCHEDULERS[scheduler]

    assert len(prompts) == len(seeds)

    first_prompt, *prompts = prompts
    embeds_a = pipeline.embed_text(first_prompt)

    first_seed, *seeds = seeds
    latents_a = torch.randn(
        (1, pipeline.unet.in_channels, height // 8, width // 8),
        device=pipeline.device,
        generator=torch.Generator(device=pipeline.device).manual_seed(first_seed),
    )

    if do_loop:
        prompts.append(first_prompt)
        seeds.append(first_seed)

    frame_index = 0
    for prompt, seed in zip(prompts, seeds):
        # Text
        embeds_b = pipeline.embed_text(prompt)

        # Latent Noise
        latents_b = torch.randn(
            (1, pipeline.unet.in_channels, height // 8, width // 8),
            device=pipeline.device,
            generator=torch.Generator(device=pipeline.device).manual_seed(seed),
        )

        latents_batch, embeds_batch = None, None
        for i, t in enumerate(np.linspace(0, 1, num_steps)):

            frame_filepath = output_path / (f"frame%06d{frame_filename_ext}" % frame_index)
            if resume and frame_filepath.is_file():
                frame_index += 1
                continue

            if use_lerp_for_text:
                embeds = torch.lerp(embeds_a, embeds_b, float(t))
            else:
                embeds = slerp(float(t), embeds_a, embeds_b)
            latents = slerp(float(t), latents_a, latents_b)

            embeds_batch = embeds if embeds_batch is None else torch.cat([embeds_batch, embeds])
            latents_batch = latents if latents_batch is None else torch.cat([latents_batch, latents])

            del embeds
            del latents
            torch.cuda.empty_cache()

            batch_is_ready = embeds_batch.shape[0] == batch_size or t == 1.0
            if not batch_is_ready:
                continue

            do_print_progress = (i == 0) or ((frame_index) % 20 == 0)
            if do_print_progress:
                print(f"COUNT: {frame_index}/{len(seeds)*num_steps}")

            with torch.autocast("cuda"):
                outputs = pipeline(
                    latents=latents_batch,
                    text_embeddings=embeds_batch,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    eta=eta,
                    num_inference_steps=num_inference_steps,
                    output_type='pil' if not upsample else 'numpy'
                )["sample"]

                del embeds_batch
                del latents_batch
                torch.cuda.empty_cache()
                latents_batch, embeds_batch = None, None

                if upsample:
                    images = []
                    for output in outputs:
                        images.append(upsampling_pipeline(output))
                else:
                    images = outputs
            for image in images:
                frame_filepath = output_path / (f"frame%06d{frame_filename_ext}" % frame_index)
                image.save(frame_filepath)
                frame_index += 1

        embeds_a = embeds_b
        latents_a = latents_b

    if make_video:
        return make_video_ffmpeg(output_path, f"{name}.mp4", fps=fps, frame_filename=f"frame%06d{frame_filename_ext}")


if __name__ == "__main__":
    import fire

    fire.Fire(walk)