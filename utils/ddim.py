from packaging import version as pver
from typing import List, Optional, Tuple, Union
import torch
from diffusers.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils import deprecate

class DDIMPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, unet, scheduler, v_pred = False, x_pred = False):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        self.v_pred = v_pred
        self.x_pred = x_pred

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        use_clipped_model_output: Optional[bool] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        pose = None,
        shading = None,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            eta (`float`, *optional*, defaults to 0.0):
                The eta parameter which controls the scale of the variance (0 is DDIM and 1 is one type of DDPM).
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            use_clipped_model_output (`bool`, *optional*, defaults to `None`):
                if `True` or `False`, see documentation for `DDIMScheduler.step`. If `None`, nothing is passed
                downstream to the scheduler. So use `None` for schedulers which don't support this argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipeline_utils.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipeline_utils.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is a list with the
            generated images.
        """

        if (
            generator is not None
            and isinstance(generator, torch.Generator)
            and generator.device.type != self.device.type
            and self.device.type != "mps"
        ):
            message = (
                f"The `generator` device is `{generator.device}` and does not match the pipeline "
                f"device `{self.device}`, so the `generator` will be ignored. "
                f'Please use `generator=torch.Generator(device="{self.device}")` instead.'
            )
            deprecate(
                "generator.device == 'cpu'",
                "0.12.0",
                message,
            )
            generator = None


        # Sample gaussian noise to begin loop
        if isinstance(self.unet.sample_size, int):
            image_shape = (batch_size, self.unet.in_channels, self.unet.sample_size, self.unet.sample_size)
        else:
            image_shape = (batch_size, self.unet.in_channels, *self.unet.sample_size)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        rand_device = "cpu" if self.device.type == "mps" else self.device
        if isinstance(generator, list):
            shape = (1,) + image_shape[1:]
            image = [
                torch.randn(shape, generator=generator[i], device=rand_device, dtype=self.unet.dtype)
                for i in range(batch_size)
            ]
            image = torch.cat(image, dim=0).to(self.device)
        else:
            image = torch.randn(image_shape, generator=generator, device=rand_device, dtype=self.unet.dtype)
            image = image.to(self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            if pose is None:
                if shading is None:
                    model_output = self.unet(image, t).sample
                else:
                    model_output = self.unet(image, t, shading = shading).sample
            else:
                if shading is None:
                    model_output = self.unet(image, t, c=pose).sample
                else:
                    model_output = self.unet(image, t, c=pose, shading = shading).sample  
                                  
            if self.v_pred or self.x_pred:
                sqrt_alpha_prod = self.scheduler.alphas_cumprod.to(image.device)[t] ** 0.5
                sqrt_one_minus_alpha_prod = (1 - self.scheduler.alphas_cumprod.to(image.device)[t]) ** 0.5
                if self.v_pred:
                    model_output = sqrt_alpha_prod * model_output + sqrt_one_minus_alpha_prod * image
                elif self.x_pred:
                    model_output = (image - sqrt_alpha_prod * model_output) / sqrt_one_minus_alpha_prod
            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            image = self.scheduler.step(
                model_output, t, image, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
            ).prev_sample

        return image
