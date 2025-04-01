from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    PNDMScheduler,
    DDIMScheduler,
    StableDiffusionPipeline,
)
from diffusers.utils.import_utils import is_xformers_available

# suppress partial model loading warning
logging.set_verbosity_error()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from guidance.prompt_processor import StableDiffusionPromptProcessor
from utils.debug import timer
from time import time 

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


class StableDiffusion(nn.Module):
    def __init__(
        self,
        config,
        fp16=True,
        vram_O=False,
        sd_version="2.1",
        hf_key=None,
        t_range=[0.02, 0.98],
    ):
        super().__init__()

        self.config = config
        self.device = config.device
        self.sd_version = sd_version

        if hf_key is not None:
            print(f"[INFO] using hugging face custom model key: {hf_key}")
            model_key = hf_key
        elif self.sd_version == "2.1":
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == "2.0":
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == "1.5":
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(
                f"Stable-diffusion version {self.sd_version} not supported."
            )

        self.dtype = torch.float16 if fp16 else torch.float32

        # Create model
        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, torch_dtype=self.dtype
        )

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)

        self.vae = pipe.vae

        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype
        )

        self.pipe = pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        if(self.config.use_view_dependent_prompt):
            self.prompt_processor = StableDiffusionPromptProcessor(self.config, self.pipe)
        self.embeddings = None

    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts):
        pos_embeds = self.encode_text(prompts)  # [1, 77, 768]
        neg_embeds = self.encode_text(negative_prompts)
        self.embeddings = torch.cat([neg_embeds, pos_embeds], dim=0)  # [2, 77, 768]
        return self.embeddings
    
    def set_text_embeds(self, embeddings):
        self.embeddings = embeddings
    
    def encode_text(self, prompt):
        # prompt: [str]
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings

    @torch.no_grad()
    def refine(self, pred_rgb,
               guidance_scale=100, steps=50, strength=0.8,
        ):

        batch_size = pred_rgb.shape[0]
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        latents = self.encode_imgs(pred_rgb_512.to(self.dtype))

        self.scheduler.set_timesteps(steps)
        init_step = int(steps * strength)
        latents = self.scheduler.add_noise(latents, torch.randn_like(latents), self.scheduler.timesteps[init_step])

        for i, t in enumerate(self.scheduler.timesteps[init_step:]):
    
            latent_model_input = torch.cat([latents] * 2)

            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=self.embeddings,
            ).sample

            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        imgs = self.decode_latents(latents) # [1, 3, 512, 512]
        return imgs
    
    def refine_step(self, pred_rgb, sd_rgb):
        H, W = pred_rgb.shape[2:]
        sd_rgb = F.interpolate(sd_rgb, (H, W), mode="bilinear", align_corners=False)
        return F.mse_loss(pred_rgb, sd_rgb)
    
    def apply_rescale(self, pos, neg, weight=7.5, rescale=0.7):
            # Apply regular classifier-free guidance.
            cfg = neg + weight * (pos - neg)
            # Calculate standard deviations.
            std_pos = pos.std([1,2,3], keepdim=True)
            std_cfg = cfg.std([1,2,3], keepdim=True)
            # Apply guidance rescale with fused operations.
            factor = std_pos / std_cfg
            factor = rescale * factor + (1 - rescale)
            return cfg * factor

    def train_step(
        self,
        pred_rgb,
        curr_step=None,
        guidance_scale=100,
        as_latent=False,
        camera_data=None,
        determ_step = None,
        q_unet = None,
        pose = None,
        embeddings = None,
        prompt_processor = None,
    ):
        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.dtype)

        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode="bilinear", align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode="bilinear", align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        if determ_step is not None:
            t = determ_step*torch.ones((batch_size,), dtype=torch.long, device=self.device)
        elif self.config.scheduler.dreamlike:
            step_ratio = curr_step / self.config.max_iter
            # dreamtime-like
            t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
        elif(self.config.scheduler.random):
            t = torch.randint(int(0.02*self.max_step), int(0.98*self.max_step), (batch_size,), dtype=torch.long, device=self.device)
        elif(self.config.scheduler.linear_uniform):
            # step_ratio = curr_step / self.config.scheduler.max_step_rescale
            # max_step_percent = (1-step_ratio)*0.98 + step_ratio*self.config.scheduler.max_scale_linear
            # max_step_percent = np.max([max_step_percent, self.config.scheduler.max_scale_linear])
            # t = torch.randint(int(self.max_step*0.02), int(max_step_percent*self.max_step), (batch_size,), dtype=torch.long, device=self.device)
            
            if(curr_step < self.config.scheduler.max_step_rescale):
                max_step_percent = 1
            else:
                max_step_percent = self.config.scheduler.max_scale_linear
            t = torch.randint(int(self.max_step*0.02), int(max_step_percent*self.max_step), (batch_size,), dtype=torch.long, device=self.device)
        
            # max_step_percent = self.config.scheduler.max_scale_linear
            # if(curr_step < self.config.scheduler.max_step_rescale):
            #     t = torch.randint(int(max_step_percent*self.max_step), int(1*self.max_step), (batch_size,), dtype=torch.long, device=self.device)
            # else:
            #     t = torch.randint(int(self.max_step*0.02), int(max_step_percent*self.max_step), (batch_size,), dtype=torch.long, device=self.device)
        # w(t), sigma_t^2
        print(t)
        w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)


        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)

            if(self.config.use_view_dependent_prompt and camera_data is not None): 
                azimuth, elevation, camera_distances = camera_data
                prompt_proc = prompt_processor if prompt_processor is not None else self.prompt_processor
                embeds = prompt_proc().get_text_embedding(elevation, azimuth, camera_distances, 
                                                                True)
                # print(f"Azimuth: {azimuth}, Elevation: {elevation}, Camera Distance: {camera_distances}")
                embeds = embeds.type(self.dtype)
                noise_pred = self.unet(
                    latent_model_input, tt, 
                    encoder_hidden_states=embeds
                ).sample
            else:
                if(embeddings is not None):
                    embeds = embeddings
                else:
                    embeds = torch.repeat_interleave(self.embeddings, batch_size, dim=0)
                noise_pred = self.unet(
                    latent_model_input, tt, 
                    encoder_hidden_states=embeds
                ).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)

            if(self.config.rescale.enable):
                noise_pred = self.apply_rescale(noise_pred_pos, noise_pred_uncond, 
                                                weight=self.config.course_guidance, 
                                                rescale=self.config.rescale.scale)
            else:
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_pos - noise_pred_uncond
                )
            
            if(q_unet is not None):
                latents_noisy = latents_noisy.type(q_unet.dtype)
                pose = pose.type(q_unet.dtype)
                noise_pred_q = q_unet(latents_noisy, t, textemb=embeds, c = pose).sample
                if(self.config.vsd.v_pred):
                    sqrt_alpha_prod = self.scheduler.alphas_cumprod.to(self.device)[t] ** 0.5
                    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
                    while len(sqrt_alpha_prod.shape) < len(latents_noisy.shape):
                        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
                    sqrt_one_minus_alpha_prod = (1 - self.scheduler.alphas_cumprod.to(self.device)[t]) ** 0.5
                    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
                    while len(sqrt_one_minus_alpha_prod.shape) < len(latents_noisy.shape):
                        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
                    noise_pred_q = sqrt_alpha_prod * noise_pred_q + sqrt_one_minus_alpha_prod * latents_noisy
                    noise_pred_q = noise_pred_q.type(self.dtype)
          
        if(q_unet is not None):
            grad = w * (noise_pred - noise_pred_q)
        else:
            grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum') / latents.shape[0]


        return loss, latents
    
    # def train_step(
    #     self,
    #     pred_rgb,
    #     curr_step=None,
    #     guidance_scale=100,
    #     as_latent=False,
    #     camera_data=None,
    #     determ_step = None,
    #     q_unet = None,
    #     pose = None,
    #     embeddings = None,
    #     prompt_processor = None,
    # ):
    #     batch_size = pred_rgb.shape[0]
    #     pred_rgb = pred_rgb.to(self.dtype)

    #     if as_latent:
    #         latents = F.interpolate(pred_rgb, (64, 64), mode="bilinear", align_corners=False) * 2 - 1
    #     else:
    #         # interp to 512x512 to be fed into vae.
    #         pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode="bilinear", align_corners=False)
    #         # encode image into latents with vae, requires grad!
    #         latents = self.encode_imgs(pred_rgb_512)

    #     if determ_step is not None:
    #         t = determ_step*torch.ones((batch_size,), dtype=torch.long, device=self.device)
    #     elif self.config.scheduler.dreamlike:
    #         step_ratio = curr_step / self.config.max_iter
    #         # dreamtime-like
    #         t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
    #         t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
    #     elif(self.config.scheduler.random):
    #         t = torch.randint(int(0.02*self.max_step), int(0.98*self.max_step), (batch_size,), dtype=torch.long, device=self.device)
    #     elif(self.config.scheduler.linear_uniform):
    #         # step_ratio = curr_step / self.config.scheduler.max_step_rescale
    #         # max_step_percent = (1-step_ratio)*0.98 + step_ratio*self.config.scheduler.max_scale_linear
    #         # max_step_percent = np.max([max_step_percent, self.config.scheduler.max_scale_linear])
    #         # t = torch.randint(int(self.max_step*0.02), int(max_step_percent*self.max_step), (batch_size,), dtype=torch.long, device=self.device)
            
    #         if(curr_step < self.config.scheduler.max_step_rescale):
    #             max_step_percent = 1
    #         else:
    #             max_step_percent = self.config.scheduler.max_scale_linear
    #         t = torch.randint(int(self.max_step*0.02), int(max_step_percent*self.max_step), (batch_size,), dtype=torch.long, device=self.device)
        
    #         # max_step_percent = self.config.scheduler.max_scale_linear
    #         # if(curr_step < self.config.scheduler.max_step_rescale):
    #         #     t = torch.randint(int(max_step_percent*self.max_step), int(1*self.max_step), (batch_size,), dtype=torch.long, device=self.device)
    #         # else:
    #         #     t = torch.randint(int(self.max_step*0.02), int(max_step_percent*self.max_step), (batch_size,), dtype=torch.long, device=self.device)
    #     # w(t), sigma_t^2
    #     print(t)
    #     t = torch.cat([t[0:len(t)//2]] * 2)
    #     w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)


    #     # predict the noise residual with unet, NO grad!
    #     with torch.no_grad():
    #         # add noise
    #         noise = torch.randn_like(latents[0:len(latents)//2])
    #         noise = torch.cat([noise] * 2)
    #         latents_noisy = self.scheduler.add_noise(latents, noise, t)
    #         # pred noise
    #         # latent_model_input = torch.cat([latents_noisy] * 2)
    #         latent_model_input = latents_noisy
    #         tt = torch.cat([t[0:len(t)//2]] * 2)

    #         if(self.config.use_view_dependent_prompt and camera_data is not None): 
    #             azimuth, elevation, camera_distances = camera_data
    #             prompt_proc = prompt_processor if prompt_processor is not None else self.prompt_processor
    #             embeds = prompt_proc().get_text_embedding(elevation, azimuth, camera_distances, 
    #                                                             True)
    #             # print(f"Azimuth: {azimuth}, Elevation: {elevation}, Camera Distance: {camera_distances}")
    #             embeds = embeds.type(self.dtype)
    #             noise_pred = self.unet(
    #                 latent_model_input, tt, 
    #                 encoder_hidden_states=embeds
    #             ).sample
    #         else:
    #             if(embeddings is not None):
    #                 embeds = embeddings
    #             else:
    #                 embeds = torch.repeat_interleave(self.embeddings, batch_size, dim=0)
    #             noise_pred = self.unet(
    #                 latent_model_input, tt, 
    #                 encoder_hidden_states=embeds
    #             ).sample

    #         # perform guidance (high scale from paper!)
    #         noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)

    #         if(self.config.rescale.enable):
    #             noise_pred = self.apply_rescale(noise_pred_pos, noise_pred_uncond, 
    #                                             weight=self.config.course_guidance, 
    #                                             rescale=self.config.rescale.scale)
    #         else:
    #             noise_pred = noise_pred_uncond + guidance_scale * (
    #                 noise_pred_pos - noise_pred_uncond
    #             )
            
    #         if(q_unet is not None):
    #             latents_noisy = latents_noisy.type(q_unet.dtype)
    #             pose = pose.type(q_unet.dtype)
    #             noise_pred_q = q_unet(latents_noisy, t, textemb=embeds, c = pose).sample
    #             if(self.config.vsd.v_pred):
    #                 sqrt_alpha_prod = self.scheduler.alphas_cumprod.to(self.device)[t] ** 0.5
    #                 sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    #                 while len(sqrt_alpha_prod.shape) < len(latents_noisy.shape):
    #                     sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
    #                 sqrt_one_minus_alpha_prod = (1 - self.scheduler.alphas_cumprod.to(self.device)[t]) ** 0.5
    #                 sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    #                 while len(sqrt_one_minus_alpha_prod.shape) < len(latents_noisy.shape):
    #                     sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
    #                 noise_pred_q = sqrt_alpha_prod * noise_pred_q + sqrt_one_minus_alpha_prod * latents_noisy
    #                 noise_pred_q = noise_pred_q.type(self.dtype)
          
    #     print(noise_pred.shape, noise.shape)
    #     if(q_unet is not None):
    #         grad = w * (noise_pred - noise_pred_q)
    #     else:
    #         grad = w[0:len(w)//2] * (noise_pred - noise[0:len(noise)//2])
    #     grad = torch.nan_to_num(grad)

    #     latents = latents[0:len(latents)//2]
    #     target = (latents - grad).detach()
    #     loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum') / latents.shape[0]


    #     return loss, latents

    def train_step_mod(
        self,
        pred_rgb,
        curr_step=None,
        guidance_scale=100,
        as_latent=False,
        camera_data=None,
        determ_step = None,
        q_unet = None,
        pose = None,
        embeddings = None,
        prompt_processor = None,
    ):
        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.dtype)

        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode="bilinear", align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode="bilinear", align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        if determ_step is not None:
            t = determ_step*torch.ones((batch_size,), dtype=torch.long, device=self.device)
        elif self.config.scheduler.dreamlike:
            step_ratio = curr_step / self.config.max_iter
            # dreamtime-like
            t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
        elif(self.config.scheduler.random):
            t = torch.randint(int(0.02*self.max_step), int(0.98*self.max_step), (batch_size,), dtype=torch.long, device=self.device)
        elif(self.config.scheduler.linear_uniform):
            # step_ratio = curr_step / self.config.scheduler.max_step_rescale
            # max_step_percent = (1-step_ratio)*0.98 + step_ratio*self.config.scheduler.max_scale_linear
            # max_step_percent = np.max([max_step_percent, self.config.scheduler.max_scale_linear])
            # t = torch.randint(int(self.max_step*0.02), int(max_step_percent*self.max_step), (batch_size,), dtype=torch.long, device=self.device)
            if(curr_step < self.config.scheduler.max_step_rescale):
                max_step_percent = 1
            else:
                max_step_percent = self.config.scheduler.max_scale_linear
            t = torch.randint(int(self.max_step*0.02), int(max_step_percent*self.max_step), (batch_size,), dtype=torch.long, device=self.device)
        # w(t), sigma_t^2
        w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)

            if(self.config.use_view_dependent_prompt and camera_data is not None): 
                azimuth, elevation, camera_distances = camera_data
                prompt_proc = prompt_processor if prompt_processor is not None else self.prompt_processor
                embeds = prompt_proc().get_text_embedding(elevation, azimuth, camera_distances, 
                                                                True)
                print(f"Azimuth: {azimuth}, Elevation: {elevation}, Camera Distance: {camera_distances}")
                embeds = embeds.type(self.dtype)
                noise_pred = self.unet(
                    latent_model_input, tt, 
                    encoder_hidden_states=embeds
                ).sample
            else:
                if(embeddings is not None):
                    embeds = embeddings
                else:
                    embeds = torch.repeat_interleave(self.embeddings, batch_size, dim=0)
                noise_pred = self.unet(
                    latent_model_input, tt, 
                    encoder_hidden_states=embeds
                ).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)

            if(self.config.rescale.enable):
                noise_pred = self.apply_rescale(noise_pred_pos, noise_pred_uncond, 
                                                weight=self.config.course_guidance, 
                                                rescale=self.config.rescale.scale)
            else:
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_pos - noise_pred_uncond
                )
            noise_pred = 100 * (noise_pred_pos)

        t2 = torch.clip(t+1, 0, int(1*self.max_step))
        w2 = (1 - self.alphas[t2]).view(batch_size, 1, 1, 1)
        with torch.no_grad():
            # add noise
            # noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t2)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t2] * 2)

            if(self.config.use_view_dependent_prompt and camera_data is not None): 
                azimuth, elevation, camera_distances = camera_data
                prompt_proc = prompt_processor if prompt_processor is not None else self.prompt_processor
                embeds = prompt_proc().get_text_embedding(elevation, azimuth, camera_distances, 
                                                                True)
                print(f"Azimuth: {azimuth}, Elevation: {elevation}, Camera Distance: {camera_distances}")
                embeds = embeds.type(self.dtype)
                noise_pred2 = self.unet(
                    latent_model_input, tt, 
                    encoder_hidden_states=embeds
                ).sample
            else:
                if(embeddings is not None):
                    embeds = embeddings
                else:
                    embeds = torch.repeat_interleave(self.embeddings, batch_size, dim=0)
                noise_pred2 = self.unet(
                    latent_model_input, tt, 
                    encoder_hidden_states=embeds
                ).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond2, noise_pred_pos2 = noise_pred2.chunk(2)

            if(self.config.rescale.enable):
                noise_pred = self.apply_rescale(noise_pred_pos, noise_pred_uncond, 
                                                weight=self.config.course_guidance, 
                                                rescale=self.config.rescale.scale)
            else:
                print("check2")
                noise_pred2 = noise_pred_uncond2 + guidance_scale * (
                    noise_pred_pos2 - noise_pred_uncond2
                )
            noise_pred2 = 100 * (noise_pred_pos2)

        grad = w * (noise_pred - noise_pred_pos2)
        print("check")
        # grad = w * (noise_pred)
        grad = torch.nan_to_num(grad)

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum') / latents.shape[0]


        return loss, latents

    @torch.no_grad()
    def produce_latents(
        self,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):
        if latents is None:
            latents = torch.randn(
                (
                    self.embeddings.shape[0] // 2,
                    self.unet.in_channels,
                    height // 8,
                    width // 8,
                ),
                device=self.device,
            )

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=self.embeddings
            ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def prompt_to_img(
        self,
        prompts,
        negative_prompts="",
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        self.get_text_embeds(prompts, negative_prompts)
        
        # Text embeds -> img latents
        latents = self.produce_latents(
            height=height,
            width=width,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype("uint8")

        return imgs


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("--negative", default="", type=str)
    parser.add_argument(
        "--sd_version",
        type=str,
        default="2.1",
        choices=["1.5", "2.0", "2.1"],
        help="stable diffusion version",
    )
    parser.add_argument(
        "--hf_key",
        type=str,
        default=None,
        help="hugging face Stable diffusion model key",
    )
    parser.add_argument("--fp16", action="store_true", help="use float16 for training")
    parser.add_argument(
        "--vram_O", action="store_true", help="optimization for low VRAM usage"
    )
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device("cuda")

    sd = StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()
