import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from torch.utils.tensorboard import SummaryWriter
from pytorch3d.ops import knn_points
import os
from omegaconf import OmegaConf
import random

from comp3dgs.gs_renderer2 import GaussianRenderer, GaussianModel, MiniCam
from utils.cam import orbit_camera, OrbitCamera, gen_random_minicam
from point_e.point_e_guidance import PointEGuidance
from utils.pc_helper import farthest_point_sampling
from utils.losses import lower_bound_knn, smooth_depth_loss, knn_loss
from utils.alpha_hull import alpha_hull_ids
from utils.debug import timer
from utils.ddim import DDIMPipeline

class ObjectTrainer(object):

    def __init__(self, config, guidance=None, vsd_unet = None, pointe_guidance=None) -> None:

        self.config = config
        self.curr_iter = 0
        self.max_iter = config.max_iter
        self.course_steps = config.course_steps
        self.device = torch.device(config.device)
        self.optimizer = None
        # init rendering params
        self.cam = OrbitCamera(self.config.render.ref_size, self.config.render.ref_size, 
                               r=self.config.render.radius, fovy=self.config.render.fovy)
        # init guidance
        self.config.negative_prompt = config.negative_prompt if(config.negative_prompt!="None") else ""
        self.guide = None if guidance is None else guidance
        # init 3dgs
        # self.renderer = Renderer(config=self.config)
        self.renderer = GaussianRenderer(config=self.config) 
        self.gaussians = GaussianModel(sh_degree=0, config=self.config, device=self.device)
        # point-e
        if(self.config.point_e.enable):
            self.point_e_xyz = None
            if(self.config.point_e.sd_loss):
                self.point_e_guide = PointEGuidance(self.config.point_e)
                self.point_e_guide.set_text(self.config.prompt)
        # fps points
        self.fps_idxs = None
        # init tensorboard
        self.tb_writer = SummaryWriter(self.config.save_path)
        # GUI data
        self.training = self.config.training
        self.gui = edict({'train_time': 0, 'loss': 0})
        # save config
        with open(os.path.join(self.config.save_path, "config.yaml"), 'w') as f:
            OmegaConf.save(config, f.name)

        self.text_embeds = self.prepare_train()

        if(self.config.vsd.enable):
            if(vsd_unet is not None):
                self.unet = vsd_unet[0]
                self.unet_scheduler = vsd_unet[1]
                self.unet_optimizer = vsd_unet[2]
            else:
                self.initialize_lora_unet()
        else:
            self.unet = None

        # convert stable diffusion back to fp16
        # self.guide.unet = self.guide.unet.half()
        # self.guide.vae = self.guide.vae.half()
        # self.guide.text_encoder = self.guide.text_encoder.half()


    def prepare_train(self):
        # ---------- Setup Training ------------
        # initialize 3dgs
        if(self.config.load_path is not None):
            self.gaussians.initialize(self.config.load_path)
        elif(self.config.floor):
            self.gaussians.initialize("floor")
        elif(self.config.point_e.enable):
            self.point_e_xyz = self.gaussians.initialize("point-e").to(self.device)
        else:
            self.gaussians.initialize(num_pts=self.config.gs3d.num_pts, radius=self.config.gs3d.init_radius)
        self.gaussians.training_setup(self.config.gs3d)
        self.gaussians.active_sh_degree = self.gaussians.max_sh_degree
        self.optimizer = self.gaussians.optimizer
        # load guidance
        if(self.guide is not None):
            print("SD already loaded!")
        else:
            print(f"[INFO] loading SD...")
            from guidance.sd_utils import StableDiffusion
            self.guide = StableDiffusion(self.config, fp16=True)
            print(f"[INFO] loaded SD!")
        # prepare embeddings
        with torch.no_grad():
            text_embeds = self.guide.get_text_embeds([self.config.prompt], [self.config.negative_prompt])
        # init fps points
        _, idx = farthest_point_sampling(self.gaussians.get_xyz, self.config.fps_k)
        self.fps_idxs = idx

        if(self.config.alpha_loss):
            self.step_alpha() 

        self.scaler = torch.cuda.amp.GradScaler()


        return text_embeds

    def initialize_lora_unet(self):
        if not self.config.vsd.lora:
            if self.config.vsd.q_cond:
                from utils.conditional_unet import CondUNet2DModel
                unetname = CondUNet2DModel
            else:
                from diffusers import UNet2DModel
                unetname = UNet2DModel
            self.unet = unetname(
                sample_size=64, # render height for NeRF in training, assert self.config.vsd.h==self.config.vsd.w==64
                in_channels=4,
                out_channels=4,
                layers_per_block=2,
                block_out_channels=(128, 256, 384, 512),
                down_block_types=(
                    "DownBlock2D",
                    "AttnDownBlock2D",
                    "AttnDownBlock2D",
                    "AttnDownBlock2D",
                ),
                up_block_types=(
                    "AttnUpBlock2D",
                    "AttnUpBlock2D",
                    "AttnUpBlock2D",
                    "UpBlock2D",
                ),
            )    
        else:
            # use lora
            from utils.lora_unet import UNet2DConditionModel     
            from diffusers.loaders import AttnProcsLayers
            from diffusers.models.attention_processor import LoRAAttnProcessor
            import einops
            if(self.config.vsd.v_pred):
                _unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="unet", low_cpu_mem_usage=False, device_map=None).to(self.device)
            else:
                _unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="unet", low_cpu_mem_usage=False, device_map=None).to(self.device)
            _unet.requires_grad_(False)
            lora_attn_procs = {}
            for name in _unet.attn_processors.keys():
                cross_attention_dim = None if name.endswith("attn1.processor") else _unet.config.cross_attention_dim
                if name.startswith("mid_block"):
                    hidden_size = _unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(_unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = _unet.config.block_out_channels[block_id]
                lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            _unet.set_attn_processor(lora_attn_procs)
            lora_layers = AttnProcsLayers(_unet.attn_processors)

            textemb = self.text_embeds[1]
            device = torch.device(self.config.device)
            class LoraUnet(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.unet = _unet
                    self.sample_size = 64
                    self.in_channels = 4
                    self.device = device
                    self.dtype = torch.float32
                    self.textemb = textemb.type(self.dtype)
                def forward(self,x,t, textemb=None, c=None,shading="albedo"):
                    if textemb is None:
                        textemb = self.textemb.expand(x.shape[0], self.textemb.shape[0], self.textemb.shape[1]).contiguous()
                    else:
                        textemb = textemb.type(self.dtype)
                        textemb = textemb[len(textemb)//2:] # remove null tokens
                    return self.unet(x,t,encoder_hidden_states=textemb,c=c,shading=shading)
            self._unet = _unet
            self.lora_layers = lora_layers
            self.unet = LoraUnet().to(self.config.device)   
            # self.unet.half()                  

        self.unet = self.unet.to(self.device)
        params = [
            {'params': self.lora_layers.parameters()},
            {'params': self._unet.camera_emb.parameters()},
            {'params': self._unet.lambertian_emb},
            {'params': self._unet.textureless_emb},
            {'params': self._unet.normal_emb},
        ] 
        self.unet_optimizer = torch.optim.AdamW(params, lr=self.config.vsd.unet_lr) # naive adam
        warm_up_lr_unet = lambda iter: iter / (self.config.vsd.warmup_steps*self.config.vsd.K+1) if iter <= (self.config.vsd.warmup_steps*self.config.vsd.K+1) else 1
        self.unet_scheduler = torch.optim.lr_scheduler.LambdaLR(self.unet_optimizer, warm_up_lr_unet)

        # TODO
        # if lr_scheduler is None:
        # self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        # else:
        #     self.lr_scheduler = lr_scheduler(self.optimizer)

    @torch.no_grad()
    def step_camera(self):
        # calculate bounding box of scene
        xyz = self.gaussians.get_xyz
        bbox = torch.stack([torch.min(xyz, dim=0)[0], torch.max(xyz, dim=0)[0]], dim=0)
        bbox = bbox.cpu().numpy()
        # find center of scene
        new_center = (bbox[0] + bbox[1]) / 2
        # find new radius
        largest_axis = np.max(bbox[1] - bbox[0])/2
        new_radius = 1.10*(largest_axis) / np.tan(self.cam.fovy/2)
        # update camera
        self.cam.center = new_center
        self.cam.radius = new_radius


    def step_pointe(self):
        pointe_loss = self.point_e_guide(self.gaussians)
        return pointe_loss
    
    def step_fps(self):
        if self.curr_iter % self.config.fps_interval == 0:
            _, idx = farthest_point_sampling(self.gaussians.get_xyz, self.config.fps_k)
            self.fps_idxs = idx
    
    def step_alpha(self):
        if self.curr_iter % self.config.alpha_interval == 0 or \
           self.curr_iter==1 or \
           self.curr_iter % self.config.gs3d.densification_interval == 1:
            idx = alpha_hull_ids(self.gaussians.get_xyz, alpha=5)
            self.alpha_idxs = idx

    def step_gaussians(self, render_data):
        if self.curr_iter >= self.config.gs3d.density_start_iter and self.curr_iter <= self.config.gs3d.density_end_iter:
            viewspace_point_tensor, visibility_filter, radii = render_data["viewspace_points"], render_data["visibility_filter"], render_data["radii"]
            self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if self.curr_iter % self.config.gs3d.densification_interval == 0:
                self.gaussians.densify_and_prune(self.config.gs3d.densify_grad_threshold, min_opacity=0.05, extent=2.0, max_screen_size=1)
                if(self.config.dynamic_camera):
                    self.step_camera()
                self.config.knn_k += self.config.knn_multiplier

            if self.curr_iter % self.config.gs3d.opacity_reset_interval == 0:
                self.gaussians.reset_opacity()

    def train_step(self):

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        
        # update SD text_embeddings
        self.guide.set_text_embeds(self.text_embeds)

        # update iter
        self.curr_iter += 1

        # update lr
        self.gaussians.update_learning_rate(self.curr_iter)
        loss = 0
        loss_dict = {}
        tb_imgs = {}

        ### Stage 1: guidance loss
        images = []
        uc_images = []
        depths = []
        hors = []
        vers = []
        cam_rads = []
        poses = []
        # Sample views
        for _ in range(self.config.batch_size):
            enable_fov_jitter = self.config.render.fov_jitter.enable and \
                                self.curr_iter > self.config.render.fov_jitter.step_start
            cam_data, cur_cam = gen_random_minicam(self.cam, self.cam.radius, 
                                                    center_elevation=self.config.render.elevation,
                                                    min_ver=self.config.render.min_ver,
                                                    max_ver=self.config.render.max_ver,
                                                    device=self.device, fov_jitter=enable_fov_jitter)

            bg_color = np.random.rand(3)
            # bg_color = np.array([0.0, 0.0, 0.0])
            out = self.renderer.render(self.gaussians, cur_cam, bg_color = bg_color)
            image = out["image"].unsqueeze(0)# [1, 3, H, W] in [0, 1]
            uc_image = out["unclamped_image"].unsqueeze(0)# [1, 3, H, W] in [0, 1]
            depth = out["depth"].unsqueeze(0)# [1, 1, H, W] in [0, 1]
            images.append(image)
            uc_images.append(uc_image)
            depths.append(depth)
            vers.append(cam_data[0])
            hors.append(cam_data[1]-self.config.reorient)
            poses.append(cam_data[2])
            cam_rads.append(self.cam.radius)
        images = torch.cat(images, dim=0)
        uc_images = torch.cat(uc_images, dim=0)
        depths = torch.cat(depths, dim=0)
        poses = torch.tensor(poses)
        poses = torch.flatten(poses, start_dim=1).to(self.device)
        cam_data = [torch.tensor(hors, device=self.device), torch.tensor(vers, device=self.device), torch.tensor(cam_rads, device=self.device)]
        embeds = torch.repeat_interleave(self.text_embeds, self.config.batch_size, dim=0)
        # calculate loss
        # with torch.autocast(device_type='cuda', dtype=torch.float16):
        course_sd_loss, latents = self.guide.train_step(images, self.curr_iter, 
                                                        guidance_scale=self.config.course_guidance,
                                                        camera_data=cam_data,
                                                        q_unet = self.unet,
                                                        pose=poses)
        
        loss_dict['course_sd_loss'] = course_sd_loss.item()
        loss += self.config.course_sd_loss*course_sd_loss

        # optimize step
        self.scaler.scale(loss).backward()
        # self.optimizer.step()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        # update gaussians clones/splits/prunes
        self.step_gaussians(out)  

        # VSD Update
        vsd_conf = self.config.vsd
        if self.curr_iter % vsd_conf.K2 == 0 and vsd_conf.enable:
            for _ in range(vsd_conf.K):
                self.unet_optimizer.zero_grad()
                timesteps = torch.randint(0, 1000, (self.config.batch_size,), device=self.device).long() # temperarily hard-coded for simplicity
                with torch.no_grad():
                    latents_clean = latents.expand(self.config.batch_size, latents.shape[1], latents.shape[2], latents.shape[3]).contiguous()
                    latents_clean = latents_clean.type(self.unet.dtype)
                    if vsd_conf.q_cond:
                        if random.random() < 0.1:
                            poses = torch.zeros(self.config.batch_size, 16).to(self.device)
                noise = torch.randn(latents_clean.shape, device=self.device, dtype=self.unet.dtype)
                latents_noisy = self.guide.scheduler.add_noise(latents_clean, noise, timesteps)
                latents_noisy = latents_noisy.type(self.unet.dtype)
                poses = poses.type(self.unet.dtype)
                embeds = None
                if vsd_conf.q_cond:
                    model_output = self.unet(latents_noisy, timesteps, textemb=embeds, c = poses).sample
                else:
                    model_output = self.unet(latents_noisy, timesteps).sample
                if vsd_conf.v_pred:
                    loss_unet = F.mse_loss(model_output, self.guide.scheduler.get_velocity(latents_clean, noise, timesteps))
                else:
                    loss_unet = F.mse_loss(model_output, noise)
                loss_unet.backward()
                self.unet_optimizer.step()
                if vsd_conf.scheduler:
                    self.unet_scheduler.step()              

        # update tensorboard
        tb_imgs['3dgs_img'] = images[0] 
        self.training_report(self.tb_writer, self.curr_iter, self.config.log_interval, 
                        loss_dict, self.gaussians, tb_imgs)
        # save model
        if(self.curr_iter % self.config.save_interval == 1):
            ply_save_path = os.path.join(self.config.save_path, f"iter_{self.curr_iter}_{self.config.prompt}.ply")
            self.gaussians.save_ply(ply_save_path)
            # Save Lora model
            # if(self.config.vsd.enable):
            #     torch.save(self.unet.state_dict(), os.path.join(self.config.save_path, f"iter_{self.curr_iter}_{self.config.prompt}.pth"))

        ender.record()
        torch.cuda.synchronize()
        # update gui info
        self.gui.train_time = starter.elapsed_time(ender)
        self.gui.loss = loss.item()
        
    def training_report(self, tb_writer, iteration, log_interval, loss, gaussians, images):
        if tb_writer:
            for key in loss.keys():
                tb_writer.add_scalar(f'train_loss_patches/{key}', loss[key], iteration)
            tb_writer.add_scalar('total_points', gaussians.get_xyz.shape[0], iteration)
        if(iteration % log_interval == 0):
            tb_writer.add_histogram("scene/opacity_histogram", gaussians.get_opacity, iteration)
            tb_writer.add_histogram("scene/scaling_histogram", torch.max(gaussians.get_scaling, dim=-1).values, iteration)
            grads = torch.norm(gaussians.xyz_gradient_accum, dim=-1)
            grads_filter = grads < torch.std(grads)*5
            denom = gaussians.denom[grads_filter]
            denom = denom[:,0]
            grads = grads[grads_filter]
            if(len(grads) > 1000):
                tb_writer.add_histogram("scene/view_gradients_histogram", grads, iteration)
                tb_writer.add_histogram("scene/view_norm_gradients_histogram", grads/denom, iteration)
                tb_writer.add_histogram("scene/denom_histogram", denom, iteration)
            for key in images.keys():
                img = images[key] 
                if(len(img.shape) == 4):
                    img = img[0]
                tb_writer.add_image(f"scene/{key}", img, iteration)
            if(self.config.vsd.enable):
                pipeline = DDIMPipeline(unet=self.unet, scheduler=self.guide.scheduler, v_pred = True)
                poses = torch.zeros(1, 16).to(self.device)
                with torch.no_grad():
                    images = pipeline(batch_size=1, output_type="numpy", shading = "albedo", pose=poses)
                    rgb = self.guide.decode_latents(images.type(torch.float16))
                    img = rgb.detach().permute(0,2,3,1).cpu().numpy()
                    img = torch.tensor(img.transpose(0,3,1,2), dtype=torch.float32)
                    tb_writer.add_image(f"scene/vsd_vpred", img[0], iteration)
                pipeline = DDIMPipeline(unet=self.unet, scheduler=self.guide.scheduler, v_pred = False)
                with torch.no_grad():
                    images = pipeline(batch_size=1, output_type="numpy", shading = "albedo", pose=poses)
                    rgb = self.guide.decode_latents(images.type(torch.float16))
                    img = rgb.detach().permute(0,2,3,1).cpu().numpy()
                    img = torch.tensor(img.transpose(0,3,1,2), dtype=torch.float32)
                    tb_writer.add_image(f"scene/vsd_npred", img[0], iteration)
            torch.cuda.empty_cache()