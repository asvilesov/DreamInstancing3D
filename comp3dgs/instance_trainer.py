import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import cv2
from tqdm import tqdm
import numpy as np
from easydict import EasyDict as edict
from omegaconf import OmegaConf
import random

import torch
from torch import nn
import torch.nn.functional as F


from utils.cam import OrbitCamera, gen_random_minicam
# from comp3dgs.gs_renderer import RendererComposition
from comp3dgs.instance_renderer2 import InstanceCompositionModel, GaussianRenderer, SH2RGB, RGB2SH
from utils.losses import lower_bound_knn, smooth_depth_loss, knn_loss, knn_rigidity_loss, knn_proximity_loss
from guidance.prompt_processor import StableDiffusionPromptProcessor
# from comp3dgs.instance_renderer import RendererInstanceComposition

import matplotlib.pyplot as plt 
from torch.utils.tensorboard import SummaryWriter

from comp3dgs.object_trainer import ObjectTrainer

from utils.pc_helper import knn
from utils.debug import timer

import scipy

from shapely.geometry import Polygon




def intersection_loss_max_vec(p1, p2, cent1, cent2, subsamp_len=5000, device="cuda:0"):
    ## p1 is the anchor object, p2 is the moving object

    p1_dash = p1[torch.randperm(p1.shape[0])[:subsamp_len]]
    p2_dash = p2[torch.randperm(p2.shape[0])[:subsamp_len]]


    with torch.no_grad():
        _, _, idx = knn(p2_dash, 1, p1_dash) 


    pt_vecs = p1_dash[idx.squeeze()] - p2_dash ## Vector from every point in p2 to closest point in p1

    pt_vecs = pt_vecs/torch.norm(pt_vecs,dim=-1,keepdim=True)

    ref_vecs = p1_dash[idx.squeeze()] - cent2.squeeze(0) ## Vector from center of p2 to closest point in p1

    ref_vecs = ref_vecs/torch.norm(ref_vecs,dim=-1,keepdim=True)

    dots = torch.sum(pt_vecs*ref_vecs,dim=-1)
    mask_dots = (dots<0).float()

    l_agg = -0.05*torch.sum(mask_dots*dots)
    count_agg = torch.sum(mask_dots)

    if count_agg==0:
        return torch.tensor(0, dtype=torch.float32, device=device)
    else:
        return l_agg/count_agg



## define the gravity loss

def gravity_loss(pts, floor=-0.5, comb_factor = 2000):
    disps = pts[..., 1] - floor

    mask_pos = (disps >= 0).float()
    mask_neg = (disps < 0).float()
    
    if torch.sum(mask_neg) == 0:
        return torch.sum(disps*mask_pos)/(torch.sum(mask_pos)+1e-5) 

    else:
        return (torch.sum(disps*mask_pos)/(torch.sum(mask_pos)+1e-5))/comb_factor + torch.sum(-1*disps*mask_neg)/(torch.sum(mask_neg)+1e-5)


    
def get_floor(pts, frac = 0.001):
    y_coords = pts[..., 1]
    # get median of lowest frac % of points
    y_coords = torch.sort(y_coords, dim=0)[0]
    y_coords = y_coords[:int(frac*y_coords.shape[0])]
    return torch.median(y_coords)

def get_bound(pts,direction='min',dim=0,frac = 0.001):
    cur_coords = pts[..., dim]
    # get median of lowest frac % of points
    if direction=='min':
        cur_coords = torch.sort(cur_coords, dim=0, descending=False)[0]
    else:
        cur_coords = torch.sort(cur_coords, dim=0, descending=True)[0]
    cur_coords = cur_coords[:int(frac*cur_coords.shape[0])]
    return torch.median(cur_coords)


def centering_correction(cent1, cent2, angle=np.pi/3, v_step=0.3):
    ## cent1 is the center of the anchor object, cent2 is the center of the secondary object (of interest)
    correct_x = (cent1[...,0]-cent2[...,0]).detach().squeeze()
    correct_z = (cent1[...,2]-cent2[...,2]).detach().squeeze()

    h_len = (correct_x.pow(2)+correct_z.pow(2)).sqrt().squeeze()
    v_len = torch.tan(torch.tensor([angle]))*h_len

    scl = v_step/v_len

    correct_x = correct_x*scl
    correct_z = correct_z*scl
    correct_y = v_step
    
    return correct_x, correct_y, correct_z




class InstanceCompTrainer(object):

    def __init__(self, config, groups, rand_init=True, vsd_unet=None) -> None:
        self.config = config
        self.curr_iter = 0
        self.max_iter = config.max_iter
        self.device = torch.device(config.device)
        self.optimizer = None
        self.rand_init = rand_init
        # init rendering params
        self.cam = OrbitCamera(self.config.render.ref_size, self.config.render.ref_size, 
                               r=self.config.render.radius, fovy=self.config.render.fovy)
        # init guidance
        self.config.negative_prompt = config.negative_prompt if(config.negative_prompt!="None") else ""
        self.guide = None
        self.text_embeds = None
        # init 3dgs
        self.renderer = GaussianRenderer(config=self.config) 
        self.gaussians = InstanceCompositionModel(config=self.config, groups=groups, device=self.device)
        # init tensorboard
        self.tb_writer = SummaryWriter(self.config.save_path)
        # GUI data
        self.training = self.config.training
        self.gui = edict({'train_time': 0, 'loss': 0})

        if(self.config.load_path is None):

            # initialize object Trainers
            self.obj_trainers = []

            self.prepare_train()

            if(self.config.vsd.enable):
                if(vsd_unet is not None):
                    self.unet = vsd_unet[0]
                    self.unet_scheduler = vsd_unet[1]
                    self.unet_optimizer = vsd_unet[2]
                else:
                    self.initialize_lora_unet()
            else:
                self.unet = None

        # clear cache
        torch.cuda.empty_cache()

    def prepare_train(self):
        # load guidance
        print(f"[INFO] loading SD...")
        from guidance.sd_utils import StableDiffusion
        self.guide = StableDiffusion(self.config)
        print(f"[INFO] loaded SD!")
        # prepare embeddings
        with torch.no_grad():
            self.text_embeds = self.guide.get_text_embeds([self.config.prompt], [self.config.negative_prompt])

        self.instance_prompt_processors = []
        for i in range(len(self.gaussians.groups)):
            obj_config = self.gaussians.groups[i]["config"]
            negative_prompt = obj_config.negative_prompt if(obj_config.negative_prompt!="None") else ""
            self.gaussians.groups[i]["embeds"] = self.guide.get_text_embeds([obj_config.prompt], [negative_prompt])
            self.instance_prompt_processors.append(StableDiffusionPromptProcessor(
                                                                self.gaussians.groups[i]["config"], 
                                                                self.guide.pipe)
            )

        # ---------- Setup Training ------------
        # initialize optimizer
        self.configure_instance_trainer()

        self.scaler = torch.cuda.amp.GradScaler()

    def configure_instance_trainer(self):
        self.gaussians.training_setup()
        self.gaussians.active_sh_degree = self.gaussians.sh_degree
        self.optimizer = self.gaussians.optimizer

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

            device = torch.device(self.config.device)
            class LoraUnet(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.unet = _unet
                    self.sample_size = 64
                    self.in_channels = 4
                    self.device = device
                    self.dtype = torch.float32
                def forward(self,x,t, textemb, c=None,shading="albedo"):
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
    
    def vsd_update(self, latents):
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
                print(latents_noisy.dtype, noise.dtype, timesteps.dtype, poses.dtype)
                if vsd_conf.q_cond:
                    model_output = self.unet(latents_noisy, timesteps, c = poses).sample
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

    def get_camera_views(self, gaussians, config, class_id):
        images = []
        uc_images = []
        poses = []
        vers = []
        hors = []
        cam_rads = []
        embeddings = []
        for _ in range(config.batch_size):
            (ver, hor, pose), cur_cam = gen_random_minicam(self.cam, config.render.radius, 
                                                    center_elevation=config.render.elevation,
                                                    min_ver = config.render.min_ver,
                                                    max_ver = config.render.max_ver,
                                                    device=self.device,
                                                    fov_jitter=True)
            if(np.random.rand() < 0.5):
                bg_color = np.random.rand(3)
            else:
                bg_color = np.ones(3)
            out = self.renderer.render(gaussians, cur_cam, bg_color = bg_color)
            image = out["image"].unsqueeze(0)# [1, 3, H, W] in [0, 1]
            uc_image = out["unclamped_img"].unsqueeze(0)
            vers.append(ver)
            hors.append(hor-config.reorient)
            cam_rads.append(self.cam.radius)
            images.append(image)
            uc_images.append(uc_image)
            poses.append(pose)
            embeddings.append(self.gaussians.groups[class_id]["embeds"])
        images = torch.cat(images, dim=0)
        uc_images = torch.cat(uc_images, dim=0)
        cam_data = [torch.tensor(hors, device=self.device), torch.tensor(vers, device=self.device), torch.tensor(cam_rads, device=self.device)]
        poses = torch.tensor(poses)
        poses = torch.flatten(poses, start_dim=1).to(self.device)
        embeddings = torch.cat(embeddings, dim=0)

        return images, out, poses, cam_data, embeddings, uc_images

    def get_camera_views_scene(self, gaussians, config, class_id, obj_only=False):
        images = []
        uc_images = []
        poses = []
        vers = []
        hors = []
        cam_rads = []
        embeddings = []
        for _ in range(config.batch_size):
            offset, scale, rot_deg, gauss_idxs = self.gaussians.grab_random_object_bounds(class_id)

            (ver, hor, pose), cur_cam = gen_random_minicam(self.cam, config.render.radius * scale, 
                                                    center_elevation=config.render.elevation,
                                                    min_ver = config.render.min_ver,
                                                    max_ver = config.render.max_ver,
                                                    device=self.device,
                                                    fov_jitter=True,
                                                    offset=offset)
            if(np.random.rand() < 0.5):
                bg_color = np.random.rand(3)
            else:
                bg_color = np.ones(3)
            if(obj_only):
                out = self.renderer.render(gaussians, cur_cam, bg_color = bg_color, gauss_idxs=gauss_idxs)
            else:
                out = self.renderer.render(gaussians, cur_cam, bg_color = bg_color)

            image = out["image"].unsqueeze(0)# [1, 3, H, W] in [0, 1]
            uc_image = out["unclamped_img"].unsqueeze(0)
            images.append(image)
            uc_images.append(uc_image)
            poses.append(pose)
            vers.append(ver)
            hors.append(hor-config.reorient+rot_deg)
            cam_rads.append(self.cam.radius)
            embeddings.append(self.gaussians.groups[class_id]["embeds"])
        images = torch.cat(images, dim=0)
        uc_images = torch.cat(uc_images, dim=0)
        poses = torch.tensor(poses)
        poses = torch.flatten(poses, start_dim=1).to(self.device)
        cam_data = [torch.tensor(hors, device=self.device), torch.tensor(vers, device=self.device), torch.tensor(cam_rads, device=self.device)]
        embeddings = torch.cat(embeddings, dim=0)

        return images, out, poses, cam_data, gauss_idxs, embeddings, uc_images
    
    def get_camera_views_scene_bg(self, gaussians, config, class_id, obj_only=False):
        images = []
        images_bg = []
        uc_images = []
        poses = []
        vers = []
        hors = []
        cam_rads = []
        embeddings = []
        for _ in range(config.batch_size):
            offset, scale, rot_deg, gauss_idxs = self.gaussians.grab_random_object_bounds(class_id)

            (ver, hor, pose), cur_cam = gen_random_minicam(self.cam, config.render.radius * scale, 
                                                    center_elevation=config.render.elevation,
                                                    min_ver = config.render.min_ver,
                                                    max_ver = config.render.max_ver,
                                                    device=self.device,
                                                    fov_jitter=True,
                                                    offset=offset)
            if(np.random.rand() < 0.5):
                bg_color = np.random.rand(3)
            else:
                bg_color = np.ones(3)

            out_bg = self.renderer.render(gaussians, cur_cam, bg_color = bg_color)
            out = self.renderer.render(gaussians, cur_cam, bg_color = bg_color, gauss_idxs=gauss_idxs)

            image = out["image"].unsqueeze(0)# [1, 3, H, W] in [0, 1]
            image_bg = out_bg["image"].unsqueeze(0)# [1, 3, H, W] in [0, 1]
            uc_image = out["unclamped_img"].unsqueeze(0)
            images.append(image)
            images_bg.append(image_bg)
            uc_images.append(uc_image)
            poses.append(pose)
            vers.append(ver)
            hors.append(hor-config.reorient+rot_deg)
            cam_rads.append(self.cam.radius)
            embeddings.append(self.gaussians.groups[class_id]["embeds"])
        images = torch.cat(images, dim=0)
        images_bg = torch.cat(images_bg, dim=0)
        # images = torch.cat([images, images_bg], dim=0)
        images = torch.cat([images_bg, images], dim=0)
        uc_images = torch.cat(uc_images, dim=0)
        poses = torch.tensor(poses)
        poses = torch.flatten(poses, start_dim=1).to(self.device)
        cam_data = [torch.tensor(hors, device=self.device), torch.tensor(vers, device=self.device), torch.tensor(cam_rads, device=self.device)]
        embeddings = torch.cat(embeddings, dim=0)

        return images, out, poses, cam_data, gauss_idxs, embeddings, uc_images

    def update_camera_views(self):
        if(self.curr_iter % 1500 == 0):
            for i in range(len(self.gaussians.groups)):
                if(self.gaussians.groups[i]["type"] != "floor"):
                    self.gaussians.groups[i]["config"].render.radius -= 0.1

    
    def train_step(self):
        # A. Initialization
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        self.curr_iter += 1
        loss = 0
        loss_dict = {}
        tb_imgs = {}
        if(self.gaussians.groups[-1]["type"] == "globe"):
            num_objs = len(self.gaussians.groups) - 1 #remove globe
        else:
            num_objs = len(self.gaussians.groups)

        stop_canonical_object = 2000
        warmup = 200

        # if(self.curr_iter >= stop_canonical_object):
        #     for i in range(num_objs):
        #         self.gaussians.groups[i]["config"].batch_size = 4
        
        self.update_camera_views()

        # Test whether coloring matters
        for optimizer in self.gaussians.optimizer_list:
            for group in optimizer.param_groups:
                if(group["name"] == "f_dc"):
                    print(group["params"][0].shape)
                    if(self.curr_iter == stop_canonical_object):
                        group["params"][0].data = RGB2SH(torch.rand_like(group["params"][0]))


        # B. Run Forward Pass for Scene/Objects:
        masks_list = []
        meta_data = []
        for i in range(num_objs):
            obj_type = self.gaussians.groups[i]["type"]
            # disable gradients on objects not being currently trained 
            for j in range(num_objs):
                if(j != i and obj_type != "floor"):
                    self.gaussians.disable_gradients_delta(group_id=j)
            # Update Learning Rates
            if(obj_type == "floor"):
                self.gaussians.update_learning_rate_floor(self.curr_iter)
            # if(self.curr_iter > stop_canonical_object):
            #     lr_dict = self.gaussians.update_learning_rate(self.curr_iter-stop_canonical_object)
            #     print(lr_dict)

                    
            # Update and Cache scene gaussians
            self.gaussians.need_update()
            obj_loss = 0
            gaussian_object_metadata = []
            # Sample Scene
            obj_config = self.gaussians.groups[i]["config"]
            self.guide.set_text_embeds(self.gaussians.groups[i]["embeds"])

            scene_images, out, pose, cam_data, gauss_idxs, embeddings, uc_images = self.get_camera_views_scene(self.gaussians, 
                                                                                    self.gaussians.groups[i]['config'], 
                                                                                    i)
            gaussian_object_metadata.append({"images": scene_images, "out": out, "pose": pose, 
                                                "cam_data": cam_data, "gauss_idxs": gauss_idxs, "embeds": embeddings, "uc_images": uc_images})    
                
            # if(self.curr_iter <= stop_canonical_object and obj_type != "floor"):
            #     scene_images, out, pose, cam_data, embeddings, uc_images = self.get_camera_views(self.gaussians.gaussian_objects[i], 
            #                                                     self.gaussians.groups[i]['config'],
            #                                                     i)
            #     gaussian_object_metadata.append({"images": scene_images, "out": out, "pose": pose,
            #                                         "cam_data": cam_data, "embeds": embeddings, "uc_images": uc_images})
            # if(self.curr_iter > stop_canonical_object or obj_type == "floor"):
            #     # Sample from the scene
            #     if((self.curr_iter % 2 == 0) or obj_type == "floor"):
            #         scene_images, out, pose, cam_data, gauss_idxs, embeddings, uc_images = self.get_camera_views_scene(self.gaussians, 
            #                                                                         self.gaussians.groups[i]['config'], 
            #                                                                         i)
            #         gaussian_object_metadata.append({"images": scene_images, "out": out, "pose": pose, 
            #                                             "cam_data": cam_data, "gauss_idxs": gauss_idxs, "embeds": embeddings, "uc_images": uc_images})    
            #     # Sample from the object only
            #     if((self.curr_iter % 2 == 1) and obj_type != "floor"):
            #         scene_images, out, pose, cam_data, gauss_idxs, embeddings, uc_images = self.get_camera_views_scene(self.gaussians, 
            #                                                                         self.gaussians.groups[i]['config'], 
            #                                                                         i, obj_only=True)
            #         gaussian_object_metadata.append({"images": scene_images, "out": out, "pose": pose, 
            #                                             "cam_data": cam_data, "embeds": embeddings, "uc_images": uc_images})    
        
            # Combine into batch
            batch_size = len(gaussian_object_metadata)
            scene_images_batch = torch.cat([gaussian_object_metadata[i]["images"] for i in range(batch_size)], dim=0)
            pose_batch = torch.cat([gaussian_object_metadata[i]["pose"] for i in range(batch_size)], dim=0)
            hor_batch = torch.cat([gaussian_object_metadata[i]["cam_data"][0] for i in range(batch_size)], dim=0)
            ver_batch = torch.cat([gaussian_object_metadata[i]["cam_data"][1] for i in range(batch_size)], dim=0)
            rad_batch = torch.cat([gaussian_object_metadata[i]["cam_data"][2] for i in range(batch_size)], dim=0)
            cam_batch = [hor_batch, ver_batch, rad_batch] if obj_config.use_view_dependent_prompt else None
            embeddings_batch = torch.cat([gaussian_object_metadata[i]["embeds"] for i in range(batch_size)], dim=0)
            embeddings_batch = torch.cat([embeddings_batch[0::2], embeddings_batch[1::2]], dim=0)

            course_sd_loss, latents = self.guide.train_step(scene_images_batch, self.curr_iter,
                                                        guidance_scale=obj_config.course_guidance,
                                                        camera_data=cam_batch,
                                                        q_unet=self.unet,
                                                        pose = pose_batch,
                                                        embeddings = embeddings_batch,
                                                        prompt_processor = self.instance_prompt_processors[i])

            obj_loss += obj_config.course_sd_loss*course_sd_loss
            obj_loss.backward()

            #   optimize canonical objects
            if(self.curr_iter <= stop_canonical_object):
                self.gaussians.gaussian_objects[i].optimizer.step() #ablation 1
            if(self.gaussians.groups[i]["type"] == "floor"):
                if(self.gaussians.groups[-1]["type"] == "globe"):
                    self.gaussians.gaussian_objects[-1].optimizer.step() #optimize globe
            for j in range(len(self.gaussians.groups)): #ablation 1
                self.gaussians.gaussian_objects[j].optimizer.zero_grad()
            #   optimize instance objects
            if(self.curr_iter > stop_canonical_object):
                offset_obj = 0
                for j in range(len(self.gaussians.groups)):
                    if(self.gaussians.groups[j]["type"] == "floor"):
                        self.gaussians.groups[j]["sim"] = 0.8
                    elif(self.gaussians.groups[j]["type"] == "object"):
                        self.gaussians.groups[j]["sim"] = 0.0
                    for k in range(self.gaussians.groups[j]["N"]):
                        self.gaussians.optimizer_list[offset_obj+k].step()
                        self.gaussians.optimizer_list[offset_obj+k].zero_grad()
                    offset_obj += self.gaussians.groups[j]["N"]

            meta_data.append(gaussian_object_metadata)
            
            # enable gradients again
            for j in range(num_objs):
                self.gaussians.enable_gradients_delta(group_id=j)

            #   update tensorboard
            loss_dict[f'course_sd_loss_{i}'] = course_sd_loss.item()
            if(len(gaussian_object_metadata) == 1):
                tb_imgs[f'obj_only_{i}_img'] = gaussian_object_metadata[0]["images"][0]
            else:
                tb_imgs[f'obj_{i}_img'] = gaussian_object_metadata[0]["images"][0]
                tb_imgs[f'obj_only_{i}_img'] = gaussian_object_metadata[1]["images"][0]

        # step gaussians meta data
        if(self.curr_iter <= stop_canonical_object):
            for i in range(num_objs):
                if("gauss_idxs" in meta_data[i][0].keys()):
                    masks = self.gaussians.gaussian_objects[i].step_gaussians(self.curr_iter, meta_data[i][0]["out"], 
                                                                            meta_data[i][0]["gauss_idxs"])
                else:
                    masks = self.gaussians.gaussian_objects[i].step_gaussians(self.curr_iter, meta_data[i][0]["out"])
                masks_list.append(masks)
            # step gaussians for scene's delta parameters
            if self.curr_iter % self.config.gs3d.densification_interval == 0 and self.curr_iter <= self.config.gs3d.density_end_iter:
                self.gaussians.densify_and_prune_instances(masks_list)    

        # VSD Update
        vsd_conf = self.config.vsd
        if self.curr_iter % vsd_conf.K2 == 0 and vsd_conf.enable:
            for _ in range(vsd_conf.K):
                self.unet_optimizer.zero_grad()
                timesteps = torch.randint(0, 1000, (latents.shape[0],), device=self.device).long() # temperarily hard-coded for simplicity
                with torch.no_grad():
                    latents_clean = latents.expand(latents.shape[0], latents.shape[1], latents.shape[2], latents.shape[3]).contiguous()
                    latents_clean = latents_clean.type(self.unet.dtype)
                    if vsd_conf.q_cond:
                        if random.random() < 0.1:
                            pose_batch = torch.zeros(latents.shape[0], 16).to(self.device)
                noise = torch.randn(latents_clean.shape, device=self.device, dtype=self.unet.dtype)
                latents_noisy = self.guide.scheduler.add_noise(latents_clean, noise, timesteps)
                latents_noisy = latents_noisy.type(self.unet.dtype)
                pose_batch = pose_batch.type(self.unet.dtype)
                if vsd_conf.q_cond:
                    model_output = self.unet(latents_noisy, timesteps, textemb=embeddings_batch, c = pose_batch).sample
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

        # D. Save Data
        #   save model
        if(self.curr_iter % self.config.save_interval == 0):
            save_path = os.path.join(self.config.save_path, f"iter_{self.curr_iter}")
            print(f"Saving Checkpoint - {save_path} ...")
            self.gaussians.save_ply_global(save_path)
        #   save tensorboard
        training_report(self.tb_writer, self.curr_iter, self.config.log_interval, 
                        loss_dict, self.gaussians, tb_imgs)
        self.gui.loss = 0 #loss.item()
        ender.record()
        torch.cuda.synchronize()
        self.gui.train_time = starter.elapsed_time(ender)

    def train_step_mod(self):
        # A. Initialization
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        self.curr_iter += 1
        loss = 0
        loss_dict = {}
        tb_imgs = {}
        if(self.gaussians.groups[-1]["type"] == "globe"):
            num_objs = len(self.gaussians.groups) - 1 #remove globe
        else:
            num_objs = len(self.gaussians.groups)

        stop_canonical_object = 2000
        warmup = 200

        # if(self.curr_iter >= stop_canonical_object):
        #     for i in range(num_objs):
        #         self.gaussians.groups[i]["config"].batch_size = 4
        
        self.update_camera_views()

        # Test whether coloring matters
        for optimizer in self.gaussians.optimizer_list:
            for group in optimizer.param_groups:
                if(group["name"] == "f_dc"):
                    print(group["params"][0].shape)
                    if(self.curr_iter == stop_canonical_object):
                        group["params"][0].data = RGB2SH(torch.rand_like(group["params"][0]))


        # B. Run Forward Pass for Scene/Objects:
        masks_list = []
        meta_data = []
        for i in range(num_objs):
            obj_type = self.gaussians.groups[i]["type"]
            # disable gradients on objects not being currently trained 
            for j in range(num_objs):
                if(j != i and obj_type != "floor"):
                    self.gaussians.disable_gradients_delta(group_id=j)
            # Update Learning Rates
            if(obj_type == "floor"):
                self.gaussians.update_learning_rate_floor(self.curr_iter)
            # if(self.curr_iter > stop_canonical_object):
            #     lr_dict = self.gaussians.update_learning_rate(self.curr_iter-stop_canonical_object)
            #     print(lr_dict)

                    
            # Update and Cache scene gaussians
            self.gaussians.need_update()
            obj_loss = 0
            gaussian_object_metadata = []
            # Sample Scene
            obj_config = self.gaussians.groups[i]["config"]
            self.guide.set_text_embeds(self.gaussians.groups[i]["embeds"])

            scene_images, out, pose, cam_data, gauss_idxs, embeddings, uc_images = self.get_camera_views_scene_bg(self.gaussians, 
                                                                                    self.gaussians.groups[i]['config'], 
                                                                                    i)
            gaussian_object_metadata.append({"images": scene_images, "out": out, "pose": pose, 
                                                "cam_data": cam_data, "gauss_idxs": gauss_idxs, "embeds": embeddings, "uc_images": uc_images})    
                
            print(scene_images.shape)
            # Combine into batch
            batch_size = len(gaussian_object_metadata)
            scene_images_batch = torch.cat([gaussian_object_metadata[i]["images"] for i in range(batch_size)], dim=0)
            pose_batch = torch.cat([gaussian_object_metadata[i]["pose"] for i in range(batch_size)], dim=0)
            hor_batch = torch.cat([gaussian_object_metadata[i]["cam_data"][0] for i in range(batch_size)], dim=0)
            ver_batch = torch.cat([gaussian_object_metadata[i]["cam_data"][1] for i in range(batch_size)], dim=0)
            rad_batch = torch.cat([gaussian_object_metadata[i]["cam_data"][2] for i in range(batch_size)], dim=0)
            cam_batch = [hor_batch, ver_batch, rad_batch] if obj_config.use_view_dependent_prompt else None
            embeddings_batch = torch.cat([gaussian_object_metadata[i]["embeds"] for i in range(batch_size)], dim=0)
            embeddings_batch = torch.cat([embeddings_batch[0::2], embeddings_batch[1::2]], dim=0)

            course_sd_loss, latents = self.guide.train_step(scene_images_batch, self.curr_iter,
                                                        guidance_scale=obj_config.course_guidance,
                                                        camera_data=cam_batch,
                                                        q_unet=self.unet,
                                                        pose = pose_batch,
                                                        embeddings = embeddings_batch,
                                                        prompt_processor = self.instance_prompt_processors[i])

            obj_loss += obj_config.course_sd_loss*course_sd_loss
            obj_loss.backward()

            #   optimize canonical objects
            if(self.curr_iter <= stop_canonical_object):
                if(self.gaussians.groups[i]["type"] == "object" and self.curr_iter > warmup):
                    self.gaussians.gaussian_objects[i].optimizer.step() #ablation 1
                else:
                    self.gaussians.gaussian_objects[i].optimizer.step() #ablation 1
            if(self.gaussians.groups[i]["type"] == "floor"):
                if(self.gaussians.groups[-1]["type"] == "globe"):
                    self.gaussians.gaussian_objects[-1].optimizer.step() #optimize globe
            for j in range(len(self.gaussians.groups)): #ablation 1
                self.gaussians.gaussian_objects[j].optimizer.zero_grad()
            #   optimize instance objects
            if(self.curr_iter > stop_canonical_object):
                offset_obj = 0
                for j in range(len(self.gaussians.groups)):
                    if(self.gaussians.groups[j]["type"] == "floor"):
                        self.gaussians.groups[j]["sim"] = 0.8
                    elif(self.gaussians.groups[j]["type"] == "object"):
                        self.gaussians.groups[j]["sim"] = 0.0
                    for k in range(self.gaussians.groups[j]["N"]):
                        self.gaussians.optimizer_list[offset_obj+k].step()
                        self.gaussians.optimizer_list[offset_obj+k].zero_grad()
                    offset_obj += self.gaussians.groups[j]["N"]

            meta_data.append(gaussian_object_metadata)
            
            # enable gradients again
            for j in range(num_objs):
                self.gaussians.enable_gradients_delta(group_id=j)

            #   update tensorboard
            loss_dict[f'course_sd_loss_{i}'] = course_sd_loss.item()
            if(len(gaussian_object_metadata) == 1):
                tb_imgs[f'obj_only_{i}_img'] = gaussian_object_metadata[0]["images"][0]
            else:
                tb_imgs[f'obj_{i}_img'] = gaussian_object_metadata[0]["images"][0]
                tb_imgs[f'obj_only_{i}_img'] = gaussian_object_metadata[1]["images"][0]

        # step gaussians meta data
        if(self.curr_iter <= stop_canonical_object):
            for i in range(num_objs):
                if("gauss_idxs" in meta_data[i][0].keys()):
                    masks = self.gaussians.gaussian_objects[i].step_gaussians(self.curr_iter, meta_data[i][0]["out"], 
                                                                            meta_data[i][0]["gauss_idxs"])
                else:
                    masks = self.gaussians.gaussian_objects[i].step_gaussians(self.curr_iter, meta_data[i][0]["out"])
                masks_list.append(masks)
            # step gaussians for scene's delta parameters
            if self.curr_iter % self.config.gs3d.densification_interval == 0 and self.curr_iter <= self.config.gs3d.density_end_iter:
                self.gaussians.densify_and_prune_instances(masks_list)    

        # VSD Update
        vsd_conf = self.config.vsd
        if self.curr_iter % vsd_conf.K2 == 0 and vsd_conf.enable:
            for _ in range(vsd_conf.K):
                self.unet_optimizer.zero_grad()
                timesteps = torch.randint(0, 1000, (latents.shape[0],), device=self.device).long() # temperarily hard-coded for simplicity
                with torch.no_grad():
                    latents_clean = latents.expand(latents.shape[0], latents.shape[1], latents.shape[2], latents.shape[3]).contiguous()
                    latents_clean = latents_clean.type(self.unet.dtype)
                    if vsd_conf.q_cond:
                        if random.random() < 0.1:
                            pose_batch = torch.zeros(latents.shape[0], 16).to(self.device)
                noise = torch.randn(latents_clean.shape, device=self.device, dtype=self.unet.dtype)
                latents_noisy = self.guide.scheduler.add_noise(latents_clean, noise, timesteps)
                latents_noisy = latents_noisy.type(self.unet.dtype)
                pose_batch = pose_batch.type(self.unet.dtype)
                if vsd_conf.q_cond:
                    model_output = self.unet(latents_noisy, timesteps, textemb=embeddings_batch, c = pose_batch).sample
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

        # D. Save Data
        #   save model
        if(self.curr_iter % self.config.save_interval == 0):
            save_path = os.path.join(self.config.save_path, f"iter_{self.curr_iter}")
            print(f"Saving Checkpoint - {save_path} ...")
            self.gaussians.save_ply_global(save_path)
        #   save tensorboard
        training_report(self.tb_writer, self.curr_iter, self.config.log_interval, 
                        loss_dict, self.gaussians, tb_imgs)
        self.gui.loss = 0 #loss.item()
        ender.record()
        torch.cuda.synchronize()
        self.gui.train_time = starter.elapsed_time(ender)
        
def training_report(tb_writer, iteration, log_interval, loss, gaussians, images):
    if tb_writer:
        for key in loss.keys():
            tb_writer.add_scalar(f'train_loss_patches/{key}', loss[key], iteration)
        tb_writer.add_scalar('total_points', gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    # if iteration in testing_iterations:
    if(iteration % log_interval == 1):
        tb_writer.add_histogram("scene/opacity_histogram", gaussians.get_opacity, iteration)
        tb_writer.add_histogram("scene/scaling_histogram", torch.norm(gaussians.get_scaling, dim=-1), iteration)
        tb_writer.add_histogram("scene/scaling_log_histogram", torch.log(torch.norm(gaussians.get_scaling, dim=-1)+1e-4), iteration)
        for key in images.keys():
            img = images[key] 
            if(len(img.shape) == 4):
                img = img[0]
            tb_writer.add_image(f"scene/{key}", img, iteration)
        torch.cuda.empty_cache()



 # if(self.curr_iter > self.config.start_iter or self.curr_iter % 100==1):
        #     for i in range(num_objs):
        #         self.guide.set_text_embeds(self.gaussians.groups[i]["embeds"])
        #         obj_config = self.gaussians.groups[i]["config"]
        #         scene_images, out, pose, gauss_idxs = self.get_camera_views_scene(self.gaussians, 
        #                                                                         self.gaussians.groups[i]['config'], 
        #                                                                         i)
        #         course_sd_loss, _ = self.guide.train_step(scene_images, step_ratio, 
        #                                                   guidance_scale=obj_config.course_guidance,
        #                                                   q_unet=self.unet,
        #                                                   pose = pose)
        #         loss_dict[f'course_sd_loss_{i}'] = course_sd_loss.item()
        #         obj_loss += obj_config.course_sd_loss*course_sd_loss
        #         if(obj_config.point_e.knn_loss):
        #             loss_knn =  knn_loss(mean=self.gaussians.gaussian_objects[i].get_xyz,
        #                             radius=self.gaussians.gaussian_objects[i].get_scaling.max(dim=1).values.detach(),
        #                             mean2=self.gaussians.gaussian_objects[i].anchor_knn, K=3)
        #             loss_dict[f'loss_knn_pointe_{i}'] = loss_knn.item()
        #             obj_loss += obj_config.point_e.knn_loss*loss_knn
        #         gaussian_object_metadata.append({"images": scene_images, "out": out, "pose": pose, "gauss_idxs": gauss_idxs})
        # if(self.config.train_obj):
        #     for i in range(num_objs):
        #         self.guide.set_text_embeds(self.gaussians.groups[i]["embeds"])
        #         obj_config = self.gaussians.groups[i]["config"]
        #         scene_images, out, pose = self.get_camera_views(self.gaussians.gaussian_objects[i], 
        #                                                                         self.gaussians.groups[i]['config'])
        #         course_sd_loss, _ = self.guide.train_step(scene_images, step_ratio, 
        #                                                   guidance_scale=obj_config.course_guidance,
        #                                                   q_unet=self.unet,
        #                                                   pose = pose)
        #         loss_dict[f'course_sd_loss_obj_{i}'] = course_sd_loss.item()
        #         obj_loss += obj_config.course_sd_loss*course_sd_loss
        #         gaussian_object_metadata.append({"images": scene_images, "out": out, "pose": pose})




        #             obj_loss += self.config.course_sd_loss*course_sd_loss

        #     obj_loss.backward()

        #     # C. Optimize step
        #     #   optimize scene
        #     self.optimizer.step()
        #     self.optimizer.zero_grad()
        #     #   optimize objects
        #     for i in range(len(self.gaussians.groups)):
        #         self.gaussians.gaussian_objects[i].optimizer.step()
        #     #   update gaussian clone/split + lr
        #     masks_list = []
        #     if(self.curr_iter > self.config.start_iter or self.curr_iter % 100==1):
        #         masks = self.gaussians.gaussian_objects[i].step_gaussians(self.curr_iter, gaussian_object_metadata[i]["out"], 
        #                                                       gaussian_object_metadata[i]["gauss_idxs"])
        #     else:
        #         masks = self.gaussians.gaussian_objects[i].step_gaussians(self.curr_iter, gaussian_object_metadata[i]["out"])
        #     masks_list.append(masks)
        # for i in range(len(self.gaussians.groups)):
        #     self.gaussians.gaussian_objects[i].optimizer.zero_grad()
        # self.gui.loss = 0 #loss.item()
        # if self.curr_iter % self.config.gs3d.densification_interval == 0 and self.curr_iter <= self.config.gs3d.density_end_iter:
        #     self.gaussians.densify_and_prune_instances(masks_list)            

        # # D. Save Data
        # #   save model
        # if(self.curr_iter % self.config.save_interval == 0):
        #     print("Saving Checkpoint...")
        #     ply_save_path = os.path.join(self.config.save_path, f"iter_{self.curr_iter}_{self.config.prompt}.ply")
        #     glob_ply_save_path = os.path.join(self.config.save_path, f"iter_{self.curr_iter}_{self.config.prompt}_glob.ply")
        #     self.gaussians.save_ply(ply_save_path)
        #     self.gaussians.save_ply_global(glob_ply_save_path)
                
        # #   update tensorboard
        # loss_dict[f'course_sd_loss'] = course_sd_loss.item()
        # for i in range(num_objs):
        #     if(self.curr_iter > self.config.start_iter or self.curr_iter % 100==1):
        #         tb_imgs[f'obj_{i}_img'] = gaussian_object_metadata[i]["images"][0]
        #         if(self.config.train_obj):
        #             tb_imgs[f'obj_only_{i}_img'] = gaussian_object_metadata[i+num_objs]["images"][0]
        #     else:
        #         tb_imgs[f'obj_only_{i}_img'] = gaussian_object_metadata[i]["images"][0]

        # training_report(self.tb_writer, self.curr_iter, self.config.log_interval, 
        #                 loss_dict, self.gaussians, tb_imgs)
        # #   update gui info
        # ender.record()
        # torch.cuda.synchronize()
        # self.gui.train_time = starter.elapsed_time(ender)