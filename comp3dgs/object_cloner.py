import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from torch.utils.tensorboard import SummaryWriter
from pytorch3d.ops import knn_points
import os
from omegaconf import OmegaConf

from comp3dgs.gs_renderer import Renderer, MiniCam
from utils.cam import orbit_camera, OrbitCamera, gen_random_minicam
from point_e.point_e_guidance import PointEGuidance

from utils.pc_helper import farthest_point_sampling
from utils.losses import lower_bound_knn, smooth_depth_loss, knn_loss
from utils.alpha_hull import alpha_hull_ids
from utils.debug import timer

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

class ObjectCloner(object):

    def __init__(self, config, guidance=None, pointe_guidance=None) -> None:

        self.config = config
        self.curr_iter = 0
        self.max_iter = config.max_iter
        self.course_steps = config.course_steps
        self.device = torch.device(config.device)
        self.optimizer = None
        # init rendering params
        self.cam = OrbitCamera(self.config.render.ref_size, self.config.render.ref_size, 
                               r=self.config.render.radius, fovy=self.config.render.fovy)
        self.renderer_ref = Renderer(config=self.config)
        self.renderer = Renderer(config=self.config)
        # init tensorboard
        self.tb_writer = SummaryWriter(self.config.save_path)
        # GUI data
        self.training = self.config.training
        self.gui = edict({'train_time': 0, 'loss': 0})
        # save config
        with open(os.path.join(self.config.save_path, "config.yaml"), 'w') as f:
            OmegaConf.save(config, f.name)

        self.prepare_train()

    def prepare_train(self):
        # ---------- Setup Training ------------
        # initialize 3dgs
        if(self.config.load_path is not None):
            self.renderer_ref.initialize(self.config.load_path) 
        else:
            raise Exception("Must provide load_path to initialize 3dgs!")
        self.renderer_ref.gaussians.training_setup(self.config.gs3d)
        self.renderer_ref.gaussians.active_sh_degree = self.renderer_ref.gaussians.max_sh_degree

        self.renderer.initialize(num_pts=self.config.gs3d.num_pts, radius=self.config.gs3d.init_radius)
        self.renderer.gaussians.training_setup(self.config.gs3d)
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
        self.optimizer = self.renderer.gaussians.optimizer

    @torch.no_grad()
    def step_camera(self):
        # calculate bounding box of scene
        xyz = self.renderer.gaussians.get_xyz
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

    def step_gaussians(self, render_data):
        # densify and prune quickly at the start
        if self.curr_iter >= self.config.gs3d.density_start_iter and self.curr_iter <= self.config.gs3d.density_end_iter:
            viewspace_point_tensor, visibility_filter, radii = render_data["viewspace_points"], render_data["visibility_filter"], render_data["radii"]
            self.renderer.gaussians.max_radii2D[visibility_filter] = torch.max(self.renderer.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            self.renderer.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if self.curr_iter % self.config.gs3d.densification_interval == 0:
                self.renderer.gaussians.densify_and_prune(self.config.gs3d.densify_grad_threshold, min_opacity=0.05, extent=2.0, max_screen_size=1)
                if(self.config.dynamic_camera):
                    self.step_camera()

            if self.curr_iter % self.config.gs3d.opacity_reset_interval == 0:
                self.renderer.gaussians.reset_opacity()

    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        
        # update iter
        self.curr_iter += 1

        # update lr
        self.renderer.gaussians.update_learning_rate(self.curr_iter)
        loss = 0
        loss_dict = {}
        tb_imgs = {}

        enable_fov_jitter = self.config.render.fov_jitter.enable and \
                                    self.curr_iter > self.config.render.fov_jitter.step_start
        
        _, cur_cam = gen_random_minicam(self.cam, self.cam.radius, 
                                                center_elevation=self.config.render.elevation,
                                                device=self.device, fov_jitter=enable_fov_jitter)
        bg_color = np.random.rand(3)
        clone_out = self.renderer.render(cur_cam, bg_color = bg_color)
        clone_img = clone_out["image"].unsqueeze(0)# [1, 3, H, W] in [0, 1]
        with torch.no_grad():
            orig_out = self.renderer_ref.render(cur_cam, bg_color = bg_color)
            orig_img = orig_out["image"].unsqueeze(0)# [1, 3, H, W] in [0, 1]

        if(self.curr_iter%500==1):
            print(f"Original={len(self.renderer_ref.gaussians._xyz)}, New={len(self.renderer.gaussians._xyz)}")

        # calculate loss
        mse_loss = F.mse_loss(clone_img, orig_img)
        loss_dict["mse"] = mse_loss.item()

        loss += mse_loss

        # optimize step
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # update gaussians clones/splits/prunes
        self.step_gaussians(clone_out)  

        # update tensorboard
        tb_imgs['clone_img'] = clone_img
        training_report(self.tb_writer, self.curr_iter, self.config.log_interval, 
                        loss_dict, self.renderer.gaussians, tb_imgs)
        
        # save model
        if(self.curr_iter % self.config.save_interval == 0):
            ply_save_path = os.path.join(self.config.save_path, f"iter_{self.curr_iter}_{self.config.prompt}.ply")
            self.renderer.gaussians.save_ply(ply_save_path)

        ender.record()
        torch.cuda.synchronize()
        # update gui info
        self.gui.train_time = starter.elapsed_time(ender)
        self.gui.loss = loss.item()

    def eval_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        num_steps = 500
        psnr_list = []
        mse_list = []
        print("Starting Evaluation...")
        for i in range(num_steps):
            _, cur_cam = gen_random_minicam(self.cam, self.cam.radius, 
                                                    center_elevation=self.config.render.elevation,
                                                    device=self.device)
            bg_color = np.random.rand(3)
            with torch.no_grad():
                clone_out = self.renderer.render(cur_cam, bg_color = bg_color)
                clone_img = clone_out["image"].unsqueeze(0)# [1, 3, H, W] in [0, 1]
                orig_out = self.renderer_ref.render(cur_cam, bg_color = bg_color)
                orig_img = orig_out["image"].unsqueeze(0)# [1, 3, H, W] in [0, 1]
            
            psnr_list.append(psnr(clone_img, orig_img))
            mse_list.append(mse(clone_img, orig_img))

        psnr_list = torch.tensor(psnr_list).detach().cpu().numpy()
        mse_list = torch.tensor(mse_list).detach().cpu().numpy()
        
        psnr_total = np.mean(np.array(psnr_list))
        mse_total = np.mean(np.array(mse_list))

        print(f"PSNR={psnr_total} dB, MSE={mse_total}")

        
        
def training_report(tb_writer, iteration, log_interval, loss, gaussians, images):
    if tb_writer:
        for key in loss.keys():
            tb_writer.add_scalar(f'train_loss_patches/{key}', loss[key], iteration)
        tb_writer.add_scalar('total_points', gaussians.get_xyz.shape[0], iteration)
    if(iteration % log_interval == 1):
        tb_writer.add_histogram("scene/opacity_histogram", gaussians.get_opacity, iteration)
        tb_writer.add_histogram("scene/scaling_histogram", torch.norm(gaussians.get_scaling, dim=-1), iteration)
        grads = torch.norm(gaussians.xyz_gradient_accum, dim=-1)
        grads_filter = grads < torch.std(grads)*5
        denom = gaussians.denom[grads_filter]
        denom = denom[:,0]
        grads = grads[grads_filter]
        if(len(grads) > 00):
            tb_writer.add_histogram("scene/view_gradients_histogram", grads, iteration)
            tb_writer.add_histogram("scene/view_norm_gradients_histogram", grads/denom, iteration)
            tb_writer.add_histogram("scene/denom_histogram", denom, iteration)
        for key in images.keys():
            img = images[key] 
            if(len(img.shape) == 4):
                img = img[0]
            tb_writer.add_image(f"scene/{key}", img, iteration)
        torch.cuda.empty_cache()