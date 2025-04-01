import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import cv2
from tqdm import tqdm
import numpy as np
import torch
from easydict import EasyDict as edict
from omegaconf import OmegaConf

from torch import nn


from utils.cam import OrbitCamera, gen_random_minicam_span, gen_random_minicam_comp
from comp3dgs.gs_renderer import RendererComposition


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




class CompTrainer(object):

    def __init__(self, config,rand_init=True) -> None:
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
        self.renderer = RendererComposition(config=self.config.composite)      
        # init tensorboard
        self.tb_writer = SummaryWriter(self.config.save_path)
        # GUI data
        self.training = True
        self.gui = edict({'train_time': 0, 'loss': 0})

        # initialize object Trainers
        self.obj_trainers = []

        # Flag to check when compositional training starts
        self.comp_start = False

        ## Params for the recentering impulse
        self.recenter_count = 0
        self.rrad = 0
        self.main_floor = 0

        self.max_recenter_count = 5
        self.recenter_area_ratio_l = 0.4
        self.recenter_area_ratio_h = 0.95

        ## Scale optimization parameters
        self.init_iter = 0
        self.init_cam_rad = 4.5 
        self.init_scale_lim = 0.3

        self.init_cam_rad2 = 2.5
        self.init_scale_lim2 = 0.2
        self.scale_switch_threshold = 0.05

        ## For the viewspace grad. normalization
        self.viewspace_exponent = 0.7

        self.prepare_train()

    def prepare_train(self):
        # load guidance
        print(f"[INFO] loading SD...")
        from guidance.sd_utils import StableDiffusion
        self.guide = StableDiffusion(self.config)
        print(f"[INFO] loaded SD!")
        # prepare embeddings
        with torch.no_grad():
            self.text_embeds = self.guide.get_text_embeds([self.config.prompt], [self.config.negative_prompt])

        for i in range(self.config.composite.num_objs):
            # create copy of object-level config
            object_config = OmegaConf.load(self.config.composite.object_config_path)
            object_config.prompt = self.config.composite.prompts[i]
            object_config.pointe_prompt = self.config.composite.pointe_prompts[i]
            object_config.point_e.knn_loss = self.config.composite.pointe_knn[i]
            object_config.negative_prompt = self.config.composite.negative_prompts[i]
            object_config.load_path = self.config.composite.obj_paths[i] if(self.config.composite.obj_paths[i]!="None") else None
            object_config.save_path = os.path.join(self.config.save_path, f"obj_{i}_{self.config.composite.prompts[i]}")
            self.obj_trainers.append(ObjectTrainer(object_config, self.guide)) #guide should be used by reference
        
        # ---------- Setup Training ------------
        # initialize 3dgs
        self.renderer.initialize(self.obj_trainers)

        

    def configure_compositional_trainer(self):
        if self.rand_init:
            ## Estimate the max sampling radius
            self.rrad = self.get_sampling_radius()
            # print('*************Sampling Radius********************')
            # print(self.rrad)

            ## Get floor
            self.main_floor = get_floor(self.renderer.gaussians.get_xyz[0:self.renderer.gaussians.point_per_obj[0]])
            # print('****************Floor*******************')
            # print(self.main_floor)

            self.select_init_fixedCt_pruneOcc(flag='both',num_init_samples=150,g_scale=100)

            
            for _ in range(3):

                self.select_init_fixedCt_pruneOcc(flag='trans',num_init_samples=50)

                cur_sc = self.renderer.gaussians.glob_scale_list[1].detach().cpu().squeeze().numpy()

                print(cur_sc)
                if cur_sc<self.init_scale_lim+self.scale_switch_threshold:
                    self.init_cam_rad = self.init_cam_rad2
                    self.init_scale_lim = self.init_scale_lim2
                    print('*******Changed the init camera radius********')

                self.select_init_fixedCt_final(flag='scale',num_init_samples=50,g_scale=100)
                

        self.renderer.gaussians.training_setup(self.config.composite)
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.sh_degree
        self.optimizer = self.renderer.gaussians.optimizer


        

    def objs_step(self):
        if self.config.composite.train_objects and (self.curr_iter <= self.config.composite.obj_stop_thresh):
            for i in range(self.config.composite.num_objs):
                self.obj_trainers[i].train_step()
    
    
    def get_rand_direction(self):
        elev = np.random.uniform(-np.pi/2,np.pi/2)
        azim = np.random.uniform(-np.pi, np.pi)
        return torch.tensor([np.cos(elev)*np.sin(azim), -np.sin(elev), np.cos(elev)*np.cos(azim)], device=self.device).unsqueeze(0).unsqueeze(0).float()
    

    def get_sampling_radius(self):
        x_min_0 = get_bound(self.renderer.gaussians.get_xyz[0:self.renderer.gaussians.point_per_obj[0]],direction='min',dim=0,frac = 0.001).detach().cpu().numpy()
        x_max_0 = get_bound(self.renderer.gaussians.get_xyz[0:self.renderer.gaussians.point_per_obj[0]],direction='max',dim=0,frac = 0.001).detach().cpu().numpy()
        
        y_min_0 = get_bound(self.renderer.gaussians.get_xyz[0:self.renderer.gaussians.point_per_obj[0]],direction='min',dim=1,frac = 0.001).detach().cpu().numpy()
        y_max_0 = get_bound(self.renderer.gaussians.get_xyz[0:self.renderer.gaussians.point_per_obj[0]],direction='max',dim=1,frac = 0.001).detach().cpu().numpy()
        
        z_min_0 = get_bound(self.renderer.gaussians.get_xyz[0:self.renderer.gaussians.point_per_obj[0]],direction='min',dim=2,frac = 0.001).detach().cpu().numpy()
        z_max_0 = get_bound(self.renderer.gaussians.get_xyz[0:self.renderer.gaussians.point_per_obj[0]],direction='max',dim=2,frac = 0.001).detach().cpu().numpy()

        cent_0 = torch.mean(self.renderer.gaussians.get_xyz[0:self.renderer.gaussians.point_per_obj[0]],dim=0).detach().cpu().numpy()

        x_min_0 = x_min_0-cent_0[0]
        x_max_0 = x_max_0-cent_0[0]
        y_min_0 = y_min_0-cent_0[1]
        y_max_0 = y_max_0-cent_0[1]
        z_min_0 = z_min_0-cent_0[2]
        z_max_0 = z_max_0-cent_0[2]


        
        ## Get the numbers for anchor object (object 1)
        x_min_1 = get_bound(self.renderer.gaussians.get_xyz[self.renderer.gaussians.point_per_obj[0]:],direction='min',dim=0,frac = 0.001).detach().cpu().numpy()
        x_max_1 = get_bound(self.renderer.gaussians.get_xyz[self.renderer.gaussians.point_per_obj[0]:],direction='max',dim=0,frac = 0.001).detach().cpu().numpy()

        
        y_min_1 = get_bound(self.renderer.gaussians.get_xyz[self.renderer.gaussians.point_per_obj[0]:],direction='min',dim=1,frac = 0.001).detach().cpu().numpy()
        y_max_1 = get_bound(self.renderer.gaussians.get_xyz[self.renderer.gaussians.point_per_obj[0]:],direction='max',dim=1,frac = 0.001).detach().cpu().numpy()

        
        z_min_1 = get_bound(self.renderer.gaussians.get_xyz[self.renderer.gaussians.point_per_obj[0]:],direction='min',dim=2,frac = 0.001).detach().cpu().numpy()
        z_max_1 = get_bound(self.renderer.gaussians.get_xyz[self.renderer.gaussians.point_per_obj[0]:],direction='max',dim=2,frac = 0.001).detach().cpu().numpy()

        cent_1 = torch.mean(self.renderer.gaussians.get_xyz[self.renderer.gaussians.point_per_obj[0]:],dim=0).detach().cpu().numpy()

        x_min_1 = x_min_1-cent_1[0]
        x_max_1 = x_max_1-cent_1[0]
        y_min_1 = y_min_1-cent_1[1]
        y_max_1 = y_max_1-cent_1[1]
        z_min_1 = z_min_1-cent_1[2]
        z_max_1 = z_max_1-cent_1[2]

        x_b_0 = np.max([np.abs(x_min_0),np.abs(x_max_0)])
        y_b_0 = np.max([np.abs(y_min_0),np.abs(y_max_0)])
        z_b_0 = np.max([np.abs(z_min_0),np.abs(z_max_0)])

        x_b_1 = np.max([np.abs(x_min_1),np.abs(x_max_1)])
        y_b_1 = np.max([np.abs(y_min_1),np.abs(y_max_1)])
        z_b_1 = np.max([np.abs(z_min_1),np.abs(z_max_1)])

        rad = np.max([x_b_0+x_b_1,y_b_0+y_b_1,z_b_0+z_b_1])

        return rad
    
    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        # update iter
        self.curr_iter += 1
        step_ratio = min(1, self.curr_iter / self.max_iter)

        loss = 0
        loss_dict = {}
        tb_imgs = {}

        # Run a step of the object level trainings
        self.objs_step()

        ## Set the text embed to be the global text embed
        self.guide.set_text_embeds(self.text_embeds)

        

        ### Stage 1: guidance loss
        if(self.curr_iter <= self.config.course_steps) and (self.curr_iter >= self.config.composite.comp_start_thresh) and (self.curr_iter < self.config.composite.comp_stop_thresh):
            
            if self.comp_start==False:
                self.configure_compositional_trainer()
                self.comp_start = True

            self.optimizer.zero_grad()
            ### novel view (manual batch)
            images = []
            for _ in range(self.config.batch_size):
                ver, hor, cur_cam = gen_random_minicam_comp(self.cam, self.config.render.radius, 
                                                       center_elevation=self.config.render.elevation,
                                                       device=self.device)
                bg_color = np.array([1,1,1]) #np.random.rand(3)
                out = self.renderer.render(cur_cam, bg_color = bg_color)
                image = out["image"].unsqueeze(0)# [1, 3, H, W] in [0, 1]
                images.append(image)
            images = torch.cat(images, dim=0)

            course_sd_loss = self.config.course_sd_loss*self.guide.train_step(images, step_ratio, 
                                                                              guidance_scale=self.config.course_guidance)
            loss_dict['course_sd_loss'] = course_sd_loss.item()
            loss += course_sd_loss


            tb_imgs['3dgs_img'] = images[0]

            if self.curr_iter<=(self.config.composite.comp_start_thresh + self.config.stage1_steps):
                pass 
            else:
                floor = get_floor(self.renderer.gaussians.get_xyz[0:self.renderer.gaussians.point_per_obj[0]])
                gl = gravity_loss(self.renderer.gaussians.get_xyz[self.renderer.gaussians.point_per_obj[0]:], floor=floor)
                

                t1 = torch.mean(self.renderer.gaussians.get_xyz[0:self.renderer.gaussians.point_per_obj[0]],dim=0).unsqueeze(0).unsqueeze(0)
                t2 = torch.mean(self.renderer.gaussians.get_xyz[self.renderer.gaussians.point_per_obj[0]:],dim=0).unsqueeze(0).unsqueeze(0)

                l_temp = intersection_loss_max_vec(self.renderer.gaussians.get_xyz[0:self.renderer.gaussians.point_per_obj[0]],
                                             self.renderer.gaussians.get_xyz[self.renderer.gaussians.point_per_obj[0]:],
                                             t1,t2)
                


                loss = loss + 10000*gl
                loss = loss + (gl.item()/0.55)*15000*l_temp 


                if l_temp.item()>0:

                    pc1 = self.renderer.gaussians.get_xyz[0:self.renderer.gaussians.point_per_obj[0]].detach().cpu().numpy()
                    pc2 = self.renderer.gaussians.get_xyz[self.renderer.gaussians.point_per_obj[0]:].detach().cpu().numpy()

                    pc1 = np.concatenate((pc1[:,0:1],pc1[:,2:3]),axis=1)
                    pc2 = np.concatenate((pc2[:,0:1],pc2[:,2:3]),axis=1)

                    bd1 = scipy.spatial.ConvexHull(pc1)
                    bd2 = scipy.spatial.ConvexHull(pc2)

                    bd1 = pc1[bd1.vertices]
                    bd2 = pc2[bd2.vertices]

                    ## Get overlap area %
                    p = Polygon(bd1)
                    q = Polygon(bd2)

                    if p.intersects(q):
                        area_ratio = p.intersection(q).area/q.area
                        print(area_ratio)

                        if area_ratio>self.recenter_area_ratio_l and area_ratio<self.recenter_area_ratio_h and self.recenter_count<=self.max_recenter_count: 
                            self.recenter_count+=1
                            c_x,c_y,c_z = centering_correction(t1, t2) 
                            with torch.no_grad():
                                self.renderer.gaussians.glob_trans_list[1][...,0]+=c_x
                                self.renderer.gaussians.glob_trans_list[1][...,1]+=c_y
                                self.renderer.gaussians.glob_trans_list[1][...,2]+=c_z

                    

            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.gui.loss = loss.item()
        




        ## Save plys
        if(self.curr_iter % self.config.save_interval == 0):
            ply_save_path = os.path.join(self.config.save_path, f"iter_{self.curr_iter}_{self.config.prompt}.ply")
            glob_ply_save_path = os.path.join(self.config.save_path, f"iter_{self.curr_iter}_{self.config.prompt}_glob.ply")
            self.renderer.gaussians.save_ply(ply_save_path)
            self.renderer.gaussians.save_ply_global(glob_ply_save_path)

        # update tensorboard
        training_report(self.tb_writer, self.curr_iter, self.config.log_interval, 
                        loss_dict, self.renderer.gaussians, tb_imgs)

        ender.record()
        torch.cuda.synchronize()
        # update gui info
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