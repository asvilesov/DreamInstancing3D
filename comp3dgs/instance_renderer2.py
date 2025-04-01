from comp3dgs.gs_renderer2 import *
from utils.debug import timer
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R
import quaternion
from utils.debug import timer
import time 

# Notes:
# 1. Coordinate System - Left Handed
#    Y - Up
#    X - Right
#    Z - Forward (away from camera)

def create_grid(grid_size, grid_resolution, device):
    """Create a grid of points in the range [-1, 1] x [-1, 1] x [-1, 1]"""
    grid = torch.meshgrid(
        grid_size[0]*torch.linspace(-0.5, 0.5, grid_resolution[0], device=device),
        grid_size[1]*torch.linspace(-0.5, 0.5, grid_resolution[1], device=device),
    )
    N = len(grid[0].flatten())
    y_axis = torch.zeros(N, device=device)
    print(grid[0].shape)
    grid = torch.stack([grid[0].flatten(), y_axis, grid[1].flatten()], dim=-1)
    print(f"grid.shape: {grid.shape}")
    return grid

class InstanceCompositionModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation, rotation_global):
            L = build_scaling_rotation_global(scaling_modifier * scaling, rotation, rotation_global)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
    
    def grab_random_object_bounds(self, class_id):
        # Grab random object uniformely from classes
        K = len(self.groups) # num types of objects 
        instance_id = np.random.randint(0, self.groups[class_id]["N"])
        obj_id = 0
        for i in range(class_id):
            obj_id += self.groups[i]["N"]
        obj_id += instance_id
        # print(K, class_id, instance_id, obj_id)
        # Grab object
        offset = self.glob_trans_list[obj_id][0].detach().cpu().numpy()
        scale = self.glob_scale_list[obj_id][0].detach().cpu().numpy()
        rot_deg = self.groups[class_id]["R_degrees"][instance_id]
        y_max = scale*torch.max(self.gaussian_objects[class_id]._xyz[:,1]).detach().cpu().numpy() + offset[1]
        y_min = scale*torch.min(self.gaussian_objects[class_id]._xyz[:,1]).detach().cpu().numpy() + offset[1]
        rand_y = np.random.uniform()*(y_max-y_min) + y_min
        rand_y = 0 #TODO
        # Grab object gaussian index range
        start_idx = torch.sum(torch.tensor(self.point_per_obj[0:obj_id])).type(torch.int64)
        end_idx = torch.sum(torch.tensor(self.point_per_obj[0:obj_id+1])).type(torch.int64)

        return np.array([offset[0], offset[1], offset[2]]), scale, rot_deg, (start_idx, end_idx)

    def __init__(self, config, groups, device):

        self.config = config
        self.sh_degree = config.sh_degree
        self.grid_size = np.array(config.grid_size)
        assert self.grid_size.shape[0] == 2
        self.grid_resolution = np.array(config.grid_resolution)
        assert self.grid_resolution.shape[0] == 2
        self.load_path = config.load_path
        self.device = device

        if(self.load_path is not None):
            self.load_ply_global(self.load_path)
        else:
            self.groups = groups
            self.init_base_gaussians()

            # update variable for efficient scene construction
            self.update_scene = True
            self.update_gaussians = {'xyz':True, 'features':True, 'scaling':True, 'rotation':True, 'opacity':True}

            # create grid for rendering
            self.no_objs = 0
            self.glob_trans_list = []
            self.glob_rot_list = []
            self.glob_scale_list = []
            self.glob_opacity_list = []
            self.point_per_obj = []
            self._xyz = torch.empty(size=(0,3), device=self.device)
            self._features_dc = torch.empty(size=(0,1, 3), device=self.device)
            self._features_rest = torch.empty(size=(0, 0, 3), device=self.device)
            self._scaling = torch.empty(size=(0, 3), device=self.device)
            self._rotation = torch.empty(size=(0, 4), device=self.device)
            self._opacity = torch.empty(size=(0, 1), device=self.device)

            for j, obj in enumerate(groups):
                for i in range(obj["N"]):
                    self.glob_trans_list.append(nn.Parameter(obj["T"][i].unsqueeze(0).float(), requires_grad=obj["opt_pose"]))
                    self.glob_rot_list.append(nn.Parameter(obj["R"][i].unsqueeze(0).float(), requires_grad=obj["opt_pose"]))
                    self.glob_scale_list.append(nn.Parameter(obj["S"][i].float(), requires_grad=obj["opt_pose"]))
                    self.glob_opacity_list.append(nn.Parameter(torch.tensor(1, device=self.device).unsqueeze(0).float(), requires_grad=False))
                self.point_per_obj += [self.gaussian_objects[j]._xyz.shape[0]]*obj["N"]
                self.no_objs += obj["N"]
                
                self._xyz = torch.concat((self._xyz, self.gaussian_objects[j]._xyz.expand(obj["N"], -1, 3).flatten(0,1)), dim=0)
                self._features_dc = torch.concat((self._features_dc, self.gaussian_objects[j]._features_dc.expand(obj["N"], -1, 1, 3).flatten(0, 1)), dim=0)
                self._features_rest = torch.concat((self._features_rest, self.gaussian_objects[j]._features_rest.expand(obj["N"], -1, 0, 3).flatten(0, 1)), dim=0)
                self._scaling = torch.concat((self._scaling, self.gaussian_objects[j]._scaling.expand(obj["N"], -1, 3).flatten(0, 1)), dim=0)
                self._rotation = torch.concat((self._rotation, self.gaussian_objects[j]._rotation.expand(obj["N"], -1, 4).flatten(0, 1)), dim=0)
                self._opacity = torch.concat((self._opacity, self.gaussian_objects[j]._opacity.expand(obj["N"], -1, 1).flatten(0, 1)), dim=0)

            self.training_setup()

            
        ## Initialize parameter values for the composed object
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.active_sh_degree = self.sh_degree
        self.setup_functions()

    def init_base_gaussians(self):
        from utils.pointe_helper import point_e_intialize
        self.gaussian_objects = []
        for i in range(len(self.groups)):
            self.groups[i]["config"].prompt = self.groups[i]["text"]
            self.groups[i]["config"].negative_prompt = self.groups[i]["neg_text"] if "neg_text" in self.groups[i] else "None"
            self.groups[i]["config"].reorient = self.groups[i]["reorient"]
            self.groups[i]["config"].use_view_dependent_prompt = self.groups[i]["pose_emb"]
            if(self.groups[i]["type"] == "object"):
                self.groups[i]["config"].pointe_prompt = self.groups[i]["pointe"]
                gaussians = GaussianModel(self.sh_degree, self.groups[i]["config"], self.device)
                gaussians.initialize("point-e")
                min_y1 = torch.min(gaussians.get_xyz[:,1])
                #TODO not multi-scale per instance compatible
                self.groups[i]["T"] -= self.groups[i]["S"][0]*torch.tensor([0, min_y1, 0], device=self.device).unsqueeze(0)
                self.groups[i]["T"] += self.groups[i]["T_y"]
            elif(self.groups[i]["type"] == "floor"):
                nx, ny = (70, 70)
                x = np.linspace(-0.5, 0.5, nx)
                y = np.linspace(-0.5, 0.5, ny)
                xv, yv = np.meshgrid(x, y)
                xv = xv.flatten()
                yv = yv.flatten()
                z = np.zeros_like(xv)
                xyz = np.stack((xv, z, yv), axis=1)
                print("Floor shape", xyz.shape)
                num_pts = xyz.shape[0]
                shs = np.random.random((num_pts, 3))
                pcd = BasicPointCloud(
                    points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
                )
                gaussians = GaussianModel(self.sh_degree, self.groups[i]["config"], self.device)
                gaussians.create_from_pcd(pcd, 1, spatial_scale=0.5) #1.2
                gaussians.anchor_knn = gaussians.get_xyz.detach().clone()
            elif(self.groups[i]["type"] == "globe"):
                gaussians = GaussianModel(self.sh_degree, self.groups[i]["config"], self.device)
                gaussians.initialize(input=None, num_pts=30000, radius = 1.2, spatial_scale=1)
                gaussians.training_setup(self.groups[i]["config"].gs3d)
            else:
                raise NotImplementedError
            
            gaussians.training_setup(self.groups[i]["config"].gs3d)
            self.gaussian_objects.append(gaussians)

    def need_update(self):
        self.update_scene = True
        self.update_gaussians = {'xyz':True, 'features':True, 'scaling':True, 'rotation':True, 'opacity':True}   
        self.update_coords()     

    def capture(self):
        self.update_coords()
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def get_radius(self,cur_obj):
        xyz = cur_obj._xyz
        rads = torch.norm(xyz, dim=-1, keepdim=True)
        return torch.max(rads)
    
    def get_rand_direction(self):
        elev = np.random.uniform(-np.pi/6, -np.pi/4)
        azim = np.random.uniform(-np.pi, np.pi)
        return torch.tensor([np.cos(elev)*np.sin(azim), -np.sin(elev), np.cos(elev)*np.cos(azim)], device=self.device).unsqueeze(0)

    def setup_delta_gaussians(self):
        self.delta_xyz = []
        self.delta_features_dc = []
        self.delta_features_rest = []
        self.delta_scaling = []
        self.delta_rotation = []
        self.delta_opacity = []
        for j, obj in enumerate(self.groups):
            for k in range(obj["N"]):
                self.delta_xyz.append(nn.Parameter(torch.zeros_like(self.gaussian_objects[j]._xyz, device=self.device).requires_grad_(True)))
                self.delta_features_dc.append(nn.Parameter(RGB2SH(torch.rand_like(self.gaussian_objects[j]._features_dc, device=self.device)).requires_grad_(True)))
                self.delta_features_rest.append(nn.Parameter(torch.rand_like(self.gaussian_objects[j]._features_rest, device=self.device).requires_grad_(True)))
                self.delta_scaling.append(nn.Parameter(torch.zeros_like(self.gaussian_objects[j]._scaling, device=self.device).requires_grad_(True)))
                self.delta_rotation.append(nn.Parameter(torch.zeros_like(self.gaussian_objects[j]._rotation, device=self.device).requires_grad_(True)))
                self.delta_opacity.append(nn.Parameter(torch.zeros_like(self.gaussian_objects[j]._opacity, device=self.device).requires_grad_(True)))

    def disable_gradients_delta(self, group_id):
        self.delta_features_dc[group_id].requires_grad_(False)
        self.delta_features_rest[group_id].requires_grad_(False)
        # self.delta_xyz[group_id].requires_grad_(False)
        # self.delta_scaling[group_id].requires_grad_(False)
        # self.delta_rotation[group_id].requires_grad_(False)
        # self.delta_opacity[group_id].requires_grad_(False)
    
    def disable_gradients_pose(self, group_id):
        offset = 0
        for i in range(group_id):
            offset += self.groups[i]["N"]
        for i in range(self.groups[group_id]["N"]):
            self.glob_trans_list[offset+i].requires_grad_(False)
            self.glob_rot_list[offset+i].requires_grad_(False)
            self.glob_scale_list[offset+i].requires_grad_(False)
            self.glob_opacity_list[offset+i].requires_grad_(False)
        
    
    def enable_gradients_delta(self, group_id):
        self.delta_features_dc[group_id].requires_grad_(True)
        self.delta_features_rest[group_id].requires_grad_(True)
        # self.delta_xyz[group_id].requires_grad_(True)
        # self.delta_scaling[group_id].requires_grad_(True)
        # self.delta_rotation[group_id].requires_grad_(True)
        # self.delta_opacity[group_id].requires_grad_(True)

    def enable_gradients_pose(self, group_id):
        if(self.groups[group_id]["opt_pose"]):
            offset = 0
            for i in range(group_id):
                offset += self.groups[i]["N"]
            for i in range(self.groups[group_id]["N"]):
                self.glob_trans_list[offset+i].requires_grad_(True)
                self.glob_rot_list[offset+i].requires_grad_(True)
                self.glob_scale_list[offset+i].requires_grad_(True)
                self.glob_opacity_list[offset+i].requires_grad_(True)

    def update_coords(self):
        if(self.update_scene):
            self.point_per_obj = []
            self._xyz = torch.empty(size=(0,3), device=self.device)
            self._features_dc = torch.empty(size=(0,1, 3), device=self.device)
            self._features_rest = torch.empty(size=(0, 0, 3), device=self.device)
            self._scaling = torch.empty(size=(0, 3), device=self.device)
            self._rotation = torch.empty(size=(0, 4), device=self.device)
            self._opacity = torch.empty(size=(0, 1), device=self.device)
            for j, obj in enumerate(self.groups):
                self.point_per_obj += [self.gaussian_objects[j]._xyz.shape[0]]*obj["N"]
                self._xyz = torch.concat((self._xyz, self.gaussian_objects[j]._xyz.expand(obj["N"], -1, 3).flatten(0,1)), dim=0)
                self._features_dc = torch.concat((self._features_dc, self.gaussian_objects[j]._features_dc.expand(obj["N"], -1, 1, 3).flatten(0, 1)), dim=0)
                self._features_rest = torch.concat((self._features_rest, self.gaussian_objects[j]._features_rest.expand(obj["N"], -1, 0, 3).flatten(0, 1)), dim=0)
                self._scaling = torch.concat((self._scaling, self.gaussian_objects[j]._scaling.expand(obj["N"], -1, 3).flatten(0, 1)), dim=0)
                self._rotation = torch.concat((self._rotation, self.gaussian_objects[j]._rotation.expand(obj["N"], -1, 4).flatten(0, 1)), dim=0)
                self._opacity = torch.concat((self._opacity, self.gaussian_objects[j]._opacity.expand(obj["N"], -1, 1).flatten(0, 1)), dim=0)
            self.update_scene = False
        else:
            pass

    @property
    def get_scaling(self):
        self.update_coords()
        if(self.update_gaussians['scaling']):
            # Scale according to global scaling for each object
            temp = torch.empty(size=(0,3), device=self.device)
            offset = 0
            offset_obj = 0
            for j in range(len(self.groups)):
                stride = self.gaussian_objects[j].get_scaling.shape[0]
                for i in range(self.groups[j]["N"]):
                    temp = torch.cat((temp, self.glob_scale_list[offset_obj] * 
                                            self.scaling_activation(self._scaling[offset:offset+stride] + self.delta_scaling[offset_obj])), dim=0)
                    offset += stride
                    offset_obj += 1
            self._scaling_world = temp
            self.update_gaussians['scaling'] = False
        return self._scaling_world
    
    @property
    def get_raw_scaling(self):
        self.update_coords()
        # Scale according to global scaling for each object
        for i in range(self.no_objs):
            if i==0:
                temp = self.scaling_inverse_activation(self.glob_scale_list[i] * self.scaling_activation(self._scaling[0:self.point_per_obj[i]]))
            else:
                temp = torch.cat((temp, self.scaling_inverse_activation(self.glob_scale_list[i] * self.scaling_activation(self._scaling[torch.sum(torch.tensor(self.point_per_obj[0:i])):torch.sum(torch.tensor(self.point_per_obj[0:i+1]))]))), dim=0)

        return temp
    
    @property
    def get_xyz(self):
        self.update_coords()
        if(self.update_gaussians['xyz']):
            # we need to do scaling, rotation and translation
            temp = torch.empty(size=(0, 3), device=self.device)
            offset = 0
            offset_obj = 0

            rotation_matrices = build_rotation(torch.concat(self.glob_rot_list))
            scales = torch.concat(self.glob_scale_list).unsqueeze(1).unsqueeze(1)
            translations = torch.concat(self.glob_trans_list).unsqueeze(1)

            for j in range(len(self.groups)):
                stride = self.gaussian_objects[j]._features_dc.shape[0]
                num_instances = self.groups[j]["N"]
                #   1. scale
                xyz_local = torch.reshape(self._xyz[offset:offset+num_instances*stride], shape=(num_instances, -1, 3))
                delta_xyz_local = torch.stack(self.delta_xyz[offset_obj:offset_obj+num_instances])
                temp2 = scales[offset_obj:offset_obj+num_instances]*(xyz_local+delta_xyz_local)
                #   2. rotation
                temp2 = (temp2 @ rotation_matrices[offset_obj:offset_obj+num_instances])
                #   3. translation
                temp2 = (temp2 + translations[offset_obj:offset_obj+num_instances])
                temp = torch.cat((temp,torch.flatten(temp2, start_dim=0, end_dim=1)), dim=0)
                offset += num_instances*stride
                offset_obj += num_instances
            self._xyz_world = temp
            self.update_gaussians['xyz'] = False
        return self._xyz_world
    
    @property
    def get_features(self):
        self.update_coords()
        if(self.update_gaussians['features']): 
            temp = torch.empty(size=(0,1, 3), device=self.device)
            temp2 = torch.empty(size=(0, 0, 3), device=self.device)
            offset = 0
            offset_obj = 0
            for j in range(len(self.groups)):
                stride = self.gaussian_objects[j]._features_dc.shape[0]
                for i in range(self.groups[j]["N"]):
                    theta = self.groups[j]["sim"]
                    temp = torch.cat((temp, theta*self._features_dc[offset:offset+stride] +
                                        (1-theta)*self.delta_features_dc[offset_obj]), dim=0)
                    temp2 = torch.cat((temp2, theta*self._features_rest[offset:offset+stride] + 
                                        (1-theta)*self.delta_features_rest[offset_obj]), dim=0)                    
                    offset += stride
                    offset_obj += 1
            self._features_dc_world = temp
            self._features_rest_world = temp2
            self.update_gaussians['features'] = False
        return torch.cat((self._features_dc_world, self._features_rest_world), dim=1)
    
    @property
    def get_opacity(self):
        self.update_coords()
        if(self.update_gaussians['opacity']):
            temp = torch.empty(size=(0,1), device=self.device)
            offset = 0
            offset_obj = 0
            for j in range(len(self.groups)):
                stride = self.gaussian_objects[j]._opacity.shape[0]
                for i in range(self.groups[j]["N"]):
                    temp = torch.cat((temp, self.opacity_activation(self._opacity[offset:offset+stride] +
                                      self.delta_opacity[offset_obj])), dim=0)
                    offset += stride
                    offset_obj += 1
            
            self._opacity_world = temp
            self.update_gaussians['opacity'] = False
        return self._opacity_world #self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        self.update_coords()
        if(self.update_gaussians['rotation'] or scaling_modifier != 1):
            temp = torch.empty(size=(0,6), device=self.device)
            offset = 0
            offset_obj = 0
            unroll_delta_rotation = torch.empty(size=(0, 4))
            unroll_global_rotation = torch.empty(size=(0, 4))
            for j in range(len(self.groups)):
                stride = self.gaussian_objects[j].get_scaling.shape[0]
                for i in range(self.groups[j]["N"]):
                    # print(self.delta_scaling[j][:,i].shape)
                    unroll_delta_rotation = torch.concat((unroll_delta_rotation, self.delta_rotation[offset_obj]))
                    unroll_global_rotation = torch.concat((unroll_global_rotation, self.glob_rot_list[offset_obj].expand(stride, 4)))
                    offset += stride
                    offset_obj += 1
            
            temp = self.covariance_activation(self.get_scaling, 
                                                scaling_modifier,
                                                self._rotation+unroll_delta_rotation,
                                                unroll_global_rotation)

            self._rotation_world = temp
            self.update_gaussians['rotation'] = False
        return self._rotation_world
    
    def training_setup(self):
        self.setup_delta_gaussians()
        self.optimizer_list = []
        offset_obj = 0
        for j, obj in enumerate(self.groups):
            for i in range(obj["N"]):
                if(obj["type"] == "object"):
                    l_obj = [
                        {'params': self.delta_xyz[offset_obj+i], 'lr': 0.0001, "name": "xyz"},
                        {'params': self.delta_features_dc[offset_obj+i], 'lr': 0.04, "name": "f_dc"},
                        {'params': self.delta_features_rest[offset_obj+i], 'lr': self.config.gs3d.feature_lr / 20.0, "name": "f_rest"},
                        {'params': self.delta_opacity[offset_obj+i], 'lr': 0.05, "name": "opacity"},
                        {'params': self.delta_scaling[offset_obj+i], 'lr': 0.006, "name": "scaling"},
                        {'params': self.delta_rotation[offset_obj+i], 'lr': 0.006, "name": "rotation"}
                    ]
                    optimizer = torch.optim.Adam(l_obj, betas=(0.90, 0.95), lr=1e-7, eps=1e-7)
                else:
                    l_env = [
                        {'params': self.delta_xyz[offset_obj+i], 'lr': 0.0002, "name": "xyz"},
                        {'params': self.delta_features_dc[offset_obj+i], 'lr': 0.01, "name": "f_dc"},
                        {'params': self.delta_features_rest[offset_obj+i], 'lr': self.config.gs3d.feature_lr / 20.0, "name": "f_rest"},
                        {'params': self.delta_opacity[offset_obj+i], 'lr': 0.05, "name": "opacity"},
                        {'params': self.delta_scaling[offset_obj+i], 'lr': 0.001, "name": "scaling"},
                        {'params': self.delta_rotation[offset_obj+i], 'lr': 0.003, "name": "rotation"}
                    ]
                    optimizer = torch.optim.Adam(l_env, betas=(0.90, 0.999), lr=1e-7, eps=1e-7)
                self.optimizer_list.append(optimizer)
            offset_obj += obj["N"]
        
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=0.002,
                                                    lr_delay_steps= 000,
                                                    lr_final=0.001,
                                                    lr_delay_mult=0.01,
                                                    max_steps=1000)
        self.c_scheduler_args = get_expon_lr_func(lr_init=0.05,
                                                    lr_delay_steps= 000,
                                                    lr_final=0.05,
                                                    lr_delay_mult=0.01,
                                                    max_steps=1000)
        self.o_scheduler_args = get_expon_lr_func(lr_init=0.05,
                                                    lr_delay_steps= 000,
                                                    lr_final=0.05,
                                                    lr_delay_mult=0.01,
                                                    max_steps=1000)
        self.s_scheduler_args = get_expon_lr_func(lr_init=0.006,
                                                    lr_delay_steps= 000,
                                                    lr_final=0.006,
                                                    lr_delay_mult=0.01,
                                                    max_steps=1000)
        self.r_scheduler_args = get_expon_lr_func(lr_init=0.006,
                                                    lr_delay_steps= 000,
                                                    lr_final=0.006,
                                                    lr_delay_mult=0.01,
                                                    max_steps=1000)
        
        self.xyz_scheduler_floor_args = get_expon_lr_func(lr_init=0.001,
                                                    lr_delay_steps= 000,
                                                    lr_final=0.0001,
                                                    lr_delay_mult=0.01,
                                                    max_steps=1000)
        
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        lr_dict = {}
        offset_obj = 0
        for j, obj in enumerate(self.groups):
            for i in range(obj["N"]):
                if(obj["type"] == "object"):
                    for param_group in self.optimizer_list[offset_obj+i].param_groups:
                        if param_group["name"] == "xyz":
                            lr = self.xyz_scheduler_args(iteration)
                            param_group['lr'] = lr
                            lr_dict["xyz"] = lr
                        elif param_group["name"] == "f_dc":
                            lr = self.c_scheduler_args(iteration)
                            param_group['lr'] = lr
                            lr_dict["f_dc"] = lr
                        elif param_group["name"] == "f_rest":
                            lr = self.c_scheduler_args(iteration)
                            param_group['lr'] = lr
                            lr_dict["f_rest"] = lr
                        elif param_group["name"] == "opacity":
                            lr = self.o_scheduler_args(iteration)
                            param_group['lr'] = lr
                            lr_dict["opacity"] = lr
                        elif param_group["name"] == "scaling":
                            lr = self.s_scheduler_args(iteration)
                            param_group['lr'] = lr
                            lr_dict["scaling"] = lr
                        elif param_group["name"] == "rotation":
                            lr = self.r_scheduler_args(iteration)
                            param_group['lr'] = lr
                            lr_dict["rotation"] = lr
            offset_obj += obj["N"]
        return lr_dict
            
    def update_learning_rate_floor(self, iteration):
        ''' Learning rate scheduling per step '''
        offset_obj = 0
        for j, obj in enumerate(self.groups):
            for i in range(obj["N"]):
                if(obj["type"] == "floor"):
                    for param_group in self.optimizer_list[offset_obj+i].param_groups:
                        if param_group["name"] == "xyz":
                            lr = self.xyz_scheduler_floor_args(iteration)
                            param_group['lr'] = lr
            offset_obj += obj["N"]

    def update_learning_rate_trans(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "glob_trans":
                lr = self.trans_scheduler_args_trans(iteration)
                param_group['lr'] = lr
                return lr
            
    def update_learning_rate_rot(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "glob_rot":
                lr = self.trans_scheduler_args_rot(iteration)
                param_group['lr'] = lr
                return lr
            
    def update_learning_rate_scale(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "glob_scale":
                lr = self.trans_scheduler_args_scale(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply_global(self, path):
        """Save meta data for analysis or reloading later:
        K Gaussian Object Types - (x,y,z, f_dc, f_rest, opacity, scale, rotation)
        (N_1, N_2,...N_K) Delta Instance Data
        Translations, Rotations, Scales

        Args:
            path (_type_): _description_
        """
        os.makedirs(path, exist_ok=True)

        save_dict = {}
        # Add global transformation parameters
        glob_trans = torch.stack(self.glob_trans_list, dim=0).detach().cpu().numpy()
        glob_rot = torch.stack(self.glob_rot_list, dim=0).detach().cpu().numpy()
        glob_scale = torch.stack(self.glob_scale_list, dim=0).detach().cpu().numpy()
        print("extrinsics:", glob_trans.shape, glob_rot.shape, glob_scale.shape)

        save_delta_xyz = []
        save_delta_features_dc = []
        save_delta_features_rest = []
        save_delta_scaling = []
        save_delta_rotation = []
        save_delta_opacity = []
        for j, obj in enumerate(self.groups):
            save_delta_xyz.append(self.delta_xyz[j].detach().cpu().numpy())
            save_delta_features_dc.append(self.delta_features_dc[j].detach().cpu().numpy())
            save_delta_features_rest.append(self.delta_features_rest[j].detach().cpu().numpy())
            save_delta_scaling.append(self.delta_scaling[j].detach().cpu().numpy())
            save_delta_rotation.append(self.delta_rotation[j].detach().cpu().numpy())
            save_delta_opacity.append(self.delta_opacity[j].detach().cpu().numpy())
        
        save_dict = {
            "groups": self.groups,
            "glob_trans": glob_trans,
            "glob_rot": glob_rot,
            "glob_scale": glob_scale,
            "delta_xyz": save_delta_xyz,
            "delta_features_dc": save_delta_features_dc,
            "delta_features_rest": save_delta_features_rest,
            "delta_scaling": save_delta_scaling,
            "delta_rotation": save_delta_rotation,
            "delta_opacity": save_delta_opacity,
        }

        # save the object gaussians
        for j, obj in enumerate(self.groups):
            obj_path = os.path.join(path, "object_{}.ply".format(j)) 
            print(obj_path)
            self.gaussian_objects[j].save_ply(obj_path)

        final_path = os.path.join(path, "delta_global.npy")
        print(final_path)
        np.save(final_path, save_dict, allow_pickle=True)

    def load_ply_global(self, path):

        load_dict = np.load(os.path.join(path, "delta_global.npy"), allow_pickle=True).item()
        print(load_dict.keys())

        self.groups = load_dict["groups"]
        self.glob_trans_list = [torch.tensor(i, device=self.device) for i in load_dict["glob_trans"]]
        self.glob_rot_list = [torch.tensor(i, device=self.device) for i in load_dict["glob_rot"]]
        self.glob_scale_list = [torch.tensor(i, device=self.device) for i in load_dict["glob_scale"]]
        self.delta_xyz = [torch.tensor(xyz, device=self.device) for xyz in load_dict["delta_xyz"]]
        self.delta_features_dc = [torch.tensor(f_dc, device=self.device) for f_dc in load_dict["delta_features_dc"]]
        self.delta_features_rest = [torch.tensor(f_rest, device=self.device) for f_rest in load_dict["delta_features_rest"]]
        self.delta_scaling = [torch.tensor(scaling, device=self.device) for scaling in load_dict["delta_scaling"]]
        self.delta_rotation = [torch.tensor(rotation, device=self.device) for rotation in load_dict["delta_rotation"]]
        self.delta_opacity = [torch.tensor(opacity, device=self.device) for opacity in load_dict["delta_opacity"]]

        self.gaussian_objects = []
        for i in range(len(self.groups)):
            print(i)
            gaussians = GaussianModel(self.sh_degree, self.groups[i]["config"], self.device)
            gaussians.load_ply(os.path.join(path, "object_{}.ply".format(i)))
            self.gaussian_objects.append(gaussians)

        # self.groups = self.groups[0:-1]
        # self.gaussian_objects = self.gaussian_objects[0:-1]

        self.need_update()

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors[name]

    def _prune_optimizer(self, obj_id, mask, tensor_names):
        optimizable_tensors = {}
        for group in self.optimizer_list[obj_id].param_groups:
            if(group["name"] in tensor_names):
                stored_state = self.optimizer_list[obj_id].state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer_list[obj_id].state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer_list[obj_id].state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask, obj_id, instance_id):
        valid_points_mask = ~mask
        d = ["xyz", "f_dc", "f_rest", "opacity", "scaling" , "rotation"]
        optimizable_tensors = self._prune_optimizer(instance_id, valid_points_mask, d)

        self.delta_xyz[instance_id] = optimizable_tensors["xyz"]
        self.delta_features_dc[instance_id] = optimizable_tensors["f_dc"]
        self.delta_features_rest[instance_id] = optimizable_tensors["f_rest"]
        self.delta_opacity[instance_id] = optimizable_tensors["opacity"]
        self.delta_scaling[instance_id] = optimizable_tensors["scaling"]
        self.delta_rotation[instance_id] = optimizable_tensors["rotation"]

    def cat_tensors_to_optimizer(self, obj_id, tensors_dict, mask):
        optimizable_tensors = {}
        for group in self.optimizer_list[obj_id].param_groups:
            if(group["name"] in tensors_dict.keys()):
                extension_tensor = tensors_dict[group["name"]]
                print(len(self.optimizer_list))
                print(group["params"])
                print(obj_id)
                stored_state = self.optimizer_list[obj_id].state.get(group['params'][0], None)
                if stored_state is not None:

                    # stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    # stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], stored_state["exp_avg"][mask]), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], stored_state["exp_avg_sq"][mask]), dim=0)

                    del self.optimizer_list[obj_id].state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer_list[obj_id].state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, postfix_mask, instance_id, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(instance_id, d, postfix_mask)
        self.delta_xyz[instance_id] = optimizable_tensors["xyz"]
        self.delta_features_dc[instance_id] = optimizable_tensors["f_dc"]
        self.delta_features_rest[instance_id] = optimizable_tensors["f_rest"]
        self.delta_opacity[instance_id] = optimizable_tensors["opacity"]
        self.delta_scaling[instance_id] = optimizable_tensors["scaling"]
        self.delta_rotation[instance_id] = optimizable_tensors["rotation"]

    def densify_and_split(self, selected_pts_mask, obj_id, instance_id, N=2):

        new_xyz = self.delta_xyz[instance_id][selected_pts_mask].repeat(N,1)
        new_scaling = self.delta_scaling[instance_id][selected_pts_mask].repeat(N,1) # TODO
        # new_scaling = self.scaling_inverse_activation(self.scaling_activation(self.delta_scaling[instance_id][selected_pts_mask]).repeat(N,1) / (1.1))
        new_rotation = self.delta_rotation[instance_id][selected_pts_mask].repeat(N,1)
        new_features_dc = self.delta_features_dc[instance_id][selected_pts_mask].repeat(N,1,1)
        new_features_rest = self.delta_features_rest[instance_id][selected_pts_mask].repeat(N,1,1)
        new_opacity = self.delta_opacity[instance_id][selected_pts_mask].repeat(N,1)

        print("old_xyz", self.delta_xyz[instance_id].shape)
        print("new_xyz", new_xyz.shape)


        postfix_mask = torch.nonzero(selected_pts_mask).squeeze()
        postfix_mask = postfix_mask.repeat(N,1).flatten()
        self.densification_postfix(postfix_mask, instance_id, new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)


        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=self.device, dtype=bool)))
        self.prune_points(prune_filter, obj_id, instance_id)
        return selected_pts_mask

    def densify_and_clone(self, selected_pts_mask, obj_id, instance_id):
        new_xyz = self.delta_xyz[instance_id][selected_pts_mask]
        new_features_dc = self.delta_features_dc[instance_id][selected_pts_mask]
        new_features_rest = self.delta_features_rest[instance_id][selected_pts_mask]
        new_opacities = self.delta_opacity[instance_id][selected_pts_mask]
        new_scaling = self.delta_scaling[instance_id][selected_pts_mask]
        new_rotation = self.delta_rotation[instance_id][selected_pts_mask]

        self.densification_postfix(selected_pts_mask, instance_id, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
        return selected_pts_mask
    
    def densify_and_prune_instances(self, masks_list):
        self.need_update()
        obj_offset = 0
        for j in range(len(masks_list)):
            for k in range(self.groups[j]["N"]):
                self.densify_and_clone(masks_list[j][0], j, obj_offset+k)
                self.densify_and_split(masks_list[j][1], j, obj_offset+k)
                self.prune_points(masks_list[j][2], j, obj_offset+k)
            obj_offset += self.groups[j]["N"]
        torch.cuda.empty_cache()

class GaussianRenderer:
    def __init__(self, config):
        
        self.config = config
        self.device = config.device
        torch.set_default_device(self.device)
        self.sh_degree = config.sh_degree

        self.bg_color = torch.tensor(
            [1, 1, 1],
            dtype=torch.float32,
            device=self.device,
        )

    def render(
        self,
        gaussians,
        viewpoint_camera,
        scaling_modifier=1,
        bg_color=None,
        override_color=None,
        compute_cov3D_python=True,
        convert_SHs_python=False,
        gauss_idxs=None,
    ):
        if(gauss_idxs is not None):
            pass
        else:
            gauss_idxs = (0, gaussians.get_xyz.shape[0])
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                gaussians.get_xyz[gauss_idxs[0]:gauss_idxs[1]],
                dtype=gaussians.get_xyz.dtype,
                requires_grad=True,
                device=self.device,
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        if(bg_color is None):
            bg_color = self.bg_color
        else:
            bg_color = torch.tensor(
                                    bg_color,
                                    dtype=torch.float32,
                                    device=self.device,
                                )

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = gaussians.get_xyz
        means2D = screenspace_points
        opacity = gaussians.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = gaussians.get_covariance(scaling_modifier)
        else:
            scales = gaussians.get_scaling
            rotations = gaussians.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if convert_SHs_python:
                shs_view = gaussians.get_features.transpose(1, 2).view(
                    -1, 3, (gaussians.max_sh_degree + 1) ** 2
                )
                dir_pp = gaussians.get_xyz - viewpoint_camera.camera_center.repeat(
                    gaussians.get_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(
                    gaussians.active_sh_degree, shs_view, dir_pp_normalized
                )
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = gaussians.get_features
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
            means3D=means3D[gauss_idxs[0]:gauss_idxs[1]],
            means2D=means2D,
            shs=shs[gauss_idxs[0]:gauss_idxs[1]],
            colors_precomp=colors_precomp,
            opacities=opacity[gauss_idxs[0]:gauss_idxs[1]],
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp[gauss_idxs[0]:gauss_idxs[1]],
        )

        rendered_image_clamp = rendered_image.clamp(0, 1)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "image": rendered_image_clamp,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "unclamped_img": rendered_image,
        }
    
