from comp3dgs.gs_renderer import *
from utils.debug import timer

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

    def __init__(self, config, device):

        self.sh_degree = config.sh_degree
        self.grid_size = np.array(config.grid_size)
        assert self.grid_size.shape[0] == 2
        self.grid_resolution = np.array(config.grid_resolution)
        assert self.grid_resolution.shape[0] == 2
        self.obj_path = config.load_path
        self.device = device

        # temporary for one object
        self.gaussians = GaussianModel(self.sh_degree, self.device)
        self.gaussians.load_ply(self.obj_path)
        self.objects = []
        # floor 
        self.floor_gaussians = GaussianModel(self.sh_degree, self.device)
        # uniformaly sample a floor
        nx, ny = (30, 30)
        x = np.linspace(-0.5*self.grid_size[0]/(self.grid_resolution[0]-1), 0.5*self.grid_size[0]/(self.grid_resolution[0]-1), nx)
        y = np.linspace(-0.5*self.grid_size[1]/(self.grid_resolution[1]-1), 0.5*self.grid_size[1]/(self.grid_resolution[1]-1), ny)
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
        self.floor_gaussians.create_from_pcd(pcd, 1, spatial_scale=0.8)


        # update variable for efficient scene construction
        self.update_scene = True
        self.update_gaussians = {'xyz':True, 'features':True, 'scaling':True, 'rotation':True, 'opacity':True}

        # create grid for rendering
        self.grid = create_grid(self.grid_size, self.grid_resolution, device)

        self.glob_trans_list = []
        self.glob_rot_list = []
        self.glob_scale_list = []
        self.glob_opacity_list = []

        # Set the object-level coordinates and world-level coordinate transformations
        self.point_per_obj = []
        N_instances = len(self.grid)
        self.no_objs = N_instances
        for i in range(N_instances):
            cur_obj = self.gaussians #GaussianModel(sh_degree)
            self.objects.append(cur_obj)

            # root_rad = self.get_radius(cur_obj)
            self.glob_trans_list.append(nn.Parameter(self.grid[i].unsqueeze(0).float().requires_grad_(True)))
            self.glob_rot_list.append(nn.Parameter(torch.tensor([1,0,0,0], device=self.device).unsqueeze(0).float().requires_grad_(True)))
            self.glob_scale_list.append(nn.Parameter(torch.tensor([0.4], device=self.device).unsqueeze(0).float().requires_grad_(True)))
            self.glob_opacity_list.append(nn.Parameter(torch.tensor(0.01, device=self.device).unsqueeze(0).float().requires_grad_(True)))

            if i==0:
                self._xyz = cur_obj._xyz
                self._features_dc = cur_obj._features_dc
                self._features_rest = cur_obj._features_rest
                self._scaling = cur_obj._scaling
                self._rotation = cur_obj._rotation
                self._opacity = cur_obj._opacity

            else:
                self._xyz = torch.cat((self._xyz, cur_obj._xyz), dim=0)
                self._features_dc = torch.cat((self._features_dc, cur_obj._features_dc), dim=0)
                self._features_rest = torch.cat((self._features_rest, cur_obj._features_rest), dim=0)
                self._scaling = torch.cat((self._scaling, cur_obj._scaling), dim=0)
                self._rotation = torch.cat((self._rotation, cur_obj._rotation), dim=0)
                self._opacity = torch.cat((self._opacity, cur_obj._opacity), dim=0)

            self.point_per_obj.append(cur_obj._xyz.shape[0])
        
        self.no_objs += N_instances
        for j in range(N_instances):
            cur_obj = self.floor_gaussians
            self.objects.append(cur_obj)

            # root_rad = self.get_radius(cur_obj)
            self.glob_trans_list.append(nn.Parameter((self.grid[j]-torch.tensor([0,0.5,0])).unsqueeze(0).float().requires_grad_(False), requires_grad=False))
            self.glob_rot_list.append(nn.Parameter(torch.tensor([1,0,0,0], device=self.device).unsqueeze(0).float().requires_grad_(False), requires_grad=False))
            self.glob_scale_list.append(nn.Parameter(torch.tensor([1], device=self.device).unsqueeze(0).float().requires_grad_(False), requires_grad=False))
            self.glob_opacity_list.append(nn.Parameter(torch.tensor(1, device=self.device).unsqueeze(0).float().requires_grad_(False), requires_grad=False))

            if i==0:
                self._xyz = cur_obj._xyz
                self._features_dc = cur_obj._features_dc
                self._features_rest = cur_obj._features_rest
                self._scaling = cur_obj._scaling
                self._rotation = cur_obj._rotation
                self._opacity = cur_obj._opacity

            else:
                self._xyz = torch.cat((self._xyz, cur_obj._xyz), dim=0)
                self._features_dc = torch.cat((self._features_dc, cur_obj._features_dc), dim=0)
                self._features_rest = torch.cat((self._features_rest, cur_obj._features_rest), dim=0)
                self._scaling = torch.cat((self._scaling, cur_obj._scaling), dim=0)
                self._rotation = torch.cat((self._rotation, cur_obj._rotation), dim=0)
                self._opacity = torch.cat((self._opacity, cur_obj._opacity), dim=0)

            self.point_per_obj.append(cur_obj._xyz.shape[0])
        
        self._features_dc = torch.nn.Parameter(self._features_dc.contiguous().requires_grad_(True), requires_grad=True)
        self._scaling = torch.nn.Parameter(self._scaling.contiguous().requires_grad_(True), requires_grad=True)    
        self._xyz = torch.nn.Parameter(self._xyz.contiguous().requires_grad_(True), requires_grad=True)    
        self._rotation = torch.nn.Parameter(self._rotation.contiguous().requires_grad_(True), requires_grad=True)    
        self._opacity = torch.nn.Parameter(self._opacity.contiguous().requires_grad_(True), requires_grad=True)        
        # Set the world-level coordinates 
        self._xyz_world = torch.clone(self._xyz)
        self._features_dc_world = torch.clone(self._features_dc)
        self._features_rest_world = torch.clone(self._features_rest)
        self._scaling_world = torch.clone(self._scaling)
        self._rotation_world = torch.clone(self._rotation)
        self._opacity_world = torch.clone(self._opacity)
        
        print(self.glob_opacity_list)
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

    def need_update(self):
        self.update_scene = True
        self.update_gaussians = {'xyz':True, 'features':True, 'scaling':True, 'rotation':True, 'opacity':True}        

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

    def update_coords(self):
        if(self.update_scene):
            # self.point_per_obj = []
            # for i in range(self.no_objs):
            #     cur_obj = self.objects[i]
            #     if i==0:
            #         self._xyz = cur_obj._xyz
            #         features_dc = cur_obj._features_dc
            #         self._features_rest = cur_obj._features_rest
            #         self._scaling = cur_obj._scaling
            #         self._rotation = cur_obj._rotation
            #         self._opacity = cur_obj._opacity

            #     else:
            #         self._xyz = torch.cat((self._xyz, cur_obj._xyz), dim=0)
            #         features_dc = torch.cat((features_dc, cur_obj._features_dc), dim=0)
            #         self._features_rest = torch.cat((self._features_rest, cur_obj._features_rest), dim=0)
            #         self._scaling = torch.cat((self._scaling, cur_obj._scaling), dim=0)
            #         self._rotation = torch.cat((self._rotation, cur_obj._rotation), dim=0)
            #         self._opacity = torch.cat((self._opacity, cur_obj._opacity), dim=0)
            #     self.point_per_obj.append(cur_obj._xyz.shape[0])
            # self.optimizer.param_groups["dc"] = features_dc 

            self.update_scene = False
        else:
            pass

    @property
    def get_scaling(self):
        self.update_coords()
        if(self.update_gaussians['scaling']):
            # Scale according to global scaling for each object
            for i in range(self.no_objs):
                if i==0:
                    temp = self.glob_scale_list[i] * self.scaling_activation(self._scaling[0:self.point_per_obj[i]])
                else:
                    temp = torch.cat((temp, self.glob_scale_list[i] * self.scaling_activation(self._scaling[torch.sum(torch.tensor(self.point_per_obj[0:i])):torch.sum(torch.tensor(self.point_per_obj[0:i+1]))])), dim=0)
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
            for i in range(self.no_objs):
                glob_rot_mat = build_rotation(self.glob_rot_list[i])
                if i==0:
                    # 1. scaling
                    temp2 = self.glob_scale_list[i]*self._xyz[0:self.point_per_obj[i]]
                    # 2. rotation
                    temp2 = (glob_rot_mat @ temp2.unsqueeze(-1)).squeeze(-1)
                    # 3. translation
                    temp = (temp2 + self.glob_trans_list[i]).squeeze(0)
                else:
                    temp2 = self.glob_scale_list[i]*self._xyz[torch.sum(torch.tensor(self.point_per_obj[0:i])):torch.sum(torch.tensor(self.point_per_obj[0:i+1]))]
                    temp2 = (glob_rot_mat @ temp2.unsqueeze(-1)).squeeze(-1)
                    temp = torch.cat((temp,(temp2 + self.glob_trans_list[i]).squeeze(0)), dim=0)
            self._xyz_world = temp
            self.update_gaussians['xyz'] = False
        return self._xyz_world
    
    @property
    def get_features(self):
        self.update_coords()
        if(self.update_gaussians['features']):
            features_dc = self._features_dc
            features_rest = self._features_rest
            self._features_dc_world = features_dc
            self._features_rest_world = features_rest
            self.update_gaussians['features'] = False
        return torch.cat((self._features_dc_world, self._features_rest_world), dim=1)
    
    @property
    def get_opacity(self):
        self.update_coords()
        if(self.update_gaussians['opacity']):
            for i in range(self.no_objs):
                if i==0:
                    temp = self.glob_opacity_list[i] * self.opacity_activation(self._opacity[0:self.point_per_obj[i]])
                else:
                    temp = torch.cat((temp, self.glob_opacity_list[i] * self.opacity_activation(self._opacity[torch.sum(torch.tensor(self.point_per_obj[0:i])):torch.sum(torch.tensor(self.point_per_obj[0:i+1]))])), dim=0)
            self._opacity_world = temp
            self.update_gaussians['opacity'] = False
        return self._opacity_world #self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        self.update_coords()
        if(self.update_gaussians['rotation']):
            for i in range(self.no_objs):
                if i==0:
                    temp = self.covariance_activation(self.get_scaling[0:self.point_per_obj[i]], scaling_modifier, self._rotation[0:self.point_per_obj[i]], self.glob_rot_list[i])

                else:
                    temp = torch.cat((temp, self.covariance_activation(self.get_scaling[torch.sum(torch.tensor(self.point_per_obj[0:i])):torch.sum(torch.tensor(self.point_per_obj[0:i+1]))], scaling_modifier, self._rotation[torch.sum(torch.tensor(self.point_per_obj[0:i])):torch.sum(torch.tensor(self.point_per_obj[0:i+1]))], self.glob_rot_list[i])), dim=0)
            self._rotation_world = temp
            self.update_gaussians['rotation'] = False
        return self._rotation_world
    
    def training_setup(self):
        # self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device = self.device)
        # self.denom = torch.zeros((self.get_xyz.shape[0], 1), device = self.device)

        l = [
            {'params': self.glob_trans_list, 'lr': 2e-5, "name": "glob_trans"},
            {'params': self.glob_rot_list, 'lr': 2e-5, "name": "glob_rot"},
            {'params': self.glob_scale_list, 'lr': 1e-5, "name": "glob_scale"},
            {'params': self.glob_opacity_list, 'lr': 5e-6, "name": "glob_opacity"},
            {'params': [self._features_dc], 'lr': 2e-2, "name": "dc"},
            {'params': [self._scaling], 'lr': 2e-3, "name": "scaling"},
            {'params': [self._rotation], 'lr': 2e-3, "name": "rotation"},
            {'params': [self._opacity], 'lr': 1e-2, "name": "opacity"},
            {'params': [self._xyz], 'lr': 4e-4, "name": "xyz"},
        ]

        ## Freeze gradients for the anchor object
        # self.glob_trans_list[0].requires_grad_(False)
        # self.glob_rot_list[0].requires_grad_(False)
        # self.glob_scale_list[0].requires_grad_(False)
        # self.glob_scale_list[1].requires_grad_(False)

        self.optimizer = torch.optim.SGD(l, lr=0.1)

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

    def save_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        self.update_coords()

        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self.get_raw_scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_ply_global(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        self.update_coords()
        # Add global transformation parameters
        glob_trans = torch.stack(self.glob_trans_list, dim=0).squeeze(1).squeeze(1).detach().cpu().numpy()
        glob_rot = torch.stack(self.glob_rot_list, dim=0).squeeze(1).detach().cpu().numpy()
        glob_scale = torch.stack(self.glob_scale_list, dim=0).squeeze(1).detach().cpu().numpy()

        l_atts = ['tx','ty','tz','r0','r1','r2','r3','scale']
        dtype_full = [(attribute, 'f4') for attribute in l_atts]

        elements = np.empty(glob_trans.shape[0], dtype=dtype_full)

        attributes = np.concatenate((glob_trans, glob_rot, glob_scale), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply_global(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["tx"]),
                        np.asarray(plydata.elements[0]["ty"]),
                        np.asarray(plydata.elements[0]["tz"])),  axis=1)
        
        rot = np.stack((np.asarray(plydata.elements[0]["r0"]),
                        np.asarray(plydata.elements[0]["r1"]),
                        np.asarray(plydata.elements[0]["r2"]),
                        np.asarray(plydata.elements[0]["r3"])),  axis=1)
        
        scale = np.asarray(plydata.elements[0]["scale"])[..., np.newaxis]

        self.glob_trans_list = []
        self.glob_rot_list = []
        self.glob_scale_list = []
        for ii in range(xyz.shape[0]):
            self.glob_trans_list.append(nn.Parameter(torch.tensor(xyz[ii], device=self.device).unsqueeze(0).unsqueeze(0).float().requires_grad_(True)))    
            self.glob_rot_list.append(nn.Parameter(torch.tensor(rot[ii], device=self.device).unsqueeze(0).float().requires_grad_(True)))
            self.glob_scale_list.append(nn.Parameter(torch.tensor(scale[ii], device=self.device).unsqueeze(0).float().requires_grad_(True)))








class RendererInstanceComposition:
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
    
    def initialize(self):
        # load checkpoint
        self.gaussians = InstanceCompositionModel(self.config, self.device)
    
    def render(
        self,
        viewpoint_camera,
        scaling_modifier=1,
        bg_color=None,
        override_color=None,
        compute_cov3D_python=True,
        convert_SHs_python=False,
    ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                self.gaussians.get_xyz,
                dtype=self.gaussians.get_xyz.dtype,
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
            sh_degree=self.gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.gaussians.get_xyz
        means2D = screenspace_points
        opacity = self.gaussians.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = self.gaussians.get_covariance(scaling_modifier)
        else:
            scales = self.gaussians.get_scaling
            rotations = self.gaussians.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if convert_SHs_python:
                shs_view = self.gaussians.get_features.transpose(1, 2).view(
                    -1, 3, (self.gaussians.max_sh_degree + 1) ** 2
                )
                dir_pp = self.gaussians.get_xyz - viewpoint_camera.camera_center.repeat(
                    self.gaussians.get_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(
                    self.gaussians.active_sh_degree, shs_view, dir_pp_normalized
                )
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = self.gaussians.get_features
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
        rendered_image = rendered_image.clamp(0, 1)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "image": rendered_image,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }