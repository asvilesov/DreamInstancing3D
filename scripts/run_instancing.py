import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from tqdm import tqdm
import dearpygui.dearpygui as dpg
from datetime import datetime
import torch
from utils.gui import DisplayGUI
from comp3dgs.instance_trainer import InstanceCompTrainer
import numpy as np
import random

from dataclasses import dataclass


def create_grid(grid_size:int, grid_resolution:int, device) -> torch.Tensor:
    """Create a grid of points in the range [-1, 1] x [-1, 1] x [-1, 1]"""
    grid = torch.meshgrid(
        (grid_size[0]/grid_resolution[0])*(torch.arange(0, grid_resolution[0], device=device)-(grid_resolution[0]-1)/2),
        (grid_size[1]/grid_resolution[1])*(torch.arange(0, grid_resolution[1], device=device)-(grid_resolution[1]-1)/2),
    )
    N = len(grid[0].flatten())
    y_axis = torch.zeros(N, device=device)
    print(grid[0].shape)
    grid = torch.stack([grid[0].flatten(), y_axis, grid[1].flatten()], dim=-1)
    print(f"grid.shape: {grid.shape}")
    return grid

seed_value = 1
torch.manual_seed(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()
    # override default config from cli
    config = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))
    
    now = datetime.now() # current date and time
    config.save_path = os.path.join(config.save_path, now.strftime("%m-%d-%Y_%H-%M-%S_")+config.prompt)
    os.makedirs(config.save_path, exist_ok=False)
    torch.set_default_device(config.device)

    # Instance Configuration:
    from scipy.spatial.transform import Rotation as R
    import quaternion
    

    ############## Cars and Trucks on a freeway

    grid_size = np.array([16, 2])
    grid_resolution = np.array([8, 1])
    grid = create_grid(grid_size, grid_resolution, config.device)
    N = grid_resolution[0]*grid_resolution[1]
    N_sample1 = 1

    T = torch.tensor([[0.0, 0.0, 0.4],
                      [3.5, 0.0, 0.4],
                      [-2.0, 0.0, -0.4],
                      [-5.0, 0.0, -0.4]])

    Rot = torch.tensor([1,0,0,0]).expand(N, 4)
    S = torch.tensor([1]).expand(N, 1)
    Rs = []

    rot_degrees_floor = np.zeros(N)

    rot_degrees = np.array([0, 0, 180, 180])
    for i in range(N_sample1):
        quat = np.quaternion(1, 0, 0, 0)
        r = R.from_euler('y', rot_degrees[i:i+1], degrees=True)[0]
        r = r.as_matrix()
        r[1:3] = -r[1:3]
        r = R.from_matrix(r)
        b = r.as_quat()
        quat2 = np.quaternion(b[0], b[1], b[2], b[3])
        quat = quat2
        quat = quaternion.as_float_array(quat)
        Rs.append(quat)
    Rs = torch.tensor(Rs)

    groups = [
        {"text": "DSLR photo of a sailboat.",
         "reorient": 90, 
         "pointe": "a sailboat", 
            "N": N_sample1, "T": T[0:N_sample1], "R": Rs[0:N_sample1], "S": 0.6*S[0:N_sample1], 
            "opt_pose": True, "pose_emb": True, "R_degrees": rot_degrees[0:N_sample1], "T_y": -0.1,
            "config": OmegaConf.load("./configs/instance/text3d_object.yaml"),
            "sim": 1.00, 
            "type": "object"},
        {"text": "DSLR photo of the ocean.", 
         "reorient": 0,
            "N": N, "T": grid - torch.tensor([0.0, 0.20, 0.0]), "R": Rot, "S": grid_size[0]/grid_resolution[0] * S, 
            "opt_pose": False, "pose_emb": False, "R_degrees": rot_degrees_floor, "T_y": 0.0,
            "config": OmegaConf.load("./configs/instance/text3d_floor.yaml"),
            "sim": 1.0,
            "type": "floor"},
        # {"text": "weather", 
        #     "N": 1, "T": torch.tensor([[0.0, 0.0, 0.0]]), "R": Rot[0:1], "S": 50*S, "opt_pose": False, 
        #     "config": OmegaConf.load("./configs/instance/text3d_globe.yaml"),
        #     "sim": 0.0,
        #     "type": "globe"},
        # {"text": "rock", "N": N, "T": grid, "R": R, "S": S},
    ]

    ############## Train on a railway track

    # grid_size = np.array([32, 2])
    # grid_resolution = np.array([16, 1])
    # grid = create_grid(grid_size, grid_resolution, config.device)
    # N = grid_resolution[0]*grid_resolution[1]
    # N_sample1 = 1
    # N_sample2 = 4

    # T = torch.tensor([[-3.5, 0.0, 0],
    #                   [-0.5, 0.0, 0],
    #                   [2, 0.0, 0],
    #                   [4.5, 0.0, 0],
    #                   [7, 0.0, 0]
    #                   ]) - torch.tensor([2, 0.0, 0.0])

    # Rot = torch.tensor([1,0,0,0]).expand(N, 4)
    # S = torch.tensor([1]).expand(N, 1)
    # Rs = []

    # rot_degrees_floor = np.zeros(N)

    # rot_degrees = np.array([180,0,0,0,0])
    # for i in range(N_sample1+N_sample2):
    #     quat = np.quaternion(1, 0, 0, 0)
    #     r = R.from_euler('y', rot_degrees[i:i+1], degrees=True)[0]
    #     r = r.as_matrix()
    #     r[1:3] = -r[1:3]
    #     r = R.from_matrix(r)
    #     b = r.as_quat()
    #     quat2 = np.quaternion(b[0], b[1], b[2], b[3])
    #     quat = quat2
    #     quat = quaternion.as_float_array(quat)
    #     Rs.append(quat)
    # Rs = torch.tensor(Rs)

    # groups = [
    #     {"text": "A DSLR photo of a streaming engine train, high resolution.",
    #      "reorient": 90, 
    #      "pointe": "a train", 
    #         "N": N_sample1, "T": T[0:N_sample1], "R": Rs[0:N_sample1], "S": 1.2*S[0:N_sample1], 
    #         "opt_pose": True, "pose_emb": True, "R_degrees": rot_degrees[0:N_sample1], "T_y": 0.1,
    #         "config": OmegaConf.load("./configs/instance/text3d_object.yaml"),
    #         "sim": 0.03, 
    #         "type": "object"},
    #     {"text": "A DSLR photo of a train wagon", 
    #      "reorient": 90,
    #      "pointe": "a chest", 
    #         "N": N_sample2, "T": T[N_sample1:N_sample1+N_sample2], "R": Rs[N_sample1:N_sample1+N_sample2], "S": 1.2*S[N_sample1:N_sample1+N_sample2], 
    #         "opt_pose": True, "pose_emb": True, "R_degrees": rot_degrees[N_sample1:N_sample1+N_sample2], "T_y": 0.05,
    #         "config": OmegaConf.load("./configs/instance/text3d_object.yaml"),
    #         "sim": 0.2, 
    #         "type": "object"},
    #     {"text": "A photo of train tracks.", 
    #      "reorient": 0,
    #         "N": N, "T": grid - torch.tensor([0.0, 0.20, 0.0]), "R": Rot, "S": grid_size[0]/grid_resolution[0] * S, 
    #         "opt_pose": False, "pose_emb": False, "R_degrees": rot_degrees_floor, "T_y": 0.0,
    #         "config": OmegaConf.load("./configs/instance/text3d_floor.yaml"),
    #         "sim": 0.5,
    #         "type": "floor"}
    # ]




    ############## Sailboat and jetskis in the sea

    # grid_size = np.array([20, 20])
    # grid_resolution = np.array([10, 10])
    # grid = create_grid(grid_size, grid_resolution, config.device)
    # N = grid_resolution[0]*grid_resolution[1]
    # N_sample1 = 2
    # N_sample2 = 4

    # T = torch.tensor([[-7, 0.0, -7],
    #                   [-6, 0.0, 5],
    #                   [-2.0, 0.0, 1],
    #                   [-3, 0.0, -1.5],
    #                   [5.5, 0.0, 5],
    #                   [5.0, 0.0, 3]])

    # Rot = torch.tensor([1,0,0,0]).expand(N, 4)
    # S = torch.tensor([1]).expand(N, 1)
    # Rs = []

    # rot_degrees_floor = np.zeros(N)

    # rot_degrees = np.array([70, 270, 40, 50, 30, 200])
    # for i in range(N_sample1+N_sample2):
    #     quat = np.quaternion(1, 0, 0, 0)
    #     r = R.from_euler('y', rot_degrees[i:i+1], degrees=True)[0]
    #     r = r.as_matrix()
    #     r[1:3] = -r[1:3]
    #     r = R.from_matrix(r)
    #     b = r.as_quat()
    #     quat2 = np.quaternion(b[0], b[1], b[2], b[3])
    #     quat = quat2
    #     quat = quaternion.as_float_array(quat)
    #     Rs.append(quat)
    # Rs = torch.tensor(Rs)

    # groups = [
    #     {"text": "photo of a sailboat",
    #      "reorient": 90, 
    #      "pointe": "a sailboat", 
    #         "N": N_sample1, "T": T[0:N_sample1], "R": Rs[0:N_sample1], "S": 4.0*S[0:N_sample1], 
    #         "opt_pose": True, "pose_emb": True, "R_degrees": rot_degrees[0:N_sample1], "T_y": 0.5,
    #         "config": OmegaConf.load("./configs/instance/text3d_object.yaml"),
    #         "sim": 1.0, 
    #         "type": "object"},
    #     {"text": "photo of a jetski", 
    #      "reorient": 90,
    #      "pointe": "a jetski", 
    #         "N": N_sample2, "T": T[N_sample1:N_sample1+N_sample2], "R": Rs[N_sample1:N_sample1+N_sample2], "S": 1*S[N_sample1:N_sample1+N_sample2], 
    #         "opt_pose": True, "pose_emb": True, "R_degrees": rot_degrees[N_sample1:N_sample1+N_sample2], "T_y": 0.03,
    #         "config": OmegaConf.load("./configs/instance/text3d_object.yaml"),
    #         "sim": 0.1, 
    #         "type": "object"},
    #     {"text": "photo of the sea", 
    #      "reorient": 0,
    #         "N": N, "T": grid - torch.tensor([0.0, 0.20, 0.0]), "R": Rot, "S": grid_size[0]/grid_resolution[0] * S, 
    #         "opt_pose": False, "pose_emb": False, "R_degrees": rot_degrees_floor, "T_y": 0.0,
    #         "config": OmegaConf.load("./configs/instance/text3d_floor.yaml"),
    #         "sim": 0.5,
    #         "type": "floor"},
    #     {"text": "weather", 
    #      "reorient": 0,
    #         "N": 1, "T": torch.tensor([[0.0, 0.0, 0.0]]), "R": Rot[0:1], "S": 50*S, 
    #         "opt_pose": False, "pose_emb": False, "R_degrees": rot_degrees_floor[0:1], "T_y": 0.0,
    #         "config": OmegaConf.load("./configs/instance/text3d_globe.yaml"),
    #         "sim": 0.0,
    #         "type": "globe"},
    # ]

    ############### Deer in a forest
    # grid_size = np.array(config.grid_size)
    # grid_resolution = np.array(config.grid_resolution)
    # grid = create_grid(grid_size, grid_resolution, config.device)
    # N = grid_resolution[0]*grid_resolution[1]
    # Rot = torch.tensor([1,0,0,0]).expand(N, 4)
    # S = torch.tensor([1]).expand(N, 1)

    # N_sample1 = 10
    # N_sample2 = 3
    # rand_idxs = np.random.choice(N, N_sample1+N_sample2, replace=False)
    # rand_idxs = np.random.choice(N, N_sample1+N_sample2, replace=False)
    # rand_idxs = np.random.choice(N, N_sample1+N_sample2, replace=False)
    # rand_idxs = np.random.choice(N, N_sample1+N_sample2, replace=False)
    # rand_idxs = np.random.choice(N, N_sample1+N_sample2, replace=False)
    # rand_idxs = np.random.choice(N, N_sample1+N_sample2, replace=False)



    # rand_idxs1 = rand_idxs[:N_sample1]
    # rand_idxs2 = rand_idxs[N_sample1:]
    # Rs = []

    # rot_degrees_floor = np.zeros(N)
    # rot_degrees = 360*np.random.rand(N)
    # for i in range(N):
    #     quat = np.quaternion(1, 0, 0, 0)
    #     r = R.from_euler('y', rot_degrees[i:i+1], degrees=True)[0]
    #     r = r.as_matrix()
    #     r[1:3] = -r[1:3]
    #     r = R.from_matrix(r)
    #     b = r.as_quat()
    #     quat2 = np.quaternion(b[0], b[1], b[2], b[3])
    #     quat = quat2
    #     quat = quaternion.as_float_array(quat)
    #     Rs.append(quat)
    # Rs = torch.tensor(Rs)

    # groups = [
    #     # {"text": "DSLR photo of a deer", 
    #     #  "reorient": 90,
    #     #  "pointe": "a deer", 
    #     #     "N": N_sample2, "T": grid[rand_idxs2], "R": Rs[rand_idxs2], "S": 1*S[rand_idxs2], 
    #     #     "opt_pose": True, "pose_emb": True, "R_degrees": rot_degrees[rand_idxs2], "T_y": 0.3,
    #     #     "config": OmegaConf.load("./configs/instance/text3d_object.yaml"),
    #     #     "sim": 0.1, #0.1,
    #     #     "type": "object"},
    #     {"text": "DSLR photo of a pine tree", 
    #      "pointe": "A tree.", 
    #      "reorient": 0,
    #         "N": N_sample1, "T": grid[rand_idxs1], "R": Rs[rand_idxs1], "S": 3.0*S[rand_idxs1], 
    #         "opt_pose": False, "pose_emb": False, "R_degrees": rot_degrees[rand_idxs1], "T_y": 0.0,
    #         "config": OmegaConf.load("./configs/instance/text3d_object.yaml"),
    #         "sim": 0.3,
    #         "type": "object"},
    #     # {"text": "DSLR photo of a grass field", 
    #     #  "reorient": 0, 
    #     #     "N": N, "T": grid - torch.tensor([0.0, 0.20, 0.0]), "R": Rot, "S": grid_size[0]/grid_resolution[0] * S, 
    #     #     "opt_pose": False, "pose_emb": False, "R_degrees": rot_degrees_floor, "T_y": 0.0,
    #     #     "config": OmegaConf.load("./configs/instance/text3d_floor.yaml"),
    #     #     "sim": 0.5,
    #     #     "type": "floor"},
    #     # {"text": "weather", 
    #     #     "N": 1, "T": torch.tensor([[0.0, 0.0, 0.0]]), "R": Rot[0:1], "S": 50*S, "opt_pose": False, 
    #     #     "config": OmegaConf.load("./configs/instance/text3d_globe.yaml"),
    #     #     "sim": 0.0,
    #     #     "type": "globe"},
    #     # {"text": "rock", "N": N, "T": grid, "R": R, "S": S},
    # ]
    # groups = [
    #     {"text": "DSLR photo of a deer", 
    #      "reorient": 90,
    #      "pointe": "a deer", 
    #         "N": N_sample2, "T": grid[rand_idxs2], "R": Rs[rand_idxs2], "S": 1*S[rand_idxs2], 
    #         "opt_pose": True, "pose_emb": True, "R_degrees": rot_degrees[rand_idxs2], "T_y": 0.3,
    #         "config": OmegaConf.load("./configs/instance/text3d_object.yaml"),
    #         "sim": 0.2, #0.1,
    #         "type": "object"},
    #     # {"text": "a tree, cartoon style", 
    #     #  "pointe": "a tree", 
    #     #  "reorient": 0,
    #     #     "N": N_sample1, "T": grid[rand_idxs1], "R": Rs[rand_idxs1], "S": 3.0*S[rand_idxs1], 
    #     #     "opt_pose": False, "pose_emb": False, "R_degrees": rot_degrees[rand_idxs1], "T_y": 0.0,
    #     #     "config": OmegaConf.load("./configs/instance/text3d_object.yaml"),
    #     #     "sim": 0.3,
    #     #     "type": "object"},
    #     # {"text": "a grass field, cartoon style", 
    #     #  "reorient": 0, 
    #     #     "N": N, "T": grid - torch.tensor([0.0, 0.20, 0.0]), "R": Rot, "S": grid_size[0]/grid_resolution[0] * S, 
    #     #     "opt_pose": False, "pose_emb": False, "R_degrees": rot_degrees_floor, "T_y": 0.0,
    #     #     "config": OmegaConf.load("./configs/instance/text3d_floor.yaml"),
    #     #     "sim": 0.5,
    #     #     "type": "floor"},
    #     # {"text": "weather", 
    #     #     "N": 1, "T": torch.tensor([[0.0, 0.0, 0.0]]), "R": Rot[0:1], "S": 50*S, "opt_pose": False, 
    #     #     "config": OmegaConf.load("./configs/instance/text3d_globe.yaml"),
    #     #     "sim": 0.0,
    #     #     "type": "globe"},
    #     # {"text": "rock", "N": N, "T": grid, "R": R, "S": S},
    # ]

    trainer = InstanceCompTrainer(config=config, groups=groups)
    gui = DisplayGUI(gui_config=config.gui, render_config=config.render, trainer=trainer)

    for step in tqdm(range(config.max_iter), total=config.max_iter):
        if config.gui.enable:
            while(not trainer.training):
                gui.render()
            else:
                trainer.train_step()
                gui.render()
        gui.need_update = True

    while(dpg.is_dearpygui_running()):
        gui.render()
