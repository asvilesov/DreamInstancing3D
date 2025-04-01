import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from tqdm import tqdm
import dearpygui.dearpygui as dpg
from datetime import datetime
import torch
from utils.gui import DisplayGUI
# import matplotlib.pyplot as plt 
# from pytorch3d.ops import knn_points
# from comp3dgs.compositional_trainer import CompTrainer
from comp3dgs.instance_trainer import InstanceCompTrainer
import numpy as np

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


    trainer = InstanceCompTrainer(config=config, groups=None)
    gui = DisplayGUI(gui_config=config.gui, render_config=config.render, trainer=trainer)

    for step in tqdm(range(config.max_iter), total=config.max_iter):

        if(step == 5):
            break
        if config.gui.enable:
            while(not trainer.training):
                gui.render()
            else:
                trainer.train_step()
                gui.render()
        gui.need_update = True

    trainer.gaussians.save_ply_global(config.save_path)
    trainer.gaussians.load_ply_global(config.save_path)

    while(dpg.is_dearpygui_running()):
        gui.render()
