import os
import argparse
from omegaconf import OmegaConf
import dearpygui.dearpygui as dpg
from tqdm import tqdm
from datetime import datetime
import torch
import random
import numpy as np

from utils.gui import DisplayGUI
from comp3dgs.object_trainer import ObjectTrainer

seed_value = 1
torch.manual_seed(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()
    # override default config from cli
    config = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))
    now = datetime.now() # current date and time
    save_path = os.path.join(config.save_path, now.strftime("%m-%d-%Y_%H-%M-%S_")+config.prompt)
    k = 4
    obj_trainers = []
    for i in range(k):
        config = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))
        config.save_path = save_path + "_" + str(i)
        os.makedirs(config.save_path, exist_ok=False)
        if(i == 0):
            obj_trainers.append(ObjectTrainer(config=config))
        else:
            obj_trainers.append(ObjectTrainer(config=config, guidance=obj_trainers[0].guide, 
                                              vsd_unet=(obj_trainers[0].unet, obj_trainers[0].unet_scheduler, obj_trainers[0].unet_optimizer)))
    torch.set_default_device(config.device)

    gui = DisplayGUI(gui_config=config.gui, render_config=config.render, trainer=obj_trainers[0])

    # start training
    for step in tqdm(range(config.max_iter), total=config.max_iter):
        if config.gui.enable:
            while(not obj_trainers[0].training):
                gui.render()
            else:
                gui.render()

        for i in range(k):
            obj_trainers[i].train_step()
        gui.need_update = True

    while(dpg.is_dearpygui_running()):
        gui.render()
