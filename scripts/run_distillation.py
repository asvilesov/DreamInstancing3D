import os
import argparse
from omegaconf import OmegaConf
import dearpygui.dearpygui as dpg
from tqdm import tqdm
from datetime import datetime
import torch

from utils.gui import DisplayGUI
from comp3dgs.object_cloner import ObjectCloner

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()
    # override default config from cli
    config = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))
    
    now = datetime.now() # current date and time
    config.save_path = os.path.join(config.save_path, now.strftime("%m-%d-%Y_%H-%M-%S_cloner"))
    os.makedirs(config.save_path, exist_ok=False)
    torch.set_default_device(config.device)

    trainer = ObjectCloner(config=config)
    gui = DisplayGUI(gui_config=config.gui, render_config=config.render, trainer=trainer)

    # start training
    for step in tqdm(range(config.max_iter), total=config.max_iter):
        if config.gui.enable:
            while(not trainer.training):
                gui.render()
            else:
                gui.render()

        trainer.train_step()
        gui.need_update = True

    trainer.eval_step()
    gui.need_update = True

    while(dpg.is_dearpygui_running()):
        gui.render()