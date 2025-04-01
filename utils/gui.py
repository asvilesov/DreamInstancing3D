from utils.cam import orbit_camera, OrbitCamera
from comp3dgs.gs_renderer import Renderer, MiniCam

import dearpygui.dearpygui as dpg
import numpy as np
import os
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
import imageio as iio

class DisplayGUI:
    def __init__(self, gui_config, render_config, trainer):
        self.display_c = gui_config  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.render_c = render_config
        self.WIDTH = gui_config.WIDTH
        self.HEIGHT = gui_config.HEIGHT
        self.cam = OrbitCamera(gui_config.WIDTH, gui_config.HEIGHT, r=self.render_c.radius+1, fovy=self.render_c.fovy)
        self.gui_image = np.ones((self.WIDTH, self.HEIGHT, 3), dtype=np.float32)
        self.need_update = True  

        self.trainer = trainer
        self.mode = "image"
        self.gaussain_scale_factor = 1
        self.bg_gui_color = 1

        self.curr_elevation = 0
        self.curr_azimuth = 0

        self.save_path = "./media/"
        self.img_num = 0
        self.video_num = 0
        self.currently_recording = False
        self.save_on_update_only = False
        self.video_buffer = []

        dpg.create_context()
        self.register_dpg()
        
        dpg.create_viewport(
            title="Gaussian3D",
            width=self.WIDTH + 600,
            height=self.HEIGHT + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        ### register a larger font
        # get it from: https://github.com/lxgw/LxgwWenKai/releases/download/v1.300/LXGWWenKai-Regular.ttf
        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)

        # dpg.show_metrics()

        dpg.show_viewport()

    def __del__(self):
        if self.display_c.enable:
            dpg.destroy_context()

    def register_dpg(self):
        #---------------------register texture---------------------

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.WIDTH,
                self.HEIGHT,
                self.gui_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        #---------------------register window---------------------

        # the rendered image, as the primary window
        with dpg.window(
            tag="_primary_window",
            width=self.WIDTH,
            height=self.HEIGHT,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        #---------------------control window---------------------
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=600,
            height=self.HEIGHT,
            pos=[self.WIDTH, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            def callback_setattr_gui(sender, app_data, user_data):
                setattr(self, user_data, app_data)
            
            def callback_setattr_trainer(sender, app_data, user_data):
                setattr(self.trainer, user_data, app_data)

            #---------------------Trainer---------------------
            with dpg.collapsing_header(label="Trainer", default_open=True):
                dpg.add_input_text(
                    label="prompt",
                    default_value=self.trainer.config.prompt,
                    callback=callback_setattr_trainer,
                    user_data="prompt",
                )
                dpg.add_input_text(
                    label="negative",
                    default_value=self.trainer.config.negative_prompt,
                    callback=callback_setattr_trainer,
                    user_data="negative_prompt",
                )
                with dpg.group(horizontal=True):
                    dpg.add_text("Train: ")

                    def callback_train(sender, app_data):
                        if self.trainer.training:
                            self.trainer.training = False
                            dpg.configure_item("_button_train", label="start")
                        else:
                            self.trainer.training = True
                            dpg.configure_item("_button_train", label="stop")
                    label_training = "stop" if self.trainer.training else "start"
                    dpg.add_button(
                        label=label_training, tag="_button_train", callback=callback_train
                    )
                    dpg.bind_item_theme("_button_train", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="_log_train_time")
                    dpg.add_text("", tag="_log_train_log")

                # add save button
                with dpg.group(horizontal=True):
                    dpg.add_text("Save: ")

                    def callback_save(sender, app_data, user_data):
                        ply_save_path = os.path.join(self.trainer.config.save_path, f"user_iter_{self.trainer.curr_iter}_{self.trainer.config.prompt}.ply")
                        self.trainer.gaussians.save_ply(ply_save_path)
                        print(f"Ply saved to {ply_save_path}!")

                    dpg.add_button(
                        label="model",
                        tag="_button_save_model",
                        callback=callback_save,
                        user_data='model',
                    )

            #---------------------Rendering---------------------
            with dpg.collapsing_header(label="Rendering", default_open=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(
                    ("image", "depth", "alpha", "alpha control points"),
                    label="mode",
                    default_value=self.mode,
                    callback=callback_change_mode,
                )

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = np.deg2rad(app_data)
                    self.need_update = True

                dpg.add_slider_int(
                    label="FoV (vertical)",
                    min_value=1,
                    max_value=120,
                    format="%d deg",
                    default_value=np.rad2deg(self.cam.fovy),
                    callback=callback_set_fovy,
                )

                # radius slider
                def callback_set_radius(sender, app_data):
                    print(app_data)
                    self.cam.radius = app_data
                    self.need_update = True

                dpg.add_slider_float(
                    label="radius",
                    min_value=0.5,
                    max_value=10,
                    format="%.2f m",
                    default_value=self.cam.radius,
                    callback=callback_set_radius,
                )

                # elevation slider
                def callback_set_elevation(sender, app_data):
                    self.curr_elevation = app_data
                    self.cam.rot = R.from_matrix(orbit_camera(elevation=self.curr_elevation, azimuth=self.curr_azimuth, 
                                                radius=self.render_c.radius)[0:3, 0:3])
                    self.need_update = True

                dpg.add_slider_int(
                    label="Elevation",
                    min_value=-90,
                    max_value=90,
                    format="%d deg",
                    default_value=np.rad2deg(0),
                    callback=callback_set_elevation,
                )

                # azimuth slider
                def callback_set_azimuth(sender, app_data):
                    self.curr_azimuth = app_data
                    self.cam.rot = R.from_matrix(orbit_camera(elevation=self.curr_elevation, azimuth=self.curr_azimuth, 
                                                radius=self.render_c.radius)[0:3, 0:3])
                    self.need_update = True

                dpg.add_slider_int(
                    label="Azimuth",
                    min_value=-180,
                    max_value=180,
                    format="%d deg",
                    default_value=np.rad2deg(0),
                    callback=callback_set_azimuth
                )

                def callback_set_gaussain_scale(sender, app_data):
                    self.gaussain_scale_factor = app_data
                    self.need_update = True

                dpg.add_slider_float(
                    label="gaussain scale",
                    min_value=0,
                    max_value=1,
                    format="%.2f",
                    default_value=self.gaussain_scale_factor,
                    callback=callback_set_gaussain_scale,
                )

                def callback_set_bg_color(sender, app_data):
                    self.bg_gui_color = app_data
                    self.need_update = True

                dpg.add_slider_float(
                    label="background color",
                    min_value=0,
                    max_value=1,
                    format="%.2f",
                    default_value=self.bg_gui_color,
                    callback=callback_set_bg_color,
                )


            #---------------------Save Image // Video---------------------
            with dpg.collapsing_header(label="Save Media", default_open=True):

                dpg.add_input_text(
                    label="save path",
                    default_value=self.save_path,
                    callback=callback_setattr_gui,
                    user_data="save_path",
                )
                
                with dpg.group(horizontal=True):
                    dpg.add_text("Save video during update only: ")

                    def callback_save_on_update_toggle(sender, app_data):
                        if self.save_on_update_only:
                            self.save_on_update_only = False
                            dpg.configure_item("_button_save_on_update_toggle", label="off")
                        else:
                            self.save_on_update_only = True
                            dpg.configure_item("_button_save_on_update_toggle", label="on")
                    label_record = "off" if not self.save_on_update_only else "on"
                    dpg.add_button(
                        label=label_record, tag="_button_save_on_update_toggle", callback=callback_save_on_update_toggle
                    )
                    dpg.bind_item_theme("_button_save_on_update_toggle", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("Record Video: ")

                    def callback_record(sender, app_data):
                        if self.currently_recording:
                            self.currently_recording = False
                            save_video_path = os.path.join(self.save_path, f"video_{self.video_num}.mp4")
                            self.video_num += 1
                            print("Saving video...")
                            iio.mimwrite(save_video_path, self.video_buffer, fps=30)
                            self.video_buffer = []
                            print(f"Video saved to {save_video_path}!")
                            dpg.configure_item("_button_record", label="start")
                        else:
                            self.currently_recording = True
                            dpg.configure_item("_button_record", label="stop")
                    label_record = "stop" if self.currently_recording else "start"
                    dpg.add_button(
                        label=label_record, tag="_button_record", callback=callback_record
                    )
                    dpg.bind_item_theme("_button_record", theme_button)

                # with dpg.group(horizontal=True):
                #     dpg.add_text("", tag="_log_train_time")
                #     dpg.add_text("", tag="_log_train_log")

                # add save image button
                with dpg.group(horizontal=True):
                    dpg.add_text("Take Image: ")

                    def callback_save_img(sender, app_data, user_data):
                        img_save_path = os.path.join(self.save_path, f"img_{self.img_num}.png")
                        # convert to uint8
                        save_img = (self.gui_image*255).astype(np.uint8)
                        print("Saving image...")
                        iio.imsave(img_save_path, save_img)
                        self.img_num += 1
                        print(f"Image saved to {img_save_path}!")

                    dpg.add_button(
                        label="image",
                        tag="_button_save_image",
                        callback=callback_save_img,
                        user_data='image',
                    )

        ### register camera handler
        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

        def callback_set_mouse_loc(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            # just the pixel coordinate in image
            self.mouse_loc = np.array(app_data)

        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

    def render(self):
        assert self.display_c.enable
        if(dpg.is_dearpygui_running()):
            self.test_step()
            dpg.render_dearpygui_frame()
    
    @torch.no_grad()
    def test_step(self):

        # handle recording
        if self.currently_recording:
            if self.save_on_update_only:
                if self.need_update:
                    self.video_buffer.append((self.gui_image*255).astype(np.uint8))
                    print(f"Recording frame {len(self.video_buffer)}")
            else:
                self.video_buffer.append((self.gui_image*255).astype(np.uint8))
                print(f"Recording frame {len(self.video_buffer)}")

        # ignore if no need to update
        if not self.need_update:
            return

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        # should update image
        if self.need_update:
            # render image

            cur_cam = MiniCam(
                self.cam.pose,
                self.WIDTH,
                self.HEIGHT,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
                device=self.trainer.config.device
            )

            if(self.mode == "alpha control points"):
                copy_scales = self.trainer.gaussians._scaling.clone()
                copy_feature_dc = self.trainer.gaussians._features_dc.clone()
                copy_opacity = self.trainer.gaussians._opacity.clone()

                self.trainer.gaussians._opacity += 100
                print(torch.min(self.trainer.gaussians._opacity))


            #bg
            bg_color = self.bg_gui_color*torch.tensor([1,1,1], 
                                                     dtype=torch.float32, 
                                                     device=self.trainer.config.device)

            out = self.trainer.renderer.render(self.trainer.gaussians, cur_cam, self.gaussain_scale_factor, bg_color=bg_color)

            if self.mode == "alpha control points": # The gaussians scales are not being copied back correctly
                gui_image = out['image']
                self.trainer.gaussians._scaling = copy_scales
                self.trainer.gaussians._features_dc = copy_feature_dc
                self.trainer.gaussians._opacity = copy_opacity
            else:
                gui_image = out[self.mode]  # [3, H, W]

            if self.mode in ['depth', 'alpha']:
                gui_image = gui_image.repeat(3, 1, 1)
                if self.mode == 'depth':
                    gui_image = (gui_image - gui_image.min()) / (gui_image.max() - gui_image.min() + 1e-20)

            gui_image = F.interpolate(
                gui_image.unsqueeze(0),
                size=(self.HEIGHT, self.WIDTH),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            self.gui_image = (
                gui_image.permute(1, 2, 0)
                .contiguous()
                .clamp(0, 1)
                .contiguous()
                .detach()
                .cpu()
                .numpy()
            )

            self.need_update = False

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        if self.display_c.enable:
            dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS)")
            dpg.set_value(
                "_texture", self.gui_image
            )  # buffer must be contiguous, else seg fault!
            dpg.set_value("_log_train_time", f"{self.trainer.gui.train_time:.4f}ms")
            dpg.set_value(
                "_log_train_log",
                f"step = {self.trainer.curr_iter: 5d} (+{self.trainer.max_iter: 2d}) loss = {self.trainer.gui.loss:.4f}",
            )