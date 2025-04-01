from pathlib import Path
import numpy as np
import torch

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config


def point_e_generate_pcd_from_text(text, num_points=4096, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print("Using Point-E on device:", device)

    print("creating base model...")
    base_name = "base40M-textvec"
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

    print("creating upsample model...")
    upsampler_model = model_from_config(MODEL_CONFIGS["upsample"], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS["upsample"])

    print("downloading base checkpoint...")
    base_model.load_state_dict(load_checkpoint(base_name, device))

    print("downloading upsampler checkpoint...")
    upsampler_model.load_state_dict(load_checkpoint("upsample", device))
    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, num_points - 1024],
        aux_channels=["R", "G", "B"],
        guidance_scale=[3.0, 0.0],
        model_kwargs_key_filter=("texts", ""),  # Do not condition the upsampler at all
    )
    # Set a prompt to condition on.
    prompt = text

    # Produce a sample from the model.
    samples = None
    for x in sampler.sample_batch_progressive(
        batch_size=1, model_kwargs=dict(texts=[prompt])
    ):
        samples = x
    pc = sampler.output_to_point_clouds(samples)[0]

    xyz = torch.from_numpy(pc.coords).to(torch.float32)
    rgb = torch.from_numpy(
        np.stack([pc.channels[c] for c in ["R", "G", "B"]], axis=-1)
    ).to(torch.float32)

    pc = torch.cat([xyz, rgb], dim=-1)

    return pc


def point_e_intialize(cfg):
    prompt = cfg.pointe_prompt
    pcd = point_e_generate_pcd_from_text(prompt, 4096, cfg.device)
    pcd = pcd
    pcd = pcd[::]
    xyz, rgb = pcd[:, :3], pcd[:, 3:]

    xyz -= (xyz.max(dim=0, keepdim=True).values+xyz.min(dim=0, keepdim=True).values)/2

    xyz = xyz / (xyz.norm(dim=-1).max() + 1e-5)
    xyz = xyz * 0.8

    x, y, z = xyz.chunk(3, dim=-1)
    xyz = torch.cat([x,z,-y], dim=-1)

    rgb = torch.rand_like(rgb)
    print("point-e rgb", rgb)

    z_scale = 1
    xyz[..., 2] *= z_scale

    initial_values = {}
    initial_values["mean"] = xyz
    initial_values["color"] = rgb
    initial_values["svec"] = get_svec(len(xyz))
    initial_values["qvec"] = get_qvec(len(xyz))
    initial_values["alpha"] = get_alpha(len(xyz))

    return initial_values


def get_qvec(num_points):
    qvec = torch.zeros(num_points, 4, dtype=torch.float32)
    qvec[:, 0] = 1.0
    return qvec


def get_svec(num_points):
    svec = torch.ones(num_points, 3, dtype=torch.float32) * 0.02
    return svec


def get_alpha(num_points):
    alpha = torch.ones(num_points, dtype=torch.float32) * 0.8
    return alpha
