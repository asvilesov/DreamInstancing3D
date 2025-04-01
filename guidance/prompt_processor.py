from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline
import gc
import os
from dataclasses import dataclass, field

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BertForMaskedLM, CLIPTextModel

from typing import *
from jaxtyping import *
from torch import Tensor
from rich.console import Console

console = Console()


def hash_prompt(model: str, prompt: str) -> str:
    import hashlib

    identifier = f"{model}-{prompt}"
    return hashlib.md5(identifier.encode()).hexdigest()


@dataclass
class DirectionConfig:
    name: str
    prompt: Callable[[str], str]
    negative_prompt: Callable[[str], str]
    condition: Callable[
        [Float[Tensor, "B"], Float[Tensor, "B"], Float[Tensor, "B"]],
        Float[Tensor, "B"],
    ]


@dataclass
class PromptEmbedding:
    text_embedding: Float[Tensor, "B D"]
    uncond_text_embedding: Float[Tensor, "B D"]
    text_embedding_view_dependent: Float[Tensor, "B D"]
    uncond_text_embedding_view_dependent: Float[Tensor, "B D"]
    directions: List[DirectionConfig]
    direction2idx: Dict[str, int]
    use_perp_negative: bool = False
    debug: bool = False

    # perp neg interpolation params, adapted from threestudio (https://github1s.com/threestudio-project/threestudio/blob/HEAD/threestudio/models/prompt_processors/base.py)
    perp_neg_f_sb: Tuple[float, float, float] = (1, 0.5, -0.606)
    perp_neg_f_fsb: Tuple[float, float, float] = (1, 0.5, +0.967)
    perp_neg_f_fs: Tuple[float, float, float] = (
        4,
        0.5,
        -2.426,
    )  # f_fs(1) = 0, a, b > 0
    perp_neg_f_sf: Tuple[float, float, float] = (4, 0.5, -2.426)

    def get_text_embedding(
        self,
        elevation,
        azimuth,
        camera_distances,
        use_view_dependent_prompt=False,
    ):
        bs = elevation.shape[0]

        if use_view_dependent_prompt:
            direction_idx = torch.zeros_like(elevation, dtype=torch.long)
            for d in self.directions:
                direction_idx[
                    d.condition(elevation, azimuth, camera_distances)
                ] = self.direction2idx[d.name]

            # Get text embeddings
            text_emb = self.text_embedding_view_dependent[direction_idx]
            uncond_text_emb = self.uncond_text_embedding_view_dependent[direction_idx]
        else:
            text_emb = self.text_embedding.expand(bs, -1, -1)
            uncond_text_emb = self.uncond_text_embedding.expand(bs, -1, -1)

        if self.debug:
            return {
                "direction_idx": direction_idx,
                "text_embedding": torch.cat([text_emb, uncond_text_emb], dim=0),
            }
        # mind the order, corresponding to `chunck` in stable_diffusion.py fn `compute_grad_sds`
        return torch.cat([uncond_text_emb, text_emb], dim=0)


def shift_azimuth_deg(azimuth: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    # shift azimuth angle (in degrees), to [-180, 180]
    return (azimuth + 180) % 360 - 180


class BasePromptProcessor(nn.Module):
    def __init__(self, cfg, guidance_model=None):
        super().__init__()
        self.cfg = cfg
        self.device = self.cfg.device
        self.pretrained_model_name_or_path = cfg.pretrained_model_name_or_path
        self.prompt = cfg.prompt
        self.negative_prompt = cfg.negative_prompt
        self.guidance_model = guidance_model

        self.use_cache = cfg.use_cache
        if cfg.use_cache:
            self.cache_dir = "./.cache/text_prompt_embeddings"
            os.makedirs(self.cache_dir, exist_ok=True)

        # prepare directions, adapted from threestudio
        self.directions: List[DirectionConfig]
        if cfg.view_dependent_prompt_front:
            self.directions = [
                DirectionConfig(
                    "side",
                    lambda s: f"side view of {s}",
                    lambda s: s,
                    lambda ele, azi, dis: torch.ones_like(ele, dtype=torch.bool),
                ),
                DirectionConfig(
                    "front",
                    lambda s: f"front view of {s}",
                    lambda s: s,
                    lambda ele, azi, dis: (
                        shift_azimuth_deg(azi) > -self.cfg.front_threshold
                    )
                    & (shift_azimuth_deg(azi) < self.cfg.front_threshold),
                ),
                DirectionConfig(
                    "back",
                    lambda s: f"backside view of {s}",
                    lambda s: s,
                    lambda ele, azi, dis: (
                        shift_azimuth_deg(azi) > 180 - self.cfg.back_threshold
                    )
                    | (shift_azimuth_deg(azi) < -180 + self.cfg.back_threshold),
                ),
                DirectionConfig(
                    "overhead",
                    lambda s: f"overhead view of {s}",
                    lambda s: s,
                    lambda ele, azi, dis: ele > self.cfg.overhead_threshold,
                ),
            ]
        else:
            self.directions = [
                DirectionConfig(
                    "side",
                    lambda s: f"{s}, side view",
                    lambda s: s,
                    lambda ele, azi, dis: torch.ones_like(ele, dtype=torch.bool),
                ),
                DirectionConfig(
                    "front",
                    lambda s: f"{s}, front view",
                    lambda s: s,
                    lambda ele, azi, dis: (
                        shift_azimuth_deg(azi) > -self.cfg.front_threshold
                    )
                    & (shift_azimuth_deg(azi) < self.cfg.front_threshold),
                ),
                DirectionConfig(
                    "back",
                    lambda s: f"{s}, back view",
                    lambda s: s,
                    lambda ele, azi, dis: (
                        shift_azimuth_deg(azi) > 180 - self.cfg.back_threshold
                    )
                    | (shift_azimuth_deg(azi) < -180 + self.cfg.back_threshold),
                ),
                DirectionConfig(
                    "overhead",
                    lambda s: f"{s}, overhead view",
                    lambda s: s,
                    lambda ele, azi, dis: ele > self.cfg.overhead_threshold,
                ),
            ]

        self.direction2idx = {d.name: i for i, d in enumerate(self.directions)}

        if cfg.use_prompt_debiasing:
            assert (
                self.cfg.prompt_side is None
                and self.cfg.prompt_back is None
                and self.cfg.prompt_overhead is None
            ), "Do not manually assign prompt_side, prompt_back or prompt_overhead when using prompt debiasing"
            prompts = self.get_debiased_prompt(self.prompt)
            self.prompts_view_dependent = [
                d.prompt(prompt) for d, prompt in zip(self.directions, prompts)
            ]
        else:
            self.prompts_view_dependent = [
                d.prompt(self.cfg.get(f"prompt_{d.name}", None) or self.prompt)  # type: ignore
                for d in self.directions
            ]

        prompts_vd_display = "\n".join(
            [
                f"[{d.name}]:[{prompt}]"
                for prompt, d in zip(self.prompts_view_dependent, self.directions)
            ]
        )

        self.negative_prompts_view_dependent = [
            d.negative_prompt(self.negative_prompt) for d in self.directions
        ]

        self.prepare_prompts()
        self.load_prompt_embeddings()

    def load_from_cache(self, prompt):
        cache_path = os.path.join(
            self.cache_dir,
            f"{hash_prompt(self.cfg.pretrained_model_name_or_path, prompt)}.pt",
        )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Text embedding file {cache_path} for model {self.cfg.pretrained_model_name_or_path} and prompt [{prompt}] not found."
            )

        return torch.load(cache_path, map_location=self.device)

    def prepare_text_encoder(self):
        raise NotImplementedError

    def encode_prompts(self, prompts):
        raise NotImplementedError

    def load_prompt_embeddings(self):
        self.text_embedding = self.load_from_cache(self.prompt)[None, ...]
        self.uncond_text_embedding = self.load_from_cache(self.negative_prompt)[
            None, ...
        ]
        self.text_embedding_view_dependent = torch.stack(
            [self.load_from_cache(prompt) for prompt in self.prompts_view_dependent],
            dim=0,
        )
        self.uncond_text_embedding_view_dependent = torch.stack(
            [
                self.load_from_cache(prompt)
                for prompt in self.negative_prompts_view_dependent
            ],
            dim=0,
        )

    def prepare_prompts(self):
        # NOTE: self.guidance_model is None means initialize the text encoder and tokenizer separetely from unet and vae
        self.prepare_text_encoder(self.guidance_model)
        prompts = (
            [
                self.prompt,
                self.negative_prompt,
            ]
            + self.prompts_view_dependent
            + self.negative_prompts_view_dependent
        )
        prompts_to_process = []
        for prompt in prompts:
            if self.use_cache:
                cache_path = os.path.join(
                    self.cache_dir,
                    f"{hash_prompt(self.cfg.pretrained_model_name_or_path, prompt)}.pt",
                )
                if os.path.exists(cache_path):
                    continue
            prompts_to_process.append(prompt)

        if len(prompts_to_process) > 0:
            prompt_embeddings = self.encode_prompts(prompts_to_process)

            for prompt, embedding in zip(prompts_to_process, prompt_embeddings):
                if self.use_cache:
                    cache_path = os.path.join(
                        self.cache_dir,
                        f"{hash_prompt(self.cfg.pretrained_model_name_or_path, prompt)}.pt",
                    )
                    torch.save(embedding, cache_path)

    def get_prompt_embedding(self) -> PromptEmbedding:
        return PromptEmbedding(
            text_embedding=self.text_embedding,
            uncond_text_embedding=self.uncond_text_embedding,
            text_embedding_view_dependent=self.text_embedding_view_dependent,
            uncond_text_embedding_view_dependent=self.uncond_text_embedding_view_dependent,
            directions=self.directions,
            direction2idx=self.direction2idx,
            use_perp_negative=self.cfg.use_perp_negative,
            debug=self.cfg.debug,
        )

    def get_debiased_prompt(self, prompt):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.pretrained_model_name_or_path_prompt_debiasing
        )
        model = BertForMaskedLM.from_pretrained(
            self.cfg.pretrained_model_name_or_path_prompt_debiasing
        )

        views = [d.name for d in self.directions]
        view_ids = tokenizer(" ".join(views), return_tensors="pt").input_ids[0]
        view_ids = view_ids[1:5]

        def modulate(prompt):
            prompt_vd = f"This image is depicting a [MASK] view of {prompt}"
            tokens = tokenizer(
                prompt_vd,
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            mask_idx = torch.where(tokens.input_ids == tokenizer.mask_token_id)[1]

            logits = model(**tokens).logits
            logits = F.softmax(logits[0, mask_idx], dim=-1)
            logits = logits[0, view_ids]
            probes = logits / logits.sum()
            return probes

        prompts = [prompt.split(" ") for _ in range(4)]
        full_probe = modulate(prompt)
        n_words = len(prompt.split(" "))
        prompt_debiasing_mask_ids = (
            self.cfg.prompt_debiasing_mask_ids
            if self.cfg.prompt_debiasing_mask_ids is not None
            else list(range(n_words))
        )
        words_to_debias = [prompt.split(" ")[idx] for idx in prompt_debiasing_mask_ids]
        console.print(f"Words that can potentially be removed: {words_to_debias}")
        for idx in prompt_debiasing_mask_ids:
            words = prompt.split(" ")
            prompt_ = " ".join(words[:idx] + words[(idx + 1) :])
            part_probe = modulate(prompt_)

            pmi = full_probe / torch.lerp(part_probe, full_probe, 0.5)
            for i in range(pmi.shape[0]):
                if pmi[i].item() < 0.95:
                    prompts[i][idx] = ""

        debiased_prompts = [" ".join([word for word in p if word]) for p in prompts]
        for d, debiased_prompt in zip(views, debiased_prompts):
            console.print(f"Debiased prompt of the {d} view is [{debiased_prompt}]")

        del tokenizer, model
        self.cleanup()
        gc.collect()
        torch.cuda.empty_cache()

        return debiased_prompts

    def update(self, step):
        raise NotImplementedError("Update not implemented")

    def forward(self):
        return self.get_prompt_embedding()

    def cleanup(self):
        del self.tokenizer
        del self.text_encoder

class StableDiffusionPromptProcessor(BasePromptProcessor):
    def prepare_text_encoder(self, guidance_model=None):
        if guidance_model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="tokenizer",
                cache_dir="./.cache",
            )
            self.text_encoder = CLIPTextModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="text_encoder",
                device_map="auto",
                cache_dir="./.cache",
            )
        else:
            self.tokenizer = guidance_model.tokenizer
            self.text_encoder = guidance_model.text_encoder

    def encode_prompts(self, prompts):
        with torch.no_grad():
            print(prompts)
            tokens = self.tokenizer(
                prompts,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).to(self.device)
            text_embeddings = self.text_encoder(tokens.input_ids)[0]

        return text_embeddings

    def update(self, step):
        pass