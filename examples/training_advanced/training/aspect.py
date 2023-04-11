#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import hashlib
import itertools
import logging
import math
import os
import warnings
from pathlib import Path
from typing import Optional
import random

import accelerate
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import kornia


import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

from packaging import version

from torch.distributions import Gamma

from diffusers import StableDiffusionPipeline


prompts_with_size = [
    {
        "prompt": "Perfect Breaking wave shapes; A dramatic breaking wave; tilt-shift zoom; film; super image; AR: 4:3",
        "negativePrompt": "bad wave, bad wave shape, bad wave; bad image; bad crop; bad image; zoom crop; random crop",
        "size": (1024, 768),
    },
    {
        "prompt": "a full body photo of 25 y.o brown tan skinned American woman with glasses, Short Curly hair, wearing gypsy style clothes, modestly clothed, fully clothed, background is in traditional Turkish house, high detailed skin, realistic sunlight lighting, thick thigh, wide hips, visible face pores, AR: 3:4",
        "negativePrompt": "deformed iris, deformed pupils, semi-realistic, CGI, 3d, render, sketch, cartoon, drawing, anime, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, erotic, sexy, nudity, half-naked, half naked, camera flash, oversaturated, hard lighting, diffuser lighting",
        "size": (768, 1024)
    },
    {
        "prompt": "A dramatic wave breaking; super image; super wave AR: 1024x768",
        "negativePrompt": "lowres, bad image, bad corp, zoom crop, error, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, ugly",
        "size": (1024, 768),
    },
    {
        "prompt": "Storied Sorrows, A women sits beneath a soft light, sobbing in a chair, dimly lit hotel room; film; super image; good crop AR: 4:3",
        "negativePrompt": "bad image, bad crop, zoom crop, random crop, bad lighting, bad film, bad angle, misformed person, bad person, bad people",
        "size": (1024, 768),
    },
    {
        "prompt": "a cute bird; super image; good crop, AR: 1:1",
        "size": (768, 768),
        "negativePrompt": "bad image; bad crop"
    },
    {
        "prompt": "a smiling woman; super image, super eyes; AR: 4:3",
        "size": (768, 512),
        "negativePrompt": "bad image, bad crop, bad zoom, bad eyes, bad face"
    },
    {
        "prompt": "Cyber eye; a warrior princess; Jeweled Crown; super image; dramatic fisheye; cool crop; unique good crop; AR: 3:4",
        "size": (512, 768),
        "negativePrompt": "bad image, zoom crop, bad crop"
    },
    {
        "prompt": "A cat that looks like a bear; super animal; super image; realistic; good crop AR: 4:5",
        "size": (768, 884),
        "negativePrompt": "bad image, zoom crop, bad crop; lowres, low quality, bad angle, misformed animal, misformed person"
    },
    {
        "prompt": "Barren Scowls; film; super image; super camera angle; good crop, AR: 4:3",
        "size": (1024, 768),
        "negativePrompt": "bad image, bad crop; bad film, bad camera angle; misformed faces, misaligned eyes, malformed faces, malformed limbs, misformed animal"
    },
    {
        "prompt": "A leopard fights a man; film; super realistic; super image; dratmatic shot; good crop; AR: 4:3",
        "size": (1024, 768),
        "negativePrompt": "bad image, bad painting, zoom crop, bad crop, lowres; low quality, too dark, too bright; ugly, misformed person; misformed animal",
    },
    {
        "prompt": "Halls of Space, scifi, digital painting; super painting; super image; AR: 4:3, painterly",
        "negativePrompt": "bad painting, bad image; bad crop; zoom crop; random crop",
        "size": (1024, 768),
    },
    {
        "prompt": "The friend inside your mind; Anthony bourdain and Obama enjoying dinner at a diner, AR: 3:4",
        "negativePrompt": "bad people, malformed people, misfigured persons",
        "size": (1024, 884),
    },
    {
        "prompt": "portrait of obama; super image; super photo, good crop AR: 3:4",
        "size": (702, 884),
        "negativePrompt": "bad image, bad crop, zoom crop, bad person, bad eyes, misaligned eyes, malformed person",
    },
    {
            "prompt": "A mystical dragon walrus super animal hybrid, fantasy; super image; super film; super photo; realistic; painterly ; good crop AR: 4:3",
        "negativePrompt": "cropped image, misfigured animal, bad painting, misfigured person, ugly limbs, zoom crop; bad crop; center crop",
        "size": (1024, 768)
    },
    {
        "prompt": "He Roars within his kingdom; A mystical dragon rpg; super image; realistic; super realistic, AR: 4:3",
        "negativePrompt": "cropped image, bad animal, bad painting",
        "size": (1024, 768)
    },
    {
        "prompt": "Bizzare Magical Landscape; fantasy; rpg; high aesthetic; super image; AR: 3:4",
        "negativePrompt": "bad image, bad crop",
        "size": (768, 1024)
    },
    {
        "prompt": "Cyberpunk Chipmunk; super image; super animal; AR: 2:3",
        "negativePrompt": "bad image, bad crop",
        "size": (512, 768)
    },
    {
        "prompt": "Cyberpunk Chipmunk; super image; super animal; AR: 1:1",
        "negativePrompt": "bad image, bad crop",
        "size": (256, 256)
    },
    {
        "prompt": "Cyberpunk Hedgehog; super image; super animal, good crop AR: 3:4",
        "negativePrompt": "bad image, bad crop",
        "size": (384 , 512)
    },

]
tags = [
    "nerd product",
    "high fashion person",
    "high fashion product",
    "high architecture",
    "modern",
    "minimal"
]

for tag in tags:
    prompts_with_size.append(
        {
            "prompt": f"{tag}; super image, AR: 1:1",
            "negativePrompt": "bad image, bad crop",
            "size": (512, 512)
        }
    )



def get_polynomial_decay_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    lr_end=1e-7,
    power=1.0,
    last_epoch=-1,
):
    """
        optimizer ([`~torch.optim.Optimizer`]):
        num_warmup_steps (`int`):
        num_training_steps (`int`):
        lr_end (`float`, *optional*, defaults to 1e-7):
        power (`float`, *optional*, defaults to 1.0):
            Power factor.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Note: *power* defaults to 1.0 as in the fairseq implementation, which in turn is based on the original BERT
    implementation at
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_init = optimizer.defaults["lr"]
    if not (lr_init > lr_end):
        raise ValueError(
            f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})"
        )

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining**power + lr_end
            lr = decay / lr_init  # as LambdaLR multiplies by lr_init

            return lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


logger = get_logger(__name__)

import math

from dctorch import functional as DF
import torch


def sqrtm(x):
    vals, vecs = torch.linalg.eigh(x)
    return vecs * vals.sqrt() @ vecs.T


def colored_noise(
    shape, power=2.0, mean=None, color=None, device="cpu", dtype=torch.float32
):
    mean = torch.zeros([shape[-3]]) if mean is None else mean
    color = torch.eye(shape[-3]) if color is None else color
    f_h = math.pi * torch.arange(shape[-2], device=device, dtype=dtype) / shape[-2]
    f_w = math.pi * torch.arange(shape[-1], device=device, dtype=dtype) / shape[-1]
    freqs_sq = f_h[:, None] ** 2 + f_w[None, :] ** 2
    freqs_sq[..., 0, 0] = freqs_sq[..., 0, 1]
    spd = freqs_sq ** -(power / 2)
    spd /= spd.mean()
    noise = torch.randn(shape, device=device, dtype=dtype)
    noise = torch.einsum("...chw,cd->...dhw", noise, color.to(device, dtype))
    noise = DF.idct2(noise * spd.sqrt())
    noise = noise + mean.to(device, dtype)[..., None, None]
    return noise


def random_gamma(shape, alpha, beta=1.0):
    alpha = torch.ones(shape) * torch.tensor(alpha)
    beta = torch.ones(shape) * torch.tensor(beta)
    gamma_distribution = Gamma(alpha, beta)

    return gamma_distribution.sample()


def make_pink_noise(latents):
    power = 2.4477
    mean = torch.tensor([0.4811, 0.4575, 0.4078], device=latents.device)
    cov = torch.tensor(
        [[0.0802, 0.0700, 0.0631], [0.0700, 0.0763, 0.0721], [0.0631, 0.0721, 0.0839]],
        device=latents.device,
    )

    x = colored_noise(latents.shape, power, device=latents.device)
    return x


# Takes the ideas from offset noise(1) and adds uses two gamma distributions to
# alter the noise schedule adding different amounts of offset noise at different frequencies
# [1] https://www.crosslabs.org/blog/diffusion-with-offset-noise
def gamma_offset_noise(latents, discount=1.0):
    device = latents.device
    g2 = torch.clamp(
        random_gamma((latents.shape[0], latents.shape[1], 1, 1), alpha=0.5, beta=10),
        0,
        1,
    ).to(device)

    # g1 is the random freq
    # g2 is the proportion to shift the noise
    noise = torch.randn_like(latents, device=latents.device)
    thick_noise = torch.randn(
        latents.shape[0], latents.shape[1], 1, 1,
        device=device
    )

    return noise + discount * g2 * thick_noise

def create_checkerboard_pixels(pixel_w, pixel_h, block_size):
    # Compute the checkerboard grid size and block size in pixels
    grid_w = (pixel_w + block_size // 2) // block_size
    grid_h = (pixel_h + block_size // 2) // block_size
    pixel_block_size = max(1, block_size // 2)

    # Compute the dimensions of the output grid and checkerboard tiles
    output_h, output_w = grid_h * block_size, grid_w * block_size
    tile_h, tile_w = 2 * pixel_block_size, 2 * pixel_block_size

    # Create a checkerboard pattern using small tiles
    pattern = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    tile = pattern.view(1, 2, 2).repeat(1, pixel_block_size, pixel_block_size)
    tile = F.interpolate(
        tile.unsqueeze(0), size=(tile_h, tile_w), mode="bilinear"
    ).squeeze()

    # Repeat the pattern to create the checkerboard grid
    checkerboard = tile.repeat(grid_h, grid_w)
    checkerboard = checkerboard[:pixel_h, :pixel_w]

    # Apply antialiasing to the checkerboard grid
    checkerboard = F.interpolate(
        checkerboard.unsqueeze(0).unsqueeze(0),
        size=(output_h, output_w),
        mode="bilinear",
    ).squeeze()

    # Apply a half cell offset
    row_offset = block_size // 2
    col_offset = block_size // 2
    checkerboard = torch.roll(
        checkerboard, shifts=(row_offset, col_offset), dims=(0, 1)
    )

    return checkerboard.unsqueeze(0)


def rotate_3d_and_project(checkerboard, angle, padding_mode="zeros"):
    height, width = checkerboard.shape[-2:]
    device = checkerboard.device

    # Create a 3x3 rotation matrix around the Z-axis
    angle_rad = torch.tensor(
        angle * (3.14159265 / 180.0), dtype=torch.float32, device=device
    )
    cos_angle, sin_angle = torch.cos(angle_rad), torch.sin(angle_rad)
    rotation_matrix = torch.tensor(
        [[cos_angle, -sin_angle, 0], [sin_angle, cos_angle, 0], [0, 0, 1]],
        device=device,
    ).unsqueeze(0)

    # Apply the rotation
    rotated_2d = kornia.geometry.transform.warp_perspective(
        checkerboard.unsqueeze(0),
        rotation_matrix,
        (height, width),
        padding_mode="reflection",
    )

    # Anti-alias and pad the image
    padding = torch.nn.ReflectionPad2d(1)
    if padding_mode == "zeros":
        padding = torch.nn.ZeroPad2d(1)

    rotated_2d_padded = padding(rotated_2d)
    rotated_2d_aa = kornia.filters.box_blur(rotated_2d_padded, (3, 3))

    return rotated_2d_aa.squeeze(0)


def random_high_freq(latents):
    B, C, H, W = latents.shape
    output = torch.zeros_like(latents)

    for i in range(B):
        for j in range(C):
            # Create checkerboard
            random_cell_size = random.randint(1, 5)

            checkerboard = create_checkerboard_pixels(H, W, random_cell_size)

            # Apply random rotation
            random_angle = random.uniform(0, 360)
            rotated_checkerboard = rotate_3d_and_project(checkerboard, random_angle)

            # Resize the rotated checkerboard to match the size of latents
            resized_checkerboard = F.interpolate(
                rotated_checkerboard.unsqueeze(0), size=(H, W), mode="bilinear"
            ).squeeze()

            # Assign the rotated checkerboard to the output tensor
            output[i, j] = resized_checkerboard.squeeze()

    return output


DISCOUNT_HIGH = 0.08
DISCOUNT_LOW = 0.01

LOW_FLIP = 0.2
HIGH_FLIP = 0.05


def high_noise(latents):
    return torch.randn_like(latents) + DISCOUNT_HIGH * random_high_freq(latents)


# low noise
def offset_noise(latents):
    return torch.randn_like(latents) + DISCOUNT_LOW * torch.randn(
        latents.shape[0], latents.shape[1], 1, 1, device=latents.device
    )


def hi_lo_noise(latents):
    flip = random.random()
    if flip < LOW_FLIP:
        return offset_noise(latents)
    elif flip < LOW_FLIP + HIGH_FLIP:
        return high_noise(latents)
    else:
        return torch.randn_like(latents)


def gamma_pink_noise(latents):
    device = latents.device
    g2 = torch.clamp(
        random_gamma((latents.shape[0], latents.shape[1], 1, 1), alpha=0.5, beta=10),
        0,
        1,
    ).to(device)

    # g1 is the random freq
    # g2 is the proportion to shift the noise
    noise = torch.randn_like(latents, device=latents.device)
    return torch.randn_like(latents) + g2 * make_pink_noise(latents)


def pyramid_noise_like(x, discount=0.9):

    b, c, w, h = x.shape
    u = torch.nn.Upsample(size=(w, h), mode="bilinear")
    noise = torch.randn_like(x)

    for i in range(10):
        r = random.random() * 2 + 2  # Rather than always going 2x,
        w, h = max(1, int(w / (r**i))), max(1, int(h / (r**i)))
        noise += u(torch.randn(b, c, w, h).to(x)) * discount**i
        if w == 1 or h == 1:
            break  # Lowest resolution is 1x1

    return noise / noise.std()  # Scaled back to roughly unit variance


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import (
            RobertaSeriesModelWithTransformation,
        )

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=False,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--use_filename_as_label",
        action="store_true",
        help="Uses the filename as the image labels instead of the instance_prompt, useful for regularization when training for styles with wide image variance",
    )
    parser.add_argument(
        "--use_txt_as_label",
        action="store_true",
        help="Uses the filename.txt file's content as the image labels instead of the instance_prompt, useful for regularization when training for styles with wide image variance",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument("--super_image_batch_size", type=int, default=4),
    parser.add_argument("--inner_batch_size", type=int, default=4),
    parser.add_argument("--super_image_dir", type=str, default=None),
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more details"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--step_size_up", type=float, default=1000, help="step size up for cyclical LR"
    )
    parser.add_argument(
        "--step_size_down",
        type=float,
        default=1000,
        help="step size down for cyclical LR",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_end", type=float, default=1e-9, help="end rate of polynomial lr"
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--stop_text",
        type=int,
        default=999999999,
        help="Stop training the text encoder",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--enable_xformers_vae", action="store_true", help="add xformers on vae"
    )
    parser.add_argument(
        "--flash_attention",
        action="store_true",
        help="set memory_effecient_attention to flash attention",
    )
    parser.add_argument(
        "--channels_last",
        action="store_true",
        help="Whether or not to use channels last.",
    )
    parser.add_argument(
        "--enable_vae_tiling",
        action="store_true",
        help="Whether or not to use vae tiling.",
    )
    parser.add_argument(
        "--enable_attention_slicing",
        action="store_true",
        help="Enable Attention Slicing",
    )
    parser.add_argument(
        "--cpu_model_offload",
        action="store_true",
        help="Enable cpu model offload",
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument("--save_model_every_n_steps", type=int)
    parser.add_argument("--sample_model_every_n_steps", type=int)
    parser.add_argument("--pink_noise", type=bool, default=False)
    parser.add_argument("--gamma_offset_noise", type=float, default=1.0)
    parser.add_argument("--pyramid_noise", type=bool, default=False)
    parser.add_argument("--offset_noise", type=bool, default=False)
    parser.add_argument("--hi_lo_noise", type=bool, default=False)
    parser.add_argument("--gamma_pink_noise", type=bool, default=False)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn(
                "You need not use --class_data_dir without --with_prior_preservation."
            )
        if args.class_prompt is not None:
            warnings.warn(
                "You need not use --class_prompt without --with_prior_preservation."
            )

    return args


# turns a path into a filename without the extension
def get_filename(path):
    return path.stem


def get_label_from_txt(path):
    txt_path = path.with_suffix(".txt")  # get the path to the .txt file
    if txt_path.exists():
        with open(txt_path, "r") as f:
            return f.read()
    else:
        return ""


def all_images(image_dir):
    image_path = Path(image_dir)
    return (
        list(image_path.glob("*.jpg"))
        + list(image_path.glob("*.png"))
        + list(image_path.glob("*.PNG"))
        + list(image_path.glob("*.webp"))
        + list(image_path.glob("*.jpeg"))
        + list(image_path.glob("*.JPG"))
        + list(image_path.glob("*.JPEG"))
    )


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        super=None,
        size=512,
        use_filename_as_label=False,
        use_txt_as_label=False,
        inner_batch_size=None,
        super_image_dir=None,
        super_image_batch_size=4,
    ):
        # TODO: Read this carefully and refacto and improve
        self.size = size
        self.tokenizer = tokenizer
        self.inner_batch_size = inner_batch_size

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(
                f"Instance {self.instance_data_root} images root doesn't exists."
            )

        all_files = [f for f in self.instance_data_root.glob('**/*')]
        self.num_instance_images = len(all_files)
        self.instance_prompt = instance_prompt

        self.use_filename_as_label = use_filename_as_label
        self.use_txt_as_label = use_txt_as_label
        self._length = self.num_instance_images

        self.super_image_dir = super_image_dir
        self.super_image_batch_size = super_image_batch_size

        if self.super_image_dir:
            self.super_image_dir = Path(super_image_dir)

        self.aspect_ratio_dirs = sorted(list(self.instance_data_root.iterdir()))
        self.folder_weights = [
            len(list(ar_dir.iterdir())) / sum(len(list(ar.iterdir())) for ar in self.aspect_ratio_dirs)
            for ar_dir in self.aspect_ratio_dirs
        ]



    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        aspect_ratio_index = random.choices(range(len(self.aspect_ratio_dirs)), weights=self.folder_weights)[0]
        aspect_ratio_dir = self.aspect_ratio_dirs[aspect_ratio_index]
        aspect_ratio_dir_name = Path(aspect_ratio_dir).name
        image_files = list((self.instance_data_root / aspect_ratio_dir_name).iterdir())
        superimage_files = list((self.super_image_dir / aspect_ratio_dir_name).iterdir())

        width, height = map(int, aspect_ratio_dir.name.split("x"))
        batch_size_adjust = (512 * 512) / (width * height)

        image_filepaths = random.sample(image_files, int(self.inner_batch_size * batch_size_adjust))
        superimage_filepaths = random.sample(superimage_files, int(self.super_image_batch_size * batch_size_adjust))

        size = (width, height)

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )


        images = []
        prompt_ids = []

        for path in image_filepaths + superimage_filepaths:
            abs_path = os.path.abspath(path)
            real_path = os.path.realpath(path)

            prompt = (
                get_filename(path) if self.use_filename_as_label else self.instance_prompt
            )
            prompt = get_label_from_txt(path) if self.use_txt_as_label else prompt


            def minimal_aspect_ratio(path):
                filename = Path(path).name
                width, height = map(int, filename.split('x'))
                fraction = Fraction(width, height)
                return f"{fraction.numerator}:{fraction.denominator}"

            aspect = minimal_aspect_ratio(aspect_ratio_dir)
            prompt = prompt + f"; AR: {aspect}"

            if os.path.exists(abs_path):
                try:
                    instance_image = Image.open(abs_path)
                    if not instance_image.mode == "RGB":
                        instance_image = instance_image.convert("RGB")

                    images.append(self.image_transforms(instance_image))
                    prompt_ids.append(self.tokenizer(
                        prompt,
                        truncation=True,
                        padding="max_length",
                        max_length=self.tokenizer.model_max_length,
                        return_tensors="pt",
                    ).input_ids)
                except Exception as e:
                    print(f"Error loading image {abs_path}: {e}")
            else:
                print(f"File not found: {abs_path}")


        example["images"] = images
        example["prompt_ids"] = prompt_ids

        return example


def collate_fn(examples):
    input_ids = [example["prompt_ids"] for example in examples]
    pixel_values = [example["images"] for example in examples]

    import itertools
    input_ids = list(itertools.chain(*input_ids))
    pixel_values = list(itertools.chain(*pixel_values))

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }

    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def get_full_repo_name(
    model_id: str, organization: Optional[str] = None, token: Optional[str] = None
):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


# keep track of saved steps and skip if already done
saved_steps = {}
def save_model(accelerator, unet, text_encoder, args, step=None):
    if saved_steps.get(step):
        return
    else:
        saved_steps[step] = True

    unet = accelerator.unwrap_model(unet)
    text_encoder = accelerator.unwrap_model(text_encoder)

    if step == None:
        folder = args.output_dir
    else:
        folder = args.output_dir + "-Step-" + str(step)

    print("Saving Model Checkpoint...")
    print("Directory: " + folder)

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unet,
            text_encoder=text_encoder,
            revision=args.revision,
        )
        pipeline.save_pretrained(folder)

        if args.push_to_hub:
            repo.push_to_hub(
                commit_message="End of training", blocking=False, auto_lfs_prune=True
            )


import wandb

PHOTO_COUNT = 2

test_inference_steps = 32

# record to not repeat
sampled_steps = {}

def sample_model(accelerator, unet, text_encoder, vae, args, step=None):
    if sampled_steps.get(step):
        return
    else:
        sampled_steps[step] = True

    torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
    if args.mixed_precision == "fp32":
        torch_dtype = torch.float32
    elif args.mixed_precision == "fp16":
        torch_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        torch_dtype = torch.bfloat16

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        text_encoder=text_encoder,
        vae=vae,
        torch_dtype=torch_dtype,
    )

    schedulers = [
        ("dpm-multi", DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config), 28, 6),
        ("euler-a", EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config), 35, 4),
    ]


    pipeline.to(accelerator.device)

    for prompt in prompts_with_size:
        for (schedule_name, scheduler, steps, gscale) in schedulers:
            pipeline.scheduler = scheduler
            with torch.autocast(device_type="cuda", dtype=vae.dtype):
                # Reset Seed
                seed = 123123
                generator = torch.Generator("cuda").manual_seed(seed)

                size = prompt["size"]
                text = prompt["prompt"]
                negative_prompt = prompt.get("negativePrompt")
                width = size[0]
                height = size[1]

                def round_to_nearest_multiple(number, multiple):
                    return int(round(number / multiple) * multiple)

                width = round_to_nearest_multiple(width, 8)
                height = round_to_nearest_multiple(height, 8)

                if not (negative_prompt is None):
                    negative_prompt = [negative_prompt] * PHOTO_COUNT

                images = pipeline(
                    [text] * PHOTO_COUNT,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=gscale,
                    width=width,
                    height=height,
                ).images

                label_negative = (
                    "None" if negative_prompt is None else str(negative_prompt[0:20])
                )

                # Limit the length of text, label_negative, and guidance_scale
                label_text = text[0:100]
                label_negative = label_negative[0:30]
                label_guidance_scale = str(gscale)

                label = f"{label_text} | neg: {label_negative} | scale: {label_guidance_scale} | steps: {steps} | sch: {schedule_name}"

                wandb.log(
                    {label: [wandb.Image(image, caption=label) for image in images]},
                    step=step,
                )

from pathlib import Path
from fractions import Fraction

def main(args):
    logging_dir = Path(args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if (
        args.train_text_encoder
        and args.gradient_accumulation_steps > 1
        and accelerator.num_processes > 1
    ):
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

        # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Generate class images if prior preservation is enabled.
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = (
                torch.float16 if accelerator.device.type == "cuda" else torch.float32
            )
            if args.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif args.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif args.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=args.revision,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(
                sample_dataset, batch_size=args.sample_batch_size
            )

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader,
                desc="Generating class images",
                disable=not accelerator.is_local_main_process,
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = (
                        class_images_dir
                        / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    )
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(
                    Path(args.output_dir).name, token=args.hub_token
                )
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(
                args.output_dir, clone_from=repo_name, token=args.hub_token
            )

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, revision=args.revision, use_fast=False
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

        # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            for model in models:
                sub_dir = "unet" if type(model) == type(unet) else "text_encoder"
                model.save_pretrained(os.path.join(output_dir, sub_dir))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                if type(model) == type(text_encoder):
                    # load transformers style into model
                    load_model = text_encoder_cls.from_pretrained(
                        input_dir, subfolder="text_encoder"
                    )
                    model.config = load_model.config
                else:
                    # load diffusers style into model
                    load_model = UNet2DConditionModel.from_pretrained(
                        input_dir, subfolder="unet"
                    )
                    model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.enable_vae_tiling:
        vae.enable_tiling()

    if args.enable_attention_slicing:
        unet.set_attention_slice(slice_size)

    if args.cpu_model_offload:
        for model in [self.unet, self.text_encoder, self.vae]:
            model.enable_model_cpu_offload()

    if args.channels_last:
        unet.to(memory_format=torch.channels_last)


    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
                if args.flash_attention:
                    unet.enable_xformers_memory_efficient_attention(
                        attention_op=MemoryEfficientAttentionFlashAttentionOp
                    )
            else:
                unet.enable_xformers_memory_efficient_attention()


            if args.enable_xformers_vae:
                vae.enable_xformers_memory_efficient_attention(attention_op=None)
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

        if (
            args.train_text_encoder
            and accelerator.unwrap_model(text_encoder).dtype != torch.float32
        ):
            raise ValueError(
                f"Text encoder loaded as datatype {accelerator.unwrap_model(text_encoder).dtype}."
                f" {low_precision_error_string}"
            )

            # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

        # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

            optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters())
        if args.train_text_encoder
        else unet.parameters()
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_config(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        tokenizer=tokenizer,
        use_filename_as_label=args.use_filename_as_label,
        use_txt_as_label=args.use_txt_as_label,
        super_image_dir=args.super_image_dir,
        super_image_batch_size=args.super_image_batch_size,
        inner_batch_size=args.inner_batch_size,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    total_steps = args.max_train_steps * args.gradient_accumulation_steps

    if args.lr_scheduler == "polynomial":
        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            args.lr_warmup_steps,
            total_steps,
            lr_end=args.lr_end,
            power=args.lr_power,
            last_epoch=-1,
        )
    elif args.lr_scheduler == "cyclical":
        clr_fn = lambda x: 1 / (5 ** (x * 0.0001))
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=args.lr_end,
            max_lr=args.learning_rate,
            step_size_up=args.step_size_up,
            step_size_down=args.step_size_down,
            scale_mode="iterations",
        )
    elif args.lr_scheduler == "logarithmic":
        class LogarithmicLR(torch.optim.lr_scheduler._LRScheduler):
            def __init__(self, optimizer, base_lr, max_lr, num_iterations, last_epoch=-1, verbose=False):
                self.base_lr = base_lr
                self.max_lr = max_lr
                self.num_iterations = num_iterations
                super(LogarithmicLR, self).__init__(optimizer, last_epoch, verbose)

            def get_lr(self):
                step_ratio = self.last_epoch / self.num_iterations
                lr_diff = self.max_lr - self.base_lr
                new_lr = self.base_lr + (lr_diff * np.log10(1 + 9 * step_ratio))
                return [new_lr for _ in self.base_lrs]

        lr_scheduler = LogarithmicLR(
            optimizer,
            base_lr=args.lr_end,
            max_lr=args.learning_rate,
            num_iterations=args.total_iterations,
        )
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=total_steps,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

        # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        (
            unet,
            text_encoder,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    # only swap to eval once for performance
    text_off = False

    for epoch in range(first_epoch, args.num_train_epochs):
        if args.train_text_encoder:
            text_encoder.train()

        for step, batch in enumerate(train_dataloader):
            if step > args.stop_text and text_off is False:
                text_encoder.requires_grad_(False)
                text_encoder.eval()
                text_off = True

            # Skip steps until we reach the resumed step
            if (
                args.resume_from_checkpoint
                and epoch == first_epoch
                and step < resume_step
            ):
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=weight_dtype)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                if args.pink_noise:
                    noise = make_pink_noise(latents)
                elif args.gamma_pink_noise:
                    noise = gamma_pink_noise(latents)
                elif args.offset_noise:
                    noise = offset_noise(latents)
                elif args.hi_lo_noise:
                    noise = hi_lo_noise(latents)
                elif args.gamma_offset_noise:
                    noise = gamma_offset_noise(latents, discount=args.gamma_offset_noise)
                elif args.pyramid_noise:
                    noise = pyramid_noise_like(latents, discount=0.8)
                else:
                    noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute instance loss
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )

                    # Compute prior loss
                    prior_loss = F.mse_loss(
                        model_pred_prior.float(), target_prior.float(), reduction="mean"
                    )

                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if args.train_text_encoder
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            if accelerator.is_main_process:
                if (
                    args.save_model_every_n_steps != None
                    and (global_step % args.save_model_every_n_steps) == 0
                ):
                    save_model(accelerator, unet, text_encoder, args, global_step)

                if args.sample_model_every_n_steps != None:
                    if (
                        global_step % args.sample_model_every_n_steps
                    ) == 0 and global_step > 5:
                        sample_model(
                            accelerator, unet, text_encoder, vae, args, global_step
                        )

        accelerator.wait_for_everyone()

    save_model(accelerator, unet, text_encoder, args, step=None)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
