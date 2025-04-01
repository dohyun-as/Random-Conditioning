# ------------------------------------------------------------------------------------
# Copyright 2023. Nota Inc. All Rights Reserved.
# Code modified from https://github.com/huggingface/diffusers/blob/v0.15.0/examples/text_to_image/train_text_to_image.py
# ------------------------------------------------------------------------------------
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
import os
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_TIMEOUT'] = '3600'
os.environ['NCCL_TIMEOUT_MS'] = '3600000'  # 개별 NCCL 작업의 타임아웃을 20분으로 설정
import argparse
import logging
import math
import random
from pathlib import Path
from typing import Optional
import subprocess
import sys
import shutil
import uuid
from datetime import timedelta

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Subset
import transformers
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from eval_clip_score_ddp import evaluate_clip_score, evaluate_clip_score_unseen_setting
from generate_ddp2 import sample_images_30k, sample_images_41k
from eval_score_wandb_log import log_eval_scores, log_eval_scores_unseen_setting
from torchvision.utils import save_image, make_grid


import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate
from diffusers.utils.import_utils import is_xformers_available

import csv
import time
import copy

from src.dataset import rand_cond_dataset, collate_fn
from funcs import MultiConv1x1, get_layer_output_channels, count_parameters

import warnings
warnings.filterwarnings("ignore")

# try to import wandb
try:
    import wandb
    has_wandb = True
except:
    has_wandb = False

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.15.0")

logger = get_logger(__name__, log_level="INFO")

def get_activation(mem, name):
    def get_output_hook(module, input, output):
        mem[name] = output

    return get_output_hook

def add_hook(net, mem, mapping_layers):
    for n, m in net.named_modules():
        if n in mapping_layers:
            m.register_forward_hook(get_activation(mem, n))

########################################################################## MIIL ##############################################################################################3
def get_input_activation(mem, name):
    def get_input_hook(module, input):
        if isinstance(input, tuple):
            print(input)
        # input은 튜플일 수 있으므로 첫 번째 요소를 저장
        mem[name] = input[0] if isinstance(input, tuple) else input
        
        print(f"Hooking input for {name}: type(input) = {type(input)}")
        print(mem[name])
    return get_input_hook

def add_pre_hook(net, mem, mapping_layers):
    for n, m in net.named_modules():
        if n in mapping_layers:
            # forward_pre_hook을 사용하여 입력값을 후킹
            m.register_forward_pre_hook(get_input_activation(mem, n))
########################################################################## MIIL ##############################################################################################3


def copy_weight_from_teacher(unet_stu, unet_tea, student_type):

    connect_info = {} # connect_info['TO-student'] = 'FROM-teacher'
    if student_type in ["bk_base", "bk_small"]:
        connect_info['up_blocks.0.resnets.1.'] = 'up_blocks.0.resnets.2.'
        connect_info['up_blocks.1.resnets.1.'] = 'up_blocks.1.resnets.2.'
        connect_info['up_blocks.1.attentions.1.'] = 'up_blocks.1.attentions.2.'
        connect_info['up_blocks.2.resnets.1.'] = 'up_blocks.2.resnets.2.'
        connect_info['up_blocks.2.attentions.1.'] = 'up_blocks.2.attentions.2.'
        connect_info['up_blocks.3.resnets.1.'] = 'up_blocks.3.resnets.2.'
        connect_info['up_blocks.3.attentions.1.'] = 'up_blocks.3.attentions.2.'
    elif student_type in ["bk_tiny"]:
        connect_info['up_blocks.0.resnets.0.'] = 'up_blocks.1.resnets.0.'
        connect_info['up_blocks.0.attentions.0.'] = 'up_blocks.1.attentions.0.'
        connect_info['up_blocks.0.resnets.1.'] = 'up_blocks.1.resnets.2.'
        connect_info['up_blocks.0.attentions.1.'] = 'up_blocks.1.attentions.2.'
        connect_info['up_blocks.0.upsamplers.'] = 'up_blocks.1.upsamplers.'
        connect_info['up_blocks.1.resnets.0.'] = 'up_blocks.2.resnets.0.'
        connect_info['up_blocks.1.attentions.0.'] = 'up_blocks.2.attentions.0.'
        connect_info['up_blocks.1.resnets.1.'] = 'up_blocks.2.resnets.2.'
        connect_info['up_blocks.1.attentions.1.'] = 'up_blocks.2.attentions.2.'
        connect_info['up_blocks.1.upsamplers.'] = 'up_blocks.2.upsamplers.'
        connect_info['up_blocks.2.resnets.0.'] = 'up_blocks.3.resnets.0.'
        connect_info['up_blocks.2.attentions.0.'] = 'up_blocks.3.attentions.0.'
        connect_info['up_blocks.2.resnets.1.'] = 'up_blocks.3.resnets.2.'
        connect_info['up_blocks.2.attentions.1.'] = 'up_blocks.3.attentions.2.'       
    else:
        raise NotImplementedError


    for k in unet_stu.state_dict().keys():
        flag = 0
        k_orig = k
        for prefix_key in connect_info.keys():
            if k.startswith(prefix_key):
                flag = 1
                k_orig = k_orig.replace(prefix_key, connect_info[prefix_key])            
                break

        if flag == 1:
            print(f"** forced COPY {k_orig} -> {k}")
        else:
            print(f"normal COPY {k_orig} -> {k}")
        unet_stu.state_dict()[k].copy_(unet_tea.state_dict()[k_orig])

    return unet_stu

def parse_args():
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
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--extra_text_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the extra text data for random conditioning."
        ),
    )    
    
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    
    parser.add_argument("--random_conditioning", action='store_true', help='perform condition sharing')
    parser.add_argument("--random_conditioning_lambda", type=float, default=5, help="condition share lambda")
    parser.add_argument("--sigmoid", action='store_true', help='perform condition sharing')
    
    parser.add_argument("--gpt_caption", action='store_true', help='use GPT caption')
    
    parser.add_argument("--max_extra_text_samples", type=int, default=None, help='limit extra text data')

    parser.add_argument("--use_unseen_setting", action='store_true', help='use unseen setting(train_data, eval_data)')

    
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1000)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
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
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
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
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
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
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=50000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument("--evaluation_step", type=int, default=25000, help=("evlauation step"),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
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
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    parser.add_argument("--unet_config_path", type=str, default="./src/unet_config")     
    parser.add_argument("--unet_config_name", type=str, default="bk_small", 
                        choices=["bk_base", "bk_small", "bk_tiny", "C_base", "C_small", "C_tiny", "C_micro"])   
    parser.add_argument("--lambda_sd", type=float, default=1.0, help="weighting for the denoising task loss")  
    parser.add_argument("--lambda_kd_output", type=float, default=1.0, help="weighting for output KD loss")  
    parser.add_argument("--lambda_kd_feat", type=float, default=1.0, help="weighting for feature KD loss")  
    parser.add_argument("--valid_steps", type=int, default=10000)
    parser.add_argument("--num_valid_images", type=int, default=2)
    parser.add_argument("--use_copy_weight_from_teacher", action="store_true", help="Whether to initialize unet student with teacher's weights",)
    parser.add_argument("--valid_prompt", type=str, default="a golden vase with different flowers")

    # parser.add_argument("--valid_prompts", type=str, nargs='+', default=[
    #     "A cat crawling into a white toilet seat.",
    #     "a small elephant walks in a lush forest",
    #     "A couple of elephants walking across a river.",
    #     "A white horse looking through the window of a tall brick building.",
    #     "A dog with goggles is in a motorcycle side car.",
    #     "A large elephant walking next to a fallen tree.",
    #     "A striped cat curled up above a comforter.",
    #     "A cat laying bed in a small room.",
    #     "A horse with a blanket on it eating hay.",
    #     "Dog in bag with mesh and items around",
    #     "A young sexy woman holding a tennis racquet on a tennis court.",
    #     "The torso of a man who is holding a knife.",
    #     "a person that is standing up in a tennis court",
    #     "A man swinging a tennis racquet on a tennis court.",
    #     "a young man in a grey shirt is going to cut his hair",
    #     "a red fire hydrant near a dirt road with trees in the background",
    #     "a tennis player attempting to reach a tennis ball",
    #     "A bathroom with a tiled floor and a sink.",
    #     "Bride and grooms arms cutting the wedding cake with fruit on top.",
    #     "A bathroom with a brown shower curtain and white toilet"
    # ])

    parser.add_argument("--valid_prompts", type=str, nargs='+', default=[
    "Two cakes sitting on a class table near a candle",
    "The fresh fruit is left out on the counter",
    "A couple of buses that are lined up by some buildings",
    "A bicycle with a basket and a blue and pink umbrella",
    "A kitchen with a stove top oven next to a white fridge.",
    "A desk with a laptop, computer monitor, keyboard and mouse",
    "A wooden and metal bench near a over grown bush",
    "Two men in suits and ties next to plant",
    "A woman standing next to the ocean flying a colorful kite",
    "A baseball player wearing a leather glove standing in the dirt",
    "A bird flies over a large body of water",
    "A cat sitting underneath a blue and white umbrella",
    "A dog jumps in order to catch a ring",
    "A horse drawn carriage driving down a small road",
    "A sheep standing on top of a rock",
    "A brown cow walking through a tree filled forest",
    "An elephant is carrying some plants in its tusk",
    "A bear peaks out behind the grass in the woods",
    "A zebra rolling on its back on the ground",
    "A giraffe standing next to a tall tree"
    ])
    
    
    parser.add_argument("--channel_mapping", action="store_true", help="channel mapping",)
    
    
    parser.add_argument("--drop_text", action="store_true", help="distillation null text",)
    parser.add_argument("--drop_text_p", type=float, default=0.1, help="null text ratio",)
    
    # arguments for evaluation
    parser.add_argument("--model_id", type=str, default="nota-ai/bk-sdm-base", help="Path to the pretrained model or checkpoint directory.")
    parser.add_argument("--unet_path", type=str, default="/home/work/StableDiffusion/T2I_distill1_GPU4/results/toy_ddp_bk_base/checkpoint-25000", help="Model checkpoint for evaluate")
    parser.add_argument("--img_sz", type=int, default=512)
    parser.add_argument("--img_resz", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--batch_sz", type=int, default=10)    


    parser.add_argument("--data_list", type=str, default="./data/mscoco_val2014_30k/metadata.csv")
    parser.add_argument("--valid41k_dir", type=str, default="./data/mscoco_val2014_41k")
    

    parser.add_argument('--clip_device', type=str, default='cuda', help='Device to use, cuda or cpu')
    parser.add_argument('--clip_seed', type=int, default=1234, help='Random seed for reproducibility')
    parser.add_argument('--clip_batch_size', type=int, default=50, help='Batch size for processing images')

    parser.add_argument("--use_sd_loss", action="store_true", help="Whether to calculate sd_loss (denoising task loss).")
    parser.add_argument("--SDXL",action="store_true",help="compress SDXL")
        
    args = parser.parse_args()
    
    args.save_dir = os.path.join(args.output_dir, "generated_images")
    
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
        
    run_id = uuid.uuid4().hex
    
    logging_dir = os.path.join(args.output_dir, args.logging_dir, f"run_{run_id}")
    os.makedirs(logging_dir, exist_ok=True)
    
    wandb_dir = os.path.join(args.output_dir, "wandb_logs", f"run_{run_id}")
    os.makedirs(wandb_dir, exist_ok=True)
    
    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
    
    ipg_handler = InitProcessGroupKwargs(
            timeout=timedelta(seconds=5400)
            )
    
    accelerator = Accelerator(
        kwargs_handlers=[ipg_handler],
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
    )
    
    world_size = accelerator.num_processes  
    local_rank = accelerator.local_process_index
    
    # Add custom csv logger and validation image folder
    val_img_dir = os.path.join(args.output_dir, 'val_img')
    os.makedirs(val_img_dir, exist_ok=True)


    csv_log_path = os.path.join(args.output_dir, 'log_loss.csv')
    print(csv_log_path)
    if not os.path.exists(csv_log_path):
        with open(csv_log_path, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['epoch', 'step', 'global_step',
                                'loss_total', 'loss_sd', 'loss_kd_output', 'loss_kd_feat',
                                'lr', 'lamb_sd', 'lamb_kd_output', 'lamb_kd_feat'])

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed+accelerator.local_process_index)

    # Handle the repository creation
    if accelerator.is_main_process and (args.output_dir is not None):
        os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)

    # Define teacher and student
    unet_teacher = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )

    config_student = UNet2DConditionModel.load_config(args.unet_config_path, subfolder=args.unet_config_name)
    unet = UNet2DConditionModel.from_config(config_student, revision=args.non_ema_revision)

    # Copy weights from teacher to student
    if args.use_copy_weight_from_teacher:
        copy_weight_from_teacher(unet, unet_teacher, args.unet_config_name)
   

    # Freeze student's vae and text_encoder and teacher's unet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet_teacher.requires_grad_(False)

   
    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_config(config_student, revision=args.revision)
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                if isinstance(model, UNet2DConditionModel):
                    model.save_pretrained(os.path.join(output_dir, "unet"))
                elif isinstance(model, MultiConv1x1):
                    model.save_pretrained(os.path.join(output_dir, "conv_layers"))
                else:
                    print(f"Warning: model of type {type(model)} is not handled in save_model_hook")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                if isinstance(model, UNet2DConditionModel):
                    # load diffusers style into model
                    load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                    del load_model
                elif isinstance(model, MultiConv1x1):
                    # student_channels_list와 teacher_channels_list를 다시 생성해야 합니다.
                    # 이를 위해 필요한 정보를 args나 다른 곳에서 가져옵니다.
                    teacher_channels_list = get_layer_output_channels(unet_teacher, mapping_layers_tea)
                    student_channels_list = get_layer_output_channels(unet, mapping_layers_stu)
                    # MultiConv1x1 모델 로드
                    load_model = MultiConv1x1.from_pretrained(os.path.join(input_dir, "conv_layers"),
                                                            student_channels_list, teacher_channels_list)
                    model.load_state_dict(load_model.state_dict())
                    del load_model
                else:
                    print(f"Warning: model of type {type(model)} is not handled in load_model_hook")


        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # Add hook for feature KD
    acts_tea = {}
    acts_stu = {}
    
    if args.unet_config_name in ["C_base", "C_small", "C_tiny", "C_micro"]:
        
        mapping_layers = ['up_blocks.0.upsamplers.0.conv', 'up_blocks.1.upsamplers.0.conv', 'up_blocks.2.upsamplers.0.conv', 'up_blocks.3.attentions.2.proj_out',
                          'mid_block.attentions.0.proj_out',                          
                        'down_blocks.0.downsamplers.0.conv', 'down_blocks.1.downsamplers.0.conv', 'down_blocks.2.downsamplers.0.conv', 'down_blocks.3.resnets.1.conv2']    
        mapping_layers_tea = copy.deepcopy(mapping_layers)
        mapping_layers_stu = copy.deepcopy(mapping_layers)
        
    elif args.unet_config_name in ["bk_base", "bk_small"]:
        mapping_layers = ['up_blocks.0', 'up_blocks.1', 'up_blocks.2', 'up_blocks.3',
                        'down_blocks.0', 'down_blocks.1', 'down_blocks.2', 'down_blocks.3']    
        mapping_layers_tea = copy.deepcopy(mapping_layers)
        mapping_layers_stu = copy.deepcopy(mapping_layers)
        # mapping_layers_tea = ['up_blocks.0.upsamplers.0.conv', 'up_blocks.1.upsamplers.0.conv', 'up_blocks.2.upsamplers.0.conv', 'up_blocks.3.attentions.2.proj_out',
        #                 'down_blocks.0.downsamplers.0.conv', 'down_blocks.1.downsamplers.0.conv', 'down_blocks.2.downsamplers.0.conv', 'down_blocks.3.resnets.1.conv2']  
        # mapping_layers_stu = ['up_blocks.0.upsamplers.0.conv', 'up_blocks.1.upsamplers.0.conv', 'up_blocks.2.upsamplers.0.conv', 'up_blocks.3.attentions.1.proj_out',
        #                 'down_blocks.0.downsamplers.0.conv', 'down_blocks.1.downsamplers.0.conv', 'down_blocks.2.downsamplers.0.conv', 'down_blocks.3.resnets.0.conv2']  

    elif args.unet_config_name in ["bk_tiny"]:
        mapping_layers_tea = ['down_blocks.0', 'down_blocks.1', 'down_blocks.2.attentions.1.proj_out',
                                'up_blocks.1', 'up_blocks.2', 'up_blocks.3']    
        mapping_layers_stu = ['down_blocks.0', 'down_blocks.1', 'down_blocks.2.attentions.0.proj_out',
                                'up_blocks.0', 'up_blocks.1', 'up_blocks.2']  


    if args.channel_mapping:
        teacher_channels_list = get_layer_output_channels(unet_teacher, mapping_layers_tea)
        student_channels_list = get_layer_output_channels(unet, mapping_layers_stu)
        conv_layers = MultiConv1x1(student_channels_list, teacher_channels_list)
        parameters = list(unet.parameters()) + list(conv_layers.parameters())
    else:
        parameters = unet.parameters()

    # print(f"Number of parameters in text_encoder: {count_parameters(text_encoder):,}")
    # print(f"Number of parameters in vae: {count_parameters(vae):,}")
    # print(f"Number of parameters in unet_teacher: {count_parameters(unet_teacher):,}")
    # print(f"Number of parameters in unet (student): {count_parameters(unet):,}")
    
    # with accelerator.main_process_first():
    #     for name, module in unet.named_modules():
    #         print(name)
    num_parameters_unet = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in unet (student): {num_parameters_unet:,}")
    
    optimizer = optimizer_cls(
        parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = rand_cond_dataset(data_dir=args.train_data_dir, extra_text_dir=args.extra_text_dir,n_T=noise_scheduler.num_train_timesteps, 
                               random_conditioning=args.random_conditioning, random_conditioning_lambda=args.random_conditioning_lambda, 
                               world_size=world_size, rank=local_rank, drop_text=args.drop_text, drop_text_p=args.drop_text_p, 
                               use_unseen_setting=args.use_unseen_setting, gpt_caption = args.gpt_caption, max_extra_text_samples=args.max_extra_text_samples,
                               sigmoid=args.sigmoid)

    if args.max_train_samples is not None:
        original_seed = random.getstate()
        random.seed(42)  # 원하는 시드 값 설정
        indices = random.sample(range(len(train_dataset)), args.max_train_samples)
        random.setstate(original_seed)
        print("all:", len(train_dataset))
        train_dataset = Subset(train_dataset, indices)
        print("Subset:", len(train_dataset))

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn(tokenizer),
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    if args.channel_mapping:
        # Prepare everything with our `accelerator`.
        unet, conv_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, conv_layers, optimizer, train_dataloader, lr_scheduler
        )
        if hasattr(conv_layers, 'module'):
            conv_module = conv_layers.module
        else:
            conv_module = conv_layers
        
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )
        
    
    if torch.cuda.device_count() > 1:
        print(f"use multi-gpu: # gpus {torch.cuda.device_count()}")
        # revise the hooked feature names for student (to consider ddp wrapper)
        for i, m_stu in enumerate(mapping_layers_stu):
            mapping_layers_stu[i] = 'module.'+m_stu

    add_hook(unet_teacher, acts_tea, mapping_layers_tea)
    add_hook(unet, acts_stu, mapping_layers_stu)
    
    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move student's text_encode and vae and teacher's unet to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet_teacher.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(
            "text2image-fine-tune",
            config=vars(args),
            init_kwargs={"wandb": {"dir": wandb_dir, "name": f"run_{run_id}"}}
        )
        
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
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
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    progress_bar = tqdm(
    range(global_step, args.max_train_steps), 
    initial=global_step,  # 초기값을 현재 global_step으로 설정
    total=args.max_train_steps,  # 전체 스텝을 max_train_steps로 설정
    disable=not accelerator.is_local_main_process,
    dynamic_ncols=True
)
    progress_bar.set_description("Steps")

    # get wandb_tracker (if it exists)
    wandb_tracker = accelerator.get_tracker("wandb")

    for epoch in range(first_epoch, args.num_train_epochs):

        unet.train()

        train_loss = 0.0
        train_loss_sd = 0.0
        train_loss_kd_output = 0.0
        train_loss_kd_feat = 0.0

        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = batch["latents"].to(weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                # Sample a random timestep for each image
                timesteps = batch["timesteps"]
                
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                # encoder_hidden_states = batch["text_embs"].to(weight_dtype)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                ################################################## loss calculation ####################################################################

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                if args.use_sd_loss:
                    paireds = batch['paireds']
                    mask = (paireds == 1)
                    model_pred_ = model_pred[mask]
                    target_ = target[mask]
                    if model_pred_.numel() > 0:  # 필터링된 요소가 있는지 확인
                        loss_sd = F.mse_loss(model_pred_.float(), target_.float(), reduction="mean")
                    else:
                        loss_sd = torch.tensor(0.0, device=accelerator.device)
                else:
                    loss_sd = torch.tensor(0.0, device=accelerator.device)  # If not using, set to zero

                # Predict output-KD loss
                model_pred_teacher = unet_teacher(noisy_latents, timesteps, encoder_hidden_states).sample
                loss_kd_output = F.mse_loss(model_pred.float(), model_pred_teacher.float(), reduction="mean")

                # Predict feature-KD loss
                losses_kd_feat = []
                #print("##########################################################################################")
                for i, (m_tea, m_stu) in enumerate(zip(mapping_layers_tea, mapping_layers_stu)):

                    a_tea = acts_tea[m_tea]
                    a_stu = acts_stu[m_stu]
                    
                    if type(a_tea) is tuple: a_tea = a_tea[0]                        
                    if type(a_stu) is tuple: a_stu = a_stu[0]
                    
                    if args.channel_mapping:
                        a_stu_ = conv_module.convs[i](a_stu.to(conv_module.convs[i].weight.dtype))
                        # a_stu_ = a_stu_.to(a_tea.dtype)
                        tmp = F.mse_loss(a_stu_.float(), a_tea.detach().float(), reduction="mean")
                    else:
                        # print(f"Layer teacher: {m_tea}, student: {m_stu}")
                        # print(f"a_tea shape: {a_tea.shape}, a_stu shape: {a_stu.shape}")
                        # print(f"After channel mapping, a_stu_ shape: {a_stu.shape}, a_tea shape: {a_tea.shape}")
                        tmp = F.mse_loss(a_stu.float(), a_tea.detach().float(), reduction="mean")
                    losses_kd_feat.append(tmp)
                
                    #print(a_tea.shape)
                #print("##########################################################################################")

                loss_kd_feat = sum(losses_kd_feat)

                # Compute the final loss
                loss = args.lambda_sd * loss_sd + args.lambda_kd_output * loss_kd_output + args.lambda_kd_feat * loss_kd_feat
                # loss = args.lambda_kd_output * loss_kd_output + args.lambda_kd_feat * loss_kd_feat

                ################################################## loss calculation ####################################################################

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                avg_loss_sd = accelerator.gather(loss_sd.repeat(args.train_batch_size)).mean()
                train_loss_sd += avg_loss_sd.item() / args.gradient_accumulation_steps

                avg_loss_kd_output = accelerator.gather(loss_kd_output.repeat(args.train_batch_size)).mean()
                train_loss_kd_output += avg_loss_kd_output.item() / args.gradient_accumulation_steps

                avg_loss_kd_feat = accelerator.gather(loss_kd_feat.repeat(args.train_batch_size)).mean()
                train_loss_kd_feat += avg_loss_kd_feat.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log(
                    {
                        "train_loss": train_loss, 
                        "train_loss_sd": train_loss_sd,
                        "train_loss_kd_output": train_loss_kd_output,
                        "train_loss_kd_feat": train_loss_kd_feat,
                        "lr": lr_scheduler.get_last_lr()[0]
                    }, 
                    step=global_step
                )

                if accelerator.is_main_process:
                    with open(csv_log_path, 'a') as logfile:
                        logwriter = csv.writer(logfile, delimiter=',')
                        logwriter.writerow([epoch, step, global_step,
                                            train_loss, train_loss_sd, train_loss_kd_output, train_loss_kd_feat,
                                            lr_scheduler.get_last_lr()[0],
                                            args.lambda_sd, args.lambda_kd_output, args.lambda_kd_feat])

                train_loss = 0.0
                train_loss_sd = 0.0
                train_loss_kd_output = 0.0
                train_loss_kd_feat = 0.0

                # torch.cuda.empty_cache()
                if global_step % args.checkpointing_steps == 0:
                    torch.cuda.empty_cache()
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    if accelerator.is_main_process:
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        os.makedirs(args.save_dir, exist_ok=True)
                    accelerator.wait_for_everyone()
                if global_step % args.evaluation_step==0:      
                    ################################################ Evaluate IS, FID, CLIP SCORE - Unseen Setting #################################################
                    if args.use_unseen_setting:
                        sample_images_41k(args, accelerator, save_path)  # None으로 바뀌어 있으니까 test끝나고 바로 None 지워야함!! 
                        print(f"rank {local_rank} is done")
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            try:
                                subprocess.run(
                                    [
                                        "sh", "./scripts/eval_scores_ddp_unseen_setting.sh",
                                        args.save_dir,          # SAVE_DIR
                                        str(args.img_sz),       # IMG_SZ
                                        str(args.img_resz),     # IMG_RESZ
                                        args.valid41k_dir        # valid_dir
                                    ],
                                    check=True, stdout=sys.stdout, stderr=sys.stderr
                                )
                            except subprocess.CalledProcessError as e:
                                print(f"Error occurred while running script: {e}")

                        # Wait for all ranks to complete the evaluation
                        accelerator.wait_for_everyone()
                        evaluate_clip_score_unseen_setting(args, accelerator)
                        log_eval_scores_unseen_setting(accelerator, args, global_step)
                    ################################################################################################################################################                    
     
                    ################################################ Evaluate IS, FID, CLIP SCORE - Base Setting ###################################################
                    else:
                        sample_images_30k(args, accelerator, SDXL = args.SDXL, save_path = save_path)
                        print(f"rank {local_rank} is done")
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            try:
                                subprocess.run(
                                    [
                                        "sh", "./scripts/eval_scores_ddp.sh",
                                        args.save_dir,          # SAVE_DIR
                                        str(args.img_sz),       # IMG_SZ
                                        str(args.img_resz),     # IMG_RESZ
                                        args.data_list          # DATA_LIST
                                    ],
                                    check=True, stdout=sys.stdout, stderr=sys.stderr
                                )
                            except subprocess.CalledProcessError as e:
                                print(f"Error occurred while running script: {e}")
                        else:
                            time.sleep(120)
                        # Wait for all ranks to complete the evaluation
                        accelerator.wait_for_everyone()
                        evaluate_clip_score(args, accelerator)
                        log_eval_scores(accelerator, args, global_step)
                    #################################################################################################################################################

            logs = {"step_loss": loss.detach().item(),
                    "sd_loss": loss_sd.detach().item(),
                    "kd_output_loss": loss_kd_output.detach().item(),
                    "kd_feat_loss": loss_kd_feat.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.wait_for_everyone()
            # save validation images
            if (args.valid_prompt is not None) and (global_step % args.valid_steps == 0) and accelerator.is_main_process:
                logger.info(
                    f"Running validation... \n Generating {args.num_valid_images} images with prompt:"
                    f" {args.valid_prompt}."
                )
                # create pipeline
                pipeline = StableDiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    safety_checker=None,
                    revision=args.revision,
                )
                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)
                generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

                if not os.path.exists(os.path.join(val_img_dir, "teacher_0.png")):
                    for kk in range(args.num_valid_images):
                        image = pipeline(args.valid_prompt, num_inference_steps=25, generator=generator).images[0]
                        tmp_name = os.path.join(val_img_dir, f"teacher_{kk}.png")
                        image.save(tmp_name)

                # set `keep_fp32_wrapper` to True because we do not want to remove
                # mixed precision hooks while we are still training
                pipeline.unet = accelerator.unwrap_model(unet, keep_fp32_wrapper=True).to(accelerator.device)
            
                # for kk in range(args.num_valid_images):
                #     image = pipeline(args.valid_prompt, num_inference_steps=25, generator=generator).images[0]
                #     tmp_name = os.path.join(val_img_dir, f"gstep{global_step}_epoch{epoch}_step{step}_{kk}.png")
                #     print(tmp_name)
                #     image.save(tmp_name)
                # 이미지 생성 루프
                # 리스트에 모든 이미지 저장
                all_images = []
                
                # 각 프롬프트로 이미지를 생성하고 리스트에 추가
                for kk, prompt in enumerate(args.valid_prompts):
                    image = pipeline(prompt, num_inference_steps=25, generator=generator).images[0]
                    all_images.append(transforms.ToTensor()(image))  # 이미지 배열을 텐서로 변환하여 리스트에 추가
                
                # 그리드로 결합 (nrow 파라미터로 한 줄에 몇 개의 이미지를 배치할지 결정 가능)
                grid = make_grid(all_images, nrow=10, padding=2)
                
                # 그리드 이미지를 저장할 파일 경로 지정
                grid_save_path = os.path.join(val_img_dir, f"gstep{global_step}_epoch{epoch}_step{step}_grid.png")
                
                # 그리드 이미지를 PIL 이미지로 변환하여 저장
                grid_image = transforms.ToPILImage()(grid)
                grid_image.save(grid_save_path)

                wandb.log({"Generated Image Grid": wandb.Image(grid_image, caption=f"Grid for epoch {epoch}, step {step}")})

                
                print(f"Saved grid image at: {grid_save_path}")

                del pipeline
                torch.cuda.empty_cache()

            accelerator.wait_for_everyone()

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            revision=args.revision,
        )
        pipeline.save_pretrained(args.output_dir)

    accelerator.end_training()

if __name__ == "__main__":
    main()
