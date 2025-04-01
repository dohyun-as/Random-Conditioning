import os
import argparse
import time
import torch
from accelerate import Accelerator
from tqdm import tqdm
from utils.inference_pipeline import InferencePipeline
from utils.misc import get_file_list_from_csv, change_img_size, change_img_size_ddp
import contextlib
import sys
import math
import pandas as pd
import shutil

def sample_images_30k(args, accelerator, SDXL=None, save_path=None):
    device = accelerator.device

    pipeline = InferencePipeline(weight_folder=args.model_id, seed=args.clip_seed, device=device)
    pipeline.set_pipe_and_generator(SDXL)
    
    pipeline.pipe.set_progress_bar_config(disable=True)

    if save_path is not None:  # use a separate trained unet for generation        
        from diffusers import UNet2DConditionModel 
        unet = UNet2DConditionModel.from_pretrained(save_path, subfolder='unet')
        pipeline.pipe.unet = unet.half().to(device)
        accelerator.print(f"** load unet from {save_path}")        

    save_dir_src = os.path.join(args.save_dir, f'im{args.img_sz}')  # for model's raw output images
    save_dir_tgt = os.path.join(args.save_dir, f'im{args.img_resz}')  # for resized images for benchmark

    # Create directories only on the main process
    if accelerator.is_main_process:
        os.makedirs(save_dir_src, exist_ok=True)
        os.makedirs(save_dir_tgt, exist_ok=True)

    accelerator.wait_for_everyone()

    file_list = get_file_list_from_csv(args.data_list)
    total_files = len(file_list)
    num_processes = accelerator.num_processes
    rank = accelerator.process_index

    # Distribute files evenly among ranks without leaving any unprocessed files
    files_per_process = total_files // num_processes
    remainder = total_files % num_processes

    if rank < remainder:
        start_index = rank * (files_per_process + 1)
        end_index = start_index + files_per_process + 1
    else:
        start_index = remainder * (files_per_process + 1) + (rank - remainder) * files_per_process
        end_index = start_index + files_per_process

    # Get the list of files to process for this rank
    process_file_list = file_list[start_index:end_index]
    
    total_batches = math.ceil(len(process_file_list) / args.batch_sz)
    
    # tqdm progress bar setup for rank 0 only
    if accelerator.is_main_process:
        progress_bar = tqdm(total=total_batches, desc="Generating Images",ncols=None, disable=not accelerator.is_main_process)

    # Process the assigned files in batches
    for batch_start in range(0, len(process_file_list), args.batch_sz):
        batch_end = min(batch_start + args.batch_sz, len(process_file_list))

        img_names = [file_info[0] for file_info in process_file_list[batch_start:batch_end]]
        val_prompts = [file_info[1] for file_info in process_file_list[batch_start:batch_end]]

        # Suppress output of pipeline.generate
        with contextlib.redirect_stdout(sys.stderr), contextlib.redirect_stderr(sys.stderr):
            imgs = pipeline.generate(prompt=val_prompts,
                                     guidance_scale=args.guidance_scale,
                                     n_steps=args.num_inference_steps,
                                     img_sz=args.img_sz)

        for img, img_name in zip(imgs, img_names):
            img.save(os.path.join(save_dir_src, img_name))
            img.close()

        # Update progress bar only on rank 0
        if accelerator.is_main_process:
            progress_bar.update(1)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        change_img_size(save_dir_src, save_dir_tgt, args.img_resz)
        progress_bar.close()
        accelerator.print(f"Image generation completed and resized images saved.")
    else: 
        time.sleep(300)
        
    accelerator.wait_for_everyone()
    
    pipeline.clear()
    torch.cuda.empty_cache()

def sample_images_41k(args, accelerator, save_path=None):
    device = accelerator.device

    pipeline = InferencePipeline(weight_folder=args.model_id, seed=args.clip_seed, device=device)
    pipeline.set_pipe_and_generator()
    
    pipeline.pipe.set_progress_bar_config(disable=True)

    if save_path is not None:  # use a separate trained unet for generation        
        from diffusers import UNet2DConditionModel 
        unet = UNet2DConditionModel.from_pretrained(save_path, subfolder='unet')
        pipeline.pipe.unet = unet.half().to(device)
        accelerator.print(f"** load unet from {save_path}")        

    save_dir_all = os.path.join(args.save_dir, 'all')
    save_dir_src = os.path.join(save_dir_all, f'im{args.img_sz}')  # for model's raw output images
    save_dir_tgt = os.path.join(save_dir_all, f'im{args.img_resz}')  # for resized images for benchmark

    # Create directories only on the main process
    if accelerator.is_main_process:
        os.makedirs(save_dir_all, exist_ok=True)
        os.makedirs(save_dir_src, exist_ok=True)
        os.makedirs(save_dir_tgt, exist_ok=True)

    accelerator.wait_for_everyone()
    
    data_list_41k = os.path.join(args.valid41k_dir, 'metadata_mscoco41k_all.csv')
    file_list = get_file_list_from_csv(data_list_41k)
    total_files = len(file_list)
    num_processes = accelerator.num_processes
    rank = accelerator.process_index

    # Distribute files evenly among ranks without leaving any unprocessed files
    files_per_process = total_files // num_processes
    remainder = total_files % num_processes

    if rank < remainder:
        start_index = rank * (files_per_process + 1)
        end_index = start_index + files_per_process + 1
    else:
        start_index = remainder * (files_per_process + 1) + (rank - remainder) * files_per_process
        end_index = start_index + files_per_process

    # Get the list of files to process for this rank
    process_file_list = file_list[start_index:end_index]
    
    total_batches = math.ceil(len(process_file_list) / args.batch_sz)
    
    # tqdm progress bar setup for rank 0 only
    if accelerator.is_main_process:
        progress_bar = tqdm(total=total_batches, desc="Generating Images",ncols=None, disable=not accelerator.is_main_process)
        
    local_generated_images = 0

    # Process the assigned files in batches
    for batch_start in range(0, len(process_file_list), args.batch_sz):
        batch_end = min(batch_start + args.batch_sz, len(process_file_list))

        img_names = [file_info[0] for file_info in process_file_list[batch_start:batch_end]]
        val_prompts = [file_info[1] for file_info in process_file_list[batch_start:batch_end]]

        # Suppress output of pipeline.generate
        with contextlib.redirect_stdout(sys.stderr), contextlib.redirect_stderr(sys.stderr):
            imgs = pipeline.generate(prompt=val_prompts,
                                     n_steps=args.num_inference_steps,
                                     img_sz=args.img_sz)

        for img, img_name in zip(imgs, img_names):
            img.save(os.path.join(save_dir_src, img_name))
            img.close()
            local_generated_images += 1

        # Update progress bar only on rank 0
        if accelerator.is_main_process:
            progress_bar.update(1)

    accelerator.wait_for_everyone()
    accelerator.print(f"device:{accelerator.device}, Total generated images: {local_generated_images}")

    # Convert local_generated_images to a tensor
    local_generated_images_tensor = torch.tensor([local_generated_images], device=accelerator.device)

    # Gather the total number of generated images across all ranks
    total_generated_images_tensor = accelerator.gather(local_generated_images_tensor)
    total_generated_images = total_generated_images_tensor.sum().item()
    if accelerator.is_main_process:
        #accelerator.print(f"Total generated images: {total_generated_images}")
        progress_bar.close()
        accelerator.print(f"Image generation completed!")
        accelerator.print(f"Total generated images: {total_generated_images}")    
    accelerator.wait_for_everyone()

    change_img_count = change_img_size_ddp(save_dir_src, save_dir_tgt, args.img_resz, accelerator)
    if accelerator.is_main_process:
        accelerator.print(f"resized images saved.")
        accelerator.print(f"Total resized images: {change_img_count}")
    accelerator.wait_for_everyone()


    # Load the unseen metadata CSV
    unseen_metadata_path = os.path.join(args.valid41k_dir, 'metadata_mscoco41k_unseen.csv')
    unseen_df = pd.read_csv(unseen_metadata_path)
    unseen_files = set(unseen_df['file_name'].tolist())

    # Create directories for seen and unseen images
    save_dir_unseen = os.path.join(args.save_dir, 'unseen', f'im{args.img_resz}')
    save_dir_seen = os.path.join(args.save_dir, 'seen', f'im{args.img_resz}')
    if accelerator.is_main_process:
        os.makedirs(save_dir_unseen, exist_ok=True)
        os.makedirs(save_dir_seen, exist_ok=True)

    accelerator.wait_for_everyone()

    # Distribute image files for seen and unseen categorization among ranks
    img_list = sorted([file for file in os.listdir(save_dir_tgt) if file.endswith('.jpg')])
    total_images = len(img_list)

    images_per_process = total_images // num_processes
    remainder = total_images % num_processes

    if rank < remainder:
        start_index = rank * (images_per_process + 1)
        end_index = start_index + images_per_process + 1
    else:
        start_index = remainder * (images_per_process + 1) + (rank - remainder) * images_per_process
        end_index = start_index + images_per_process

    process_img_list = img_list[start_index:end_index]

    # Copy images to seen or unseen directories
    for filename in process_img_list:
        src_path = os.path.join(save_dir_tgt, filename)
        if filename in unseen_files:
            dst_path = os.path.join(save_dir_unseen, filename)
        else:
            dst_path = os.path.join(save_dir_seen, filename)
        shutil.copy(src_path, dst_path)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        accelerator.print(f"Images have been distributed into seen and unseen categories.")
        
    pipeline.clear()
    torch.cuda.empty_cache()
