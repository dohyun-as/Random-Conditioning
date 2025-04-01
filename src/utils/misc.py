# ------------------------------------------------------------------------------------
# Copyright 2023. Nota Inc. All Rights Reserved.
# ------------------------------------------------------------------------------------

import csv
import os
from PIL import Image
import torch

def get_file_list_from_csv(csv_file_path):
    file_list = []
    with open(csv_file_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)        
        next(csv_reader, None) # Skip the header row
        for row in csv_reader: # (row[0], row[1]) = (img name, txt prompt) 
            file_list.append(row)
    return file_list

def change_img_size(input_folder, output_folder, resz=256):
    img_list = sorted([file for file in os.listdir(input_folder) if file.endswith('.jpg')])
    for i, filename in enumerate(img_list):
        img = Image.open(os.path.join(input_folder, filename))
        img.resize((resz, resz)).save(os.path.join(output_folder, filename))
        img.close()
        if i % 2000 == 0:
            print(f"{i}/{len(img_list)} | {filename}: resize to {resz}")

def change_img_size_ddp(input_folder, output_folder, resz, accelerator):
    img_list = sorted([file for file in os.listdir(input_folder) if file.endswith('.jpg')])
    
    # Distribute image list among ranks
    total_images = len(img_list)
    num_processes = accelerator.num_processes
    rank = accelerator.process_index

    images_per_process = total_images // num_processes
    remainder = total_images % num_processes

    if rank < remainder:
        start_index = rank * (images_per_process + 1)
        end_index = start_index + images_per_process + 1
    else:
        start_index = remainder * (images_per_process + 1) + (rank - remainder) * images_per_process
        end_index = start_index + images_per_process

    process_img_list = img_list[start_index:end_index]

    local_change_count = 0

    # Resize images assigned to this process
    for i, filename in enumerate(process_img_list):
        img = Image.open(os.path.join(input_folder, filename))
        img.resize((resz, resz)).save(os.path.join(output_folder, filename))
        img.close()
        local_change_count += 1
        
        if i % 1000 == 0:
            accelerator.print(f"Rank {rank}: {i}/{len(process_img_list)} | {filename}: resized to {resz}")

    accelerator.wait_for_everyone()
    # Convert local_change_count to a tensor
    local_change_count_tensor = torch.tensor([local_change_count], device=accelerator.device)

    # Gather the total number of resized images across all ranks
    total_change_count_tensor = accelerator.gather(local_change_count_tensor)

    # Sum the gathered tensors and convert it to a Python integer
    total_change_count = total_change_count_tensor.sum().item()
    if accelerator.is_main_process:
        accelerator.print(f"Total images resized: {total_change_count}")
    accelerator.wait_for_everyone()
    return total_change_count
