import os
import pandas as pd
import torch
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
class CsvDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        prompt = self.data.iloc[idx]["text"]
        file_name=self.data.iloc[idx]['file_name']
        return prompt, file_name
def generate_images(pipeline, dataloader, save_dir, metadata_file, device, accelerator):
    generator = torch.Generator(device=device).manual_seed(1234)
    metadata = []
    for i, batch in enumerate(tqdm(dataloader, desc="generating images")):
        prompts, file_names = batch
        latents = pipeline(
            list(prompts),
            num_inference_steps=25,
            guidance_scale=7.5,
            generator=generator,
            output_type="latent"
        ).images
        for latent, file_name, prompt in zip(latents, file_names, prompts):
            latent_file_name=file_name.replace('.jpg', '_latent.pt')
            torch.save(latent, os.path.join(save_dir, latent_file_name))
            metadata.append({
                "file_name": file_name,
                "text": prompt
            })
        accelerator.wait_for_everyone()     
    accelerator.wait_for_everyone()       
    all_metadata = accelerator.gather(metadata)
    if accelerator.is_local_main_process:
        # Flatten the list of metadata
        flattened_metadata = []
        for md in all_metadata:
            if isinstance(md, list):
                flattened_metadata.extend(md)
            else:
                flattened_metadata.append(md)
        df = pd.DataFrame(flattened_metadata)
        df.to_csv(metadata_file, index=False)

def main():
    accelerator = Accelerator()
    device = accelerator.device
    csv_file = "./data/laion_aes/latent_212k/metadata.csv"
    save_dir = "./data/laion_aes/latent_212k"
    metadata_file = "./data/laion_aes/test_latents/metadata.csv"
    os.makedirs(save_dir, exist_ok=True)
    dataset = CsvDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    base = "CompVis/stable-diffusion-v1-4"
    # Load model.
    pipe = StableDiffusionPipeline.from_pretrained(base, torch_dtype=torch.float16).to(device)
    pipe.safety_checker = lambda images, clip_input: (images, False) 
    # pipe.enable_vae_slicing()
        
    
    pipe.to(device)

    
    dataloader = accelerator.prepare(dataloader)
    generate_images(pipe, dataloader, save_dir, metadata_file, device, accelerator)
if __name__ == "__main__":
    main()