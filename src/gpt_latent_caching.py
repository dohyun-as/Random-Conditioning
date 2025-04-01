import os
import pandas as pd
import torch
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

class CsvDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = self.data.iloc[idx]["prompt"]
        file_name = f"{idx}"
        return prompt, file_name

def generate_latents(pipeline, dataloader, save_dir, metadata_file, device):
    generator = torch.Generator(device=device).manual_seed(1234)
    metadata = []  

    for batch in tqdm(dataloader, desc="generating latents"):
        prompts, file_names = batch
        latents = pipeline(
            list(prompts),
            num_inference_steps=25,
            generator=generator,
            output_type="latent"
        ).images
        for latent, file_name, prompt in zip(latents, file_names, prompts):
            latent_file_name = f"{file_name}_latent.pt"
            torch.save(latent, os.path.join(save_dir, latent_file_name))
            metadata.append({
                "file_name": f"{file_name}.jpg",
                "text": prompt
            })

    df = pd.DataFrame(metadata)
    df.to_csv(metadata_file, index=False)

def main():
    accelerator = Accelerator() 
    device = accelerator.device

    csv_file = "./data/laion_aes/gpt_latent_212k/212k_gpt_caption.csv"
    save_dir = "./data/laion_aes/gpt_latent_212k/latents"  
    metadata_file = "./data/laion_aes/gpt_latent_212k/gpt_metadata.csv"  
    os.makedirs(save_dir, exist_ok=True)

    dataset = CsvDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    pipeline = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        safety_checker=None,
    )
    pipeline = pipeline.to(device)

    pipeline, dataloader = accelerator.prepare(pipeline, dataloader)

    generate_latents(pipeline, dataloader, save_dir, metadata_file, device)

if __name__ == "__main__":
    main()