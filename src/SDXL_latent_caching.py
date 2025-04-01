import os
import pandas as pd
import torch
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, EulerDiscreteScheduler, UNet2DConditionModel, AutoPipelineForText2Image
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
    save_dir = "./data/laion_aes/SDXL_latent_212k/SDXL_base_latents"
    metadata_file = "./data/laion_aes/SDXL_latent_212k/metadata.csv"
    os.makedirs(save_dir, exist_ok=True)
    dataset = CsvDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # pipeline = StableDiffusionPipeline.from_pretrained(
    #     "CompVis/stable-diffusion-v1-4",
    #     safety_checker=None,
    # )
    # pipeline = pipeline.to(device)
    # pipe = StableDiffusionXLPipeline.from_pretrained("etri-vilab/koala-lightning-1b", torch_dtype=torch.float16)
    # pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    # pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16")
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_8step_unet.safetensors" # Use the correct ckpt for your step setting!
    # Load model.
    unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
    # unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
    # pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16")
    pipe = StableDiffusionXLPipeline.from_pretrained(base, torch_dtype=torch.float16, variant="fp16")
    # Ensure sampler uses "trailing" timesteps and "sample" prediction type.
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    # config 확인
    print(pipe.unet.config)
        
    
    pipe.to(device)

    
    dataloader = accelerator.prepare(dataloader)
    generate_images(pipe, dataloader, save_dir, metadata_file, device, accelerator)
if __name__ == "__main__":
    main()