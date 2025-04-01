import os
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="CompVis/stable-diffusion-v1-4",
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--revision", type=str, default=None, required=False,
                        help="Revision of pretrained model identifier from huggingface.co/models.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Truncate the number of training examples.")
    parser.add_argument("--output_dir", type=str, default="/workspace/BK-SDM/data/laion_aes/GT_latent_212k",
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--seed", type=int, default=1234, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512, help="The resolution for input images.")
    parser.add_argument("--center_crop", action="store_true", help="Whether to center crop the input images.")
    parser.add_argument("--random_flip", action="store_true", help="Whether to randomly flip images horizontally.")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Batch size (per device).")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of workers for DataLoader.")
    args = parser.parse_args()
    print('args')
    return args

def main():
    args = parse_args()
    
    # Set up Accelerator
    accelerator = Accelerator()
    device = accelerator.device

    # Set random seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    
    print('seed')
    # Load models
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)

    print('models')
    # Image preprocessing
    image_transforms = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
        transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # Dataset definition
    class ImageTextDataset(Dataset):
        def __init__(self, text_dir, image_dir):
            self.text_dir = text_dir
            self.image_dir = image_dir
            self.file_names = [f.split(".")[0] for f in os.listdir(text_dir) if f.endswith(".txt")]

        def __len__(self):
            return len(self.file_names)

        def __getitem__(self, idx):
            base_name = self.file_names[idx]
            text_path = os.path.join(self.text_dir, f"{base_name}.txt")
            image_path = os.path.join(self.image_dir, f"{base_name}.jpg")

            with open(text_path, "r") as f:
                text = f.read().strip()

            image = None
            if os.path.exists(image_path):
                image = Image.open(image_path).convert("RGB")
                image = image_transforms(image)  # 이미지 전처리

            return text, image, base_name

    # DataLoader 설정
    text_dir = "/workspace/BK-SDM/data/laion_aes/preprocessed_212k/train"
    img_dir = "/workspace/BK-SDM/data/laion_aes/preprocessed_212k/train"
    
    
    print('before dataset')
    
    dataset = ImageTextDataset(text_dir, img_dir)
    
    print('dataset')

    train_dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=lambda x: list(zip(*x))
    )

    # Prepare models for distributed training
    train_dataloader, text_encoder, vae = accelerator.prepare(train_dataloader, text_encoder, vae)
     
    print('accelerator prepare')
    def process_batch(batch):
        texts, images, base_names = batch
        
        # 텍스트 임베딩 생성
        inputs = tokenizer(list(texts), return_tensors="pt", padding="max_length", truncation=True).to(device)
        with torch.no_grad():
            encoder_hidden_states = text_encoder.module(inputs.input_ids)[0] if isinstance(text_encoder, torch.nn.parallel.DistributedDataParallel) else text_encoder(inputs.input_ids)[0]  # [B, 77, 768] 형태

        # 이미지를 배치 단위로 처리
        if images:
            image_tensors = torch.stack(images).to(device)  # [B, C, H, W] 형태
            with torch.no_grad():
                latents = vae.module.encode(image_tensors).latent_dist.sample() if isinstance(vae, torch.nn.parallel.DistributedDataParallel) else vae.encode(image_tensors).latent_dist.sample()
                latents = latents * vae.module.config.scaling_factor  # Stable Diffusion의 scaling factor
        else:
            latents = None

        return latents, encoder_hidden_states, base_names
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 배치 단위로 데이터 처리
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        latents, text_embs, base_names = process_batch(batch)

        # 결과 저장
        for latent, text_emb, base_name in zip(latents, text_embs, base_names):
            if latent is not None:
                latent_path = os.path.join(args.output_dir, f"{base_name}_latent.pt")
                torch.save(latent.cpu(), latent_path)
                print('save',latent_path)
        

if __name__ == "__main__":
    main()