import os
import math
import torch
import pandas as pd
from torch.utils.data import Dataset
from safetensors.torch import load_file
import random
import numpy as np

class rand_cond_dataset(Dataset):
    def __init__(self, data_dir, extra_text_dir=None, n_T=1000, random_conditioning = False, 
                 random_conditioning_lambda=5, world_size=1, rank=0, drop_text=True, drop_text_p=0.1, 
                 use_unseen_setting=False, gpt_caption=False, max_extra_text_samples=None, safe_tensor=None,
                 sigmoid = False):
        """
        Args:
            data_dir (str): 데이터가 저장된 폴더의 경로.
        """
        self.data_dir = data_dir
        self.n_T = n_T
        self.random_conditioning = random_conditioning
        self.random_conditioning_lambda = random_conditioning_lambda
        self.world_size = world_size
        self.rank = rank
        
        self.drop_text = drop_text
        self.drop_text_p = drop_text_p
        self.gpt_caption = gpt_caption
        self.max_extra_text_samples = max_extra_text_samples
        self.safe_tensor = safe_tensor
        self.sigmoid = sigmoid
        
        # Select metadata - Using unseen setting or not
        
        if use_unseen_setting:
            seen_metadata_path = os.path.join(data_dir, "metadata_seen_CVPR.csv") #exclude animal related texts from laion 212k dataset(metadata.csv)
            print(f"Using data dir in {seen_metadata_path}!!!!!")
            self.metadata = pd.read_csv(seen_metadata_path)
            
            
            all_metadata_path = os.path.join(data_dir, "metadata.csv") #laion 212k dataset
            all_metadata = pd.read_csv(all_metadata_path)
                    
            self.text_data = self._load_extra_text_data(all_metadata, extra_text_dir)
        else:
            metadata_path = os.path.join(data_dir, "metadata.csv") #laion 212k dataset
            print(f"Using data dir in {metadata_path}!!!!!")
            self.metadata = pd.read_csv(metadata_path)
            
            self.text_data = self._load_extra_text_data(self.metadata, extra_text_dir)

    def _load_extra_text_data(self, metameta, extra_text_dir):
        """
        추가 텍스트 데이터를 로드하고, 메타데이터의 텍스트와 결합하여 반환합니다.

        Args:
            extra_text_dir (str): 추가 텍스트 데이터가 저장된 폴더의 경로.

        Returns:
            pandas.Series: 모든 텍스트 데이터가 포함된 시리즈.
        """
        # 메타데이터의 텍스트를 먼저 가져옵니다.
        text_data_list = [metameta['text'][self.rank::self.world_size]]
        if self.gpt_caption:
            gpt_caption_path = os.path.join(self.data_dir, "gpt_caption.csv")
            gpt_caption = pd.read_csv(gpt_caption_path)
            text_data_list.append(gpt_caption['prompt'][self.rank::self.world_size].dropna())
        
        else:
            if extra_text_dir is not None:
                extra_texts = []
                # extra_text_dir 내의 모든 Parquet 파일 목록을 가져옵니다.
                parquet_files = [os.path.join(extra_text_dir, f) for f in os.listdir(extra_text_dir) if f.endswith('.parquet')]

                rank_files = parquet_files[self.rank::self.world_size]
                
                for pq_file in rank_files:
                    # Parquet 파일 로드
                    try:
                        df = pd.read_parquet(pq_file)
                        print(f"read parquet {pq_file}")
                        # 'text' 또는 'TEXT' 컬럼 확인
                        if 'text' in df.columns:
                            text_column = 'text'
                        elif 'TEXT' in df.columns:
                            text_column = 'TEXT'
                        else:
                            print(f"파일 {pq_file}에 'text' 컬럼이 없습니다. 건너뜁니다.")
                            continue
                        
                        cleaned_text_data = df[text_column].dropna()  # NaN 또는 None 값을 제거
                        extra_texts.append(cleaned_text_data)

                    except Exception as e:
                        print(f"파일 {pq_file}를 로드하는 중 에러 발생: {e}")
                        continue
                
                # extra_texts를 하나의 Series로 합침
                if extra_texts:
                    extra_texts_combined = pd.concat(extra_texts, ignore_index=True)
                else:
                    extra_texts_combined = pd.Series()

                # 텍스트 샘플의 개수를 제한
                if self.max_extra_text_samples is not None:
                    extra_texts_combined = extra_texts_combined.sample(
                        n=min(int(self.max_extra_text_samples/self.world_size), len(extra_texts_combined)),
                        random_state=42
                    )

                print("extra_text:", len(extra_texts_combined))
                # text_data_list에 추가
                text_data_list.append(extra_texts_combined)

        # 모든 텍스트 데이터를 하나의 시리즈로 결합
        combined_text_data = pd.concat(text_data_list, ignore_index=True)
        print("total_text:",len(combined_text_data))
        return combined_text_data
        

    def __len__(self):
        # 유효한 인덱스 개수를 반환합니다.+
        return len(self.metadata)

    def __getitem__(self, idx):
        # 인덱스에 해당하는 데이터 파일들을 로드하여 반환합니다.
        metadata_row = self.metadata.iloc[idx]
        file_name = metadata_row['file_name']
        text = metadata_row['text']

        if self.safe_tensor:
            latent_file_name = file_name.replace('.jpg', '_latent.safetensors')
            latent_path = os.path.join(self.data_dir, latent_file_name)
            data = load_file(latent_path)
            latent_tensor = data["tensor"]

        else:
            # Modify the file_name to get latent file name
            latent_file_name = file_name.replace('.jpg', '_latent.pt')
            latent_path = os.path.join(self.data_dir, latent_file_name)
            
            latent_tensor = torch.load(latent_path, map_location=torch.device('cpu'))

        timestep = torch.randint(0, self.n_T, (1,)).long()        
        paired = torch.tensor(1).unsqueeze(0)
        if self.random_conditioning:
            t_value = timestep.item()
            if self.sigmoid:
                p = 1 / (1 + math.exp(-20 * (t_value / self.n_T - 0.7)))
            else:
                p = math.exp(-self.random_conditioning_lambda * (1 - t_value / self.n_T))
            # p = 1 / (1 + math.exp(-20 * (t_value / self.n_T - 0.7)))
            # if t_value>self.n_T/2:
            #     p = math.exp(-self.random_conditioning_lambda * (1 - t_value / self.n_T))
            # else:
            #     p = math.exp(-self.random_conditioning_lambda * (t_value / self.n_T))
            # p = (2 * (t_value - self.n_T / 2) / self.n_T) ** 2
            # p=0.5
            if torch.rand(1).item() < p:
                #rand_index = torch.randint(0, len(self.data_indices), (1,)).item()
                random_idx = torch.randint(0, len(self.text_data), (1,)).item()
                text = self.text_data[random_idx]
                paired = torch.tensor(0).unsqueeze(0)
        
        if self.drop_text:
            if random.random() < self.drop_text_p:  # 10% 확률
                text = ""                


        return latent_tensor, text, timestep, paired


def collate_fn(tokenizer, tokenizer_2=None):
    def collate(batch):
        latents, texts, timesteps, paireds = zip(*batch)
        
        latent_tensors = torch.stack(latents)
        timesteps = torch.cat(timesteps)
        paireds = torch.cat(paireds)

        captions = []
        for caption in texts:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption))
            else:
                raise ValueError(
                    f"Caption column `{caption}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        
        input_ids = inputs.input_ids

        if tokenizer_2 is not None:
            inputs_2 = tokenizer(
                captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            )      
            input_ids_2 = inputs_2.input_ids
        else:
            input_ids_2=None
        
        return {
            "latents": latent_tensors,
            "input_ids": input_ids,
            "input_ids_2": input_ids_2,
            "timesteps": timesteps,
            "paireds":paireds
        }
    return collate