# ------------------------------------------------------------------------------------
# Copyright 2023. Nota Inc. All Rights Reserved.
# Code modified from https://github.com/huggingface/diffusers/tree/v0.15.0/examples/text_to_image
# ------------------------------------------------------------------------------------

NUM_GPUS=4

# 여러 개의 unet_path를 쉼표로 구분하여 전달 (예: "path1,path2,path3")
unet_paths="/home/work/StableDiffusion/T2I_400M_uncond/results/toy_ddp_bk_base/checkpoint-25000,\
/home/work/StableDiffusion/T2I_400M_uncond/results/toy_ddp_bk_base/checkpoint-50000,\
/home/work/StableDiffusion/T2I_400M_uncond/results/toy_ddp_bk_base/checkpoint-75000,\
/home/work/StableDiffusion/T2I_400M_uncond/results/toy_ddp_bk_base/checkpoint-100000,\
/home/work/StableDiffusion/T2I_400M/results/toy_ddp_bk_base/checkpoint-25000,\
/home/work/StableDiffusion/T2I_400M/results/toy_ddp_bk_base/checkpoint-50000,\
/home/work/StableDiffusion/T2I_400M/results/toy_ddp_bk_base/checkpoint-75000,\
/home/work/StableDiffusion/T2I_400M/results/toy_ddp_bk_base/checkpoint-100000,\
/home/work/StableDiffusion/T2I_400M/results/toy_ddp_bk_base/checkpoint-125000,\
/home/work/StableDiffusion/T2I_distill1_GPU4/results/toy_ddp_bk_base/checkpoint-25000,\
/home/work/StableDiffusion/T2I_distill1_GPU4/results/toy_ddp_bk_base/checkpoint-50000"


# 쉼표로 구분된 문자열을 배열로 변환
IFS=',' read -r -a unet_path_array <<< "$unet_paths"

StartTime=$(date +%s)

# 배열에 있는 각 unet_path에 대해 명령 실행
for unet_path in "${unet_path_array[@]}"; do
  echo "Running with UNet path: ${unet_path}"
  CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes ${NUM_GPUS} src/eval.py \
    --unet_path ${unet_path}
done

EndTime=$(date +%s)
echo "** KD training takes $(($EndTime - $StartTime)) seconds."