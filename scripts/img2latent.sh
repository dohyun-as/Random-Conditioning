CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)

# 사용 가능한 GPU 개수 자동으로 감지
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [ ${NUM_GPUS} -gt 1 ]; then
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --multi_gpu --num_processes ${NUM_GPUS} src/create_pt.py \
  --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4"
else
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch src/create_pt.py \
  --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4"
fi