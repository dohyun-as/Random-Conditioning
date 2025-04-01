CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [ ${NUM_GPUS} -gt 1 ]; then
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --multi_gpu --num_processes ${NUM_GPUS} src/std_latent_caching.py
else
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch src/std_latent_caching.py
fi