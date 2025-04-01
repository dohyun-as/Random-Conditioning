#!/bin/bash

GPU_NUM=0  # 평가 작업은 메인 GPU만 사용

# 인자로부터 변수 설정
SAVE_DIR=${1:-./results/generated_images}
IMG_SZ=${2:-512}
IMG_RESZ=${3:-256}
VALID41k_DIR=${4:-./data}

# MODEL_ID 설정
MODEL_ID=generated_images

# 경로 설정 (seen과 unseen에 대한 경로)
IMG_PATH_SEEN=$SAVE_DIR/seen/im$IMG_RESZ
IMG_PATH_UNSEEN=$SAVE_DIR/unseen/im$IMG_RESZ
IMG_PATH_ALL=$SAVE_DIR/all/im$IMG_RESZ

# === Inception Score (IS) ===
IS_TXT_SEEN=$SAVE_DIR/seen/im${IMG_RESZ}_is.txt
IS_TXT_UNSEEN=$SAVE_DIR/unseen/im${IMG_RESZ}_is.txt
IS_TXT_ALL=$SAVE_DIR/all/im${IMG_RESZ}_is.txt

# === Fréchet Inception Distance (FID) ===
FID_TXT_SEEN=$SAVE_DIR/seen/im${IMG_RESZ}_fid.txt
FID_TXT_UNSEEN=$SAVE_DIR/unseen/im${IMG_RESZ}_fid.txt
FID_TXT_ALL=$SAVE_DIR/all/im${IMG_RESZ}_fid.txt

NPZ_NAME_GT_SEEN="$VALID41k_DIR/real_im${IMG_RESZ}_seen.npz"
NPZ_NAME_GT_UNSEEN="$VALID41k_DIR/real_im${IMG_RESZ}_unseen.npz"
NPZ_NAME_GT_ALL="$VALID41k_DIR/real_im${IMG_RESZ}_all.npz"

NPZ_NAME_GEN_SEEN=$SAVE_DIR/seen/im${IMG_RESZ}_fid.npz
NPZ_NAME_GEN_UNSEEN=$SAVE_DIR/unseen/im${IMG_RESZ}_fid.npz
NPZ_NAME_GEN_ALL=$SAVE_DIR/all/im${IMG_RESZ}_fid.npz

RANK=${RANK:-0}

if [ $RANK -eq 0 ]; then
    # Inception Score for seen
    echo "=== Inception Score (IS) for SEEN ==="
    fidelity --gpu $GPU_NUM --isc --input1 $IMG_PATH_SEEN | tee $IS_TXT_SEEN
    echo "============"

    # Inception Score for unseen
    echo "=== Inception Score (IS) for UNSEEN ==="
    fidelity --gpu $GPU_NUM --isc --input1 $IMG_PATH_UNSEEN | tee $IS_TXT_UNSEEN
    echo "============"

    # Inception Score for all
    echo "=== Inception Score (IS) for ALL ==="
    fidelity --gpu $GPU_NUM --isc --input1 $IMG_PATH_ALL | tee $IS_TXT_ALL
    echo "============"
fi

if [ $RANK -eq 0 ]; then
    # Fréchet Inception Distance for seen
    echo "=== Fréchet Inception Distance (FID) for SEEN ==="
    CUDA_VISIBLE_DEVICES=$GPU_NUM python3 -m pytorch_fid --save-stats $IMG_PATH_SEEN $NPZ_NAME_GEN_SEEN 2> /dev/null
    CUDA_VISIBLE_DEVICES=$GPU_NUM python3 -m pytorch_fid $NPZ_NAME_GT_SEEN $NPZ_NAME_GEN_SEEN 2> /dev/null | tee $FID_TXT_SEEN
    echo "============"

    # Fréchet Inception Distance for unseen
    echo "=== Fréchet Inception Distance (FID) for UNSEEN ==="
    CUDA_VISIBLE_DEVICES=$GPU_NUM python3 -m pytorch_fid --save-stats $IMG_PATH_UNSEEN $NPZ_NAME_GEN_UNSEEN 2> /dev/null
    CUDA_VISIBLE_DEVICES=$GPU_NUM python3 -m pytorch_fid $NPZ_NAME_GT_UNSEEN $NPZ_NAME_GEN_UNSEEN 2> /dev/null | tee $FID_TXT_UNSEEN
    echo "============"

    # Fréchet Inception Distance for all
    echo "=== Fréchet Inception Distance (FID) for ALL ==="
    CUDA_VISIBLE_DEVICES=$GPU_NUM python3 -m pytorch_fid --save-stats $IMG_PATH_ALL $NPZ_NAME_GEN_ALL 2> /dev/null
    CUDA_VISIBLE_DEVICES=$GPU_NUM python3 -m pytorch_fid $NPZ_NAME_GT_ALL $NPZ_NAME_GEN_ALL 2> /dev/null | tee $FID_TXT_ALL
    echo "============"
fi
