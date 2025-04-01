GPU_NUM=0 
RANK=${RANK:-0}

SAVE_DIR=${1:-./results/generated_images}
IMG_SZ=${2:-512}
IMG_RESZ=${3:-256}
DATA_LIST=${4:-./data/mscoco_val2014_30k/metadata.csv}

echo "SAVE_DIR: $SAVE_DIR"
echo "IMG_SZ: $IMG_SZ"
echo "IMG_RESZ: $IMG_RESZ"
echo "DATA_LIST: $DATA_LIST"

# 경로 설정
IMG_PATH="$SAVE_DIR/im$IMG_RESZ"
IS_TXT="$SAVE_DIR/im${IMG_RESZ}_is.txt"
FID_TXT="$SAVE_DIR/im${IMG_RESZ}_fid.txt"
NPZ_NAME_gen="$SAVE_DIR/im${IMG_RESZ}_fid.npz"
NPZ_NAME_real="./data/mscoco_val2014_41k_full/real_im${IMG_RESZ}.npz"

# === Inception Score (IS) ===
if [ $RANK -eq 0 ]; then
    echo "=== Inception Score (IS) ==="
    fidelity --gpu $GPU_NUM --isc --input1 "$IMG_PATH" | tee "$IS_TXT"
    echo "============"
fi

# === Fréchet Inception Distance (FID) ===
if [ $RANK -eq 0 ]; then
    echo "=== Fréchet Inception Distance (FID) ==="
    CUDA_VISIBLE_DEVICES=$GPU_NUM python3 -m pytorch_fid --save-stats "$IMG_PATH" "$NPZ_NAME_gen" 2> /dev/null
    CUDA_VISIBLE_DEVICES=$GPU_NUM python3 -m pytorch_fid "$NPZ_NAME_real" "$NPZ_NAME_gen" 2> /dev/null | tee "$FID_TXT"
    echo "============"
fi

# === CLIP Score ===
# CLIP_TXT="$SAVE_DIR/im${IMG_RESZ}_clip.txt"
# echo "=== CLIP Score ==="
# CUDA_VISIBLE_DEVICES=$GPU_NUM python3 src/eval_clip_score_ddp.py --img_dir "$IMG_PATH" --save_txt "$CLIP_TXT" --data_list "$DATA_LIST" 2> /dev/null
# echo "============"
