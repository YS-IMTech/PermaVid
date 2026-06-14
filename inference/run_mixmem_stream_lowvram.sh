#!/bin/bash

######################### 环境设置（按需改） #########################
# 强制使用 prismworld 环境的 python（避免被系统 miniforge 的 python3.12 抢占 PATH）
PRISMWORLD_ENV="/mnt/cpfs/yangshuai/miniconda3/envs/prismworld"
export PATH="${PRISMWORLD_ENV}/bin:$PATH"
PYTHON="${PRISMWORLD_ENV}/bin/python"

export PYTHONPATH="/mnt/cpfs/yangshuai/code/PermaVid/prismworld:$PYTHONPATH"

cd /mnt/cpfs/yangshuai/code/PermaVid/prismworld

# 启动前打印实际使用的 python，便于排查环境问题
echo "🐍 使用的 Python: $("$PYTHON" -c 'import sys; print(sys.executable)')"

DEVICE_COUNT=4
overlap_threshold=0.4
reference_nums=10

MODEL_DIR="/mnt/cpfs/yangshuai/code/PermaVid/prismworld/models"

MODEL_ID="${MODEL_DIR}/Wan2.1-VACE-14B"
LOADCKPT_PATH="${MODEL_DIR}/prismworld/full/camera_memory/mix/mixref_14b_step-5800.safetensors"
TOKENIZER_PATH="${MODEL_DIR}/Wan2.1-VACE-14B/google/umt5-xxl"
OUTPUT_DIR="results_test/streaming_ue14b_mixref_memory_step5800_onlydit_v1_lowvram"

# Interactive editing / Auto-suggestion model (modify according to actual path)
QWEN_EDIT_PATH="${MODEL_DIR}/Qwen/Qwen-Image-Edit"
QWEN_VL_PATH="${MODEL_DIR}/Qwen/Qwen3-VL-8B-Instruct"

mkdir -p "$OUTPUT_DIR"

image="datasets/test_00.png"

echo "🚀 \"$image\" "
"$PYTHON" inference/infer_mem_keyboard_stream_lowvram.py \
    --device_id 2 \
    --model_id "$MODEL_ID" \
    --loadckpt_path "$LOADCKPT_PATH" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --qwen_edit_path "$QWEN_EDIT_PATH" \
    --qwen_vl_path "$QWEN_VL_PATH" \
    --num_frames 49 \
    --reference_nums $reference_nums \
    --camera_speed 0.1 \
    --overlap_threshold $overlap_threshold \
    --output_dir "$OUTPUT_DIR" \
    --input_image "$image"
