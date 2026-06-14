#!/bin/bash

# ========== 全局参数设置 ==========
INPUT_VIDEO="ditto_mini_test_videos/mini_test_videos/style_004.mp4"
OUTPUT_DIR="results_local_revisit"
LORA_PATH="/mnt/public/users/yangshuai/code/CausalWorld/outputs_train_14b_lora/ditto_local/step-16600.safetensors"
NUM_FRAMES=73

PROMPTS=(
"Change all green trees to autumn orange with red tips."
)

# PROMPTS=(
# "Change all green trees to autumn orange with red tips."
# "Replace the current green trees with weeping willow trees, keeping their position and size."
# "Swap the bright green leaves for deep forest green, making the trees look denser and darker."
# "Change the tree foliage from green to golden yellow, as if it’s peak autumn."
# "Replace the round-canopy trees with tall, thin pine trees, maintaining the same height and location."
# "Make the trees’ leaves a mix of orange and purple, giving them a surreal, cartoonish fall look."
# "Change the current trees to cherry blossom trees with light pink flowers, keeping the trunk and branches."
# "Replace all trees with bare-branched winter trees, no leaves, just dark gray branches against the sky."
# )

# ========== 自动分批执行 ==========
TOTAL_PROMPTS=${#PROMPTS[@]}
BATCH_SIZE=8
START_IDX=0

while [ $START_IDX -lt $TOTAL_PROMPTS ]; do
    END_IDX=$((START_IDX + BATCH_SIZE))
    if [ $END_IDX -gt $TOTAL_PROMPTS ]; then
        END_IDX=$TOTAL_PROMPTS
    fi

    echo "=== Processing prompts $START_IDX to $((END_IDX - 1)) ==="

    for i in $(seq $START_IDX $((END_IDX - 1))); do
        DEVICE_ID=$((i % 8))  # 循环使用 device_id 0-7
        PROMPT="${PROMPTS[$i]}"

        # # 生成输出文件名：基于 prompt 简化命名（移除空格、标点，保留关键词）
        # OUTPUT_NAME=$(echo "$PROMPT" | sed 's/[^a-zA-Z0-9]/_/g' | tr '[:upper:]' '[:lower:]' | cut -c1-60)
        # OUTPUT_FILE="$OUTPUT_DIR/${OUTPUT_NAME}.mp4"

        echo "Running on device $DEVICE_ID: $PROMPT"

        python inference/infer_ditto.py \
            --lora_path "$LORA_PATH" \
            --num_frames $NUM_FRAMES \
            --device_id $DEVICE_ID \
            --input_video "$INPUT_VIDEO" \
            --output_dir "$OUTPUT_DIR" \
            --prompt "$PROMPT" &

        sleep 0.5
    done

    wait  # 等待当前批次全部完成再进入下一批

    START_IDX=$END_IDX
done

echo "✅ All prompts processed successfully!"