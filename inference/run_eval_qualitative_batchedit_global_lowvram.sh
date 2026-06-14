#!/bin/bash
set -e

######################### 环境设置（按需改） #########################
# 强制使用 prismworld 环境的 python（避免被系统 miniforge 的 python3.12 抢占 PATH）
PRISMWORLD_ENV="/mnt/cpfs/yangshuai/miniconda3/envs/prismworld"
export PATH="${PRISMWORLD_ENV}/bin:$PATH"
PYTHON="${PRISMWORLD_ENV}/bin/python"

export PYTHONPATH="/mnt/cpfs/yangshuai/code/PermaVid/prismworld:$PYTHONPATH"

cd /mnt/cpfs/yangshuai/code/PermaVid/prismworld

# 启动前打印实际使用的 python，便于排查环境问题
echo "🐍 使用的 Python: $("$PYTHON" -c 'import sys; print(sys.executable)')"

# ========== 固定配置 ==========
DEVICE_COUNT=4
overlap_threshold=0.4
reference_nums=10

MODEL_DIR="/mnt/cpfs/yangshuai/code/PermaVid/prismworld/models"

MODEL_ID="${MODEL_DIR}/Wan2.1-VACE-14B"
LOADCKPT_PATH="${MODEL_DIR}/prismworld/full/camera_memory/mix/mixref_14b_step-5800.safetensors"
TOKENIZER_PATH="${MODEL_DIR}/Wan2.1-VACE-14B/google/umt5-xxl"

# Interactive editing / Auto-suggestion model (modify according to actual path)
QWEN_EDIT_PATH="${MODEL_DIR}/Qwen/Qwen-Image-Edit"
QWEN_VL_PATH="${MODEL_DIR}/Qwen/Qwen3-VL-8B-Instruct"

images_dir="/mnt/cpfs/yangshuai/code/PermaVid/prismworld/testset"


prompt_list=(
    "change the time to evening"
    "change the weather to sunny"
)


# 读取所有 .png 文件，按文件名排序（只需读一次）
mapfile -t all_images < <(find "$images_dir" -maxdepth 1 -type f -name "*.png" | sort -V)
total_images=${#all_images[@]}

echo "🔧 全局基础信息:"
echo "   - GPU 数量: $DEVICE_COUNT"
echo "   - 总图片数: $total_images"
echo "   - 模型ID: $MODEL_ID"
echo "   - 检查点路径: $LOADCKPT_PATH"
echo "   - 图片目录: $images_dir"
echo "   - 共 ${#prompt_list[@]} 个 prompts"
echo

# 外层循环：遍历每个 prompt
for raw_prompt in "${prompt_list[@]}"; do
    # 跳过空行（以防末尾有空行）
    [[ -z "$raw_prompt" ]] && continue

    # 构造 preset_actions：固定前缀 + 当前 prompt
    # preset_actions="left-49-1.0, right-49-1.0, right-49-1.0, edit: $raw_prompt, left-49-1.0, left-49-1.0"
    preset_actions="w-49-0.05, s-49-0.05, left-49-1.0, right-49-1.0, right-49-1.0, edit: $raw_prompt, left-49-1.0, left-49-1.0"


    # 构造 output_dir 名称：替换空格和标点为下划线，只保留字母数字和下划线
    sanitized_prompt=$(echo "$raw_prompt" | tr '[:space:][:punct:]' '_' | tr -s '_' | sed 's/^_//;s/_$//')
    OUTPUT_DIR="results_batch/Qualitative_14B_mixmem_lowvram_${sanitized_prompt}"
    mkdir -p "$OUTPUT_DIR"

    echo "========================================"
    echo "🚀 开始处理 Prompt: $raw_prompt"
    echo "📁 输出目录: $OUTPUT_DIR"
    echo "========================================"

    declare -a device_groups
    for ((i=0; i<DEVICE_COUNT; i++)); do
        device_groups[i]=""
    done

    for idx in "${!all_images[@]}"; do
        device_id=$((idx % DEVICE_COUNT))
        image_path="${all_images[$idx]}"
        device_groups[$device_id]="${device_groups[$device_id]}$image_path"$'\n'
    done


    for ((dev=0; dev<DEVICE_COUNT; dev++)); do
        (
            mapfile -t images_for_dev <<< "${device_groups[$dev]}"
            if [[ ${#images_for_dev[@]} -eq 0 || -z "${images_for_dev[0]}" ]]; then
                exit 0
            fi

            echo "[Device $dev] 负责处理 ${#images_for_dev[@]} 张图片"

            for image_path in "${images_for_dev[@]}"; do
                [[ -z "$image_path" ]] && continue
                echo "🖼️ [Device $dev] 处理: $(basename "$image_path")"
                "$PYTHON" inference/infer_mem_keyboard_stream_batch_global_lowvram.py \
                    --device_id "$dev" \
                    --model_id "$MODEL_ID" \
                    --loadckpt_path "$LOADCKPT_PATH" \
                    --tokenizer_path "$TOKENIZER_PATH" \
                    --qwen_edit_path "$QWEN_EDIT_PATH" \
                    --qwen_vl_path "$QWEN_VL_PATH" \
                    --num_frames 49 \
                    --reference_nums "$reference_nums" \
                    --camera_speed 0.1 \
                    --overlap_threshold "$overlap_threshold" \
                    --output_dir "$OUTPUT_DIR" \
                    --input_image "$image_path" \
                    --preset_actions "$preset_actions" \
                    --auto_prompt
            done
        ) &
    done

    # 等待当前 prompt 的所有设备任务完成
    wait

    echo "✅ Prompt '$raw_prompt' 完成。"
    echo
done
sleep inf
echo "🎉 所有 prompts 处理完毕！"


