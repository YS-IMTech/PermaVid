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

QWEN_EDIT_PATH="${MODEL_DIR}/Qwen/Qwen-Image-Edit"
QWEN_VL_PATH="${MODEL_DIR}/Qwen/Qwen3-VL-8B-Instruct"

images_dir="/mnt/cpfs/yangshuai/code/PermaVid/prismworld/testset"
local_prompts_dir="/mnt/cpfs/yangshuai/code/PermaVid/prismworld/testset/testset_local_prompts"

# 读取所有 .png 文件，按文件名排序（只需读一次）
mapfile -t all_images < <(find "$images_dir" -maxdepth 1 -type f -name "*.png" | sort -V)
total_images=${#all_images[@]}

echo "🔧 全局基础信息:"
echo "   - GPU 数量: $DEVICE_COUNT"
echo "   - 总图片数: $total_images"
echo "   - 模型ID: $MODEL_ID"
echo "   - 检查点路径: $LOADCKPT_PATH"
echo "   - 图片目录: $images_dir"
echo "   - Local prompts 目录: $local_prompts_dir"
echo

# 定义 local_id 到 edit_type 的映射
declare -A EDIT_TYPE_MAP
EDIT_TYPE_MAP[0]="add"
EDIT_TYPE_MAP[1]="remove"
EDIT_TYPE_MAP[2]="replace"

# 外层循环：遍历 local_id = 0, 1, 2
for local_id in 0 2 1; do
    edit_type="${EDIT_TYPE_MAP[$local_id]}"
    echo "========================================"
    echo "🚀 开始处理 local_id = $local_id (edit type: $edit_type)"
    echo "========================================"

    # 构造输出目录名称，包含语义类型
    OUTPUT_DIR="results_exp/Qualitative_14B_localgood_long_mixmem_lowvram_${edit_type}"
    mkdir -p "$OUTPUT_DIR"

    # 初始化设备分组
    declare -a device_groups
    for ((i=0; i<DEVICE_COUNT; i++)); do
        device_groups[i]=""
    done

    # 遍历所有图像，分配到设备，并检查对应 prompt 是否存在
    for idx in "${!all_images[@]}"; do
        image_path="${all_images[$idx]}"
        basename_no_ext="${image_path##*/}"
        basename_no_ext="${basename_no_ext%.png}"

        prompt_file="$local_prompts_dir/${basename_no_ext}.txt"

        if [[ ! -f "$prompt_file" ]]; then
            echo "⚠️  Warning: Prompt file not found for $image_path → skipping."
            continue
        fi

        # 读取第 (local_id + 1) 行（因为 local_id=0 对应第一行）
        raw_prompt=$(sed -n "$((local_id + 1))p" "$prompt_file" | tr -d '\r\n')
        if [[ -z "$raw_prompt" ]]; then
            echo "⚠️  Warning: Line $((local_id + 1)) is empty or missing in $prompt_file → skipping $image_path"
            continue
        fi

        device_id=$((idx % DEVICE_COUNT))
        device_groups[$device_id]="${device_groups[$device_id]}$image_path|$raw_prompt"$'\n'
    done

    # 启动每个设备的子进程
    for ((dev=0; dev<DEVICE_COUNT; dev++)); do
        (
            # 拆分 image_path 和 prompt
            mapfile -t jobs_for_dev <<< "${device_groups[$dev]}"
            if [[ ${#jobs_for_dev[@]} -eq 0 || -z "${jobs_for_dev[0]}" ]]; then
                exit 0
            fi

            echo "[Device $dev] 负责处理 ${#jobs_for_dev[@]} 张图片 (local_id=$local_id, edit_type=$edit_type)"

            for job in "${jobs_for_dev[@]}"; do
                [[ -z "$job" ]] && continue
                IFS='|' read -r image_path raw_prompt <<< "$job"

                echo "🖼️ [Device $dev] 处理: $(basename "$image_path") with prompt: '$raw_prompt'"
                preset_actions="w-49-0.05, s-49-0.05, edit: $raw_prompt, right-49-0.5, right-49-0.5, left-49-0.5, left-49-0.5, left-49-0.5"

                "$PYTHON" inference/infer_mem_keyboard_stream_batch_local_lowvram.py \
                    --device_id "$dev" \
                    --model_id "$MODEL_ID" \
                    --loadckpt_path "$LOADCKPT_PATH" \
                    --tokenizer_path "$TOKENIZER_PATH" \
                    --qwen_edit_path "$QWEN_EDIT_PATH" \
                    --qwen_vl_path "$QWEN_VL_PATH" \
                    --num_frames 49 \
                    --reference_nums "$reference_nums" \
                    --camera_speed 0.05 \
                    --angle_speed 0.5 \
                    --overlap_threshold "$overlap_threshold" \
                    --output_dir "$OUTPUT_DIR" \
                    --input_image "$image_path" \
                    --preset_actions "$preset_actions" \
                    --auto_prompt
            done
        ) &
    done

    # 等待当前 local_id 的所有设备任务完成
    wait

    echo "✅ local_id=$local_id ($edit_type) 完成。"
    echo
done

echo "🎉 所有 edit types (add/remove/replace) 处理完毕！"
