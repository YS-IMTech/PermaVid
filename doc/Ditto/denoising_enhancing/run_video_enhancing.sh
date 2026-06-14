#!/bin/bash

# The prompt could be none or the caption
export text_prompt=""
echo ${text_prompt}

step=4
video_txt="./video_list.txt"
output_dir="./enhanced_videos"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

echo "Processing with $step steps..."

# Enhancing with multiple GPUs.
torchrun --nproc_per_node=8 --master_port 29522 \
    video_enhancing_batch.py \
    --ulysses_size=8 \
    --task t2v-A14B \
    --size 720*1280 \
    --ckpt_dir /path/to/Wan-AI/Wan2.2-T2V-A14B \
    --offload_model True \
    --convert_model_dtype \
    --t5_cpu  \
    --prompt "${text_prompt}" \
    --input_video_txt ${video_txt} \
    --output_dir "$output_dir" \
    --forward_step ${step} \
    --skip_backward_step ${step} \


# Enhancing with a single GPU.
# torchrun --nproc_per_node=1 --master_port 29522 \
#     video_enhancing_batch.py \
#     --task t2v-A14B \
#     --size 720*1280 \
#     --ckpt_dir /path/to/Wan-AI/Wan2.2-T2V-A14B \
#     --offload_model True \
#     --convert_model_dtype \
#     --t5_cpu  \
#     --prompt "${text_prompt}" \
#     --input_video_txt ${video_txt} \
#     --output_dir "$output_dir" \
#     --forward_step ${step} \
#     --skip_backward_step ${step} \