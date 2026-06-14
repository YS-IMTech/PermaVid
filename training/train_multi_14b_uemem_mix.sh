#!/bin/bash
set -e

######################### 分布式相关 #########################
GPUS_PER_NODE=4
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"8675"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}

export NCCL_DEBUG=INFO
export NCCL_IB_GID_INDEX=5
export NCCL_IB_TC=138
export NCCL_IB_QPS_PER_CONNECTION=8

echo "=============================="
echo "NNODES       : $NNODES"
echo "GPUS/Node    : $GPUS_PER_NODE"
echo "This NodeRank: $NODE_RANK"
echo "MASTER       : $MASTER_ADDR:$MASTER_PORT"
echo "=============================="

######################### 模型路径 #########################
MODEL_DIR="/mnt/cpfs/yangshuai/code/PermaVid/prismworld/models"
PRISMWORLD_DIR="${MODEL_DIR}/prismworld"
VACE_14B_DIR="${MODEL_DIR}/Wan2.1-VACE-14B"

######################### 训练命令 #########################
accelerate launch \
  --config_file scripts/train/full/accelerate_config_14B.yaml \
  --num_processes $((GPUS_PER_NODE * NNODES)) \
  --num_machines ${NNODES} \
  --machine_rank ${NODE_RANK} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
    examples/wanvideo/model_training/train_ue.py \
        --dataset_base_path /mnt/cpfs/yangshuai/code/PermaVid/ue_data/ue_filter \
        --dataset_metadata_path datasets/metadata_wdepth_ue_filtered.csv \
        --data_file_keys "video,depth,poses" \
        --height 480 \
        --width 832 \
        --num_frames 81 \
        --dataset_repeat 1 \
        --model_id_with_origin_paths "${VACE_14B_DIR}:diffusion_pytorch_model*.safetensors,${VACE_14B_DIR}:models_t5_umt5-xxl-enc-bf16.pth,${VACE_14B_DIR}:Wan2.1_VAE.pth" \
        --tokenizer_path "${VACE_14B_DIR}/google/umt5-xxl" \
        --learning_rate 1e-5 \
        --num_epochs 20 \
        --remove_prefix_in_ckpt "pipe.dit.,pipe.vace." \
        --output_path "./outputs_train_14b_uecam_mixref_onlydit_v2" \
        --trainable_models "dit" \
        --extra_inputs "original_pose_height,original_pose_width" \
        --use_gradient_checkpointing_offload \
        --save_steps 200 \
        --add_control_adapter \
        --wandb_log_interval 5 \
        --wandb_name "prismworld_uecam_mixref_dit_14b_v2" \
        --train_mode "memo_mix" \
        --reference_nums 10 \
        --resume_from_checkpoint "${PRISMWORLD_DIR}/full/camera_memory/mix/mixref_14b_step-5800.safetensors"

