#!/bin/bash
set -e

source /mnt/cpfs/yangshuai/miniconda3/bin/activate prismworld

######################### 环境设置（按需改） #########################
export http_proxy=http://jdtcom:709a64b73eb3@10.119.176.202:3128
export https_proxy=http://jdtcom:709a64b73eb3@10.119.176.202:3128

# 工作目录
cd /mnt/cpfs/yangshuai/code/PermaVid/prismworld

export PYTHONPATH="/mnt/cpfs/yangshuai/code/PermaVid/prismworld:$PYTHONPATH"


######################### 分布式相关 #########################
GPUS_PER_NODE=8                           # 每台机器 GPU 数
MASTER_ADDR=${MASTER_ADDR:-"localhost"}   # SenseCore 会注入
MASTER_PORT=${MASTER_PORT:-"8358"}
NNODES=${WORLD_SIZE:-"1"}                 # 总节点数
NODE_RANK=${RANK:-"0"}                    # 当前节点编号

export NCCL_DEBUG=INFO
# 如果是 IB/RoCE，还可以加这几个环境变量
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
VACE_14B_DIR="${MODEL_DIR}/Wan2.1-VACE-14B"
PRISMWORLD_DIR="${MODEL_DIR}/prismworld"

######################### 训练命令 #########################
accelerate launch \
  --config_file scripts/train/full/accelerate_config_14B.yaml \
  --num_processes $((GPUS_PER_NODE * NNODES)) \
  --num_machines ${NNODES} \
  --machine_rank ${NODE_RANK} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
    examples/wanvideo/model_training/train.py \
        --dataset_base_path /mnt/public/users/yangshuai/datasets/spatialvid_hq_dataset \
        --dataset_metadata_path datasets/config/spatialvid/metadata.csv \
        --data_file_keys "video,vace_video,poses" \
        --height 480 \
        --width 832 \
        --num_frames 81 \
        --dataset_repeat 1 \
        --model_id_with_origin_paths "${VACE_14B_DIR}:diffusion_pytorch_model*.safetensors,${VACE_14B_DIR}:models_t5_umt5-xxl-enc-bf16.pth,${VACE_14B_DIR}:Wan2.1_VAE.pth" \
        --learning_rate 1e-5 \
        --num_epochs 1 \
        --remove_prefix_in_ckpt "pipe.dit.,pipe.vace." \
        --output_path "./outputs_train_14b_cam_noref_v1" \
        --trainable_models "dit" \
        --extra_inputs "vace_video,poses,original_pose_height,original_pose_width" \
        --use_gradient_checkpointing_offload \
        --save_steps 200 \
        --add_control_adapter \
        --wandb_log_interval 20 \
        --wandb_name "causalworld_camctrl_noref_dit_14b" \
        --resume_from_checkpoint "${PRISMWORLD_DIR}/diito_local.safetensors"
