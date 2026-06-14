#!/bin/bash
accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path /path/to/Ditto-1M/videos \
  --dataset_metadata_path /path/to/csvs_for_DiffSynth/xxx.csv \
  --data_file_keys "video,vace_video" \
  --height 480 \
  --width 832 \
  --num_frames 73 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-14B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "./exps/ditto" \
  --lora_base_model "vace" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 128 \
  --extra_inputs "vace_video" \
  --use_gradient_checkpointing_offload \
  --save_steps 1000