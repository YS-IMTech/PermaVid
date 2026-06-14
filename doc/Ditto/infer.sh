#!/bin/bash
python inference/infer_ditto.py \
--lora_path /path/to/ditto_global.safetensors \
--num_frames 73 \
--device_id 0 \
--input_video /path/to/scene_02.mp4 \
--output_video results/scene_02_lego.mp4 \
--prompt "Make it the LEGO style."