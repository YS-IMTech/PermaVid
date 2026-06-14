import torch
import decord

import argparse
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from pathlib import Path
from typing import List, Tuple
from decord import VideoReader, cpu

import os
import numpy as np
from utils.camera_convert import read_uepose_from_json, poses_intrinsics_to_coordinates, euler_to_w2c_batch, generate_camera_coordinates, RT_align
from diffsynth import load_state_dict
from diffsynth.models.wan_video_camera_controller import SimpleAdapter

from utils.utils_keyframe import extract_keyframes_indices
from transformers import  AutoProcessor, Qwen3VLForConditionalGeneration

from utils.retrieval_context import select_aligned_memory_frames, generate_points_in_sphere, find_high_overlap_history_frames

from utils.vda import init_vda, run_vda
from diffusers import QwenImageEditPipeline


# 支持的 camera 控制方向
DIRECTIONS = ["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown", "In", "Out"]

def add_info2file(filepath, line):
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
        
def images_to_numpy(img_list):
    return np.stack([np.array(img) for img in img_list], axis=0)  # (T, H, W, 3)


def save_poses_to_file(coordinates, args, filename="pose"):
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{filename}_pose.txt")
    with open(out_path, 'w') as f:
        for coord in coordinates:
            line = ' '.join(f"{float(x):.9f}" for x in coord)
            f.write(line + '\n')
    print(f"✅ Poses saved to: {out_path}")


def run(args, prompt, pipe, vda_model, epoch, file_name, device, edited_index, points_local, local_invalid_indices,
        origin, direction, speed, angle, num_frames, coordinates_bank, videos_bank, depths_bank):
    
    keyframes_indices = None    
    start_id = len(coordinates_bank) - 1
    # coordinates = generate_camera_coordinates(direction, num_frames, speed, start_id, origin)

    if direction != "Reverse":
        coordinates = generate_camera_coordinates(control=direction, 
                                                length=num_frames, 
                                                speed_trans=speed, 
                                                speed_rot=angle,
                                                start_id=start_id, origin=origin)

        # origin_update=[start_id, 0.5, 0.8667, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]

        # coordinates = generate_camera_coordinates(control=direction, 
        #                                         length=num_frames, 
        #                                         speed_trans=speed, 
        #                                         speed_rot=angle,
        #                                         start_id=start_id, origin=origin_update)
        # coordinates = RT_align(coordinates, origin)

    else:
        coordinates = coordinates_bank[-num_frames:][::-1]
        for i in range(len(coordinates)):
            coordinates[i][0] = start_id + i

    
    coordinates_bank = coordinates_bank + coordinates[1:]
    
    os.makedirs(f"{args.output_dir}/{file_name}/epoch-{epoch}", exist_ok=True)
    
    if start_id > 0:
        gen_start_idx = start_id
        gen_end_idx = gen_start_idx + num_frames

        memory_start_idx = 0
        memory_end_idx = gen_start_idx 

        print("prompt:", prompt)
        print(f"[INFO] edited_index: {edited_index} start_id:{start_id} local_invalid_indices:{local_invalid_indices}")
        # if edited_index != 0 and edited_index != start_id:
        #     memory_start_idx = edited_index
        
        if len(local_invalid_indices) > 0 :
            memory_start_idx = edited_index
        
        
        keyframes_indices = extract_keyframes_indices(coordinates = coordinates_bank, 
                                                    reference_nums = args.reference_nums,
                                                    memory_start = memory_start_idx,
                                                    memory_end = memory_end_idx,
                                                    traj_start = gen_start_idx,
                                                    traj_end = gen_end_idx,
                                                    height = args.height, 
                                                    width = args.width,
                                                    invalid_indexes=local_invalid_indices, 
                                                    overlap_threshold=args.overlap_threshold,
                                                    fast=True)
        
        if edited_index == start_id:
            local_invalid_indices = find_high_overlap_history_frames(coordinates = coordinates_bank,
                                                                    current_frame_idx = gen_start_idx,
                                                                    overlap_thred=0.4,
                                                                    device = device,
                                                                    fov_h_deg=90.0,
                                                                    fov_v_deg=90.0,
                                                                    points_local = points_local,
                                                                    )
            print("local_invalid_indices:", local_invalid_indices)

            
    if keyframes_indices is not None:
        reference_image = []
        for index in keyframes_indices:
            if index not in local_invalid_indices or edited_index==0:
                print("[INFO] Final valid local index:", index)
                videos_bank[index].save(f"{args.output_dir}/{file_name}/epoch-{epoch}/RefImage_id{index}.png")
                reference_image.append(videos_bank[index])
        if len(reference_image) == 0:
            reference_image = None
    else:
        reference_image = None
     
    
    gray = np.full((args.height, args.width, 3), 127, dtype=np.uint8)
    inference_video = [videos_bank[-1]] + [Image.fromarray(gray)] * (num_frames - 1)    
    
    # --- infer ---
    # vace_video: Optional[list[Image.Image]] = None,
    # vace_video_mask: Optional[Image.Image] = None,
    # vace_reference_image: Optional[Image.Image] = None,

    video_gen = pipe(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        num_frames=num_frames,
        seed=args.seed,
        height=args.height,
        width=args.width,
        vace_video=inference_video,
        vace_reference_image=reference_image,
        tiled=True,
        coordinates=coordinates,
    )
    
    depths_vis = run_vda(vda_model,
        input_video=video_gen, # list of PILImage
        fps=args.fps,
        save_video_path=os.path.join(args.output_dir, f"{file_name}/epoch-{epoch}-{direction}-depth.mp4"),
        grayscale=True,
        device=device
        )
    
    videos_bank += video_gen[1:]
    if len(depths_bank) > 0:
        depths_bank += depths_vis[1:]
    else:
        depths_bank += depths_vis

    output_path = os.path.join(args.output_dir, f"{file_name}/epoch-{epoch}-{direction}.mp4")
    save_video(video_gen, output_path, fps=args.fps, quality=args.quality)
    print(f"✅ Saved epoch-{epoch} generated video: {output_path}")
    epoch = epoch + 1
    return epoch, coordinates_bank, videos_bank, depths_bank, local_invalid_indices


def parse_preset_actions(preset_str: str):
    """
    Parse preset_actions string into a list of action dictionaries.
    Example: "w-49-0.1, j-49-1.0, edit: change style"
    Returns: [
        {"type": "move", "key": "w", "num_frames": 49, "param": 0.1},
        {"type": "move", "key": "j", "num_frames": 49, "param": 1.0},
        {"type": "edit", "prompt": "change style"}
    ]
    """
    actions = []
    for chunk in preset_str.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.startswith("edit:"):
            _, prompt = chunk.split(":", 1)
            prompt = prompt.strip()
            actions.append({"type": "edit", "prompt": prompt})
        else:
            parts = chunk.split('-')
            key = parts[0].strip().lower()
            num_frames = None
            param = None
            if len(parts) == 2:
                try:
                    num_frames = int(parts[1].strip())
                except ValueError:
                    pass
            if len(parts) == 3:
                try:
                    param = float(parts[2].strip())
                except ValueError:
                    pass
            actions.append({"type": "move", "key": key, "num_frames": num_frames, "param": param})
    return actions


def get_caption(model, processor, image_path):

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": "Please summarize this image in one short sentence."},
            ],
        }
    ]

    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    caption = output_text[0]
    return caption


def main():
    parser = argparse.ArgumentParser(description="Run WanVideoPipeline with camera control on an input image.")
    parser.add_argument("--output_dir", type=str, default="results_test/camera_ctrl_ys", help="Directory to save output videos.")
    parser.add_argument("--prompt", type=str, default="", help="Prompt for generation.")
    parser.add_argument("--negative_prompt", type=str, default=(
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
        "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
        "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    ), help="Negative prompt for generation.")
    
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--camera_speed", type=float, default=0.05, help="Camera control speed.")
    parser.add_argument("--angle_speed", type=float, default=0.5, help="Camera control speed.")
    
    parser.add_argument("--fps", type=int, default=20, help="FPS of the output video.")
    parser.add_argument("--quality", type=int, default=5, help="Video encoding quality (1~10, lower is better).")
    parser.add_argument("--device_id", type=int, default=0, help="The ID of the CUDA device to use (e.g., 0, 1, 2).")

    parser.add_argument("--overlap_threshold", type=float, default=0.4)
    parser.add_argument("--auto_prompt", action="store_true")

    parser.add_argument("--loadckpt_path", type=str, default=None)
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-VACE-1.3B")
    parser.add_argument("--input_image", type=str, default=None, help="Path to the input image file.")
    parser.add_argument("--input_video", type=str, default=None)

    parser.add_argument("--reference_nums", type=int, default=10)
    parser.add_argument("--num_frames", type=int, default=41)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    
    # 新增 preset_actions 参数
    parser.add_argument("--preset_actions", type=str, default=None,
                        help='Preset actions, e.g., "w-49-0.1, j-49-1.0, edit: change style to Makoto Shinkai"')

    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Local tokenizer directory (umt5-xxl). Avoids HF download.")
    parser.add_argument("--qwen_edit_path", type=str, default="/mnt/public/users/yangshuai/code/public_models/Qwen/Qwen-Image-Edit",
                        help="Path to Qwen-Image-Edit model.")
    parser.add_argument("--qwen_vl_path", type=str, default="/mnt/public/users/yangshuai/code/public_models/Qwen/Qwen3-VL-8B-Instruct",
                        help="Path to Qwen3-VL model (used with --auto_prompt).")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.device_id}"
    
    print("Loading WanVideoPipeline...")
    tokenizer_config = ModelConfig(path=args.tokenizer_path) if args.tokenizer_path else None
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id=args.model_id,
                        origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            ModelConfig(model_id=args.model_id,
                        origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(model_id=args.model_id,
                        origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
        ],
        tokenizer_config=tokenizer_config,
    )
    pipe.dit.add_control_adapter = True
    pipe.dit.control_adapter = SimpleAdapter(pipe.dit.in_dim_control_adapter, pipe.dit.dim, kernel_size=pipe.dit.patch_size[1:], stride=pipe.dit.patch_size[1:])

    pipe_edit = QwenImageEditPipeline.from_pretrained(args.qwen_edit_path).to(torch.bfloat16).to(device)
    pipe_edit.set_progress_bar_config(disable=None)

    if args.loadckpt_path:
        state_dict = load_state_dict(args.loadckpt_path, torch_dtype=torch.bfloat16, device="cpu")
        vace_state_dict = {}
        dit_state_dict = {}
        for k, v in state_dict.items():
            if "vace" in k:
                vace_state_dict[k] = v
            else:
                dit_state_dict[k] = v

        if vace_state_dict:
            pipe.vace.load_state_dict(vace_state_dict, strict=True)
            print(f"[INFO] Loaded {len(vace_state_dict)} VACE parameters from: {args.loadckpt_path}")
        if dit_state_dict:
            pipe.dit.load_state_dict(dit_state_dict, strict=True)
            print(f"[INFO] Loaded {len(dit_state_dict)} DiT parameters from: {args.loadckpt_path}")

        del state_dict, vace_state_dict, dit_state_dict
        print(f"[INFO] Checkpoint loading completed.")

    pipe.enable_vram_management()
    vda_model = init_vda(device=device)

    print("Pipeline loaded.")


    if args.input_image:
        input_image = Image.open(args.input_image).convert("RGB").resize((args.width, args.height))
        file_name = os.path.splitext(os.path.basename(args.input_image))[0]
    elif args.input_video:
        file_name = os.path.splitext(os.path.basename(args.input_video))[0]
        temp_video = VideoData(args.input_video, height=args.height, width=args.width)
        temp_video = [temp_video[i] for i in range(len(temp_video))]
        input_image = temp_video[0] 
        
    os.makedirs(f"{args.output_dir}/{file_name}", exist_ok=True)
    input_image.save(f"{args.output_dir}/{file_name}/input_image.png")

    prompts_info_path = f"{args.output_dir}/{file_name}/prompts.txt"
    
    if args.auto_prompt:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.qwen_vl_path, dtype="auto", device_map=device
        )
        processor = AutoProcessor.from_pretrained(args.qwen_vl_path)

        args.prompt = get_caption(model, processor, f"{args.output_dir}/{file_name}/input_image.png")

        add_info2file(prompts_info_path, f"initial caption prompt:{args.prompt}")
        print("Prompt:",args.prompt)

    epoch = 0
    edited_index = 0
    speed = args.camera_speed
    angle = args.angle_speed
    num_frames = args.num_frames
    interaction_prompt=" "
    file_name = os.path.splitext(os.path.basename(args.input_image or args.input_video))[0]
    points_local = generate_points_in_sphere(50000, 8.0).to(device)
    origin_position = [0, 0.5, 0.8667, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
    coordinates_bank = [origin_position]
    videos_bank = [input_image]
    depths_bank = []
    local_invalid_indices=[]

    KEY_MAP = {
        **{d.lower(): d for d in DIRECTIONS},  
        'w': "W", 'a': "A", 's': "S", 'd': "D", "v": "Reverse", 
        'i': "LookUp", 'j': "LookLeft", 'k': "LookDown", 'l': "LookRight",
        'up': "LookUp", 'left': "LookLeft", 'down': "LookDown", 'right': "LookRight",
    }

    # Parse preset actions
    if args.preset_actions:
        actions_list = parse_preset_actions(args.preset_actions)
        print(f"Parsed {len(actions_list)} preset actions.")
    else:
        actions_list = []


    for action in actions_list:
        if action["type"] == "edit":
            interaction_prompt = action["prompt"]
            add_info2file(prompts_info_path, f"edited_index-{edited_index} interactive prompts:{interaction_prompt}")
            print(f"✅ Interaction Text Prompt: {interaction_prompt}")
            source_image = videos_bank[-1]
            edited_index = len(videos_bank) - 1
            inputs = {
                "image": source_image,
                "prompt": interaction_prompt,
                "generator": torch.manual_seed(0),
                "true_cfg_scale": 4.0,
                "negative_prompt": " ",
                "num_inference_steps": 50,
            }
            with torch.inference_mode():
                edited_image = pipe_edit(**inputs)
                edited_image = edited_image.images[0].convert("RGB").resize((args.width, args.height))
                edited_image_savepath = f"{args.output_dir}/{file_name}/edited_image_id{edited_index}.png"
                edited_image.save(edited_image_savepath)
                videos_bank[-1] = edited_image
                print("✅ edited_image saved at", edited_image_savepath)

            source_image.save(f"{args.output_dir}/{file_name}/source_image_id{edited_index}.png")
            args.prompt = get_caption(model, processor, edited_image_savepath)
            print("[INFO] new prompt after edit:", args.prompt)
            add_info2file(prompts_info_path, f"updated caption prompt:{args.prompt}")
            
        elif action["type"] == "move":
            key = action["key"]
            direction = KEY_MAP.get(key)
            if not direction:
                print(f"⚠️ Unknown key: {key}")
                continue

            # 动态更新 num_frames
            local_num_frames = action["num_frames"] if action["num_frames"] is not None else num_frames

            # 动态更新 speed 或 angle
            local_speed = speed
            local_angle = angle
            if direction in ['W', 'A', 'S', 'D']:
                if action["param"] is not None:
                    local_speed = action["param"]
            elif direction in ['LookUp', 'LookDown', 'LookLeft', 'LookRight']:
                if action["param"] is not None:
                    local_angle = action["param"]

            print(f"→ Executing: key={key} → direction={direction} | num_frames={local_num_frames} | speed={local_speed} | angle={local_angle}")

            origin = coordinates_bank[-1]
            epoch, coordinates_bank, videos_bank, depths_bank, local_invalid_indices = run(args, args.prompt, pipe, vda_model, epoch, file_name, device, edited_index, points_local, local_invalid_indices,
                origin, direction, speed, angle, num_frames, coordinates_bank, videos_bank, depths_bank)

            # 更新全局状态（用于后续动作默认值）
            if key in ['w', 'a', 's', 'd'] and action["param"] is not None:
                speed = local_speed
            elif key in ['i', 'j', 'k', 'l', 'up', 'down', 'left', 'right'] and action["param"] is not None:
                angle = local_angle
            if action["num_frames"] is not None:
                num_frames = local_num_frames

    # End of preset actions

    save_poses_to_file(coordinates_bank, args, file_name)    
    output_rgb_path = os.path.join(args.output_dir, f"{file_name}/Final.mp4")
    output_depth_path = os.path.join(args.output_dir, f"{file_name}/Final_depth.mp4")

    save_video(videos_bank, output_rgb_path, fps=args.fps, quality=args.quality)
    
    
    save_video(depths_bank, output_depth_path, fps=args.fps, quality=args.quality)
    
    print(f"✅ Saved all generated video: {output_rgb_path} and {output_depth_path}")


if __name__ == "__main__":
    main()