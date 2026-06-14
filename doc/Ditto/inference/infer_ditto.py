import argparse
import torch
import os
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig


def main(args):

    device = f"cuda:{args.device_id}"

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
        ],
    )
    if args.lora_path:
        print(f"Loading Ditto LoRA model: {args.lora_path} (alpha={args.lora_alpha})")
        if not os.path.exists(args.lora_path):
            print(f"Error: LoRA file not found at {args.lora_path}")
            return
        pipe.load_lora(pipe.vace, args.lora_path, alpha=args.lora_alpha)

    pipe.enable_vram_management()

    print(f"Loading input video: {args.input_video}")
    if not os.path.exists(args.input_video):
        print(f"Error: Input video file not found at {args.input_video}")
        return
        
    video = VideoData(args.input_video, height=args.height, width=args.width)
    
    num_frames = min(args.num_frames, len(video))
    if num_frames != args.num_frames:
        print(f"Warning: Requested number of frames ({args.num_frames}) exceeds total video frames ({len(video)}). Using {num_frames} frames instead.")
        
    video = [video[i] for i in range(num_frames)]
    
    reference_image = None

    video = pipe(
        prompt=args.prompt,
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        vace_video=video,
        vace_reference_image=reference_image,
        num_frames=num_frames,
        seed=args.seed,
        tiled=True,
    )

    output_dir = os.path.dirname(args.output_video)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    save_video(video, args.output_video, fps=args.fps, quality=args.quality)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InstructV2V Pipeline.")

    parser.add_argument("--input_video", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--output_video", type=str, required=True, help="Path to save the output video file.")
    parser.add_argument("--lora_path", type=str, default=None, help="Optional path to a LoRA model file (.safetensors).")
    parser.add_argument("--device_id", type=int, default=0, help="The ID of the CUDA device to use (e.g., 0, 1, 2).")
    parser.add_argument("--prompt", type=str, required=True, help="The positive prompt describing the target style.")
    parser.add_argument("--height", type=int, default=480, help="The height to use for video processing.")
    parser.add_argument("--width", type=int, default=832, help="The width to use for video processing.")
    parser.add_argument("--num_frames", type=int, default=73, help="The number of video frames to process.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducible results.")

    parser.add_argument("--lora_alpha", type=float, default=1.0, help="The alpha (weight) value for the LoRA model.")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second (FPS) for the output video.")
    parser.add_argument("--quality", type=int, default=5, help="Quality of the output video (CRF value, lower is better).")

    args = parser.parse_args()
    main(args)