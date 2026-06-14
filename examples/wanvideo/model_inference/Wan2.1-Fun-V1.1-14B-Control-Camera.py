import torch
import argparse
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download
import os

# 支持的 camera 控制方向
DIRECTIONS = ["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown", "In", "Out"]


def main():
    parser = argparse.ArgumentParser(description="Run WanVideoPipeline with camera control on an input image.")
    parser.add_argument("--input_image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--output_dir", type=str, default="results_test", help="Directory to save output videos.")
    parser.add_argument("--prompt", type=str, default="", help="Prompt for generation.")
    parser.add_argument("--negative_prompt", type=str, default=(
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
        "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
        "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    ), help="Negative prompt for generation.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--camera_speed", type=float, default=0.01, help="Camera control speed.")
    parser.add_argument("--fps", type=int, default=15, help="FPS of the output video.")
    parser.add_argument("--quality", type=int, default=5, help="Video encoding quality (1~10, lower is better).")
    parser.add_argument("--directions", type=str, default="In")
    parser.add_argument("--device_id", type=int, default=0, help="The ID of the CUDA device to use (e.g., 0, 1, 2).")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.device_id}"


    # 加载模型
    print("Loading WanVideoPipeline...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-Control-Camera",
                        origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-Control-Camera",
                        origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-Control-Camera",
                        origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
            ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-Control-Camera",
                        origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                        offload_device="cpu"),
        ],
    )
    pipe.enable_vram_management()
    print("Pipeline loaded.")

    # 读取输入图像
    input_image = Image.open(args.input_image).convert("RGB")

    # 对每个方向生成视频
    for direction in args.directions:
        if direction not in DIRECTIONS:
            print(f"⚠️ Skipping invalid direction: {direction}")
            continue

        print(f"➡️ Generating video for direction: {direction}")
        try:
            video = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                seed=args.seed,
                tiled=True,
                input_image=input_image,
                camera_control_direction=direction,
                camera_control_speed=args.camera_speed,
            )

            # 构建输出路径：{output_dir}/{file_name}_{direction}.mp4
            file_name = os.path.splitext(os.path.basename(args.input_image))[0]
            output_path = os.path.join(args.output_dir, f"{file_name}_{direction}.mp4")
            save_video(video, output_path, fps=args.fps, quality=args.quality)
            print(f"✅ Saved: {output_path}")

        except Exception as e:
            print(f"❌ Failed for direction {direction}: {e}")

    print("🎉 All done!")


if __name__ == "__main__":
    main()