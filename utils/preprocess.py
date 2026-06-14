#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import imageio.v3 as iio
from PIL import Image
import os
import argparse


def create_srcmasks(output_path, extend, height, width, index):

    n = extend
    assert n >= 1, "extend 至少为 1"
    assert height > 0 and width > 0, "height 和 width 必须 > 0"

    if index < 0:
        index = n + index  # e.g. -1 → n-1
    if not (0 <= index < n):
        raise ValueError(f"index={index} 超出范围 [0, {n-1}]")

    white_frame = np.full((height, width, 3), 255, dtype=np.uint8)
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)
    frames = []
    for i in range(n):
        if i == index:
            frames.append(black_frame)
        else:
            frames.append(white_frame)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    try:
        iio.imwrite(
            output_path,
            frames,
            fps=20,
            codec='libx264',
            quality=8,
        )
        dur_sec = n / 20
        print(f"✅ 视频已保存: {output_path}")
        print(f"   📏 尺寸: {width} × {height} | 🎞️ 帧数: {n} | 🕒 时长: {dur_sec:.2f}s")
        print(f"   ⚫ 黑色帧索引: {index} | ⚪ 其余为白色")
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        raise


def create_srcvideo(input_image, output_path, extend, height, width):

    n = extend
    assert n >= 1, "extend 至少为 1"
    assert height > 0 and width > 0, "height 和 width 必须为正整数"

    img = np.array(Image.open(input_image).convert('RGB'))

    first_frame = np.array(Image.fromarray(img).resize((width, height)))

    gray_frame = np.full((height, width, 3), 127.5, dtype=np.uint8)

    frames = [first_frame]
    if n > 1:
        frames.extend([gray_frame] * (n - 1))

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    try:
        iio.imwrite(
            output_path,
            frames,
            fps=20,
            codec='libx264',
            quality=8,
        )
        print(f"✅ 视频已保存: {output_path} | 帧数: {n} | 尺寸: {width}×{height}")
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Extend a single image into an N-frame MP4 video: "
                    "frame 0 = resized input image, frames 1~N-1 = gray (RGB=128)."
    )
    parser.add_argument("--input", type=str, help="输入图像路径（如 input.jpg）")
    parser.add_argument("--output", type=str, help="输出 MP4 路径（如 output.mp4）")
    parser.add_argument("--index",  type=int, default=0, help="黑色帧索引 0-based 支持负数；默认: 0")
    parser.add_argument("--extend", type=int, default=73)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--task", type=str, default="srcvideo")

    args = parser.parse_args()

    if args.task == "srcvideo":
        create_srcvideo(args.input, 
                        args.output, 
                        extend=args.extend, 
                        height=args.height, 
                        width=args.width)
    elif args.task == "srcmasks":
        create_srcmasks(
                output_path=args.output,
                extend=args.extend,
                height=args.height,
                width=args.width,
                index=args.index,
            )

if __name__ == "__main__":
    main()