#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation as R

try:
    import imageio.v2 as iio
except ImportError:
    iio = None

from utils.retrieval_context import select_aligned_memory_frames, generate_points_in_sphere


# ==============================================================================
# 📄 coordinates.txt 加载
# ==============================================================================

def load_w2cs_from_coordinates_txt(txt_path: str) -> Tuple[str, List[Dict]]:
    with open(txt_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError(f"coordinates.txt 为空: {txt_path}")

    video_path = lines[0]
    w2cs = []

    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 19:
            continue
        try:
            data = list(map(float, parts))
        except ValueError:
            continue
        frame_idx = int(round(data[0]))
        w2c_flat = data[7:19]
        w2c = np.array(w2c_flat, dtype=np.float32).reshape(3, 4)
        w2c_4x4 = np.eye(4, dtype=np.float32)
        w2c_4x4[:3, :] = w2c
        c2w = np.linalg.inv(w2c_4x4)

        position = c2w[:3, 3]
        w2cs.append(w2c_4x4)

    print(f"📄 从 coordinates.txt 加载 {len(w2cs)} 帧（video: {video_path}）")
    return video_path, w2cs


# ==============================================================================
# 🔁 load_pose_and_video
# ==============================================================================

def load_pose_and_video(
    coordinates_txt_path: Optional[str] = None,
    video_path: Optional[str] = None,
) -> Tuple[List[Dict], List[np.ndarray], float, str]:




    print(f" 使用 coordinates.txt: {coordinates_txt_path}")
    inferred_video_path, w2cs = load_w2cs_from_coordinates_txt(coordinates_txt_path)
    if video_path is None:
        video_path = inferred_video_path
    else:
        print(f"⚠️  --video 覆盖 coordinates.txt 中的路径: {video_path}")

    if iio is None:
        raise ImportError("需要 imageio: pip install 'imageio[ffmpeg]'")

    if not video_path:
        raise ValueError("无法确定视频路径：需通过 --video 或 coordinates.txt 首行指定")

    reader = iio.get_reader(video_path)
    try:
        meta = reader.get_meta_data()
        fps = float(meta.get("fps", 25.0))
    except Exception:
        fps = 25.0

    try:
        frames = [frame for frame in reader]
    finally:
        reader.close()

    return w2cs, frames, fps, video_path


# ==============================================================================
# ✅ 新增：parse_range（修复 NameError）
# ==============================================================================

def parse_range(range_str: str, max_len: int) -> Tuple[int, int]:
    try:
        start_s, end_s = range_str.split(":")
        start = int(start_s)
        end = int(end_s)
    except Exception:
        raise ValueError(f"非法区间格式: {range_str}, 期望 'start:end'")
    start = max(0, start)
    end = min(max_len, end)
    if end <= start:
        raise ValueError(
            f"非法区间: {range_str}, 要求 end>start 且在 [0,{max_len}] 内"
        )
    return start, end



# ==============================================================================
# 🚀 主程序
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Camera Memory Retrieval v5.4.1 — Fixed parse_range"
    )
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="outputs_retrieval_hy")
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--pose_json", type=str, default=None)
    
    # parser.add_argument("--coordinates_txt", type=str, default="results_test/camera_ctrl_ys_ue1.3b_noref_step800/clip_006_pose.txt")
    # parser.add_argument("--memory-range", type=str, default="0:70")
    # parser.add_argument("--traj-range", type=str, default="71:100")

    parser.add_argument("--coordinates_txt", type=str, default="results_test/coordinates/AnimeCitySuburbs_3_pose.txt")
    parser.add_argument("--memory-range", type=str, default="0:6999")
    parser.add_argument("--traj-range", type=str, default="7000:7600")



    parser.add_argument("--overlap_threshold", type=float, default=0.8, help="单帧最大 overlap 阈值；低于此值的帧将被过滤掉（默认: 0.4）")

    args = parser.parse_args()

    device = 'cuda'


    print("=" * 60)
    print("📷 Camera Memory Retrieval  — Fixed parse_range + Overlap Threshold")
    print(f"🎯 overlap_threshold = {args.overlap_threshold}")

    print("=" * 60)

    if not  args.coordinates_txt:
        parser.error("请提供 --coordinates_txt")

    poses, frames, fps, actual_video_path = load_pose_and_video(
        coordinates_txt_path=args.coordinates_txt,
        video_path=args.video,
    )
    L = len(poses)

    # ✅ 此处 now works: parse_range is defined
    memory_start, memory_end = parse_range(args.memory_range, L)
    traj_start, traj_end = parse_range(args.traj_range, L)
    memory_indices = list(range(memory_start, memory_end))
    query_indices = list(range(traj_start, traj_end))

    memory_poses = [poses[i] for i in memory_indices]
    query_traj = [poses[i] for i in query_indices]
    points_local = generate_points_in_sphere(50000, 8.0).to(device)

    
    
    print(f"🧠 Memory [{memory_start}, {memory_end}) → {len(memory_poses)} 帧")
    print(f"🚶 Query  [{traj_start}, {traj_end}) → {len(query_traj)} 帧")

    N = min(args.N, len(memory_poses))
    if N < args.N:
        print(f"⚠️  N 自动裁剪为 {N}")

    num_frames = traj_end - traj_start
    
    start_time = time.time()
    retrieved_indices = select_aligned_memory_frames(w2c_list = poses, 
                                                     memory_start_id = memory_start, 
                                                         current_frame_idx = traj_start,
                                                         reference_spatial_memory_frames = N,
                                                         reference_temporal_memory_frames = 0,
                                                         num_frames_pred = num_frames,
                                                         device = device,
                                                         points_local = points_local,
                                                         filter_bool = False,
                                                         overlap_thred = args.overlap_threshold
                                                        )
    elapsed = time.time() - start_time
    print(f"⏱️  总耗时: {elapsed:.2f} 秒")

    if retrieved_indices is None:
        print("\n❌ 检索失败：无满足阈值的参考帧。退出。")
        return

    print(f"\n✅ 检索结果 ({len(retrieved_indices)} 帧):")
    print(f"   memory 内索引: {retrieved_indices}")

    if args.save and poses and frames:
        global_indices = [memory_indices[i] for i in retrieved_indices]
        print(f"   全局帧号: {global_indices}")

        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)

        if iio:
            for rank, mem_rel_idx in enumerate(retrieved_indices):
                g_idx = global_indices[rank]
                filename = (
                    f"ref_{rank:03d}_mem{mem_rel_idx:03d}_global{g_idx:06d}.png"
                )
                iio.imwrite(os.path.join(out_dir, filename), frames[g_idx])
            print(f"💾 保存 {len(retrieved_indices)} 张参考帧")

            traj_path = os.path.join(
                out_dir, f"traj_{traj_start}_{traj_end}.mp4"
            )
            writer = iio.get_writer(traj_path, fps=fps)
            try:
                for gi in query_indices:
                    writer.append_data(frames[gi])
            finally:
                writer.close()
            print(f"🎬 导出轨迹视频: {traj_path}")
        else:
            print("⚠️  未安装 imageio，跳过保存")

    print("\n🎉 完成！")


if __name__ == "__main__":
    main()