#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera Memory Retrieval v5.4 — Overlap Threshold Filtering (Fixed: parse_range added)

🎯 核心目标：
    从 memory 区间中选出 N 张 reference frames，
    使这 N 张相机视锥的「联合可视区域」尽可能覆盖整段 query trajectory 曾经看到过的 3D 空间。

✅ v5.2 新增：
   6. 🎯 --fast 模式下支持「覆盖率目标早停」

✅ v5.3 新增：
   7. 📁 支持 --coordinates_txt 输入（直接提供 w2c 矩阵）

✅ v5.4 新增：
   8. 🛑 新增 --overlap_threshold（默认 0.4）：
        对 greedy 选出的帧，**仅保留 max overlap ≥ threshold 的帧**
        → 若最终为空，返回 None 并报错提示
        → 此过滤在 greedy 后进行，不影响 early-stop 逻辑

✅ v5.4.1 Bugfix:
   🔧 补全 parse_range 函数（修复 NameError）
"""

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


# ==============================================================================
# 🔧 核心几何工具
# ==============================================================================

def euler_zyx_to_rotation_matrix(angles_deg: np.ndarray) -> np.ndarray:
    pitch, yaw, roll = angles_deg
    rot = R.from_euler("ZYX", [yaw, pitch, roll], degrees=True)
    return rot.as_matrix()


def get_camera_c2w_from_pose_dict(pose: Dict) -> np.ndarray:
    R_mat = euler_zyx_to_rotation_matrix(
        np.array(pose["rotation"], dtype=np.float32)
    )
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R_mat
    c2w[:3, 3] = np.array(pose["position"], dtype=np.float32)
    c2w = c2w[:, [1, 2, 0, 3]]
    return c2w


# ==============================================================================
# 📄 coordinates.txt 加载
# ==============================================================================

def load_poses_from_coordinates_txt(txt_path: str) -> Tuple[str, List[Dict]]:
    with open(txt_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError(f"coordinates.txt 为空: {txt_path}")

    video_path = lines[0]
    poses = []

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
        poses.append({
            "position": position,
            "rotation": None,
            "c2w": c2w,
            "frame_index": frame_idx,
        })

    print(f"📄 从 coordinates.txt 加载 {len(poses)} 帧（video: {video_path}）")
    return video_path, poses


# ==============================================================================
# 🔁 load_pose_and_video
# ==============================================================================

def load_pose_and_video(
    pose_json_path: Optional[str] = None,
    coordinates_txt_path: Optional[str] = None,
    video_path: Optional[str] = None,
) -> Tuple[List[Dict], List[np.ndarray], float, str]:
    if coordinates_txt_path:
        print(f"📥 使用 coordinates.txt: {coordinates_txt_path}")
        inferred_video_path, poses_raw = load_poses_from_coordinates_txt(coordinates_txt_path)
        if video_path is None:
            video_path = inferred_video_path
        else:
            print(f"⚠️  --video 覆盖 coordinates.txt 中的路径: {video_path}")
    elif pose_json_path:
        print(f"📥 使用 pose.json: {pose_json_path}")
        with open(pose_json_path, "r") as f:
            pose_data = json.load(f)

        if "CineCameraActor" in pose_data:
            actor_dict = pose_data["CineCameraActor"]
        else:
            first_key = next(iter(pose_data.keys()))
            actor_dict = pose_data[first_key]

        sorted_items = sorted(actor_dict.items(), key=lambda kv: int(kv[0]))
        poses_raw = [
            {
                "position": np.array(p["position"], dtype=np.float32),
                "rotation": np.array(p["rotation"], dtype=np.float32),
                "scale": np.array(p.get("scale", [1.0, 1.0, 1.0]), dtype=np.float32),
                "frame_index": idx,
            }
            for idx, p in enumerate(v for _, v in sorted_items)
        ]
    else:
        raise ValueError("必须提供 --pose_json 或 --coordinates_txt")

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

    L = min(len(poses_raw), len(frames))
    if len(poses_raw) != len(frames):
        print(f"⚠️  警告：poses({len(poses_raw)}) 与视频帧({len(frames)}) 不一致 → 截断为 {L} 帧")

    poses = poses_raw[:L]
    frames = frames[:L]
    print(f"📹 已加载 {L} 帧 (fps≈{fps:.2f}) from {os.path.basename(video_path)}")

    for p in poses:
        if "c2w" not in p:
            c2w = get_camera_c2w_from_pose_dict(p)
            p["c2w"] = c2w
        if "rotation" not in p or p["rotation"] is None:
            p["rotation"] = np.zeros(3, dtype=np.float32)

    return poses, frames, fps, video_path


# ==============================================================================
# 🧮 辅助函数
# ==============================================================================

def auto_estimate_pos_scale(poses: List[Dict], percentile: float = 50.0) -> float:
    if len(poses) < 2:
        return 500.0
    positions = np.stack([p["position"] for p in poses], axis=0)
    N = len(positions)
    max_pairs = min(5000, N * (N - 1) // 2)
    if N <= 100:
        dists = pdist(positions, metric="euclidean")
    else:
        idx1 = np.random.randint(0, N, size=max_pairs)
        idx2 = np.random.randint(0, N, size=max_pairs)
        mask = idx1 != idx2
        idx1, idx2 = idx1[mask], idx2[mask]
        dists = np.linalg.norm(positions[idx1] - positions[idx2], axis=1)
    if len(dists) == 0:
        return 500.0
    scale = np.percentile(dists, percentile)
    return max(scale, 10.0)


def sample_points_in_frustum_cam(
    K: int,
    fov_y_deg: float,
    aspect: float,
    near: float,
    far: float,
    rng: np.random.Generator,
) -> np.ndarray:
    fov_y = np.deg2rad(fov_y_deg)
    tan_half = np.tan(fov_y / 2.0)
    z = rng.uniform(near, far, size=(K,))
    u = rng.uniform(-1.0, 1.0, size=(K,))
    v = rng.uniform(-1.0, 1.0, size=(K,))
    x = u * z * tan_half * aspect
    y = v * z * tan_half
    pts_cam = np.stack([x, y, z], axis=-1)
    return pts_cam


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
# 🔁 compute_frustum_overlap_scores_mc
# ==============================================================================

def compute_frustum_overlap_scores_mc(
    memory_poses: List[Dict],
    query_poses: List[Dict],
    fov_deg: float = 90.0,
    aspect: float = 1.0,
    num_samples: int = 128,
    random_seed: int = 1234,
    memory_batch_size: int = 200,
) -> np.ndarray:
    M = len(memory_poses)
    T = len(query_poses)
    if M == 0 or T == 0:
        return np.zeros((M, T), dtype=np.float32)

    all_poses = list(memory_poses) + list(query_poses)
    positions = np.stack([p["position"] for p in all_poses], axis=0)
    scene_scale = auto_estimate_pos_scale([{**p, "rotation": [0,0,0]} for p in all_poses], percentile=60.0)

    near = max(scene_scale * 0.02, 10.0)
    far = max(scene_scale * 3.0, near * 2.0)
    print(
        f"   📏 Monte Carlo 深度范围: near≈{near:.1f}, far≈{far:.1f} (scene_scale≈{scene_scale:.1f})"
    )

    rng = np.random.default_rng(random_seed)
    fov_y = np.deg2rad(fov_deg)
    tan_half = np.tan(fov_y / 2.0)
    eps = 1e-6

    pts_cam = sample_points_in_frustum_cam(
        K=num_samples,
        fov_y_deg=fov_deg,
        aspect=aspect,
        near=near,
        far=far,
        rng=rng,
    )

    query_world_pts = np.zeros((T, num_samples, 3), dtype=np.float32)
    for t, qp in enumerate(query_poses):
        c2w = qp["c2w"]
        R_q = c2w[:3, :3]
        t_q = c2w[:3, 3]
        query_world_pts[t] = (R_q @ pts_cam.T).T + t_q

    mem_pos = np.stack([mp["c2w"][:3, 3] for mp in memory_poses])
    mem_R = np.stack([mp["c2w"][:3, :3] for mp in memory_poses])
    R_wc_T = np.transpose(mem_R, (0, 2, 1))

    scores = np.zeros((M, T), dtype=np.float32)

    for start in range(0, M, memory_batch_size):
        end = min(M, start + memory_batch_size)
        B = end - start

        batch_R_wc_T = R_wc_T[start:end]
        batch_pos = mem_pos[start:end]

        pts_world_exp = query_world_pts[None, :, :, :]
        batch_pos_exp = batch_pos[:, None, None, :]

        p_rel = pts_world_exp - batch_pos_exp
        p_cam = np.einsum("bij,btkj->btki", batch_R_wc_T, p_rel)

        x, y, z = p_cam[..., 0], p_cam[..., 1], p_cam[..., 2]
        z_pos = z > 0.0
        z_in = (z >= near) & (z <= far) & z_pos
        x_in = np.abs(x) <= (z * tan_half * aspect + eps)
        y_in = np.abs(y) <= (z * tan_half + eps)

        inside = z_in & x_in & y_in
        batch_scores = inside.astype(np.float32).mean(axis=2)

        scores[start:end, :] = batch_scores

    return np.clip(scores, 0.0, 1.0)


# ==============================================================================
# 🧭 spatial_proximity_filter
# ==============================================================================

def spatial_proximity_filter(
    memory_poses: List[Dict],
    query_poses: List[Dict],
    percentile: float = 95.0,
    min_keep_ratio: float = 0.1,
) -> np.ndarray:
    M = len(memory_poses)
    T = len(query_poses)
    if M == 0 or T == 0:
        return np.ones(M, dtype=bool)

    m_pos = np.stack([p["position"] for p in memory_poses])
    q_pos = np.stack([p["position"] for p in query_poses])
    diff = m_pos[:, None, :] - q_pos[None, :, :]
    dists_to_query = np.linalg.norm(diff, axis=2).min(axis=1)

    D_max = np.percentile(dists_to_query, percentile)
    D_max = max(D_max, 1e-3)
    mask = dists_to_query <= D_max

    min_keep = max(1, int(min_keep_ratio * M))
    if mask.sum() < min_keep:
        top_k_idx = np.argpartition(dists_to_query, min_keep - 1)[:min_keep]
        mask = np.zeros(M, dtype=bool)
        mask[top_k_idx] = True

    print(f"   🧭 空间邻域过滤: {mask.sum()}/{M} memory frames kept (≤{D_max:.1f}cm)")
    return mask


# ==============================================================================
# ✅ 修改：greedy_submodular_maximization_with_overlap_filter
# ==============================================================================

def greedy_submodular_maximization_with_overlap_filter(
    cover_score: np.ndarray,
    N: int,
    overlap_threshold: float = 0.4,
    early_stop_gain: float = 0.0,
    coverage_threshold: float = 0.3,
    coverage_target_ratio: float = 0.95,
    enable_coverage_early_stop: bool = False,
) -> Optional[Tuple[List[int], List[float]]]:
    M, T = cover_score.shape
    if M == 0 or T == 0:
        return None

    if N <= 0:
        return [], []

    single_max = np.max(cover_score, axis=1)
    candidate_order = np.argsort(-single_max)

    current_best = np.zeros(T, dtype=np.float32)
    selected: List[int] = []
    gains: List[float] = []

    for step in range(N):
        marginal_gain = np.sum(
            np.maximum(0.0, cover_score - current_best),
            axis=1,
        )
        marginal_gain[selected] = -1.0

        best_i = -1
        max_gain = -1.0
        for idx in candidate_order:
            if marginal_gain[idx] > max_gain:
                max_gain = marginal_gain[idx]
                best_i = idx
                if max_gain == marginal_gain.max():
                    break

        if max_gain <= early_stop_gain and step > 0:
            print(f"   ⏸️  边际增益早停: {max_gain:.4f} ≤ {early_stop_gain}")
            break

        selected.append(best_i)
        gains.append(float(max_gain))
        np.maximum(current_best, cover_score[best_i], out=current_best)

        if enable_coverage_early_stop:
            coverage_ratio = np.mean(current_best >= coverage_threshold)
            print(f"   📊 步骤 {len(selected)}/{N}: 覆盖率 = {coverage_ratio:.3f} "
                  f"(≥{coverage_threshold} 的 query 帧占比)")
            if coverage_ratio >= coverage_target_ratio:
                print(f"   🎯 覆盖目标达成！({coverage_ratio:.3f} ≥ {coverage_target_ratio}) → "
                      f"提前终止，共选 {len(selected)} 帧")
                break

    if not selected:
        return None

    # Post-filter by per-frame max overlap
    selected = np.array(selected)
    max_overlaps = np.max(cover_score[selected], axis=1)
    keep_mask = max_overlaps >= overlap_threshold
    filtered_selected = selected[keep_mask].tolist()
    filtered_gains = [gains[i] for i, keep in enumerate(keep_mask) if keep]

    print(f"   🧹 Overlap 阈值过滤 (≥{overlap_threshold}): "
          f"{len(filtered_selected)}/{len(selected)} 帧保留")

    if len(filtered_selected) == 0:
        print(f"   ❗ 警告：无任何帧满足 overlap ≥ {overlap_threshold}！")
        return None

    return filtered_selected, filtered_gains


# ==============================================================================
# 🖼️ viser
# ==============================================================================

def _launch_viser_visualization(
    memory_poses: List[Dict],
    query_poses: List[Dict],
    retrieved_indices: List[int],
    marginal_gains: List[float],
    host: str = "0.0.0.0",
    port: int = 8888,
    memory_images: Optional[List[np.ndarray]] = None,
    fov_deg: float = 90.0,
    aspect: float = 1.0,
    full_poses: Optional[List[Dict]] = None,
    memory_indices: Optional[List[int]] = None,
    query_indices: Optional[List[int]] = None,
) -> None:
    try:
        import viser
        import viser.transforms as tf
    except ImportError as e:
        raise ImportError("请安装 viser: pip install 'viser'") from e

    print(f"🚀 启动 Viser 服务（{host}:{port}）...")
    server = viser.ViserServer(host=host, port=port)

    prefix = f"/run_{int(time.time() * 1000)}/"

    all_positions = []
    if full_poses and len(full_poses) > 0:
        all_positions = [np.array(p["position"], dtype=np.float32) for p in full_poses]
    else:
        for p in memory_poses:
            all_positions.append(np.array(p["position"], dtype=np.float32))
        for p in query_poses:
            all_positions.append(np.array(p["position"], dtype=np.float32))

    if all_positions:
        all_positions = np.stack(all_positions, axis=0)
        center = np.mean(all_positions, axis=0)
        radius = float(np.max(np.linalg.norm(all_positions - center, axis=1)))
        radius = max(radius, 100.0)
    else:
        center = np.zeros(3, dtype=np.float32)
        radius = 100.0

    cam_dist = radius * 3.0
    base_scale = max(radius * 0.02, 1.0)
    mem_scale = base_scale
    ret_scale = base_scale * 1.2

    @server.on_client_connect
    def _on_client(client: "viser.ClientHandle") -> None:
        cam_pos = center + np.array([cam_dist, cam_dist, cam_dist], dtype=np.float32)
        with client.atomic():
            client.camera.position = cam_pos
            client.camera.look_at = center
            client.camera.fov = np.deg2rad(fov_deg)
            client.camera.near = max(radius / 100.0, 1.0)
            client.camera.far = radius * 10.0

        print(f"👀 设置初始相机: pos={client.camera.position}, look_at={center}")

    def pose_to_c2w(pose: Dict) -> np.ndarray:
        if "c2w" in pose and pose["c2w"] is not None:
            return pose["c2w"].copy()
        else:
            return get_camera_c2w_from_pose_dict(pose)

    def add_label(name: str, text: str, position: np.ndarray) -> None:
        server.scene.add_label(
            name=name,
            text=text,
            position=tuple(position.tolist()),
            font_size_mode="screen",
            font_screen_scale=1.0,
            anchor="bottom-center",
        )

    gui = server.gui
    with gui.add_folder("🔍 Camera Memory Debugger"):
        show_memory = gui.add_checkbox("Memory Frustums", False)
        show_retrieved = gui.add_checkbox("Retrieved Frames", True)
        show_traj = gui.add_checkbox("Trajectories", True)
        show_labels = gui.add_checkbox("Labels (ID / Gain / Frame)", False)

    scene = server.scene
    fov_rad = np.deg2rad(fov_deg)
    aspect_val = float(aspect)

    if full_poses and memory_indices and query_indices:
        mem_set, traj_set = set(memory_indices), set(query_indices)
        pts_other, pts_mem, pts_traj = [], [], []
        for idx, p in enumerate(full_poses):
            pos = np.array(p["position"], dtype=np.float32)
            if idx in mem_set:
                pts_mem.append(pos)
            elif idx in traj_set:
                pts_traj.append(pos)
            else:
                pts_other.append(pos)

        if len(pts_mem) > 1 and show_traj.value:
            scene.add_spline_catmull_rom(
                prefix + "traj_memory",
                np.stack(pts_mem),
                tension=0.5,
                line_width=3.0,
                color=(255, 165, 0),
            )
        if len(pts_traj) > 1 and show_traj.value:
            scene.add_spline_catmull_rom(
                prefix + "traj_query",
                np.stack(pts_traj),
                tension=0.5,
                line_width=3.0,
                color=(255, 0, 0),
            )

    if show_memory.value:
        for i, pose in enumerate(memory_poses):
            T_c2w = pose_to_c2w(pose)
            scene.add_camera_frustum(
                prefix + f"cam/memory_{i}",
                fov_rad,
                aspect_val,
                color=(150, 150, 150),
                image=None,
                wxyz=tf.SO3.from_matrix(T_c2w[:3, :3]).wxyz,
                position=T_c2w[:3, 3],
                scale=mem_scale,
                variant="wireframe",
            )
            if show_labels.value:
                frame_idx = pose.get("frame_index", i)
                label_pos = (
                    T_c2w[:3, 3]
                    + T_c2w[:3, :3] @ np.array([0.0, 0.0, mem_scale * 1.5])
                )
                add_label(
                    prefix + f"label/memory_{i}",
                    f"M{i} (f={frame_idx})",
                    label_pos,
                )

    if show_retrieved.value and retrieved_indices:
        for order_idx, mem_idx in enumerate(retrieved_indices):
            pose = memory_poses[mem_idx]
            T_c2w = pose_to_c2w(pose)
            scene.add_camera_frustum(
                prefix + f"cam/retrieved_{mem_idx}",
                fov_rad,
                aspect_val,
                color=(255, 0, 0),
                image=None,
                wxyz=tf.SO3.from_matrix(T_c2w[:3, :3]).wxyz,
                position=T_c2w[:3, 3],
                scale=ret_scale,
                variant="wireframe",
            )
            if show_labels.value:
                frame_idx = pose.get("frame_index", mem_idx)
                parts = [f"R{order_idx} ← M{mem_idx}", f"f={frame_idx}"]
                if order_idx < len(marginal_gains):
                    parts.append(f"Δ={marginal_gains[order_idx]:.2f}")
                label = "\n".join(parts)
                label_pos = (
                    T_c2w[:3, 3]
                    + T_c2w[:3, :3] @ np.array([0.0, 0.0, ret_scale * 1.7])
                )
                add_label(
                    prefix + f"label/retrieved_{mem_idx}",
                    label,
                    label_pos,
                )

    display_host = "localhost" if host == "0.0.0.0" else host
    print(f"✅ 请打开: http://{display_host}:{port}")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        server.stop()
        print("✅ 服务已停止。")


# ==============================================================================
# 🔍 retrieve_reference_frames
# ==============================================================================

def retrieve_reference_frames(
    memory_poses: List[Dict],
    query_trajectory: List[Dict],
    N: int = 20,
    visualize: bool = False,
    viser_host: str = "0.0.0.0",
    viser_port: int = 8888,
    memory_images: Optional[List[np.ndarray]] = None,
    fov_deg: float = 90.0,
    aspect: float = 1.0,
    full_poses: Optional[List[Dict]] = None,
    memory_indices: Optional[List[int]] = None,
    query_indices: Optional[List[int]] = None,
    use_spatial_filter: bool = True,
    spatial_percentile: float = 95.0,
    num_samples: int = 128,
    early_stop_gain: float = 0.0,
    memory_batch_size: int = 200,
    coverage_threshold: float = 0.3,
    coverage_target_ratio: float = 0.95,
    enable_coverage_early_stop: bool = False,
    overlap_threshold: float = 0.4,
) -> Optional[List[int]]:
    print(f"🔍 检索：M={len(memory_poses)}, T={len(query_trajectory)}, N={N}")

    cover_score = compute_frustum_overlap_scores_mc(
        memory_poses=memory_poses,
        query_poses=query_trajectory,
        fov_deg=fov_deg,
        aspect=aspect,
        num_samples=num_samples,
        memory_batch_size=memory_batch_size,
    )
    print(
        f"  → Monte Carlo FOV overlap: shape={cover_score.shape}, "
        f"nonzero={np.count_nonzero(cover_score)}/{cover_score.size}"
    )

    if use_spatial_filter and len(memory_poses) > 1:
        spatial_mask = spatial_proximity_filter(
            memory_poses,
            query_trajectory,
            percentile=spatial_percentile,
            min_keep_ratio=0.1,
        )
        cover_score_filtered = cover_score.copy()
        cover_score_filtered[~spatial_mask, :] = 0.0
        print(
            f"  → 空间过滤后: {spatial_mask.sum()} / {cover_score.shape[0]} frames remain"
        )
    else:
        cover_score_filtered = cover_score

    result = greedy_submodular_maximization_with_overlap_filter(
        cover_score_filtered,
        N,
        overlap_threshold=overlap_threshold,
        early_stop_gain=early_stop_gain,
        coverage_threshold=coverage_threshold,
        coverage_target_ratio=coverage_target_ratio,
        enable_coverage_early_stop=enable_coverage_early_stop,
    )

    if result is None:
        print(f"❗ 未检索到任何满足 overlap ≥ {overlap_threshold} 的参考帧！")
        return None

    selected_indices, gains = result
    print(f"  → 检索完成！共选 {len(selected_indices)} 帧（后过滤），总增益: {sum(gains):.2f}")

    if selected_indices:
        m_pos = np.stack([p["position"] for p in memory_poses])
        q_pos = np.stack([p["position"] for p in query_trajectory])
        avg_d = np.mean([
            np.linalg.norm(m_pos[i:i + 1] - q_pos, axis=1).min()
            for i in selected_indices
        ])
        print(f"  → 选中帧平均距 query: {avg_d:.1f} cm")

    if visualize:
        try:
            _launch_viser_visualization(
                memory_poses,
                query_trajectory,
                selected_indices,
                gains,
                host=viser_host,
                port=viser_port,
                memory_images=memory_images,
                fov_deg=fov_deg,
                aspect=aspect,
                full_poses=full_poses,
                memory_indices=memory_indices,
                query_indices=query_indices,
            )
        except Exception as e:
            print(f"⚠️  Viser 失败: {e}")

    return selected_indices


# ==============================================================================
# 🚀 主程序
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Camera Memory Retrieval v5.4.1 — Fixed parse_range"
    )
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="outputs_retrieval_cord")
    parser.add_argument("--video", type=str, default=None)

    parser.add_argument("--pose_json", type=str, default=None)
    
    # parser.add_argument("--coordinates_txt", type=str, default="results_test/camera_ctrl_ys_ue1.3b_noref_step800/clip_006_pose.txt")
    # parser.add_argument("--memory-range", type=str, default="0:70")
    # parser.add_argument("--traj-range", type=str, default="71:100")

    parser.add_argument("--coordinates_txt", type=str, default="results_test/coordinates/AnimeCitySuburbs_3_pose.txt")
    parser.add_argument("--memory-range", type=str, default="0:6999")
    parser.add_argument("--traj-range", type=str, default="7000:7600")


    parser.add_argument("--spatial-percentile", type=float, default=95.0)
    parser.add_argument("--no-spatial-filter", action="store_true")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=128,
        help="Monte Carlo 采样点数（每个 query 视锥）。",
    )
    parser.add_argument("--fast", action="store_true", help="⚡ 启用高速模式")

    parser.add_argument("--overlap_threshold", type=float, default=0.8, help="单帧最大 overlap 阈值；低于此值的帧将被过滤掉（默认: 0.4）")

    args = parser.parse_args()

    print("=" * 60)
    print("📷 Camera Memory Retrieval v5.4.1 — Fixed parse_range + Overlap Threshold")
    print(f"🎯 overlap_threshold = {args.overlap_threshold}")
    if args.fast:
        print("⚡ 启用 FAST 模式")
    print("=" * 60)

    if not (args.pose_json or args.coordinates_txt):
        parser.error("请提供 --pose_json 或 --coordinates_txt")

    poses, frames, fps, actual_video_path = load_pose_and_video(
        pose_json_path=args.pose_json,
        coordinates_txt_path=args.coordinates_txt,
        video_path=args.video,
    )
    full_poses = poses
    L = len(poses)

    # ✅ 此处 now works: parse_range is defined
    memory_start, memory_end = parse_range(args.memory_range, L)
    traj_start, traj_end = parse_range(args.traj_range, L)
    memory_indices = list(range(memory_start, memory_end))
    query_indices = list(range(traj_start, traj_end))

    memory_poses = [poses[i] for i in memory_indices]
    query_traj = [poses[i] for i in query_indices]
    memory_images = [frames[i] for i in memory_indices]
    aspect = frames[0].shape[1] / frames[0].shape[0] if frames else 1.0

    print(f"🧠 Memory [{memory_start}, {memory_end}) → {len(memory_poses)} 帧")
    print(f"🚶 Query  [{traj_start}, {traj_end}) → {len(query_traj)} 帧")

    N = min(args.N, len(memory_poses))
    if N < args.N:
        print(f"⚠️  N 自动裁剪为 {N}")

    if args.fast:
        effective_num_samples = max(32, args.num_samples // 4)
        use_spatial_filter = False
        early_stop_gain = 1e-3
        coverage_threshold = 0.8
        coverage_target_ratio = 0.95
        enable_coverage_early_stop = True
        memory_batch_size = 200
        print(f"   ⚙️  FAST 模式参数: samples={effective_num_samples}, "
              f"cov_thresh={coverage_threshold}, cov_target={coverage_target_ratio}")
    else:
        effective_num_samples = args.num_samples
        use_spatial_filter = not args.no_spatial_filter
        early_stop_gain = 1e-8
        coverage_threshold = 0.8
        coverage_target_ratio = 0.95
        enable_coverage_early_stop = False
        memory_batch_size = 500

    start_time = time.time()
    retrieved_indices = retrieve_reference_frames(
        memory_poses=memory_poses,
        query_trajectory=query_traj,
        N=N,
        use_spatial_filter=use_spatial_filter,
        spatial_percentile=args.spatial_percentile,
        visualize=args.vis,
        viser_host=args.host,
        viser_port=args.port,
        memory_images=memory_images,
        fov_deg=90.0,
        aspect=aspect,
        full_poses=full_poses,
        memory_indices=memory_indices,
        query_indices=query_indices,
        num_samples=effective_num_samples,
        early_stop_gain=early_stop_gain,
        memory_batch_size=memory_batch_size,
        coverage_threshold=coverage_threshold,
        coverage_target_ratio=coverage_target_ratio,
        enable_coverage_early_stop=enable_coverage_early_stop,
        overlap_threshold=args.overlap_threshold,
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