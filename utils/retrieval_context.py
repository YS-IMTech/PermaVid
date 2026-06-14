# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

import torch
import numpy as np
from typing import List, Tuple, Dict
import math

def generate_points_in_sphere(n_points: int, radius: float) -> torch.Tensor:
    """
        Uniformly sample points within a sphere of a specified radius.

        :param n_points: The number of points to generate.
        :param radius: The radius of the sphere.
        :return: A tensor of shape (n_points, 3), representing the (x, y, z) coordinates of the points.
    """
    samples_r = torch.rand(n_points)
    samples_phi = torch.rand(n_points)
    samples_u = torch.rand(n_points)

    r = radius * torch.pow(samples_r, 1 / 3)
    phi = 2 * math.pi * samples_phi
    theta = torch.acos(1 - 2 * samples_u)

    # transfer the coordinates from spherical to cartesian
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)

    points = torch.stack((x, y, z), dim=1)
    return points


def rotation_matrix_to_angles(R: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Estimate the Pitch and Yaw angles from a 3x3 rotation matrix R in the camera coordinate system.

        Assumed Camera Coordinate System: X=Right, Y=Up, Z=Backward 
        (or NeRF style: X=Right, Y=Down, Z=Forward).
        Here we adopt the common Computer Vision convention: Z-axis is Forward.

        Note: The angle calculations here are directly based on the conventions of your `is_inside_fov_3d_hv` function:
        - Yaw/Azimuth angle is in the XZ plane (atan2(x, z)).
        - Pitch/Elevation angle is relative to the horizontal plane (atan2(y, sqrt(x^2 + z^2))).

        For the third column R[:, 2] of the W2C matrix R (the direction of the World Z-axis in the Camera frame),
        this typically corresponds to the direction the camera is looking 
        (the representation of the world Z-axis in the camera frame).

        To simplify and match your `is_inside_fov` logic, we directly use the camera's Z-axis vector:
        Camera Z-axis direction in World Frame (Forward Vector): fwd = R_w2c_inv @ [0, 0, 1]
        More simply, the Z-axis vector of the C2W matrix is the camera's forward vector in the world frame.
        C2W = W2C_inv
    """

    R_c2w = R.T
    fwd = R_c2w[:, 2]  

    x = fwd[0]
    y = fwd[1]
    z = fwd[2]

    # compute yaw and pitch
    yaw_rad = torch.atan2(x, z)
    yaw_deg = yaw_rad * (180.0 / math.pi)
    pitch_rad = torch.atan2(y, torch.sqrt(x ** 2 + z ** 2))
    pitch_deg = pitch_rad * (180.0 / math.pi)

    return pitch_deg, yaw_deg

def is_inside_fov_3d_hv(points: torch.Tensor, center: torch.Tensor,
                        center_pitch: torch.Tensor, center_yaw: torch.Tensor,
                        fov_half_h: torch.Tensor, fov_half_v: torch.Tensor) -> torch.Tensor:
    """
        Check whether points are inside a 3D view frustum defined by a center coordinate, pitch angle, and yaw angle.

        :param points: Tensor of shape (N, 3) or (N, B, 3) representing the coordinates of the sampled points.
        :param center: Tensor of shape (3) or (B, 3) representing the camera center coordinates.
        :param center_pitch: Tensor of shape (1) or (B) representing the pitch angle of center view direction.
        :param center_yaw: Tensor of shape (1) or (B) representing the yaw angle of the center view direction.
        :param fov_half_h: The horizontal half field-of-view angle (in degrees).
        :param fov_half_v: The vertical half field-of-view angle (in degrees).
        :return: Boolean tensor of shape (N) or (N, B), indicating whether each point is inside the FOV.
    """
    if points.ndim == 2:  # N, 3
        vectors = points - center[None, :]
        C = 1  
    elif points.ndim == 3:  # N, B, 3
        vectors = points - center[None, ...]
        center_pitch = center_pitch[None, :] if center_pitch.ndim == 1 else center_pitch
        center_yaw = center_yaw[None, :] if center_yaw.ndim == 1 else center_yaw
    else:
        raise ValueError("points' shape should be (N, 3) or (N, B, 3)")

    x = vectors[..., 0]
    y = vectors[..., 1]
    z = vectors[..., 2]

    # Calculate the horizontal angle (yaw/azimuth), assuming the Z-axis is forward.
    azimuth = torch.atan2(x, z) * (180 / math.pi)

    # Calculate the vertical angle (pitch/elevation).
    elevation = torch.atan2(y, torch.sqrt(x ** 2 + z ** 2)) * (180 / math.pi)

    # Calculate the angular difference from the center view direction (handling angle wrapping).
    diff_azimuth = (azimuth - center_yaw)
    diff_azimuth = torch.remainder(diff_azimuth + 180, 360) - 180

    diff_elevation = (elevation - center_pitch)
    diff_elevation = torch.remainder(diff_elevation + 180, 360) - 180

    # Check if within FOV
    in_fov_h = diff_azimuth.abs() < fov_half_h
    in_fov_v = diff_elevation.abs() < fov_half_v

    return in_fov_h & in_fov_v

def calculate_fov_overlap_similarity(
        w2c_matrix_curr: torch.Tensor,
        w2c_matrix_hist: torch.Tensor,
        fov_h_deg: float = 105.0,
        fov_v_deg: float = 75.0,
        device=None,
        points_local=None,
) -> float:
    """
        Calculate the Field-of-View (FOV) overlap similarity between two W2C poses using Monte Carlo sampling.

        Similarity = (Number of points in Curr_FOV ∩ Hist_FOV) / (Number of points in Curr_FOV).

        :param w2c_matrix_curr: The (4, 4) W2C matrix for the current frame.
        :param w2c_matrix_hist: The (4, 4) W2C matrix for the historical frame.
        :param num_samples, radius, fov_h_deg, fov_v_deg: Sampling and FOV parameters.
        :return: The overlap ratio (a float between 0.0 and 1.0).
    """
    w2c_matrix_curr = torch.tensor(w2c_matrix_curr, device=device)
    w2c_matrix_hist = torch.tensor(w2c_matrix_hist, device=device)

    c2w_matrix_curr = torch.linalg.inv(w2c_matrix_curr)
    c2w_matrix_hist = torch.linalg.inv(w2c_matrix_hist)
    C_inv = w2c_matrix_curr

    w2c_matrix_curr = torch.linalg.inv(C_inv @ c2w_matrix_curr)
    w2c_matrix_hist = torch.linalg.inv(C_inv @ c2w_matrix_hist)

    R_curr, t_curr = w2c_matrix_curr[:3, :3], w2c_matrix_curr[:3, 3]
    R_hist, t_hist = w2c_matrix_hist[:3, :3], w2c_matrix_hist[:3, 3]
    P_w_curr = -R_curr.T @ t_curr
    P_w_hist = -R_hist.T @ t_hist

    # pitch, yaw
    pitch_curr, yaw_curr = rotation_matrix_to_angles(R_curr)
    pitch_hist, yaw_hist = rotation_matrix_to_angles(R_hist)

    fov_half_h = torch.tensor(fov_h_deg / 2.0, device=device)
    fov_half_v = torch.tensor(fov_v_deg / 2.0, device=device)

    # move to P_w_curr (N, 3)
    points_world = points_local + P_w_curr[None, :]

    in_fov_curr = is_inside_fov_3d_hv(
        points_world, P_w_curr[None, :],
        pitch_curr[None], yaw_curr[None],
        fov_half_h, fov_half_v
    )

    # compute based on angle
    in_fov_hist = is_inside_fov_3d_hv(
        points_world, P_w_hist[None, :],
        pitch_hist[None], yaw_hist[None],
        fov_half_h, fov_half_v
    )

    # compute based on distance
    dist = torch.norm(points_world - P_w_hist.reshape(1, -1), dim=1) < 8.0
    in_fov_hist = in_fov_hist.bool() & dist.reshape(1, -1).bool()

    overlap_count = (in_fov_curr.bool() & in_fov_hist.bool()).sum().float()
    fov_curr_count = in_fov_curr.sum().float()

    if fov_curr_count == 0:
        return 0.0  

    overlap_ratio = overlap_count / fov_curr_count

    return overlap_ratio.item()



def coordinates_to_w2cs(coordinates: List[Tuple]):

    if not coordinates:
        raise ValueError("Input coordinates list is empty.")

    w2cs = []
    for idx, item in enumerate(coordinates):
        parts = list(item)
        if len(parts) < 19:
            continue
        try:
            data = [float(x) for x in parts]
        except (ValueError, TypeError):
            continue

        frame_idx = int(round(data[0]))
        w2c_flat = data[7:19]  # 12 numbers
        w2c = np.array(w2c_flat, dtype=np.float32).reshape(3, 4)
        w2c_4x4 = np.eye(4, dtype=np.float32)
        w2c_4x4[:3, :] = w2c
        c2w = np.linalg.inv(w2c_4x4)

        w2cs.append(w2c_4x4)
    return w2cs

def find_high_overlap_history_frames(
    current_frame_idx: int,
    coordinates: List[Tuple] = None,
    w2c_list: List[np.ndarray] = None, # allow not input
    overlap_thred: float = 0.8,
    fov_h_deg: float = 60.0,
    fov_v_deg: float = 35.0,
    device=None,
    points_local=None,
) -> List[int]:
    """
    Find all historical frame indices in [0, current_frame_idx) whose FOV has high overlap
    with the current frame (indexed by current_frame_idx).

    Similarity is computed via Monte Carlo sampling of a spherical volume and measuring
    the fraction of points visible in both current and candidate frames.

    Args:
        current_frame_idx (int): Index of the current frame.
        coordinates (List[Tuple], optional): Raw pose data to convert to W2C matrices.
        w2c_list (List[np.ndarray], optional): Precomputed list of 4x4 W2C matrices.
        overlap_thred (float): Threshold for FOV overlap similarity (default: 0.6).
        fov_h_deg (float): Horizontal field of view in degrees (default: 60.0).
        fov_v_deg (float): Vertical field of view in degrees (default: 35.0).
        device: PyTorch device for computation.
        points_local: Pre-sampled points in local camera space (shape: N x 3). If None, auto-generated.

    Returns:
        List[int]: Indices of historical frames with FOV overlap > overlap_thred.
    """
    if w2c_list is None:
        if coordinates is None:
            raise ValueError("Either 'w2c_list' or 'coordinates' must be provided.")
        w2c_list = coordinates_to_w2cs(coordinates)

    num_total_frames = len(w2c_list)
    if current_frame_idx >= num_total_frames or current_frame_idx <= 0:
        return []

    # Ensure points_local is on correct device
    if points_local is None:
        points_local = generate_points_in_sphere(50000, 8.0).to(device)
    else:
        points_local = points_local.to(device)

    current_w2c = torch.tensor(w2c_list[current_frame_idx], dtype=torch.float32, device=device)

    selected_indices = []
    overlap_alllist = []
    
    for hist_idx in range(0, current_frame_idx):
        hist_w2c = torch.tensor(w2c_list[hist_idx], dtype=torch.float32, device=device)

        overlap = calculate_fov_overlap_similarity(
            w2c_matrix_curr=current_w2c,
            w2c_matrix_hist=hist_w2c,
            fov_h_deg=fov_h_deg,
            fov_v_deg=fov_v_deg,
            device=device,
            points_local=points_local
        )

        overlap_alllist.append(overlap)
        
        if overlap > overlap_thred:
            selected_indices.append(hist_idx)

    print("all history frames overlap scores:", overlap_alllist)
    return selected_indices

def select_aligned_memory_frames(
    # w2c_list: List[np.ndarray],
    current_frame_idx: int,
    memory_start_id: int = 0,
    coordinates: List[Tuple] = None,
    w2c_list: List[np.ndarray] = None,
    reference_spatial_memory_frames = 10,
    reference_temporal_memory_frames = 0,
    num_frames_pred = 41,
    device=None,
    points_local=None,
    filter_bool: bool = True,
    overlap_thred: float = 0.6,
    pos_weight: float = 1.0,
    ang_weight: float = 1.0,
) -> List[int]:
    """
        Input params
        - w2c_list: 所有帧的 world-to-camera 4x4 矩阵列表（按时间顺序）
        - current_frame_idx: 当前帧索引（以该帧为分界，选历史 memory + 近邻 context)
        - reference_spatial_memory_frames: 需要选取的“长期记忆帧”数量（不包含 context)
        - reference_temporal_memory_frames: context 长度（直接取当前帧之前的最近 N 帧）
        - num_frames_pred: query 窗口长度（用于评估候选与“未来窗口”的视锥重叠）
        - device / points_local: 传给 calculate_fov_overlap_similarity 的可选参数
        - filter_bool: True 开启 overlap 阈值过滤（仅过滤 memory 候选，使用缓存不额外计算）
        - overlap_thred: overlap 过滤阈值(max_overlap < 阈值的候选会被跳过，继续向后补齐）

    """
    if w2c_list is None:
        w2c_list = coordinates_to_w2cs(coordinates)
    if points_local is None:
        points_local = generate_points_in_sphere(50000, 8.0).to(device)


    num_total_frames = len(w2c_list)
    if num_total_frames == 0:
        return []

    if current_frame_idx >= num_total_frames or current_frame_idx < 0:
        raise ValueError(f"current_frame_idx out of range: {current_frame_idx}, total={num_total_frames}")

    if current_frame_idx == 0:
        return []
    # if current_frame_idx <= reference_spatial_memory_frames:
    #     return list(range(0, current_frame_idx))

    # ---------- 1) context：最近 reference_temporal_memory_frames 帧 ----------
    start_context_idx = max(0, current_frame_idx - reference_temporal_memory_frames)
    context_frames_indices = list(range(start_context_idx, current_frame_idx))

    # ---------- 2) query：从 current_frame_idx 开始往后 num_frames_pred 帧 ----------
    q_end = min(num_total_frames, current_frame_idx + num_frames_pred)
    query_clip_indices = list(range(current_frame_idx, q_end))
    if len(query_clip_indices) == 0:
        query_clip_indices = [current_frame_idx]

    # ----------  memory_start_id 合法化 ----------
    memory_start_id = max(0, memory_start_id)
    memory_start_id = min(memory_start_id, start_context_idx)


    # ---------- 3) memory candidates：所有早于 context 的历史帧（逐帧候选） ----------
    candidate_indices = list(range(memory_start_id, start_context_idx))
    if len(candidate_indices) == 0:
        return sorted(set(context_frames_indices))

    # ---------- 4) 逐帧计算：avg_dist + max_overlap（缓存用于方案A） ----------
    # dist = 1 - overlap
    candidate_stats = []  # (cand_idx, avg_dist, max_overlap)
    for cand_idx in candidate_indices:
        print("cand_idx:", cand_idx)
        cand_w2c = w2c_list[cand_idx]

        total_dist = 0.0
        max_ov = -1.0
        for query_idx in query_clip_indices:
            ov = calculate_fov_overlap_similarity(
                w2c_list[query_idx],
                cand_w2c,
                fov_h_deg=60.0,
                fov_v_deg=35.0,
                device=device,
                points_local=points_local
            )
            ov = float(ov)
            if ov > max_ov:
                max_ov = ov

            total_dist += (1.0 - ov)

        avg_dist = total_dist / max(1, len(query_clip_indices))
        candidate_stats.append((cand_idx, avg_dist, max_ov))

    # 越小越好（越相似）
    candidate_stats.sort(key=lambda x: x[1])

    # ---------- 5) 选 memory：若开启过滤，则用 max_overlap 缓存做阈值过滤并向后补齐 ----------
    memory_frames_indices: List[int] = []

    if not filter_bool:
        k = min(reference_spatial_memory_frames, len(candidate_stats))
        memory_frames_indices = [idx for idx, _, _ in candidate_stats[:k]]
    else:
        # 严格挑满 reference_spatial_memory_frames（过滤后不足则继续往后补）
        for cand_idx, _, max_ov in candidate_stats:
            if max_ov >= float(overlap_thred):
                memory_frames_indices.append(cand_idx)
                if len(memory_frames_indices) >= reference_spatial_memory_frames:
                    break
        # 如果候选本来就不足，最终可能仍然不满，这是合理的兜底

    # ---------- 6) 合并 context + memory，去重排序 ----------
    final_selected_frames = sorted(set(context_frames_indices).union(memory_frames_indices))
    if final_selected_frames:
        print(f"[INFO] Final selected frames = {final_selected_frames}")
    else:
        print("[INFO] No frames selected (empty result).")
    return final_selected_frames
