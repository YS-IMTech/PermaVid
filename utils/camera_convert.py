import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Union, Optional
from typing import Dict, List, Optional, Tuple
import json
import torch
import einops
from typing_extensions import Literal

def quaternion_to_w2c(poses: Union[str, np.ndarray]) -> np.ndarray:
    if isinstance(poses, str):
        try:
            data = np.load(poses)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {poses}") from e
    elif isinstance(poses, np.ndarray):
        data = poses
    else:
        raise TypeError("`poses` must be a file path (str) or a NumPy array.")

    if data.ndim != 2 or data.shape[1] != 7:
        raise ValueError(f"Input array must be of shape (n, 7), got {data.shape}")

    n = data.shape[0]
    
    pose = torch.tensor(data)
    
    quat_t = pose[..., :3]  # Translation
    quat_r = pose[..., 3:]  # Quaternion rotation
    w2c_matrix = torch.zeros((n, 3, 4), device=pose.device)
    w2c_matrix[..., :3, 3] = quat_t 
    w2c_matrix[..., :3, :3] = quaternion_to_matrix(quat_r)
    w2c_matrix = w2c_matrix.numpy()
    
    return w2c_matrix
    
    



# def quaternion_to_w2c(poses: Union[str, np.ndarray]) -> np.ndarray:
#     if isinstance(poses, str):
#         try:
#             data = np.load(poses)
#         except FileNotFoundError as e:
#             raise FileNotFoundError(f"File not found: {poses}") from e
#     elif isinstance(poses, np.ndarray):
#         data = poses
#     else:
#         raise TypeError("`poses` must be a file path (str) or a NumPy array.")

#     if data.ndim != 2 or data.shape[1] != 7:
#         raise ValueError(f"Input array must be of shape (n, 7), got {data.shape}")

#     n = data.shape[0]
#     w2c = np.empty((n, 3, 4), dtype=np.float32)
    
#     t = data[:, :3]           # (n, 3)
#     quat_xyzw = data[:, 3:7]   # (n, 4), [x, y, z, w]
#     rotations = R.from_quat(quat_xyzw)
#     R_mats = rotations.as_matrix()  # (n, 3, 3)

#     w2c[:, :, :3] = R_mats
#     w2c[:, :, 3] = t  

#     return w2c

def euler_to_c2w_batch(poses: List[Dict], return_34: bool = True) -> np.ndarray:
    """
    批量：输入 read_uepose_from_json 得到的 poses(List[Dict]),
    输出形状为 (N, 3, 4) 或 (N, 4, 4) 的 c2w。
    """
    mats = [euler_to_c2w(p, return_34=return_34) for p in poses]
    return np.stack(mats, axis=0)  # (N, 3/4, 4)


def euler_to_w2c_batch(poses: List[Dict], return_34: bool = True) -> np.ndarray:
    """
    批量：输入 read_uepose_from_json 得到的 poses(List[Dict]),
    输出形状为 (N, 3, 4) 或 (N, 4, 4) 的 w2c。
    """

    mats = [euler_to_w2c(p, return_34=return_34) for p in poses]
    return np.stack(mats, axis=0)  # (N, 3/4, 4)

def euler_zyx_to_c2w_rotation(angles_deg: np.ndarray) -> np.ndarray:
    """
    欧拉角 → 旋转矩阵
    输入 angles_deg: [Pitch, Yaw, Roll](度)

    使用 scipy 的 "ZYX" 顺序：
        R = Rz(yaw) * Ry(pitch) * Rx(roll)
    """
    pitch, yaw, roll = angles_deg

    rot = R.from_euler("ZYX", [yaw, pitch, roll], degrees=True)
    
    return rot.as_matrix()

# def euler_zyx_to_c2w_rotation(angles_deg: np.ndarray) -> np.ndarray:
    
#     # roll, pitch, yaw = angles_deg
#     pitch, yaw, roll = angles_deg
#     yaw_rad = np.deg2rad(yaw)
#     pitch_rad = np.deg2rad(pitch)
    
#     R_y = np.array([[np.cos(yaw_rad), 0, np.sin(yaw_rad)],
#                     [0, 1, 0],
#                     [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]])
    
#     R_z = np.array([[np.cos(pitch_rad), -np.sin(pitch_rad), 0],
#                     [np.sin(pitch_rad), np.cos(pitch_rad), 0],
#                     [0, 0, 1]])
    
#     R = np.dot(R_z, R_y)
    
    
#     return R


def read_uepose_from_json(pose_json_path: str):
    with open(pose_json_path, "r") as f:
        pose_data = json.load(f)

    if "CineCameraActor" in pose_data:
        actor_dict = pose_data["CineCameraActor"]
    else:
        first_key = next(iter(pose_data.keys()))
        actor_dict = pose_data[first_key]

    sorted_items = sorted(actor_dict.items(), key=lambda kv: int(kv[0]))
    poses_raw = [v for _, v in sorted_items]

    poses: List[Dict] = []
    for idx, p in enumerate(poses_raw):
        poses.append(
            {
                "position": np.array(p["position"], dtype=np.float32) ,  #ue单位是cm
                "rotation": np.array(p["rotation"], dtype=np.float32),
                "scale": np.array(
                    p.get("scale", [1.0, 1.0, 1.0]), dtype=np.float32
                ),
                "frame_index": idx,
            }
        )
    return poses


def euler_to_c2w(pose: Dict, return_34=True) -> np.ndarray:
    """Get camera-to-world matrix (4x4) from pose dict."""

    S = np.array([
        [1, 0, 0, 0],   # 新 x
        [0, 0, -1, 0],   # 新 y
        [0, 1, 0, 0],   # 新 z
        [0, 0, 0, 1]
    ], dtype=np.float32)

    # S_inv = np.linalg.inv(S)
    R_mat = euler_zyx_to_c2w_rotation(
        np.array(pose["rotation"], dtype=np.float32)
    )
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R_mat
    c2w[:3, 3] = np.array(pose["position"], dtype=np.float32)

    c2w = c2w[:, [1, 2, 0, 3]]

    # c2w = c2w[:, [1, 0, 2, 3]]
    # c2w[:3, 1] *= -1.
    c2w[:3, 3] /= 100
    # c2w = c2w @ S

    if return_34:
        return c2w[:3, :4]
    return c2w

def euler_to_w2c(pose: Dict, return_34=True) -> np.ndarray:
    """Get world-to-camera matrix (4x4) from pose dict."""

    c2w = euler_to_c2w(pose, return_34=False)
    w2c = np.linalg.inv(c2w)
    if return_34:
        return w2c[:3, :4]
    return w2c


def euler_to_quaternion(euler: np.ndarray, unit: str = 'deg') -> np.ndarray:
    """
    Convert Euler angles (pitch, yaw, roll) to unit quaternions (x, y, z, w).
    Input must be np.ndarray of shape (..., 3).
    """
    if euler.shape[-1] != 3:
        raise ValueError(f"Expected last dim = 3, got {euler.shape}")

    pitch, yaw, roll = euler[..., 0], euler[..., 1], euler[..., 2]

    if unit == 'deg':
        pitch = np.radians(pitch)
        yaw   = np.radians(yaw)
        roll  = np.radians(roll)
    elif unit != 'rad':
        raise ValueError("unit must be 'rad' or 'deg'")

    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)

    qx = cr * sp * cy + sr * cp * sy
    qy = sr * cp * cy - cr * sp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy

    q = np.stack([qx, qy, qz, qw], axis=-1)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    # Avoid division by zero
    norm = np.where(norm == 0.0, 1.0, norm)
    return q / norm


def quaternion_to_euler(q: np.ndarray, unit: str = 'deg') -> np.ndarray:
    """
    Convert unit quaternions (x, y, z, w) to Euler angles (pitch, yaw, roll).
    Input must be np.ndarray of shape (..., 4).
    """
    if q.shape[-1] != 4:
        raise ValueError(f"Expected last dim = 4, got {q.shape}")

    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Normalize
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    inv_norm = 1.0 / norm[..., 0]
    qx, qy, qz, qw = qx * inv_norm, qy * inv_norm, qz * inv_norm, qw * inv_norm

    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)

    pitch = np.arcsin(sinp)
    yaw   = np.zeros_like(pitch)
    roll  = np.zeros_like(pitch)

    gimbal_mask = np.abs(sinp) >= 1.0 - 1e-6

    if np.any(gimbal_mask):
        pitch = np.where(gimbal_mask, np.where(sinp > 0, np.pi/2, -np.pi/2), pitch)
        yaw   = np.where(gimbal_mask, 0.0, yaw)
        roll  = np.where(gimbal_mask, 2.0 * np.arctan2(qx, qw), roll)

    mask = ~gimbal_mask
    if np.any(mask):
        yaw[mask] = np.arctan2(
            2.0 * (qw[mask] * qz[mask] + qx[mask] * qy[mask]),
            1.0 - 2.0 * (qy[mask] * qy[mask] + qz[mask] * qz[mask])
        )
        roll[mask] = np.arctan2(
            2.0 * (qw[mask] * qx[mask] + qy[mask] * qz[mask]),
            1.0 - 2.0 * (qx[mask] * qx[mask] + qy[mask] * qy[mask])
        )

    euler = np.stack([pitch, yaw, roll], axis=-1)

    if unit == 'deg':
        euler = np.degrees(euler)
    elif unit != 'rad':
        raise ValueError("output_unit must be 'rad' or 'deg'")

    return euler


def quaternion_to_matrix(quaternions, eps: float = 1e-8):
    """
    Convert 4-dimensional quaternions to 3x3 rotation matrices.
    This is adapted from Pytorch3D:
    https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py

    Args:
        quaternions: Quaternion tensor [..., 4] (order: i, j, k, r)
        eps: Small value for numerical stability

    Returns:
        Rotation matrices [..., 3, 3]
    """

    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return einops.rearrange(o, "... (i j) -> ... i j", i=3, j=3)


def poses_intrinsics_to_coordinates(
    w2c_poses: np.ndarray,      # (n, 3, 4) → world-to-camera matrices
    intrinsics: np.ndarray,     # (n, 4) → [fx, fy, cx, cy]
) -> List[Tuple]:
    """
    Convert w2c poses and intrinsics to list of Camera entry tuples (length-19).

    Output entry format (19 elements):
        (frame_id, fx, fy, cx, cy, 0, 0, w2c_row0..., w2c_row1..., w2c_row2...)
         0         1   2   3   4    5  6   7~10       11~14        15~18

    Note: w2c is taken *as is* (flattened row-major, 3*4 → 12 numbers).
    """
    n = w2c_poses.shape[0]
    assert intrinsics.shape == (n, 4), f"intrinsics shape {intrinsics.shape} ≠ ({n}, 4)"
    assert w2c_poses.shape[1:] == (3, 4), f"poses shape {w2c_poses.shape} must be (n, 3, 4)"

    entries = []
    for i in range(n):
        fx, fy, cx, cy = intrinsics[i]
        w2c_3x4 = w2c_poses[i]  # already (3, 4), world-to-camera

        # Flatten in row-major order: row0, row1, row2 → 12 numbers
        w2c_flat = w2c_3x4.flatten()  # shape (12,)

        entry = (
            float(i),       # frame_id (as float, per example)
            fx, fy, cx, cy,
            0.0, 0.0,       # fixed placeholders (positions 5,6)
            *w2c_flat.tolist()
        )
        assert len(entry) == 19, f"Entry {i} has length {len(entry)} ≠ 19"
        entries.append(entry)

    return entries

################################ coordinates to pose(in extract keyframe) ################################



def load_poses_from_coordinates(
    coordinates: List[Tuple], 
) -> Tuple[str, List[Dict]]:

    if not coordinates:
        raise ValueError("Input coordinates list is empty.")

    poses = []
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

        position = c2w[:3, 3]
        poses.append({
            "position": position,
            "rotation": None,  # as in original; you may compute from c2w if needed
            "c2w": c2w,
            "frame_index": frame_idx,
        })

    return poses




# def generate_camera_coordinates(
#     direction: Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown", "In", "Out"],
#     length: int,
#     speed: float = 1/54,
#     start_id: int = 0,
#     origin=(0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)
#     #    (frame_id, fx, fy, cx, cy, 0, 0, w2c_row0..., w2c_row1..., w2c_row2...)
#     #     0         1   2   3   4    5  6   7~10       11~14        15~18
# ):
#     coordinates = [list(origin)]
#     id = start_id + 1
#     while len(coordinates) < length:
#         coor = coordinates[-1].copy()
#         if "Left" in direction:
#             coor[9] += speed
#         if "Right" in direction:
#             coor[9] -= speed
#         if "Up" in direction:
#             coor[13] += speed
#         if "Down" in direction:
#             coor[13] -= speed
#         if "In" in direction:
#             coor[18] -= speed
#         if "Out" in direction:
#             coor[18] += speed
#         coor[0] = id
#         coordinates.append(coor)
#         id = id + 1
#     return coordinates

#     #     if "Left" in direction:
#     #         coor[9] += speed
#     #     if "Right" in direction:
#     #         coor[9] -= speed
#     #     if "Up" in direction:
#     #         coor[13] += speed
#     #     if "Down" in direction:
#     #         coor[13] -= speed
#     #     if "In" in direction:
#     #         coor[18] -= speed
#     #     if "Out" in direction:
#     #         coor[18] += speed

#         # if "Left" in direction:
#         #     coor[10] += speed
#         # if "Right" in direction:
#         #     coor[10] -= speed
#         # if "Up" in direction:
#         #     coor[14] += speed
#         # if "Down" in direction:
#         #     coor[14] -= speed
#         # if "In" in direction:
#         #     coor[18] -= speed
#         # if "Out" in direction:
#         #     coor[18] += speed
        
        
        



def unpack_pose(vec19):
    """从19维向量中提取内参 + 4*4 W2C矩阵"""
    fx, fy, cx, cy = vec19[1:5]
    w2c_mat = np.array(vec19[7:]).reshape(3, 4)
    w2c = np.eye(4)
    w2c[:3, :] = w2c_mat
    return w2c, (fx, fy, cx, cy)

def pack_pose(w2c: np.ndarray, intrinsics, frame_id=0):
    """将 W2C 矩阵 + 内参打包为19维向量"""
    fx, fy, cx, cy = intrinsics
    w2c_mat_3x4 = w2c[:3, :]
    return [
        frame_id, fx, fy, cx, cy, 0, 0,
        *w2c_mat_3x4.flatten().tolist(),
    ]

def invert_pose(pose: np.ndarray) -> np.ndarray:
    """高效求刚体变换逆(W2C ⇄ C2W)"""
    R, t = pose[:3, :3], pose[:3, 3]
    inv = np.eye(4)
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv

def rotation_matrix(axis, angle):
    """绕任意轴旋转 angle 弧度(Rodrigues)"""
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    x, y, z = axis
    return np.array([
        [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c  ]
    ])

def rot_x(a): #up/down
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],
                     [0,c,-s],
                     [0,s,c]])

def rot_y(a): # left/right
    c, s = np.cos(a), np.sin(a)
    return np.array([[ c,0, s],
                     [ 0,1, 0],
                     [-s,0, c]])

from typing import Literal




def RT_align(
    coord_list: List[List[float]],
    new_coord0: List[float]
) -> List[List[float]]:


    RT0_new = np.array(new_coord0[7:]).reshape(3, 4)
    R0_new = RT0_new[:, :3]   # 3×3
    t0_new = RT0_new[:, 3:]   # 3×1

    aligned_coords = []
    for i, entry in enumerate(coord_list):
        if len(entry) != 19:
            raise ValueError(f"coord_list[{i}] must be length 19, got {len(entry)}.")

        meta = entry[:7]
        rt_flat = entry[7:]

        RT_i = np.array(rt_flat).reshape(3, 4)
        R_i = RT_i[:, :3]
        t_i = RT_i[:, 3:]

        # core alignment
        R_new = R0_new @ R_i
        t_new = R0_new @ t_i + t0_new
        RT_i_new = np.hstack([R_new, t_new])  # 3×4

        # flatten & combine
        new_entry = list(meta) + RT_i_new.ravel().tolist()
        aligned_coords.append(new_entry)

    return aligned_coords




def generate_camera_coordinates(
    control: Literal["W", "A", "S", "D", "LookLeft", "LookRight", "LookUp", "LookDown"],
    length: int,
    speed_trans: float = 0.1,
    speed_rot: float = 1.0,          # ← 总旋转角度（度）
    start_id: int = 0,
    origin=(
        0, 0.532139961, 0.946026558, 0.5, 0.5,
        0, 0,
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0
    )
):
    w2c0, intr = unpack_pose(origin)
    c2w0 = invert_pose(w2c0)

    R0 = c2w0[:3, :3].copy()
    C  = c2w0[:3, 3].copy()

    coordinates = [list(origin)]
    r_list = [R0.copy()]
    frame_id = start_id + 1

    angle_rad = np.deg2rad(speed_rot)

    for i in range(1, length):
        # t = i / (length - 1)          # [0, 1]
        # alpha = angle_rad * t         # 本帧“绝对角度”
        alpha = angle_rad * i

        yaw = 0.0
        pitch = 0.0

        if control == "LookLeft":
            yaw = -alpha
        elif control == "LookRight":
            yaw = +alpha
        elif control == "LookUp":
            pitch = +alpha
        elif control == "LookDown":
            pitch = -alpha

        if control.startswith("Look"):
            R_total = rot_y(yaw) @ rot_x(pitch)
            R_c2w = R0 @ R_total

        else:
            # 平移（沿当前朝向）
            R_c2w = r_list[-1]
            x_cam = R_c2w @ np.array([1, 0, 0])
            z_cam = R_c2w @ np.array([0, 0, 1])

            if control == "W":
                C += speed_trans * z_cam
            elif control == "S":
                C -= speed_trans * z_cam
            elif control == "A":
                C -= speed_trans * x_cam
            elif control == "D":
                C += speed_trans * x_cam

        c2w_new = np.eye(4)
        c2w_new[:3, :3] = R_c2w
        c2w_new[:3, 3]  = C
        w2c_new = invert_pose(c2w_new)

        coor = pack_pose(w2c_new, intr, frame_id=frame_id)
        coordinates.append(coor)
        r_list.append(R_c2w.copy())
        frame_id += 1

    return coordinates


def generate_camera_coordinates_v0(
    control: Literal["W", "A", "S", "D", "LookLeft", "LookRight", "LookUp", "LookDown"],
    length: int,
    speed_trans: float = 1.0,
    speed_rot: float = 0.02,          
    start_id: int = 0,
    origin=(
        0, 0.532139961, 0.946026558, 0.5, 0.5,
        0, 0,
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0
    )
):
    w2c0, intr = unpack_pose(origin)
    c2w0 = invert_pose(w2c0)

    R0 = c2w0[:3, :3].copy()
    C  = c2w0[:3, 3].copy()

    coordinates = [list(origin)]
    r_list = [R0.copy()]
    frame_id = start_id + 1

    if control.startswith("Look"):
        while len(coordinates) < length:
            coor = coordinates[-1].copy()
            if "LookLeft" in control:
                coor[9] += speed_rot
            if "LookRight" in control:
                coor[9] -= speed_rot
            if "LookUp" in control:
                coor[13] -= speed_rot
            if "LookDown" in control:
                coor[13] += speed_rot
            coor[0] = frame_id
            coordinates.append(coor)
            frame_id = frame_id + 1
        return coordinates
    else:
        for i in range(1, length):

            R_c2w = r_list[-1]
            x_cam = R_c2w @ np.array([1, 0, 0])
            z_cam = R_c2w @ np.array([0, 0, 1])

            if control == "W":
                C += speed_trans * z_cam
            elif control == "S":
                C -= speed_trans * z_cam
            elif control == "A":
                C -= speed_trans * x_cam
            elif control == "D":
                C += speed_trans * x_cam

            c2w_new = np.eye(4)
            c2w_new[:3, :3] = R_c2w
            c2w_new[:3, 3]  = C
            w2c_new = invert_pose(c2w_new)

            coor = pack_pose(w2c_new, intr, frame_id=frame_id)
            coordinates.append(coor)
            r_list.append(R_c2w.copy())
            frame_id += 1

        return coordinates
