import numpy as np


def degrees_to_radians(degrees):
    return degrees * np.pi / 180

def compute_rotation_list(params):
    x, y, z, yaw, pitch = params
    
    yaw_rad = degrees_to_radians(yaw)
    pitch_rad = degrees_to_radians(pitch)
    
    R_y = np.array([[np.cos(yaw_rad), 0, np.sin(yaw_rad)],
                    [0, 1, 0],
                    [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]])
    
    R_z = np.array([[np.cos(pitch_rad), -np.sin(pitch_rad), 0],
                    [np.sin(pitch_rad), np.cos(pitch_rad), 0],
                    [0, 0, 1]])
    
    R = np.dot(R_z, R_y)
    
    rotation_list = [x, y, z] + R.flatten().tolist()
    
    return rotation_list

def convert_rt_to_relative(rt_list_all, ref_rt):
    def parse_rt(rt):
        t = np.array(rt[:3]).reshape((3, 1))
        R = np.array(rt[3:]).reshape((3, 3))
        return R, t

    R_ref, T_ref = parse_rt(ref_rt)
    R_ref_inv = R_ref.T
    T_ref_inv = -R_ref_inv @ T_ref

    new_rt_list = []

    for rt in rt_list_all:
        R_i, T_i = parse_rt(rt)

        R_new = R_ref_inv @ R_i
        T_new = R_ref_inv @ T_i + T_ref_inv

        rt_new = T_new.flatten().tolist() + R_new.flatten().tolist()
        new_rt_list.append(rt_new)

    return new_rt_list