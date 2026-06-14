
import os
import argparse

import numpy as np
from .camera_convert import read_uepose_from_json, poses_intrinsics_to_coordinates, euler_to_w2c_batch

def save_poses_to_file(coordinates, args, filename="pose"):
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{filename}_pose.txt")
    with open(out_path, 'w') as f:
        f.write(args.video_path + '\n')
        for coord in coordinates:
            line = ' '.join(f"{float(x):.9f}" for x in coord)
            f.write(line + '\n')
    print(f"✅ Poses saved to: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Run WanVideoPipeline with camera control on an input image.")
    parser.add_argument("--video_path", type=str, default="ditto_mini_test_videos/test/AncientTowns_2.mp4")
    parser.add_argument("--pose_path", type=str, default="ditto_mini_test_videos/test/AncientTowns_2.json")
    parser.add_argument("--output_dir", type=str, default="results_test/coordinates")

    args = parser.parse_args()

    pose_path = args.pose_path
    video_path = args.video_path
    file_name = os.path.splitext(os.path.basename(video_path))[0]

    poses = euler_to_w2c_batch(read_uepose_from_json(pose_path), return_34=True)
    num_frames = poses.shape[0]
    intrinsics = np.tile(np.array([0.5, 0.8667, 0.5, 0.5], dtype=np.float32), (num_frames, 1)) 
    # intrinsics = np.tile(np.array([0.532139961, 0.946026558, 0.5, 0.5], dtype=np.float32), (num_frames, 1)) 

    coordinates = poses_intrinsics_to_coordinates(w2c_poses=poses, intrinsics=intrinsics)
    save_poses_to_file(coordinates, args, file_name)


if __name__ == "__main__":
    main()
    
    
# results_test/coordinates/rgb_pose.txt