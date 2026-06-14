
import argparse
import numpy as np
import os
import torch
import matplotlib.cm as cm
from PIL import Image
from external.vda.video_depth_anything.video_depth import VideoDepthAnything
from external.vda.utils.dc_utils import read_video_frames, save_video


def init_vda(device='cuda', ckpt_path='/mnt/cpfs/yangshuai/code/PermaVid/prismworld/external/vda/checkpoints/video_depth_anything_vitl.pth'):

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    checkpoint_name = 'video_depth_anything'
    encoder = 'vitl'
    video_depth_anything = VideoDepthAnything(**model_configs[encoder], metric=False)
    video_depth_anything.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(device).eval()
    return video_depth_anything

def run_vda(model,
            input_video_path=None,
            input_video=None, # list of PILImage
            fps=20,
            save_video_path=None,
            grayscale=True,
            device='cuda'
            ):

    
    if input_video is None and input_video_path:
        frames, target_fps = read_video_frames(input_video_path, process_length=-1, target_fps=-1, max_res=1280)
    elif input_video is not None:
        frames = np.stack([np.array(img) for img in input_video], axis=0).astype(np.uint8)
        target_fps = fps
    
    depths, fps = model.infer_video_depth(frames, target_fps, input_size=518, device=device)

    if save_video_path is not None:
        save_video(depths, save_video_path, fps=target_fps, is_depths=True, grayscale=grayscale)

    ### save ###
    colormap = np.array(cm.get_cmap("inferno").colors)
    d_min, d_max = depths.min(), depths.max()
    depths_vis = []
    for i in range(depths.shape[0]):
        depth = depths[i]
        depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        depth_vis = (colormap[depth_norm] * 255).astype(np.uint8) if not grayscale else np.stack([depth_norm] * 3, axis=-1)

        depths_vis.append(Image.fromarray(depth_vis))
    
    return depths_vis
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything')
    parser.add_argument('--input_video', type=str, default='/mnt/public/users/yangshuai/code/CausalWorld/benchmarks/VACE-Benchmark/assets/examples/face/out_video.mp4')
    parser.add_argument('--output_dir', type=str, default='./outputs')

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = init_vda(device=DEVICE)
    
    depths_vis = run_vda(model,
            input_video_path=args.input_video,
            input_video=None, # list of PILImage
            fps=20,
            save_video_path='depth.mp4',
            grayscale=True,
            device=DEVICE
            )