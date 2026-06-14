import random
import argparse
import numpy as np
import os

# 🔑 必须在 pyplot 之前设置非交互式后端（保障服务器保存）
import matplotlib
matplotlib.use('Agg')  # Headless-safe

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def draw_arrow_triangle(ax, start, direction, length=1.0, color='blue', linewidth=4.0, alpha=1.0, zorder=10):
    norm = np.linalg.norm(direction)
    if norm == 0:
        return
    unit_dir = direction / norm

    up = np.array([0, 1, 0])
    if np.allclose(unit_dir, up) or np.allclose(unit_dir, -up):
        up = np.array([1, 0, 0])

    right = np.cross(unit_dir, up)
    right = right / np.linalg.norm(right)

    head_width = length * 0.2
    head_length = length * 0.3

    tip = start + unit_dir * length
    left = tip - right * head_width/2 - unit_dir * head_length
    right_pt = tip + right * head_width/2 - unit_dir * head_length

    vertices = [start, left, right_pt]
    triangles = [[vertices[0], vertices[1], vertices[2]]]

    poly = Poly3DCollection(triangles, facecolors=color, edgecolors=color,
                            linewidths=linewidth, alpha=alpha)
    poly.set_zorder(zorder)
    ax.add_collection3d(poly)


class CameraPoseVisualizer:
    def __init__(self, xlim, ylim, zlim):
        self.fig = plt.figure(figsize=(18, 7))
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.set_aspect("auto")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        print('Intialized camera pose visualizer')

    def extrinsic2pyramid(self, extrinsic, color_map='red', hw_ratio=9/16, base_xval=1, zval=3):
        vertex_std = np.array([
            [0, 0, 0, 1],
            [base_xval, -base_xval * hw_ratio, zval, 1],
            [base_xval,  base_xval * hw_ratio, zval, 1],
            [-base_xval,  base_xval * hw_ratio, zval, 1],
            [-base_xval, -base_xval * hw_ratio, zval, 1]
        ])
        vertex_transformed = vertex_std @ extrinsic.T
        pts = vertex_transformed[:, :3]
        meshes = [
            [pts[0], pts[1], pts[2]],
            [pts[0], pts[2], pts[3]],
            [pts[0], pts[3], pts[4]],
            [pts[0], pts[4], pts[1]],
            [pts[1], pts[2], pts[3], pts[4]]
        ]
        color = color_map if isinstance(color_map, str) else plt.cm.rainbow(color_map)
        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, edgecolors=color,
                             linewidths=0.3, alpha=0.35)
        )
        return pts

    def customize_legend(self, list_label):
        handles = []
        for idx, label in enumerate(list_label):
            color = plt.cm.rainbow(idx / len(list_label))
            handles.append(Patch(color=color, label=label))
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=handles)

    def colorbar(self, max_frame_length):
        cmap = plt.cm.rainbow
        norm = plt.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=self.ax, orientation='vertical', label='Frame Number'
        )


def get_args():
    parser = argparse.ArgumentParser(description="Visualize camera poses from trajectory file.")
    parser.add_argument('--pose_path', required=True, help='Path to the trajectory txt file (e.g., camera_poses.txt)')
    parser.add_argument('--hw_ratio', default=1.0, type=float, help='Height/width ratio of film plane')
    parser.add_argument('--sample_stride', type=int, default=4, help='Sampling stride (every N-th frame)')
    parser.add_argument('--num_frames', type=int, default=16, help='Max number of frames to visualize')
    parser.add_argument('--all_frames', action='store_true', help='Use all frames (ignore num_frames & sample_stride)')
    parser.add_argument('--fixed_stride', action='store_true',
                        help='Force fixed stride sampling (e.g., 0, stride, 2*stride...) even if num_frames*stride > total)')
    parser.add_argument('--base_xval', type=float, default=1.0)
    parser.add_argument('--zval', type=float, default=2.0)
    parser.add_argument('--use_exact_fx', action='store_true', help='Use focal length from file as zval')
    parser.add_argument('--relative_c2w', action='store_true', help='Make first camera identity')
    parser.add_argument('--margin_ratio', type=float, default=0.2,
                        help='Extra margin ratio (e.g., 0.2 = 20%% padding around bounding box)')
    parser.add_argument('--force_square_aspect', action='store_true', default=True,
                        help='Force xlim/ylim/zlim to have same range (centered)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output image path (e.g., poses.png). Default: <pose_file_stem>_camera_poses.png')
    parser.add_argument('--show', action='store_true',
                        help='Try to show plot interactively (e.g., locally). Falls back to saving on servers.')
    parser.add_argument('--save-html', type=str, default=None,
                        help='Save interactive 3D plot as HTML (requires plotly). Example: poses.html')
    # 🔑 New: frustum toggle
    parser.add_argument('--show-frustums', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Show camera frustums (pyramids). Use --no-show-frustums to hide.')
    return parser.parse_args()


def get_c2w(w2cs, transform_matrix, relative_c2w):
    if relative_c2w:
        target_cam_c2w = np.eye(4)
        abs2rel = target_cam_c2w @ w2cs[0]
        ret_poses = [target_cam_c2w] + [abs2rel @ np.linalg.inv(w2c) for w2c in w2cs[1:]]
    else:
        ret_poses = [np.linalg.inv(w2c) for w2c in w2cs]
    ret_poses = [transform_matrix @ x for x in ret_poses]
    return np.array(ret_poses, dtype=np.float32)


def compute_bounding_box(all_points, margin_ratio=0.2):
    if len(all_points) == 0:
        raise ValueError("No points to compute bounding box!")

    mins = np.min(all_points, axis=0)
    maxs = np.max(all_points, axis=0)
    ranges = maxs - mins
    margin = ranges * margin_ratio / 2

    eps = 1e-3
    ranges = np.where(ranges < eps, 2 * eps, ranges)

    x_min, x_max = mins[0] - margin[0], maxs[0] + margin[0]
    y_min, y_max = mins[1] - margin[1], maxs[1] + margin[1]
    z_min, z_max = mins[2] - margin[2], maxs[2] + margin[2]

    return (x_min, x_max), (y_min, y_max), (z_min, z_max)


def export_html_plot(c2ws, camera_positions, frame_colors, output_html,
                     hw_ratio=1.0, base_xval=1.0, zval=2.0,
                     xlim=None, ylim=None, zlim=None, elev=30, azim=-60,
                     show_frustums=True):
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise RuntimeError("❌ 'plotly' not installed. Run: pip install plotly")

    # 🔑 Precise conversion: matplotlib elev/azim → plotly camera.eye
    r = 2.5
    theta = np.radians(azim)
    phi = np.radians(90 - elev)

    eye_x = r * np.cos(theta) * np.sin(phi)
    eye_y = r * np.sin(theta) * np.sin(phi)
    eye_z = r * np.cos(phi)

    fig = go.Figure()

    # 🔑 Conditionally add frustums
    if show_frustums:
        for idx, c2w in enumerate(c2ws):
            vertex_std = np.array([
                [0, 0, 0],
                [base_xval, -base_xval * hw_ratio, zval],
                [base_xval,  base_xval * hw_ratio, zval],
                [-base_xval,  base_xval * hw_ratio, zval],
                [-base_xval, -base_xval * hw_ratio, zval]
            ])
            pts = (vertex_std @ c2w[:3, :3].T) + c2w[:3, 3]
            xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]

            faces = [
                [0, 1, 2],
                [0, 2, 3],
                [0, 3, 4],
                [0, 4, 1],
                [1, 2, 3],
                [1, 3, 4]
            ]

            color = plt.cm.rainbow(frame_colors[idx])[:3]
            color_str = f'rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})'

            fig.add_trace(go.Mesh3d(
                x=xs, y=ys, z=zs,
                i=[f[0] for f in faces], j=[f[1] for f in faces], k=[f[2] for f in faces],
                color=color_str,
                opacity=0.35,
                flatshading=True,
                showlegend=False
            ))

    # Always add trajectory line
    offset = np.array([0.0, 3.0, 0.0])
    shifted_pos = camera_positions + offset
    fig.add_trace(go.Scatter3d(
        x=shifted_pos[:, 0], y=shifted_pos[:, 1], z=shifted_pos[:, 2],
        mode='lines+markers',
        line=dict(color='red', width=6),
        marker=dict(size=4, color='red'),
        name='Camera Trajectory'
    ))

    # 🎯 Layout: match matplotlib EXACTLY
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=xlim, title='x', autorange=False, showspikes=False),
            yaxis=dict(range=ylim, title='y', autorange=False, showspikes=False),
            zaxis=dict(range=zlim, title='z', autorange=False, showspikes=False),
            aspectmode='cube' if args.force_square_aspect else 'manual',
            aspectratio=dict(x=1, y=1, z=1) if args.force_square_aspect else 
                      dict(x=(xlim[1]-xlim[0]), y=(ylim[1]-ylim[0]), z=(zlim[1]-zlim[0])),
            camera=dict(
                eye=dict(x=eye_x, y=eye_y, z=eye_z),
                up=dict(x=0, y=1, z=0),     # ✅ Y is up
                center=dict(x=0, y=0, z=0)
            )
        ),
        title="Camera Poses (Extrinsic Parameters)",
        width=1200,
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False
    )

    fig.write_html(
        output_html,
        include_plotlyjs='cdn',
        config={'displayModeBar': True, 'scrollZoom': True, 'displaylogo': False}
    )
    print(f"✅ Saved interactive HTML to: {output_html}")


def main():
    global args
    args = get_args()

    pose_file = args.pose_path
    with open(pose_file, 'r') as f:
        lines = f.readlines()
    if len(lines) <= 1:
        raise ValueError(f"Trajectory file {pose_file} is empty or only has header.")

    w2cs = []
    fxs = []
    for i, line in enumerate(lines[1:], start=1):
        parts = line.strip().split()
        if len(parts) < 19:
            raise ValueError(f"Line {i} has insufficient tokens: {len(parts)} (need ≥19)")
        fx = float(parts[1])
        w2c_flat = [float(p) for p in parts[7:19]]
        w2c = np.array(w2c_flat).reshape(3, 4)
        w2cs.append(w2c)
        fxs.append(fx)

    total_frames = len(w2cs)
    print(f"Loaded {total_frames} camera poses from {pose_file}")

    if args.all_frames:
        frame_ind = np.arange(total_frames)
        print(f"Using all {len(frame_ind)} frames.")
    elif args.fixed_stride:
        frame_ind = np.arange(0, total_frames, args.sample_stride)
        if len(frame_ind) == 0:
            frame_ind = np.array([0])
        if args.num_frames > 0:
            frame_ind = frame_ind[:args.num_frames]
        print(f"Fixed-stride sampling: {len(frame_ind)} frames at indices {frame_ind}")
    else:
        cropped_length = args.num_frames * args.sample_stride
        if cropped_length <= total_frames:
            start_frame_ind = random.randint(0, total_frames - cropped_length)
            end_frame_ind = start_frame_ind + cropped_length
            frame_ind = np.linspace(start_frame_ind, end_frame_ind - 1, args.num_frames, dtype=int)
            print(f"Random window sampling: {args.num_frames} frames from [{start_frame_ind}, {end_frame_ind})")
        else:
            frame_ind = np.arange(0, total_frames, args.sample_stride)
            if len(frame_ind) == 0:
                frame_ind = np.array([0])
            if len(frame_ind) > args.num_frames:
                step = len(frame_ind) / args.num_frames
                frame_ind = frame_ind[np.round(np.arange(0, len(frame_ind), step)).astype(int)]
            print(f"⚠️ Warning: num_frames×stride > total. Using stride={args.sample_stride} → {len(frame_ind)} frames: {frame_ind}")

    w2cs = [w2cs[i] for i in frame_ind]
    fxs = [fxs[i] for i in frame_ind]

    last_row = np.array([[0, 0, 0, 1]], dtype=np.float32)
    w2cs = [np.vstack([w2c, last_row]) for w2c in w2cs]

    transform_matrix = np.array([
        [1,  0,  0, 0],
        [0,  0,  1, 0],
        [0,  -1,  0, 0],
        [0,  0,  0, 1]
    ], dtype=np.float32)

    c2ws = get_c2w(w2cs, transform_matrix, args.relative_c2w)
    print("c2w:", c2ws.shape)

    # 🆕 Collect points for auto-limit (only needed if showing frustums or computing bbox)
    all_points_list = []
    if args.show_frustums:
        visualizer_dummy = CameraPoseVisualizer([0, 1], [0, 1], [0, 1])
        for frame_idx, c2w in enumerate(c2ws):
            zval = fxs[frame_idx] if args.use_exact_fx else args.zval
            pts = visualizer_dummy.extrinsic2pyramid(
                c2w,
                color_map=0.5,
                hw_ratio=args.hw_ratio,
                base_xval=args.base_xval,
                zval=zval
            )
            all_points_list.append(pts)

    camera_centers = np.array([c2w[:3, 3] for c2w in c2ws])
    all_points_list.append(camera_centers)
    all_points = np.concatenate(all_points_list, axis=0)

    try:
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = compute_bounding_box(all_points, args.margin_ratio)
    except Exception as e:
        print(f"⚠️ Auto-limits failed ({e}), using [-3,3] for all axes")
        x_min = y_min = z_min = -3
        x_max = y_max = z_max = 3

    if args.force_square_aspect:
        center = np.array([(x_min + x_max)/2, (y_min + y_max)/2, (z_min + z_max)/2])
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        pad = max_range / 2
        x_min, x_max = center[0] - pad, center[0] + pad
        y_min, y_max = center[1] - pad, center[1] + pad
        z_min, z_max = center[2] - pad, center[2] + pad

    print(f"Auto-limits → x: [{x_min:.2f}, {x_max:.2f}], y: [{y_min:.2f}, {y_max:.2f}], z: [{z_min:.2f}, {z_max:.2f}]")

    # 🖼️ Real visualizer
    visualizer = CameraPoseVisualizer(
        xlim=[x_min, x_max],
        ylim=[y_min, y_max],
        zlim=[z_min, z_max]
    )

    frame_colors = [i / max(1, len(c2ws) - 1) for i in range(len(c2ws))]

    # 🔑 Conditionally render frustums in matplotlib
    if args.show_frustums:
        for frame_idx, c2w in enumerate(c2ws):
            zval = fxs[frame_idx] if args.use_exact_fx else args.zval
            visualizer.extrinsic2pyramid(
                c2w,
                color_map=frame_colors[frame_idx],
                hw_ratio=args.hw_ratio,
                base_xval=args.base_xval,
                zval=zval
            )

    # 🎯 Trajectory line (always shown)
    camera_positions = np.array([c2w[:3, 3] for c2w in c2ws])
    if len(camera_positions) > 1:
        offset = np.array([0.0, 0.0, z_max])
        shifted_positions = camera_positions + offset

        visualizer.ax.plot(
            shifted_positions[:, 0],
            shifted_positions[:, 1],
            shifted_positions[:, 2],
            color='red',
            linewidth=3,
            alpha=1.0,
            zorder=1000
        )

        last_start = camera_positions[-2] + offset
        last_dir = camera_positions[-1] - camera_positions[-2]
        last_norm = np.linalg.norm(last_dir)
        if last_norm > 0:
            unit_last_dir = last_dir / last_norm
            visualizer.ax.quiver(
                last_start[0], last_start[1], last_start[2],
                unit_last_dir[0], unit_last_dir[1], unit_last_dir[2],
                color='red',
                linewidth=4.0,
                arrow_length_ratio=0.8,
                alpha=1.0,
                length=0.8,
                zorder=1000
            )

    if args.show_frustums:
        visualizer.colorbar(len(c2ws) - 1)

    # 📁 Save PNG
    pose_stem = os.path.splitext(os.path.basename(pose_file))[0]
    output_path = args.output or f"{pose_stem}_camera_poses.png"
    title = 'Camera Poses (Extrinsic Parameters)'
    if not args.show_frustums:
        title += ' (Frustums hidden)'
    plt.title(title, fontsize=14)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"✅ Saved PNG to: {output_path} | Visualized {len(c2ws)} frames" + 
          ("" if args.show_frustums else " (frustums hidden)"))

    # 📊 Capture view
    actual_xlim = visualizer.ax.get_xlim()
    actual_ylim = visualizer.ax.get_ylim()
    actual_zlim = visualizer.ax.get_zlim()
    actual_elev = visualizer.ax.elev
    actual_azim = visualizer.ax.azim
    print(f"Used view: elev={actual_elev:.1f}°, azim={actual_azim:.1f}°")

    # 🌐 Save HTML with matching view & frustum toggle
    if args.save_html:
        export_html_plot(
            c2ws=c2ws,
            camera_positions=camera_positions,
            frame_colors=frame_colors,
            output_html=args.save_html,
            hw_ratio=args.hw_ratio,
            base_xval=args.base_xval,
            zval=args.zval,
            xlim=actual_xlim,
            ylim=actual_ylim,
            zlim=actual_zlim,
            elev=actual_elev,
            azim=actual_azim,
            show_frustums=args.show_frustums
        )

    # 👁️ Optional interactive show
    if args.show:
        print("🔍 Attempting to show plot interactively...")
        try:
            matplotlib.use('TkAgg')
            plt.figure(visualizer.fig.number)
            plt.show()
            print("🖼️  Plot displayed successfully.")
        except Exception as e:
            print(f"⚠️  Could not show plot: {e}")
            if 'DISPLAY' in str(e) or 'no display' in str(e).lower():
                print("💡 Tips:")
                print("   • Local: run with --show")
                print("   • Remote: ssh -X or download PNG/HTML")
            else:
                print(f"   Unexpected error: {e}")


if __name__ == '__main__':
    main()