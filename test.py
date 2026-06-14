import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# 设置文件夹路径
folder_path = "/mnt/public/users/yangshuai/code/CausalWorld/results_sig26_exp_seed_batch_pipeline/Qualitative_14B_teaserglobal_Change_to_Makoto_Shinkai_style_vibrant_colors_dramatic_sky_cinematic_atmosphere_detailed_background_anime_aesthetic_seed5371/test_14/Final_depth"

# 获取 Spectral_r colormap 对象
cmap = cm.get_cmap('Spectral_r')

# 生成 256 个颜色值 (RGBA 或 RGB)
colormap = cmap(np.linspace(0, 1, 256))  # shape: (256, 4)

# 如果是 RGBA，只取 RGB 三通道
if colormap.shape[1] == 4:
    colormap = colormap[:, :3]  # 去掉 alpha 通道

# 遍历文件夹内所有 .png 文件
for filename in os.listdir(folder_path):
    if filename.endswith(".png") and not filename.endswith("_color.png"):
        input_path = os.path.join(folder_path, filename)
        output_path = os.path.join(folder_path, filename.replace(".png", "_color.png"))

        # 读取灰度深度图
        depth_gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if depth_gray is None:
            print(f"无法读取图像: {input_path}")
            continue

        # 将灰度值映射到 [0, 255] 整数索引
        depth_int = depth_gray.astype(np.uint8)

        # 使用 colormap 映射每个像素值
        colored_rgb = colormap[depth_int]  # shape: (H, W, 3)

        # 转换为 uint8 并乘以 255
        colored_rgb = (colored_rgb * 255).astype(np.uint8)

        # 保存为 PNG
        cv2.imwrite(output_path, cv2.cvtColor(colored_rgb, cv2.COLOR_RGB2BGR))

        print(f"已保存: {output_path}")

print("✅ 所有深度图已使用 Spectral_r colormap 转换为彩色图！")