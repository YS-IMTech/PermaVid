from huggingface_hub import snapshot_download
import os

# 模型仓库ID
model_id = "alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera"

# 本地保存目录
save_dir = "/mnt/public/users/yangshuai/code/CausalWorld/models/PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera"

# 创建保存目录（如果不存在）
os.makedirs(save_dir, exist_ok=True)

# 下载模型
try:
    snapshot_download(
        repo_id=model_id,
        local_dir=save_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        token=True  # 如果需要认证令牌，请设置为True并提供HF_TOKEN环境变量
    )
    print(f"模型已成功下载到: {save_dir}")
except Exception as e:
    print(f"下载失败: {e}")