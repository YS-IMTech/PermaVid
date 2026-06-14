import os
from pathlib import Path
from huggingface_hub import HfApi, HfFolder

# 配置
local_dir = "/mnt/public/users/yangshuai/code/prismworld/output_train_ckpt_backup"
repo_id = "ysmikey/prismworld_ckpt_backup"
token = os.getenv("HF_TOKEN") or HfFolder.get_token()  # 从环境变量或已登录的凭证获取


# 初始化 API
api = HfApi()


# 递归上传所有文件，保留目录结构
local_path = Path(local_dir)
if not local_path.exists():
    raise FileNotFoundError(f"Local directory not found: {local_dir}")

print(f"Uploading files from {local_dir} to https://huggingface.co/{repo_id} ...")

for file_path in local_path.rglob("*"):
    if file_path.is_file():
        # 相对路径作为 repo 中的路径（保留结构）
        repo_path = file_path.relative_to(local_path)
        print(f"Uploading: {repo_path}")
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=str(repo_path),
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )

print("✅ Upload completed.")