import os
from pathlib import Path

from huggingface_hub import HfApi, HfFolder


def should_skip(rel_posix: str) -> bool:
    # Skip large/unwanted directories.
    # Note: We always skip `models/` and `output_train_ckpt_backup/` as requested.
    # Also skip upload scripts themselves (not needed in the target repo).
    if rel_posix in {"upload_bp_code.py", "upload.py"}:
        return True

    skip_prefixes = [
        "models/",
        "output_train_ckpt_backup/",
        "results*/",
        # Avoid uploading git history to Hugging Face.
        ".git/",
    ]
    # `huggingface_hub` forbids updating files under any `.cache/` folder.
    # Example encountered:
    #   benchmarks/VACE-Benchmark/.cache/huggingface/.gitignore
    parts = rel_posix.split("/")
    if ".cache" in parts:
        return True

    # Also skip any top-level `result*` directories/files.
    # Example matches: `result_foo/...`, `result123.txt`
    top = parts[0] if parts else rel_posix
    if top.startswith("result"):
        return True

    return any(rel_posix.startswith(p) for p in skip_prefixes)


def main():
    local_dir = Path(__file__).resolve().parent
    repo_id = "ysmikey/bp_code"

    # Prefer env var token (recommended), fall back to cached token.
    token = os.getenv("HF_TOKEN") or HfFolder.get_token()
    if not token:
        raise RuntimeError(
            "Missing Hugging Face token. Please set environment variable HF_TOKEN "
            "(or run `huggingface-cli login`)."
        )

    api = HfApi(token=token)
    # Ensure repo exists.
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    print(f"Uploading files from: {local_dir}")
    print(f"Target repo: {repo_id}")

    uploaded = 0
    skipped = 0

    # Upload all regular files under local_dir, preserving directory structure.
    for file_path in local_dir.rglob("*"):
        if not file_path.is_file():
            continue
        rel_path = file_path.relative_to(local_dir).as_posix()
        if should_skip(rel_path):
            skipped += 1
            continue

        # Keep repo path stable.
        print(f"Uploading: {rel_path}")
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=rel_path,
            repo_id=repo_id,
            repo_type="model",
        )
        uploaded += 1

    print(f"Upload finished. uploaded={uploaded}, skipped={skipped}")


if __name__ == "__main__":
    main()

