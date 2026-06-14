# Install PrismWorld dependencies (causalworld)

## 1) Activate the conda env
```bash
source /mnt/public/users/yangshuai/miniconda3/etc/profile.d/conda.sh
conda activate causalworld
```

## 2) Install requirements
```bash
cd /mnt/public/users/yangshuai/code/prismworld
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## 3) Quick verification
```bash
python -c "import torch, decord, PIL.Image, numpy, diffusers, transformers, huggingface_hub; print('OK')"
```

## Notes
- `requirements.txt` is intentionally designed for your current `causalworld` environment.
- It does NOT pin `torch/torchvision` to avoid CUDA wheel resolution issues. If you create a fresh env, install `torch/torchvision` first with the CUDA version you need.
