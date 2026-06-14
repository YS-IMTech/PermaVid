import safetensors
import safetensors.torch

# 模型文件路径
model_path = "outputs_train_1.3b_uecam_mixref_onlydit/step-800.safetensors"
output_path = "temp/vace_1.3b_dit_wotext.txt"
# 加载 safetensors 文件
data = safetensors.torch.load_file(model_path)

# 提取所有参数名称
param_names = list(data.keys())

# 将参数名称写入 safetensor.txt
with open(output_path, "w") as f:
    for name in param_names:
        f.write(name + "\n")

print(f"参数名称已成功写入 safetensor.txt，共 {len(param_names)} 个参数。")