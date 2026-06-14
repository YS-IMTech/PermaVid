import os
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# ========== 配置参数 ==========
INPUT_DIR = "/mnt/public/users/yangshuai/code/CausalWorld/datasets/testset"
OUTPUT_DIR = "/mnt/public/users/yangshuai/code/CausalWorld/datasets/testset_local_prompts"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 加载模型和处理器 ==========
print("Loading model and processor...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "/mnt/public/users/yangshuai/code/public_models/Qwen/Qwen3-VL-8B-Instruct",
    dtype="auto",
    device_map='cuda'
)
processor = AutoProcessor.from_pretrained("/mnt/public/users/yangshuai/code/public_models/Qwen/Qwen3-VL-8B-Instruct")
print("Model and processor loaded.")

# ========== 辅助函数：从描述中提取“主要对象” ==========
def extract_main_object(description):
    """
    一个简化的启发式函数，用于从模型生成的描述中提取可能的主要对象。
    实际应用中，可以结合更复杂的 NLP 或视觉定位技术。
    这里我们假设描述中第一个出现的、有意义的名词短语就是主要对象。
    """
    # 移除开头的 "The" 或 "A" 等冠词，并尝试找到第一个名词性短语
    description = description.strip().lower()
    if description.startswith("the "):
        description = description[4:]
    elif description.startswith("a "):
        description = description[2:]

    # 简单地按空格分割，寻找包含常见名词的词组
    words = description.split()
    main_object_parts = []

    # 常见的“对象”相关词汇（可以根据需要扩展）
    object_keywords = {'pot', 'jar', 'vase', 'bowl', 'table', 'chair', 'window', 'oven', 'shelf', 'lamp', 'kitchen', 'tree', 'flower', 'cup', 'plate', 'pan'}

    for word in words:
        # 移除标点符号
        clean_word = word.rstrip('.,!?;:()[]{}"\'')
        # 如果这个词在关键词里，或者它是一个常见的名词（这里简化处理）
        if clean_word in object_keywords or (len(clean_word) > 3 and not clean_word.endswith('ing') and not clean_word.endswith('ed')):
            main_object_parts.append(clean_word)
            break  # 只取第一个匹配的

    if main_object_parts:
        return ' '.join(main_object_parts)
    else:
        # 如果没找到，返回描述的前几个词作为 fallback
        return ' '.join(words[:3]) if len(words) >= 3 else description

# ========== 主处理循环 ==========
for filename in sorted(os.listdir(INPUT_DIR)):
    if filename.lower().endswith('.png'):
        image_path = os.path.join(INPUT_DIR, filename)
        output_txt_path = os.path.join(OUTPUT_DIR, filename.replace('.png', '.txt'))

        print(f"\nProcessing {filename}...")

        # ========== 步骤1: 让模型生成一个简洁的图像描述 ==========
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {
                        "type": "text",
                        "text": "Describe the most prominent object in the center of this image in one very short phrase, focusing on its type and location. Be concise."
                    },
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=64)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        description = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        print(f"  Model description: '{description}'")

        # ========== 步骤2: 提取主要对象 ==========
        main_object = extract_main_object(description)
        print(f"  Extracted main object: '{main_object}'")

        # ========== 步骤3: 生成三个编辑指令 ==========
        # 为了保证多样性，我们给 replace 一个具体的替换物
        # 这里简单地用一些常见的、与厨房相关的物品
        replace_options = ["a glass vase", "a metal bowl", "a wooden cutting board", "a potted plant", "a book"]

        # 选择第一个作为默认，或者可以随机选择
        replace_target = replace_options[0]

        prompt_add = f"add a small flower next to the {main_object}"
        prompt_remove = f"remove the {main_object}"
        prompt_replace = f"replace the {main_object} with {replace_target}"

        # ========== 步骤4: 保存到文件 ==========
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(prompt_add + '\n')
            f.write(prompt_remove + '\n')
            f.write(prompt_replace + '\n')

        print(f"  Prompts saved to {output_txt_path}")

print("\nAll images processed successfully!")