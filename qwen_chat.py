from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "/mnt/public/users/yangshuai/code/public_models/Qwen/Qwen3-VL-8B-Instruct", dtype="auto", device_map='cuda'
)
processor = AutoProcessor.from_pretrained("/mnt/public/users/yangshuai/code/public_models/Qwen/Qwen3-VL-8B-Instruct")

image_path = "/mnt/public/users/yangshuai/code/CausalWorld/datasets/testset/test_00.png"
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path,
            },
            {"type": "text", "text": "Please summarize this image in one short sentence."},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
prompt = output_text[0]
print("Prompt:",prompt)