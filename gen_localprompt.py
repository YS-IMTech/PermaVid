import os
import glob
import torch
import random
import numpy as np
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# ===================== 随机种子控制（确保可复现） =====================
def set_seed(seed=25):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(25)  # ← 在导入后立即设置种子

# ===================== paths =====================
model_path = "/mnt/public/users/yangshuai/code/public_models/Qwen/Qwen3-VL-8B-Instruct"
img_dir = "/mnt/public/users/yangshuai/code/CausalWorld/datasets/testset"
out_dir = "/mnt/public/users/yangshuai/code/CausalWorld/datasets/testset_local_prompts2"
os.makedirs(out_dir, exist_ok=True)

# ===================== load model =====================
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path,
    dtype="auto",
    device_map="cuda",
)
processor = AutoProcessor.from_pretrained(model_path)
model.eval()  # ← 显式设为 eval 模式（关闭 dropout 等）

@torch.inference_mode()
def gen_local_edit_prompts(image_path: str, max_new_tokens: int = 128) -> str:
    """
    Return EXACTLY 3 lines:
      add ...
      remove ...
      replace ...
    focusing on a single, salient, local object (prefer center, unique).
    """
    prompt_text = (
        "Look at the image and pick ONE most salient, unique, easy-to-edit local object "
        "(prefer the most obvious central object; avoid sky/ground/lighting/weather/season). "
        "Then output EXACTLY 3 short local edit instructions, each on its own line, "
        "starting with the lowercase keyword: add / remove / replace.\n"
        "Rules:\n"
        "- Keep each line concise and unambiguous.\n"
        "- The edit must be local (object-level), not global style or scene changes.\n"
        "- The 3 lines must target the SAME chosen object/location.\n"
        "- Output ONLY the 3 lines, nothing else.\n"
        "Example format:\n"
        "add ...\n"
        "remove ...\n"
        "replace ...\n"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # 🔑 关键：显式关闭采样，确保 deterministic 输出
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # ← 贪心解码，无随机性
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    out = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    # ---------- sanitize to exactly 3 lines ----------
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    kept = []
    for l in lines:
        ll = l.lower()
        if ll.startswith("add ") or ll == "add":
            kept.append("add" if ll == "add" else "add " + l.split(" ", 1)[1].strip())
        elif ll.startswith("remove ") or ll == "remove":
            kept.append("remove" if ll == "remove" else "remove " + l.split(" ", 1)[1].strip())
        elif ll.startswith("replace ") or ll == "replace":
            kept.append("replace" if ll == "replace" else "replace " + l.split(" ", 1)[1].strip())

    if len(kept) < 3:
        lines = lines[:3]
        forced = []
        prefixes = ["add", "remove", "replace"]
        for i in range(3):
            if i < len(lines):
                s = lines[i]
                s = s.lstrip("-*•").strip()
                low = s.lower()
                for p in prefixes:
                    if low.startswith(p):
                        s = s[len(p):].lstrip(" :.-").strip()
                        break
                forced.append(f"{prefixes[i]} {s}".strip())
            else:
                forced.append(prefixes[i])
        kept = forced

    order = {"add": 0, "remove": 1, "replace": 2}
    def key_fn(x):
        p = x.split(" ", 1)[0].lower()
        return order.get(p, 999)

    kept = sorted(kept, key=key_fn)
    kept = kept[:3]
    while len(kept) < 3:
        kept.append(["add", "remove", "replace"][len(kept)])

    final = []
    for i, p in enumerate(["add", "remove", "replace"]):
        s = kept[i].strip()
        low = s.lower()
        if low.startswith(p):
            rest = s[len(p):].lstrip(" :.-").strip()
            final.append(f"{p} {rest}".strip())
        else:
            final.append(p)

    return "\n".join(final) + "\n"

def main():
    pngs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if not pngs:
        raise FileNotFoundError(f"No .png found in: {img_dir}")

    for img_path in pngs:
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_txt = os.path.join(out_dir, f"{base}.txt")

        prompts = gen_local_edit_prompts(img_path)
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(prompts)

        print(f"[OK] {os.path.basename(img_path)} -> {out_txt}")
        print(prompts.strip(), "\n" + "-" * 60)

if __name__ == "__main__":
    main()