import os
from PIL import Image
import torch

from diffusers import QwenImageEditPipeline

height=480
width=832

pipeline = QwenImageEditPipeline.from_pretrained("/mnt/public/users/yangshuai/code/public_models/Qwen/Qwen-Image-Edit").to(torch.bfloat16).to('cuda')
pipeline.set_progress_bar_config(disable=None)
image = Image.open("temp/t1.png").convert("RGB").resize((width, height))
w, h = image.size

prompt = "The structure remains the same, only the degree of darkness changes slightly."
inputs = {
        "image": image,
        "prompt": prompt,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 50,
    }

image.save("output_image_edit_source88.png")

with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("output_image_edit88.png")
    print("done")