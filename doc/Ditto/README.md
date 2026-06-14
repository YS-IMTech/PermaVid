# Ditto: Scaling Instruction-Based Video Editing with a High-Quality Synthetic Dataset

> **Ditto: Scaling Instruction-Based Video Editing with a High-Quality Synthetic Dataset** <br>
> Qingyan Bai, Qiuyu Wang, Hao Ouyang, Yue Yu, Hanlin Wang, Wen Wang, Ka Leong Cheng, Shuailei Ma, Yanhong Zeng, Zichen Liu, Yinghao Xu, Yujun Shen, Qifeng Chen

## 🔗 **Links & Resources**

**[**[**📄 Paper**](https://arxiv.org/abs/2510.15742)**]**
**[**[**🌐 Project Page**](https://ezioby.github.io/Ditto_page/)**]**
**[**[**📦 Model Weights**](https://huggingface.co/QingyanBai/Ditto_models/tree/main)**]**
**[**[**📊 Dataset**](https://huggingface.co/datasets/QingyanBai/Ditto-1M)**]**
**[**[**🤗 Hugging Face Demo**](https://huggingface.co/spaces/QingyanBai/Ditto)**]**


</div>

## TL;DR
To solve the data scarcity problem, we introduce a scalable pipeline Ditto, for generating high-quality video editing data, which is used to train a new state-of-the-art instruction-based video editing model, Editto.

## Summary

We introduce Ditto, a holistic framework designed to tackle the fundamental challenge of instruction-based video editing. At its heart, Ditto features a novel data generation pipeline that fuses the creative diversity of a leading image editor with an in-context video generator, overcoming the limited scope of existing models. To make this process viable, our framework resolves the prohibitive cost-quality trade-off by employing an efficient, distilled model architecture augmented by a temporal enhancer, which simultaneously reduces computational overhead and improves temporal coherence. Finally, to achieve full scalability, this entire pipeline is driven by an intelligent agent that crafts diverse instructions and rigorously filters the output, ensuring quality control at scale. Using this framework, we invested over 12,000 GPU-days to build Ditto-1M, a new dataset of one million high-fidelity video editing examples. We trained our model, Editto, on Ditto-1M with a curriculum learning strategy. The results demonstrate superior instruction-following ability and establish a new SOTA in instruction-based video editing.

## Updating List
- [x] 10/29/2025 - Release the Hugging Face online demo.
- [x] 10/27/2025 - Add [codes](#denoising-enhancing) for Denoising Enhancing.
- [x] 10/22/2025 - We have uploaded the [csvs](https://huggingface.co/datasets/QingyanBai/Ditto-1M/tree/main/csvs_for_DiffSynth) that can be directly used for model training with DiffSynth-Studio, as well as the metadata [json](https://huggingface.co/datasets/QingyanBai/Ditto-1M/blob/main/training_metadata/sim2real.json) for sim2real setting.
- [x] 10/22/2025 - We finish uploading all the videos of the dataset!


## Model Usage

### 1. Using with DiffSynth

#### *Environment Setup*

```bash
# Create conda environment (if you already have a DiffSynth conda environment, you can reuse it)
conda create -n ditto python=3.10
conda activate ditto
pip install -e .
```

#### *Download Models*

Download the base model and our models from [Google Drive](https://drive.google.com/drive/folders/1SCsD-r-8QtQUNZSXdz0ALYd_Z_xXyN_6?usp=sharing) or [Hugging Face](https://huggingface.co/QingyanBai/Ditto_models/tree/main/models):
```bash
# Download Wan-AI/Wan2.1-VACE-14B from Hugging Face to models/Wan-AI/
hf download Wan-AI/Wan2.1-VACE-14B --local-dir models/Wan-AI/

# Download Ditto models
hf download QingyanBai/Ditto_models --include="models/*" --local-dir ./
```


#### *Usage*


You can either use the provided script or run Python directly:

```bash
# Option 1: Use the provided script
bash infer.sh

# Option 2: Run Python directly
python inference/infer_ditto.py \
    --input_video /path/to/input_video.mp4 \
    --output_video /path/to/output_video.mp4 \
    --prompt "Editing instruction." \
    --lora_path /path/to/model.safetensors \
    --num_frames 73 \
    --device_id 0
```

Some test cases could be found at [HF Dataset](https://huggingface.co/datasets/QingyanBai/Ditto-1M/tree/main/mini_test_videos). You can also find some reference editing prompts in `inference/example_prompts.txt`.

### 2. Using with ComfyUI
<sub>Note: While ComfyUI runs faster with lower computational requirements (832×480x73  videos need 11G GPU memory and ~4min on A6000), please note that due to the use of quantized and distilled models, there may be some quality degradation.</sub>

#### *Environment Setup*

First, follow the [ComfyUI installation guide](https://github.com/comfyanonymous/ComfyUI) to set up the base ComfyUI environment.
We strongly recommend installing [ComfyUI-Manager](https://github.com/Comfy-Org/ComfyUI-Manager) for easy custom node management:

```bash
# Install ComfyUI-Manager
cd ComfyUI/custom_nodes
git clone https://github.com/Comfy-Org/ComfyUI-Manager.git
```

After installing ComfyUI, you can either:

Option 1 (Recommended): Use ComfyUI-Manager to automatically install all required custom nodes with the function Install Missing Custom Nodes.

Option 2: Manually install the required custom nodes (you can refer to [this page](https://docs.comfy.org/installation/install_custom_node)):
<sub>
- [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)
- [KJNodes for ComfyUI](https://github.com/kijai/ComfyUI-KJNodes)
- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
</sub>

#### *Download Models*

Download the required model weights from: [Kijai/WanVideo_comfy](https://huggingface.co/Kijai/WanVideo_comfy/tree/main) to subfolders of `models/`. Required files include:
- [Wan2_1-T2V-14B_fp8_e4m3fn.safetensors](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Wan2_1-T2V-14B_fp8_e4m3fn.safetensors) to `diffusion_models/`
- [Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors) to `loras/` for inference acceleration
- [Wan2_1_VAE_bf16.safetensors](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Wan2_1_VAE_bf16.safetensors) to `vae/wan/`
- [umt5-xxl-enc-bf16.safetensors](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/umt5-xxl-enc-bf16.safetensors) to `text_encoders/`


Download our models from [Google Drive](https://drive.google.com/drive/folders/1SCsD-r-8QtQUNZSXdz0ALYd_Z_xXyN_6?usp=sharing) or [Hugging Face](https://huggingface.co/QingyanBai/Ditto_models/tree/main/models_comfy) to `diffusion_models/` (use VACE Module Select node for loading).


#### *Usage*

Use the workflow `ditto_comfyui_workflow.json` in this repo to get started.
We provided some reference prompts in the note.
Some test cases could be found at [HF Dataset](https://huggingface.co/datasets/QingyanBai/Ditto-1M/tree/main/mini_test_videos).

<sub>Note: If you want to test sim2real cases, you can try prompts like 'Turn it into the real domain'.</sub>

## Model Training

### Training Setup

To train a model, you can first download the training CSV files from the [csvs](https://huggingface.co/datasets/QingyanBai/Ditto-1M/tree/main/csvs_for_DiffSynth) directory on Hugging Face, then use the provided `train.sh` script for training.

```bash
# Download the training CSVs from HF dataset to your local directory
hf download QingyanBai/Ditto-1M --include="csvs_for_DiffSynth/*" --local-dir ./

# Run training
bash train.sh
```

### Multi-Node Training

Thanks to DiffSynth-Studio, this codebase supports multi-node training. You can consider using [DLRover](https://github.com/intelligent-machine-learning/dlrover) to support training across multiple machines.

## Denoising Enhancing

To use the denoising enhancing functionality, first install requirements for [Wan2.2](https://github.com/Wan-Video/Wan2.2):

```bash
cd denoising_enhancing
pip install -e .
```

Download the [Wan-AI/Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) model, then save the paths of videos to be processed into a txt file - the test case can be found [here](https://huggingface.co/datasets/QingyanBai/Ditto-1M/tree/main/enhancing_test_videos). Run the script:

```bash
bash run_video_enhancing.sh
```

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{bai2025ditto,
  title={Scaling Instruction-Based Video Editing with a High-Quality Synthetic Dataset},
  author={Bai, Qingyan and Wang, Qiuyu and Ouyang, Hao and Yu, Yue and Wang, Hanlin and Wang, Wen and Cheng, Ka Leong and Ma, Shuailei and Zeng, Yanhong and Liu, Zichen and Xu, Yinghao and Shen, Yujun and Chen, Qifeng},
  journal={arXiv preprint arXiv:2510.15742},
  year={2025}
}
```

## Acknowledgments

We thank [Wan](https://github.com/Wan-Video/Wan2.1) & [VACE](https://github.com/ali-vilab/VACE) & [Qwen-Image](https://github.com/QwenLM/Qwen-Image) for providing the powerful foundation model, and [QwenVL](https://github.com/QwenLM/Qwen2.5-VL) for the advanced visual understanding capabilities. We also thank [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) serving as the codebase for this repository.

## License

This project is licensed under the CC BY-NC-SA 4.0([Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/)).

The code is provided for academic research purposes only.

For any questions, please contact qingyanbai@hotmail.com.
