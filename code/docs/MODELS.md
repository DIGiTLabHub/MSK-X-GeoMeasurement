# Models and dependencies

Install the training dependencies separately from the CPU replay environment:

```bash
python -m pip install -r code/requirements-training.txt
# For frozen medical encoders, also install:
python -m pip install -r code/requirements-frozen.txt
```

The archived execution package specifies Python 3.11, PyTorch 2.4.0 and torchvision 0.19.0 (CUDA 12.1 in the executed environment). Use a CUDA-compatible PyTorch installation for your machine. Full GPU retraining is separate from verification of saved results.

| Condition | Implementation/source | Expected location relative to `MSK_WORKSPACE` |
| --- | --- | --- |
| EfficientNet-B0, ConvNeXt-T, ViT-B/16 | `timm`, ImageNet initialization | Weights fetched by `timm`; caches under `outputs/model_cache/` |
| BiomedCLIP | `open_clip`; [released model](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) | Model-hub cache; no local checkpoint copy in this repository |
| MedImageInsight | [released wrapper and weights](https://huggingface.co/lion-ai/MedImageInsights) | `MedImageInsights/medimageinsightmodel.py`, `MedImageInsights/2024.09.27/vision_model/medimageinsigt-v1.0.0.pt`, and its accompanying configuration/language files |
| MedCLIP-Swin | [official MedCLIP source](https://github.com/RyanWangZf/MedCLIP) | Source in `external_models/MedCLIP/`; released checkpoint directory `checkpoints/MedCLIP/vit/` containing `pytorch_model.bin` |
| MedSAM ViT-B encoder | [MedSAM](https://github.com/bowang-lab/MedSAM), with [SAM source](https://github.com/facebookresearch/segment-anything) | SAM source in `external_models/segment-anything/`; checkpoint `cvpr/checkpoints/medsam_vit_b.pth` (legacy executed relative path) |
| Structured foot U-Net | Implemented in this repository | Random initialization; trained readout weights are generated under the selected output directory |

Obtain upstream code, configuration and weights directly from the providers. They are not vendored here. The legacy MedSAM checkpoint path is preserved to keep the executed adapter unchanged.

The shared frozen preprocessing canvas is 224 pixels. MedImageInsight's released transform subsequently produces a 480-pixel tensor; MedSAM is a separate 1024-pixel condition. The foot U-Net uses a 256-pixel canvas. These are distinct declared interfaces.

Bone X-ray VLP was unavailable at the recorded experiment date. It is not a runnable default or replaced by ImageNet weights. Historical ResNet50, MedCLIP-RN50 and DAX records are retained only where the original bootstrap draw schedule depends on them; they are not additional primary comparisons in this release.

Provider prompts use the historical aliases recorded in the manuscript. Whether those aliases can be called now is provider-dependent; `--model` can record an explicit alternative for a separate rerun. A new interaction is not evidence that the historical outputs have been reproduced.
