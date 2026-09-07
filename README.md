# computer-vision-project

# Efficient Attention for Perceptual Similarity (DreamSim-style)

## Overview

This project explores how **efficient attention mechanisms** affect perceptual similarity estimation in a **DreamSim-like** pipeline. Starting from a pretrained Vision Transformer (ViT) backbone, we replace the standard self-attention layers in the last blocks with alternative attention variants, fine-tune each configuration on human similarity judgments, and compare them in terms of **agreement with human perception** and **computational cost** (inference time, parameter count).

Given a *reference* image and two *candidate* images (a triplet), the model produces embeddings for each image and uses **cosine similarity** to predict which candidate is perceptually closer to the reference. The prediction is compared against the fraction of human votes collected for that triplet.

Two attention configurations are implemented and evaluated against the frozen pretrained baseline:

- **Baseline** — pretrained ViT backbone, zero-shot, no fine-tuning.
- **Standard Attention** — a re-implementation of vanilla multi-head self-attention, used as a fine-tuning control group.
- **MoH Attention** — a Mixture-of-Heads attention variant, with a subset of *shared* heads and a subset of *routed* heads selected dynamically per token, plus a load-balancing auxiliary loss.

## Dataset

The project uses **NIGHTS** (Novel Image Generations with Human-Tested Similarity), the dataset introduced in the DreamSim paper: triplets of (reference, left, right) images, each annotated with the fraction of human votes indicating which of the two candidates is more similar to the reference.

- **Project page**: https://dreamsim-nights.github.io/
- **Data source**: https://data.csail.mit.edu/nights/ (metadata CSV + chunked image archives)

Images and metadata are downloaded automatically by the notebook — no manual download needed. Data is organized into `ref/` and `distort/` folders, split into `train` / `val` / `test` according to the original dataset splits.

## Method

1. **Backbone**: `vit_base_patch16_224.dino` (pretrained, via `timm`), used to extract the `[CLS]` token embedding for each image in a triplet.
2. **Partial fine-tuning**: only the last transformer blocks (and the final norm layer) are unfrozen; the rest of the backbone stays frozen.
3. **Attention replacement**: in the unfrozen blocks, the original attention module is swapped with either:
   - `StandardAttention` — functionally equivalent multi-head attention, re-initialized from the pretrained weights;
   - `MoHAttentionPretrained` — Mixture-of-Heads attention with shared/routed heads and a routing auxiliary loss.
4. **Training objective**: binary cross-entropy between the predicted preference (from the cosine similarity margin) and the human vote fraction.
5. **Evaluation**: accuracy on train/val/test splits, plus inference-time benchmarking (GPU warm-up + timed forward passes) to compare computational efficiency across configurations.

## Results

| Model               | Train Acc. | Val Acc. |
|---------------------|:----------:|:--------:|
| Baseline (zero-shot)|     –      |  0.908   |
| Standard Attention  |   0.909    |  0.913   |
| MoH Attention       |   0.901    |  0.923   |

*(See the notebook for the full training curves, test-set metrics, and inference-time comparison.)*

## Repository structure

```
.
├── computer_vision_project.ipynb   # Main notebook: data pipeline, models, training, evaluation
└── README.md
```

## Requirements

- Python 3.10+
- `torch`, `torchvision`
- `timm`
- `pandas`, `numpy`, `matplotlib`, `Pillow`
- `tqdm`

Designed to run on **Google Colab** (uses `google.colab.drive` to persist model checkpoints to Google Drive); it can also run locally by adjusting the paths in the *Globals* section and removing the Drive-mounting cell.

## How to run

1. Open `computer_vision_project.ipynb` in Google Colab (recommended, GPU runtime).
2. Run the notebook top to bottom:
   - **Imports & Globals** — sets hyperparameters, paths, and model constants.
   - **Data** — downloads and prepares the NIGHTS dataset.
   - **Network** — defines the ViT wrapper and the attention modules.
   - **Training** — fine-tunes the Standard Attention and MoH Attention models (checkpoints saved to Google Drive).
   - **Evaluation** — computes accuracy and inference-time statistics for all three configurations.
3. Trained weights are saved under `CV_project/1_StandardAttention` and `CV_project/2_MoHAttention` in Google Drive, and reloaded automatically in the evaluation section.

## References

- DreamSim: Learning New Dimensions of Human Visual Similarity using Synthetic Data
- Mixture-of-Head Attention (MoH)
