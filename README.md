# Visual Instance-aware Prompt Tuning

Official PyTorch implementation scaffold for **Visual Instance-aware Prompt Tuning** (ACM MM 2025).

This repository now exposes the core architecture used by the project: probabilistic instance-aware prompt generation, dataset-level prompts, PCA-balanced prompt propagation, a prompt-injected ViT wrapper, and lightweight training/evaluation entrypoints. Full datasets, pretrained checkpoints, and exact experiment scripts will be released separately.

## Core Idea

Classical visual prompt tuning learns a shared prompt for all images. ViaPT instead predicts mean/std statistics from the current image tokens, samples instance-aware prompts by reparameterization, concatenates them with dataset-level prompts, and uses PCA to retain important prompting information across transformer layers.

```text
image -> patch tokens -> 2-layer prompt generator -> mean/std -> instance prompts
                                      |                              |
dataset-level prompts ----------------+                              v
                         PCA-balanced prompt propagation -> ViT blocks
                                                                    |
                                                                    v
                                                       task head / classifier
```

## Repository Layout

```text
configs/vipt_base.yaml        Minimal experiment configuration
scripts/train.py              Training entrypoint
scripts/evaluate.py           Evaluation entrypoint
src/vipt/models/prompts.py    ViaPT prompt generation and PCA propagation
src/vipt/models/vit_vipt.py   Prompt-injected ViT wrapper
src/vipt/engine/trainer.py    Training and validation loop skeleton
src/vipt/data/datasets.py     Dataset placeholders and sample schema
```

## Quick Start

```bash
pip install -r requirements.txt
python scripts/train.py --config configs/vipt_base.yaml
python scripts/evaluate.py --config configs/vipt_base.yaml --checkpoint path/to/checkpoint.pt
```

The current code is intentionally a research scaffold: it documents the method interfaces and critical control flow without bundling private datasets or final training recipes.

## Citation

```bibtex
@inproceedings{xiao2025vipt,
  title={Visual Instance-aware Prompt Tuning},
  author={Xiao, Xi and Zhang, Yunbei and Li, Xingjian and Wang, Tianyang and Wang, Xiao and Wei, Yuxiang and Hamm, Jihun and Xu, Min},
  booktitle={ACM Multimedia},
  year={2025}
}
```
