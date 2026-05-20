from __future__ import annotations

import torch
from torch import nn

from .prompts import (
    DatasetPromptBank,
    PCABalancedPromptPropagation,
    ProbabilisticInstancePromptGenerator,
    PromptInjector,
)


class TransformerBlock(nn.Module):
    """Small ViT block used as a readable architecture placeholder."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(tokens)
        attn_out, _ = self.attn(normed, normed, normed)
        tokens = tokens + attn_out
        return tokens + self.mlp(self.norm2(tokens))


class VIPTModel(nn.Module):
    """ViaPT: visual instance-aware prompt tuning for ViTs.

    The implementation mirrors the paper-level control flow:
    image-token statistics -> probabilistic instance prompts -> concatenate
    with dataset-level prompts -> PCA-balanced propagation across layers.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        num_classes: int = 1000,
        num_instance_prompts: int = 4,
        num_dataset_prompts: int = 4,
        pca_dim: int = 384,
        prompt_layers: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.prompt_layers = set(prompt_layers or range(depth))
        self.num_prompts = num_instance_prompts + num_dataset_prompts
        num_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList(
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        )

        self.instance_generator = ProbabilisticInstancePromptGenerator(
            embed_dim=embed_dim,
            num_instance_prompts=num_instance_prompts,
        )
        self.dataset_prompts = DatasetPromptBank(num_dataset_prompts, embed_dim)
        self.propagation = PCABalancedPromptPropagation(
            num_prompts=self.num_prompts,
            embed_dim=embed_dim,
            pca_dim=pca_dim,
        )
        self.injector = PromptInjector()
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def encode_patches(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(images).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(images.shape[0], -1, -1)
        return torch.cat([cls, tokens], dim=1) + self.pos_embed

    def initial_prompts(self, tokens: torch.Tensor) -> torch.Tensor:
        patch_tokens = tokens[:, 1:]
        instance_prompts = self.instance_generator(patch_tokens)
        dataset_prompts = self.dataset_prompts(tokens.shape[0])
        return torch.cat([instance_prompts, dataset_prompts], dim=1)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.encode_patches(images)
        prompts = self.initial_prompts(tokens)

        for layer_id, block in enumerate(self.blocks):
            if layer_id in self.prompt_layers:
                tokens_with_prompts = self.injector.insert(tokens, prompts)
                tokens_with_prompts = block(tokens_with_prompts)
                tokens, prompt_outputs = self.injector.split_after_block(
                    tokens_with_prompts, self.num_prompts
                )
                prompts = self.propagation(prompt_outputs)
            else:
                tokens = block(tokens)
        return self.norm(tokens[:, 0])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(images))
