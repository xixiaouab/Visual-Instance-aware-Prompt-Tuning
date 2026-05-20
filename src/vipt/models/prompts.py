from __future__ import annotations

import math

import torch
from torch import nn


class DatasetPromptBank(nn.Module):
    """Dataset-level prompts shared by all inputs."""

    def __init__(self, num_prompts: int, embed_dim: int) -> None:
        super().__init__()
        self.prompts = nn.Parameter(torch.empty(1, num_prompts, embed_dim))
        nn.init.trunc_normal_(self.prompts, std=0.02)

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.prompts.expand(batch_size, -1, -1)


class ProbabilisticInstancePromptGenerator(nn.Module):
    """ViaPT instance-aware prompt generator.

    The paper uses image tokens E0 rather than raw images for efficiency. A
    lightweight 2-layer convolutional encoder predicts Gaussian statistics
    (mean and std), then lambda instance prompts are sampled by the
    reparameterization trick.
    """

    def __init__(
        self,
        embed_dim: int,
        num_instance_prompts: int,
        hidden_dim: int = 256,
        fixed_eval_seed: int = 0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_instance_prompts = num_instance_prompts
        self.encoder = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.to_stats = nn.Linear(hidden_dim, embed_dim * 2)
        generator = torch.Generator().manual_seed(fixed_eval_seed)
        eps = torch.randn(num_instance_prompts, embed_dim, generator=generator)
        self.register_buffer("fixed_eval_eps", eps)

    def _tokens_to_map(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        batch, num_tokens, dim = patch_tokens.shape
        side = int(math.sqrt(num_tokens))
        if side * side != num_tokens:
            raise ValueError("Patch tokens must form a square grid for the conv encoder.")
        return patch_tokens.transpose(1, 2).reshape(batch, dim, side, side)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        features = self.encoder(self._tokens_to_map(patch_tokens)).flatten(1)
        mean, log_std = self.to_stats(features).chunk(2, dim=-1)
        std = log_std.clamp(min=-5.0, max=2.0).exp()

        if self.training:
            eps = torch.randn(
                patch_tokens.shape[0],
                self.num_instance_prompts,
                self.embed_dim,
                device=patch_tokens.device,
                dtype=patch_tokens.dtype,
            )
        else:
            eps = self.fixed_eval_eps.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
            eps = eps.unsqueeze(0).expand(patch_tokens.shape[0], -1, -1)
        return mean[:, None, :] + eps * std[:, None, :]


class PCABalancedPromptPropagation(nn.Module):
    """Balanced prompt propagation via PCA + learnable padding.

    VPT-Deep is recovered when ``pca_dim=0``; VPT-Shallow is recovered when
    ``pca_dim=embed_dim``. ViaPT keeps the informative PCA subspace and fills
    the remaining dimensions with learnable parameters.
    """

    def __init__(self, num_prompts: int, embed_dim: int, pca_dim: int) -> None:
        super().__init__()
        if not 0 <= pca_dim <= embed_dim:
            raise ValueError("pca_dim must be in [0, embed_dim].")
        self.num_prompts = num_prompts
        self.embed_dim = embed_dim
        self.pca_dim = pca_dim
        tail_dim = embed_dim - pca_dim
        self.learnable_tail = nn.Parameter(torch.empty(1, num_prompts, tail_dim))
        if tail_dim > 0:
            nn.init.trunc_normal_(self.learnable_tail, std=0.02)

    def forward(self, previous_prompt_outputs: torch.Tensor) -> torch.Tensor:
        batch, prompt_count, dim = previous_prompt_outputs.shape
        if prompt_count != self.num_prompts or dim != self.embed_dim:
            raise ValueError("Unexpected prompt tensor shape.")

        if self.pca_dim == 0:
            retained = previous_prompt_outputs.new_zeros(batch, prompt_count, 0)
        elif self.pca_dim == self.embed_dim:
            retained = previous_prompt_outputs
        else:
            flat = previous_prompt_outputs.reshape(batch * prompt_count, dim)
            centered = flat - flat.mean(dim=0, keepdim=True)
            q = min(self.pca_dim, centered.shape[0], dim)
            _, _, components = torch.pca_lowrank(centered, q=q)
            retained = centered @ components
            if q < self.pca_dim:
                padding = retained.new_zeros(retained.shape[0], self.pca_dim - q)
                retained = torch.cat([retained, padding], dim=-1)
            retained = retained.reshape(batch, prompt_count, self.pca_dim)

        tail = self.learnable_tail.expand(batch, -1, -1)
        return torch.cat([retained, tail], dim=-1)


class PromptInjector(nn.Module):
    """Insert prompts after the CLS token and remove them after a block."""

    def insert(self, tokens: torch.Tensor, prompts: torch.Tensor) -> torch.Tensor:
        cls_token, patch_tokens = tokens[:, :1], tokens[:, 1:]
        return torch.cat([cls_token, prompts, patch_tokens], dim=1)

    def split_after_block(
        self, tokens: torch.Tensor, prompt_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cls_token = tokens[:, :1]
        prompt_outputs = tokens[:, 1 : 1 + prompt_length]
        patch_tokens = tokens[:, 1 + prompt_length :]
        image_tokens = torch.cat([cls_token, patch_tokens], dim=1)
        return image_tokens, prompt_outputs
