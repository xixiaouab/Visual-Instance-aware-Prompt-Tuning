from __future__ import annotations

import torch
from torch import nn
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, device: str = "cuda") -> dict[str, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.shape[0]

    return {"loss": total_loss / max(len(loader.dataset), 1)}


@torch.no_grad()
def validate(model, loader, device: str = "cuda") -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return {"acc": correct / max(total, 1)}
