from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset


class ManifestImageDataset(Dataset):
    """JSONL image classification dataset.

    Each line should look like:
    ``{"image": "relative/path.jpg", "label": 3}``
    """

    def __init__(
        self,
        manifest: str | Path,
        image_root: str | Path,
        transform: Callable | None = None,
    ) -> None:
        self.image_root = Path(image_root)
        self.transform = transform
        with Path(manifest).open("r", encoding="utf-8") as handle:
            self.samples = [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(self.image_root / sample["image"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {"image": image, "label": int(sample["label"]), "meta": sample}
