from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vipt.models import VIPTModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vipt_base.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    model = VIPTModel(**cfg["model"])
    params = sum(parameter.numel() for parameter in model.parameters())
    print(f"ViaPT model initialized with {params:,} parameters.")
    print("Training data/checkpoint release is pending; wire your DataLoader here.")


if __name__ == "__main__":
    main()
