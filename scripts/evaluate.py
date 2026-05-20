from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vipt_base.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    print(f"Evaluation scaffold: config={args.config}, checkpoint={args.checkpoint}")


if __name__ == "__main__":
    main()
