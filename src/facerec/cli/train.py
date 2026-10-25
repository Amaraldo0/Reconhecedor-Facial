"""Fine-tuning da FaceNet guiado por config (E03, E04, E05)."""

from __future__ import annotations

import argparse
import json

from facerec.training.train import train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    s = train(args.config)
    print(json.dumps({k: v for k, v in s.items() if k != "history"}, indent=2, default=str))


if __name__ == "__main__":
    main()
