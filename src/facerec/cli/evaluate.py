"""Avaliação LFW 10-fold + validação privada (E01–E05, E08)."""

from __future__ import annotations

import argparse
import json

from facerec.evaluation.lfw import evaluate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--checkpoint",
        choices=["best", "last"],
        default="best",
        help="best = escolhido na validação; last = última época (resultados em ultima_epoca/)",
    )
    args = ap.parse_args()
    r = evaluate(args.config, args.checkpoint)
    short = {k: v for k, v in r.items() if k != "environment"}
    for part in ("lfw", "private_val"):
        if part in short:
            short[part] = {k: v for k, v in short[part].items() if k != "roc_curve"}
    print(json.dumps(short, indent=2, default=str))


if __name__ == "__main__":
    main()
