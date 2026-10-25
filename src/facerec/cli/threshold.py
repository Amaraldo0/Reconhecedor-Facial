"""E07 — varredura de threshold na validação do dataset privado e teste único."""

from __future__ import annotations

import argparse
import json

from facerec.evaluation.private import run_threshold_experiment


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    summary = run_threshold_experiment(args.config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
