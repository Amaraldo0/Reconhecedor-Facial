"""Gera experiments/results/results_table.md e as ROC comparativas a partir dos metrics.json."""

from __future__ import annotations

import argparse

from facerec.evaluation.report import build_report


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    md = build_report()
    if md is None:
        print("Nenhum resultado em experiments/results/. Rode os experimentos primeiro.")
        return
    print(md)


if __name__ == "__main__":
    main()
