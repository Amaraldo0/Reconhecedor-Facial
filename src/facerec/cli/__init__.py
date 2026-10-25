"""Linha de comando: `python -m facerec <comando> [opções]` (ou só `facerec <comando>`).

Cada comando é um módulo com `main()` que só lê os argumentos, chama a biblioteca e imprime
o resultado. `python -m facerec <comando> --help` mostra as opções de cada um.
"""

from __future__ import annotations

import importlib
import sys

# comando -> (módulo com main(), descrição)
COMMANDS = {
    "preprocess": ("facerec.data.preprocess", "detecta e alinha as faces de um dataset"),
    "train": ("facerec.cli.train", "fine-tuning da FaceNet (E03-E05)"),
    "evaluate": ("facerec.cli.evaluate", "LFW 10-fold + validação privada (E01-E05, E08)"),
    "compare-matchers": ("facerec.cli.compare_matchers", "comparadores open-set (E06)"),
    "threshold": ("facerec.cli.threshold", "varredura de threshold + teste único (E07)"),
    "report": ("facerec.cli.report", "tabela de resultados e ROC comparativa"),
    "evaluate-folder": ("facerec.cli.evaluate_folder", "avaliação na pasta pessoas/"),
    "recognize": ("facerec.cli.recognize", "reconhece fotos usando pessoas/ como cadastro"),
    "app": ("facerec.cli.app", "aplicação web local (webcam)"),
}


def usage() -> str:
    width = max(map(len, COMMANDS))
    lines = ["uso: python -m facerec <comando> [opções]", "", "comandos:"]
    lines += [f"  {name:<{width}}  {desc}" for name, (_, desc) in COMMANDS.items()]
    lines += ["", "Opções de cada comando: python -m facerec <comando> --help"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return
    if argv[0] not in COMMANDS:
        sys.exit(f"Comando desconhecido: {argv[0]}\n\n{usage()}")
    module = importlib.import_module(COMMANDS[argv[0]][0])
    sys.argv = [f"facerec {argv[0]}", *argv[1:]]
    module.main()
