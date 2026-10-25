"""Atalho antigo, mantido por compatibilidade: `python -m facerec.train` ==
`python -m facerec train` (código em facerec/cli/train.py)."""

from facerec.cli.train import main

if __name__ == "__main__":
    main()
