"""Aplicação web local (FastAPI + webcam). Equivale a `uvicorn facerec.app.main:app`.

Escuta só em 127.0.0.1 por padrão: as imagens nunca saem da máquina.
"""

from __future__ import annotations

import argparse


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="reinicia ao editar o código")
    args = ap.parse_args()
    uvicorn.run("facerec.app.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
