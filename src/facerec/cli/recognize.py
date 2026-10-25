"""Reconhece fotos usando a pasta `pessoas/` como cadastro (sem webcam e sem banco).

    python -m facerec recognize minha_foto.jpg
    python -m facerec recognize pasta_de_teste/
    python -m facerec recognize pessoas/                 # cada foto contra as OUTRAS fotos
    python -m facerec recognize foto.jpg --pre-aligned   # fotos já recortadas no rosto

Cadastro = cada subpasta de `pessoas/` é uma pessoa (nome = nome da subpasta). Modelo,
comparador e threshold vêm de configs/app.yaml, como na aplicação. Os embeddings ficam só
em memória. Se a foto consultada estiver dentro de `pessoas/`, ela é retirada do cadastro
antes da consulta (senão a resposta seria trivial: a foto se compararia consigo mesma).

Privacidade: roda só localmente. A saída mostra o nome da pessoa reconhecida (é a função do
sistema); não compartilhe essa saída fora do grupo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from facerec.core.config import load_yaml, pick_device, resolve_path
from facerec.data.datasets import IMAGE_EXTS
from facerec.models import build_embedder
from facerec.recognition.gallery import FolderGallery
from facerec.recognition.metrics import UNKNOWN
from facerec.vision.pipeline import FaceLoader, describe


def _gallery_root(arg: str) -> Path:
    """Caminho relativo parte do diretório atual (como a foto consultada); se não existir
    ali, tenta a raiz do projeto."""
    root = Path(arg)
    if not root.is_absolute():
        cwd_path = Path.cwd() / root
        root = cwd_path if cwd_path.is_dir() else resolve_path(arg)
    if not root.is_dir():
        sys.exit(f"Pasta do cadastro não encontrada: {arg}")
    return root


def _queries(arg: str) -> list[Path]:
    q = Path(arg)
    q = q if q.is_absolute() else Path.cwd() / q
    if q.is_dir():
        return sorted(p for p in q.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if q.is_file():
        return [q]
    sys.exit(f"Não encontrado: {arg}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("query", help="foto ou pasta de fotos a reconhecer")
    ap.add_argument(
        "--gallery", default="pessoas", help="pasta do cadastro (uma subpasta por pessoa)"
    )
    ap.add_argument("--config", default="configs/app.yaml")
    ap.add_argument("--pre-aligned", action="store_true")
    ap.add_argument("--threshold", type=float, help="sobrescreve o threshold do app.yaml")
    args = ap.parse_args()

    cfg = load_yaml(resolve_path(args.config))
    device = pick_device(cfg.get("device", "auto"))
    rec = cfg.get("recognition", {})
    threshold = args.threshold if args.threshold is not None else float(rec["threshold"])
    matcher_name = rec.get("matcher", "cosine")
    embedder = build_embedder(cfg["model"], device)
    loader = FaceLoader(embedder, args.pre_aligned, device)

    gallery = FolderGallery(
        _gallery_root(args.gallery), loader, embedder, matcher_name, rec.get("matcher_kwargs") or {}
    )
    print(f"Modelo: {embedder.name} | comparador: {matcher_name} | threshold: {threshold}")
    print(
        f"Cadastro: {gallery.n_people} pessoa(s), {len(gallery.labels)} foto(s)"
        + (f", {gallery.skipped} sem face" if gallery.skipped else "")
        + (f" ({gallery.loose} foto(s) soltas ignoradas)" if gallery.loose else "")
    )

    hits = total = 0
    print()
    for path in _queries(args.query):
        face, reason = loader(path)
        shown = path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path.name
        if face is None:
            print(f"{shown}: {describe(reason)}")
            continue
        m = gallery.identify(path, embedder.embed(face)[0], threshold)
        if m is None:
            print(f"{shown}: única foto do cadastro — nada para comparar")
            continue
        result = f"Pessoa reconhecida: {m.label}" if m.label != UNKNOWN else "Pessoa desconhecida"
        extra = f" [{describe(reason)}]" if reason else ""
        check = ""
        if m.truth is not None and m.truth in m.gallery_labels:
            total += 1
            hits += m.label == m.truth
            check = "  -> CERTO" if m.label == m.truth else "  -> ERRADO"
        print(f"{shown}: {result} (score {m.score:.3f}){extra}{check}")
    if total:
        print(f"\nAcertos nas fotos do cadastro: {hits}/{total} ({100 * hits / total:.0f}%)")


if __name__ == "__main__":
    main()
