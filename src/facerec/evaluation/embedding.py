"""Embeddings de arquivos já alinhados e custo de inferência — compartilhado pelas avaliações.

Os embeddings ficam só em memória: nunca são gravados em disco pelos experimentos.
"""

from __future__ import annotations

import time

import numpy as np

from facerec.models.base import Embedder
from facerec.vision.pipeline import load_aligned


def embed_paths(
    embedder: Embedder, paths: list[str], batch_size: int = 64
) -> dict[str, np.ndarray]:
    """{caminho: embedding} para recortes produzidos pelo preprocess."""
    uniq = sorted(set(paths))
    out: dict[str, np.ndarray] = {}
    for i in range(0, len(uniq), batch_size):
        chunk = uniq[i : i + batch_size]
        emb = embedder.embed([load_aligned(p, embedder.input_size) for p in chunk])
        out.update(zip(chunk, emb, strict=True))
    return out


def pair_scores(embedder: Embedder, pairs: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
    """Pares (a, b, mesma_pessoa) -> (similaridade cosseno, rótulo 0/1)."""
    emb = embed_paths(embedder, [p for a, b, _ in pairs for p in (a, b)])
    scores = np.array([float(emb[a] @ emb[b]) for a, b, _ in pairs])
    labels = np.array([int(lbl) for _, _, lbl in pairs])
    return scores, labels


def inference_cost(embedder: Embedder, n: int = 50) -> dict:
    """Tempo médio por face (após aquecimento) e nº de parâmetros do modelo."""
    face = np.random.default_rng(0).integers(0, 255, (embedder.input_size,) * 2 + (3,), np.uint8)
    for _ in range(5):  # aquecimento
        embedder.embed(face)
    t0 = time.perf_counter()
    for _ in range(n):
        embedder.embed(face)
    return {
        "inference_ms_per_face": 1000 * (time.perf_counter() - t0) / n,
        "n_parameters": embedder.num_parameters(),
    }
