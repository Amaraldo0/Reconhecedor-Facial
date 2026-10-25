"""Cadastro em memória a partir de uma pasta `pessoas/<nome>/*.jpg` (sem banco e sem webcam).

Usado pelo `python -m facerec recognize`. Se a foto consultada fizer parte do cadastro, ela é
retirada antes da consulta — senão a resposta seria trivial (a foto se compararia consigo).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from facerec.data.datasets import list_people
from facerec.recognition.matching import Matcher, build_matcher


@dataclass
class Match:
    label: str  # nome da pessoa ou UNKNOWN
    score: float
    truth: str | None  # pessoa verdadeira, quando a foto consultada está no cadastro
    gallery_labels: set[str]  # pessoas presentes no cadastro usado nesta consulta


class FolderGallery:
    def __init__(self, root: Path, loader, embedder, matcher_name: str, matcher_kwargs: dict):
        """`loader(path) -> (recorte | None, motivo)`, como `vision.pipeline.FaceLoader`."""
        items, self.n_people, self.loose = list_people(root)
        names = sorted(p.name for p in root.iterdir() if p.is_dir())
        faces, self.labels, self.paths, self.skipped = [], [], [], 0
        for path, idx in items:
            face, _ = loader(path)
            if face is None:
                self.skipped += 1
                continue
            faces.append(face)
            self.labels.append(names[idx])
            self.paths.append(path.resolve())
        if not faces:
            raise SystemExit(f"Nenhuma foto utilizável no cadastro ({root}).")
        self.embeddings = embedder.embed(faces)
        self.matcher_name = matcher_name
        self.matcher_kwargs = matcher_kwargs
        self.full = self._fit(list(range(len(self.labels))))

    def _fit(self, keep: list[int]) -> Matcher:
        return build_matcher(self.matcher_name, **self.matcher_kwargs).fit(
            self.embeddings[keep], [self.labels[i] for i in keep]
        )

    def identify(self, path: Path, embedding: np.ndarray, threshold: float) -> Match | None:
        """None = a foto é a única do cadastro (não sobra ninguém para comparar)."""
        matcher, truth = self.full, None
        rp = path.resolve()
        if rp in self.paths:
            keep = [i for i, p in enumerate(self.paths) if p != rp]
            truth = self.labels[self.paths.index(rp)]
            if not keep:
                return None
            matcher = self._fit(keep)
        label, score = matcher.identify(embedding, threshold)
        return Match(label, score, truth, set(matcher.labels))
