"""Modelos pré-treinados disponíveis pelo nome curto (usado pela avaliação da pasta do grupo)."""

from __future__ import annotations

from facerec.models.arcface import DEFAULT_ARCFACE_PATH

MODEL_PRESETS = {
    "facenet": {"name": "facenet"},
    "arcface": {"name": "arcface", "path": DEFAULT_ARCFACE_PATH},
    "mobilefacenet": {"name": "arcface", "path": "checkpoints/public/buffalo_s/w600k_mbf.onnx"},
}
DOWNLOAD_HINT = {
    "arcface": "python scripts/download_arcface.py",
    "mobilefacenet": "python scripts/download_arcface.py --pack buffalo_s",
}
