"""Arquivo de foto -> recorte alinhado pronto para a rede, do mesmo jeito em todos os lugares.

Dois caminhos:
- `FaceDetector.detect_single` (detection.py): modo ESTRITO da aplicação — exige exatamente
  uma face de boa qualidade e levanta erros de domínio.
- `FaceLoader` (este módulo): modo TOLERANTE das avaliações e do `recognize` — nunca levanta
  erro por foto; se houver várias faces usa a maior e devolve o motivo quando descarta a foto.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# motivo -> mensagem para o usuário (as chaves também são as contagens do relatório agregado)
SKIP_MESSAGES = {
    "ilegivel": "arquivo ilegível",
    "sem_face": "nenhuma face detectada",
    "face_pequena": "face muito pequena",
    "desfocada": "imagem desfocada",
}
MULTIPLE_FACES = "varias_faces_usou_maior"
MULTIPLE_FACES_MESSAGE = "várias faces — usei a maior"


def load_aligned(path: str | Path, size: int) -> np.ndarray:
    """Recorte já produzido pelo preprocess (LFW/CelebA/privado): só garante o tamanho."""
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def load_pre_aligned(path: str | Path, size: int) -> np.ndarray:
    """Foto já recortada: corta o quadrado central e redimensiona para a entrada da rede."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return np.asarray(img.resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)


class FaceLoader:
    """`loader(path) -> (recorte | None, motivo)`.

    motivo: "" (ok), `MULTIPLE_FACES` (ok, mas havia várias faces) ou uma chave de
    `SKIP_MESSAGES` (recorte None = foto descartada).
    """

    def __init__(self, embedder, pre_aligned: bool, device: str = "cpu", quality: bool = False):
        self.size = embedder.input_size
        self.pre_aligned = pre_aligned
        self.quality = quality
        self.detector = None
        if not pre_aligned:
            from facerec.vision.detection import FaceDetector

            self.detector = FaceDetector(
                device=device,
                align_mode=embedder.align_mode,
                image_size=self.size,
                prob_threshold=0.9,
            )

    def __call__(self, path: Path) -> tuple[np.ndarray | None, str]:
        from facerec.vision.detection import laplacian_variance, load_image

        try:
            if self.pre_aligned:
                return load_pre_aligned(path, self.size), ""
            img = load_image(path)
            faces = self.detector.detect(img)
        except Exception:
            return None, "ilegivel"
        if not faces:
            return None, "sem_face"
        if self.quality and faces[0].size < self.detector.min_face_px:
            return None, "face_pequena"
        face = self.detector.align(img, faces[0])
        if self.quality and laplacian_variance(face) < self.detector.blur_threshold:
            return None, "desfocada"
        return face, MULTIPLE_FACES if len(faces) > 1 else ""


def describe(reason: str) -> str:
    """Mensagem legível para um motivo devolvido por `FaceLoader`."""
    if reason == MULTIPLE_FACES:
        return MULTIPLE_FACES_MESSAGE
    return SKIP_MESSAGES.get(reason, reason)
