"""`python -m facerec <comando>`, atalhos antigos e carregador de faces — sem dados reais."""

import importlib
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from facerec import cli
from facerec.vision.pipeline import MULTIPLE_FACES, FaceLoader, describe
from tests.conftest import FakeEmbedder


def test_usage_lists_every_command(capsys):
    cli.main([])
    out = capsys.readouterr().out
    assert all(name in out for name in cli.COMMANDS)


def test_unknown_command_exits():
    with pytest.raises(SystemExit):
        cli.main(["nao-existe"])


@pytest.mark.parametrize("name", list(cli.COMMANDS))
def test_every_command_module_has_main(name):
    assert callable(importlib.import_module(cli.COMMANDS[name][0]).main)


def test_dispatch_passes_arguments(monkeypatch):
    seen = {}
    monkeypatch.setattr(sys, "argv", ["pytest"])  # cli.main troca sys.argv; restaurado depois
    monkeypatch.setattr("facerec.cli.report.main", lambda: seen.update(argv=list(sys.argv)))
    cli.main(["report", "--x"])
    assert seen["argv"] == ["facerec report", "--x"]


@pytest.mark.parametrize(
    "legacy",
    [
        "evaluate",
        "evaluate_folder",
        "recognize",
        "train",
        "threshold",
        "compare_matchers",
        "report",
    ],
)
def test_legacy_module_commands_still_work(legacy):
    """`python -m facerec.<antigo> --help` continua funcionando."""
    r = subprocess.run(
        [sys.executable, "-m", f"facerec.{legacy}", "--help"], capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    assert "usage" in r.stdout


class FakeDetector:
    min_face_px, blur_threshold = 60, 15.0

    def __init__(self, n_faces):
        self.n_faces = n_faces

    def detect(self, img):
        from facerec.vision.detection import DetectedFace

        return [DetectedFace(np.array([0, 0, 100, 100.0]), 0.99, None)] * self.n_faces

    def align(self, img, face):
        return np.random.default_rng(0).integers(0, 255, (160, 160, 3), dtype=np.uint8)


def _loader(n_faces):
    loader = FaceLoader(FakeEmbedder(), pre_aligned=True)
    loader.pre_aligned, loader.detector = False, FakeDetector(n_faces)
    return loader


def test_face_loader_reasons(tmp_path):
    photo = tmp_path / "foto.png"
    Image.new("RGB", (120, 120)).save(photo)
    (tmp_path / "ruim.png").write_bytes(b"nao e imagem")

    assert _loader(1)(photo)[1] == ""
    face, reason = _loader(2)(photo)
    assert face is not None and reason == MULTIPLE_FACES
    assert _loader(0)(photo) == (None, "sem_face")
    assert _loader(1)(tmp_path / "ruim.png") == (None, "ilegivel")
    assert describe("sem_face") == "nenhuma face detectada"


def test_face_loader_pre_aligned_crops_center_square(tmp_path):
    photo = tmp_path / "recorte.png"
    Image.new("RGB", (200, 100)).save(photo)
    face, reason = FaceLoader(FakeEmbedder(), pre_aligned=True)(photo)
    assert reason == "" and face.shape == (FakeEmbedder.input_size,) * 2 + (3,)
