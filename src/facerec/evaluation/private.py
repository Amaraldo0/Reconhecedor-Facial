"""Experimentos no dataset privado em conjunto aberto (open-set): E06 e E07.

    python -m facerec compare-matchers --config configs/experiments/E06_matchers_open_set.yaml
    python -m facerec threshold        --config configs/experiments/E07_threshold_sweep.yaml

Protocolo (ADR-004):
- galeria  = sessões de cadastro das pessoas "conhecidas";
- validação = outras sessões dos conhecidos + pessoas "desconhecidas" (nunca cadastradas);
- teste    = sessões restantes + OUTRAS pessoas desconhecidas. Usado UMA vez, só no E07.

O threshold é escolhido sobre os scores da VALIDAÇÃO (alvos definidos em open_set.py).

Privacidade: só métricas AGREGADAS são salvas. Nada por participante.
"""

from __future__ import annotations

import json
import shutil
import time

import numpy as np

from facerec.core.config import environment_info, load_yaml, pick_device, resolve_path, set_seed
from facerec.data.datasets import list_private
from facerec.data.splits import Record, split_private_by_session
from facerec.evaluation import plots
from facerec.evaluation.embedding import embed_paths
from facerec.evaluation.open_set import (
    accept_targets,
    choose_threshold,
    decide,
    fit_matcher,
    matcher_label,
    separability,
    top_matches,
)
from facerec.models import build_embedder
from facerec.recognition.matching import Matcher
from facerec.recognition.metrics import UNKNOWN, binary_metrics, open_set_metrics
from facerec.recognition.threshold import select_threshold


def _prepare(cfg: dict):
    set_seed(cfg.get("seed", 42))
    p = cfg["private"]
    root = resolve_path(p["aligned_root"])
    if not root.exists():
        raise FileNotFoundError(
            f"{root} não existe. Organize as fotos em data/private/raw/<P01>/<sessao>/ e rode "
            "python -m facerec preprocess --dataset private --model <modelo>"
        )
    split = split_private_by_session(
        list_private(root),
        p["gallery_sessions"],
        p["val_sessions"],
        p["test_sessions"],
        p.get("n_unknown_val", 1),
        p.get("n_unknown_test", 1),
        seed=cfg.get("seed", 42),
    )
    model_cfg = dict(cfg["model"])
    if model_cfg.get("checkpoint"):
        model_cfg["checkpoint"] = str(resolve_path(model_cfg["checkpoint"]))
    embedder = build_embedder(model_cfg, pick_device(cfg.get("device", "auto")))
    paths = [r.path for part in ("gallery", "val", "test") for r in split[part]]
    emb = embed_paths(embedder, paths)  # fica só em memória, nunca vai para disco
    out_dir = resolve_path(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    return split, emb, out_dir


def _queries(records: list[Record], known: set[str], emb: dict):
    x = np.stack([emb[r.path] for r in records])
    y = [r.identity if r.identity in known else UNKNOWN for r in records]
    return x, y


def _gallery_matcher(cfg_m: dict, split, emb) -> Matcher:
    g = split["gallery"]
    return fit_matcher(cfg_m, np.stack([emb[r.path] for r in g]), [r.identity for r in g])


def _dataset_summary(split) -> dict:
    """Só contagens — nenhum identificador de participante."""
    return {
        "n_known_identities": len(split["known_identities"]),
        "n_unknown_val_identities": len(split["unknown_val"]),
        "n_unknown_test_identities": len(split["unknown_test"]),
        "n_gallery_images": len(split["gallery"]),
        "n_val_queries": len(split["val"]),
        "n_test_queries": len(split["test"]),
    }


def _write(summary: dict, out_dir) -> None:
    (out_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )


# --------------------------------------------------------------------------- E06
def run_matcher_comparison(config_path: str) -> dict:
    cfg = load_yaml(config_path)
    split, emb, out_dir = _prepare(cfg)
    shutil.copy(config_path, out_dir / "config_used.yaml")
    known = set(split["known_identities"])
    xv, yv = _queries(split["val"], known, emb)
    strategy = cfg.get("threshold_strategy", "eer")
    rows = {}
    for m_cfg in cfg["matchers"]:
        matcher = _gallery_matcher(m_cfg, split, emb)
        t0 = time.perf_counter()
        top, scores = top_matches(matcher, xv)
        ms = 1000 * (time.perf_counter() - t0) / max(len(xv), 1)
        target = accept_targets(yv, top)
        thr = choose_threshold(scores, target, strategy)
        res = open_set_metrics(yv, decide(top, scores, thr))
        res.update(threshold=thr, ms_per_query=ms, **separability(scores, target))
        rows[matcher_label(m_cfg)] = res
    best = max(rows, key=lambda k: (rows[k]["macro_f1"], rows[k]["accuracy"]))
    summary = {
        "experiment": cfg["experiment"],
        "model": cfg["model"],
        "dataset": _dataset_summary(split),
        "split_evaluated": "validation",
        "threshold_strategy": strategy,
        "matchers": rows,
        "best_matcher_on_validation": best,
        "environment": environment_info(),
    }
    _write(summary, out_dir)
    plots.save_metric_bars(
        rows,
        ["accuracy", "macro_f1", "false_accept_unknown", "false_reject_known"],
        out_dir / "matchers_comparison.png",
        "E06 — comparadores (validação, open-set)",
    )
    return summary


# --------------------------------------------------------------------------- E07
def threshold_sweep_open_set(true, top, scores, n: int = 101) -> list[dict]:
    """Métricas open-set em `n` thresholds entre o menor e o maior score."""
    sweep = []
    for t in np.linspace(scores.min(), scores.max(), n):
        m = open_set_metrics(true, decide(top, scores, t))
        sweep.append(
            {
                "threshold": float(t),
                "accuracy": m["accuracy"],
                "far_unknown": m["false_accept_unknown"],
                "frr_known": m["false_reject_known"],
                "macro_f1": m["macro_f1"],
            }
        )
    return sweep


def run_threshold_experiment(config_path: str) -> dict:
    cfg = load_yaml(config_path)
    split, emb, out_dir = _prepare(cfg)
    shutil.copy(config_path, out_dir / "config_used.yaml")
    known = set(split["known_identities"])
    matcher = _gallery_matcher(cfg["matcher"], split, emb)
    tcfg = cfg.get("threshold", {"strategy": "eer"})

    # 1) varredura e escolha na VALIDAÇÃO
    xv, yv = _queries(split["val"], known, emb)
    top_v, s_v = top_matches(matcher, xv)
    sweep = threshold_sweep_open_set(yv, top_v, s_v)
    thr = select_threshold(
        s_v, accept_targets(yv, top_v), tcfg["strategy"], tcfg.get("far_target", 0.01)
    )
    val_metrics = open_set_metrics(yv, decide(top_v, s_v, thr))

    # 2) teste — UMA única vez, com o threshold congelado
    xt, yt = _queries(split["test"], known, emb)
    top_t, s_t = top_matches(matcher, xt)
    target_t = accept_targets(yt, top_t)
    test_metrics = open_set_metrics(yt, decide(top_t, s_t, thr))
    test_binary = binary_metrics(s_t, target_t, thr)
    test_binary.update(separability(s_t, target_t))

    summary = {
        "experiment": cfg["experiment"],
        "model": cfg["model"],
        "matcher": cfg["matcher"],
        "threshold_strategy": tcfg,
        "chosen_threshold": thr,
        "dataset": _dataset_summary(split),
        "validation": val_metrics,
        "test": test_metrics,
        "test_binary_accept_reject": test_binary,
        "sweep_validation": sweep,
        "environment": environment_info(),
    }
    _write(summary, out_dir)
    plots.save_threshold_sweep(
        sweep, thr, out_dir / "threshold_sweep.png", "E07 — varredura de threshold (validação)"
    )
    plots.save_score_hist(
        s_t,
        target_t,
        out_dir / "test_score_hist.png",
        "E07 — scores no teste",
        names=("conhecido (correto)", "desconhecido/errado"),
        xlabel="score",
        bins=40,
        threshold=thr,
    )
    print(
        f"\n>>> Copie para configs/app.yaml:  recognition.threshold: {thr:.4f}  "
        f"(matcher: {cfg['matcher']['name']})"
    )
    return {k: v for k, v in summary.items() if k != "sweep_validation"}
