"""Peças comuns da identificação em conjunto aberto (pasta do grupo, E06 e E07).

Para cada consulta o matcher devolve (melhor identidade, score). A consulta é um
"acerto aceitável" (alvo 1) se for de uma pessoa CONHECIDA e a melhor identidade estiver
correta; todo o resto (desconhecido, ou conhecido confundido com outra pessoa) é alvo 0.
O threshold separa esses dois grupos: score >= threshold -> aceita a melhor identidade.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from facerec.recognition.matching import Matcher, build_matcher
from facerec.recognition.metrics import UNKNOWN, compute_eer, roc_auc
from facerec.recognition.threshold import select_threshold


def matcher_label(cfg: dict) -> str:
    """Nome do comparador no relatório: "knn_k3", "cosine", ..."""
    return cfg["name"] + (f"_k{cfg['k']}" if "k" in cfg else "")


def fit_matcher(cfg: dict, embeddings: np.ndarray, labels: Sequence[str]) -> Matcher:
    """cfg = {"name": ..., **parâmetros do matcher}."""
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return build_matcher(cfg["name"], **kwargs).fit(embeddings, list(labels))


def top_matches(matcher: Matcher, queries: np.ndarray) -> tuple[list[str], np.ndarray]:
    tops = [matcher.best(q) for q in queries]
    return [t[0] for t in tops], np.array([t[1] for t in tops])


def accept_targets(true: Sequence[str], top: Sequence[str]) -> np.ndarray:
    return np.array([int(t != UNKNOWN and t == p) for t, p in zip(true, top, strict=True)])


def decide(top: Sequence[str], scores: np.ndarray, threshold) -> list[str]:
    """Aplica o threshold (um valor, ou um por consulta) e devolve a identidade ou UNKNOWN."""
    thr = np.broadcast_to(np.asarray(threshold, dtype=float), np.shape(scores))
    return [lbl if s >= t else UNKNOWN for lbl, s, t in zip(top, scores, thr, strict=True)]


def has_both_classes(target: np.ndarray) -> bool:
    return 0 < int(np.sum(target)) < len(target)


def choose_threshold(
    scores: np.ndarray,
    target: np.ndarray,
    strategy: str = "eer",
    far_target: float = 0.01,
    default: float = 0.0,
) -> float:
    """`select_threshold`, ou `default` quando só existe uma classe (nada a separar)."""
    if not has_both_classes(target):
        return default
    return select_threshold(scores, target, strategy, far_target)


def separability(scores: np.ndarray, target: np.ndarray) -> dict:
    """ROC-AUC e EER entre aceitáveis e não aceitáveis (vazio se só houver uma classe)."""
    if not has_both_classes(target):
        return {}
    return {"roc_auc": roc_auc(scores, target), "eer": compute_eer(scores, target)[0]}


def accept_reject_metrics(target: np.ndarray, accepted: np.ndarray) -> dict:
    """Precision/recall/F1 da decisão aceitar × rejeitar e a matriz de confusão dela."""
    target, accepted = np.asarray(target).astype(bool), np.asarray(accepted).astype(bool)
    tp = int((accepted & target).sum())
    fp = int((accepted & ~target).sum())
    fn = int((~accepted & target).sum())
    tn = int((~accepted & ~target).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "confusion_accept_reject": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
