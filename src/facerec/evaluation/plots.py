"""Todos os gráficos dos experimentos (único módulo que importa o matplotlib).

Cada função salva UM arquivo PNG em `path` e fecha a figura. Os gráficos só mostram
métricas agregadas — nunca imagens ou identificadores de pessoas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SAME_DIFF = ("mesma pessoa", "pessoas diferentes")


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path: Path, dpi: int = 120) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    _plt().close(fig)


def roc_points(scores, labels, n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(fpr, tpr) da curva ROC; com `n`, subamostrada para caber no metrics.json."""
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, scores)
    if n is not None:
        idx = np.unique(np.linspace(0, len(fpr) - 1, n).astype(int))
        fpr, tpr = fpr[idx], tpr[idx]
    return fpr, tpr


def save_score_hist(
    scores,
    labels,
    path: Path,
    title: str,
    names: tuple[str, str] = SAME_DIFF,
    xlabel: str = "similaridade cosseno",
    bins: int = 50,
    threshold: float | None = None,
) -> None:
    """Histograma dos scores da classe 1 (`names[0]`) contra a classe 0 (`names[1]`)."""
    scores, labels = np.asarray(scores), np.asarray(labels)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    edges = np.linspace(scores.min(), scores.max(), bins)
    ax.hist(scores[labels == 1], bins=edges, alpha=0.6, density=True, label=names[0])
    ax.hist(scores[labels == 0], bins=edges, alpha=0.6, density=True, label=names[1])
    if threshold is not None:
        ax.axvline(threshold, color="k", ls="--", lw=0.8)
    ax.set(xlabel=xlabel, ylabel="densidade", title=title)
    ax.legend(fontsize=8)
    _save(fig, path)


def save_roc(scores, labels, auc: float, path: Path, title: str) -> None:
    fpr, tpr = roc_points(scores, labels)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="FAR (falso positivo)", ylabel="1 - FRR (verdadeiro positivo)", title=title)
    ax.legend()
    _save(fig, path)


def save_confusion(
    cm: dict,
    path: Path,
    title: str,
    xticks: tuple[str, str] = ("prev. diferentes", "prev. mesma"),
    yticks: tuple[str, str] = ("real diferentes", "real mesma"),
) -> None:
    """Matriz 2x2 {tn, fp, fn, tp}: linhas = real (0, 1), colunas = previsto (0, 1)."""
    m = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.imshow(m, cmap="Blues")
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, str(v), ha="center", va="center", color="black", fontsize=12)
    ax.set_xticks([0, 1], list(xticks))
    ax.set_yticks([0, 1], list(yticks))
    ax.set_title(title, fontsize=10)
    _save(fig, path)


def save_roc_comparison(
    curves: dict[str, tuple],
    path: Path,
    title: str,
    log_far: bool = False,
    auc_digits: int = 4,
) -> None:
    """Várias curvas ROC no mesmo gráfico. curves = {rótulo: (fpr, tpr, auc)}."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, label=f"{label} (AUC {auc:.{auc_digits}f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    if log_far:
        ax.set(xscale="log", xlim=(1e-3, 1), xlabel="FAR (escala log)")
    else:
        ax.set(xlabel="FAR")
    ax.set(ylabel="TAR = 1 - FRR", title=title)
    ax.legend(fontsize=8)
    _save(fig, path, dpi=130)


def save_metric_bars(rows: dict[str, dict], keys: list[str], path: Path, title: str) -> None:
    """Barras agrupadas: um grupo por linha de `rows`, uma barra por métrica de `keys`."""
    plt = _plt()
    names = list(rows)
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.8 / len(keys)
    for i, k in enumerate(keys):
        ax.bar(np.arange(len(names)) + i * w, [rows[n][k] or 0 for n in names], w, label=k)
    ax.set_xticks(np.arange(len(names)) + 0.4 - w / 2, names)
    ax.set(ylim=(0, 1), title=title)
    ax.legend(fontsize=8)
    _save(fig, path)


def save_threshold_sweep(sweep: list[dict], chosen: float, path: Path, title: str) -> None:
    plt = _plt()
    t = [r["threshold"] for r in sweep]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, [r["far_unknown"] for r in sweep], label="FAR (desconhecido aceito)")
    ax.plot(t, [r["frr_known"] for r in sweep], label="FRR (conhecido rejeitado)")
    ax.plot(t, [r["accuracy"] for r in sweep], label="acurácia open-set")
    ax.axvline(chosen, color="k", ls="--", lw=0.8, label=f"escolhido = {chosen:.3f}")
    ax.set(xlabel="threshold", title=title)
    ax.legend(fontsize=8)
    _save(fig, path)


def save_loss_curves(history: list[dict], path: Path) -> None:
    """Fine-tuning: loss de treino × validação e EER da validação por época."""
    plt = _plt()
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], marker="o", label="treino (loss do exp.)")
    ax[0].plot(ep, [h["val_loss"] for h in history], marker="o", label="validação (triplet)")
    ax[0].set(xlabel="época", ylabel="loss", title="Curvas de loss")
    ax[0].legend()
    ax[1].plot(ep, [h["val_eer"] for h in history], marker="o", color="tab:red")
    ax[1].set(xlabel="época", ylabel="EER", title="EER na validação")
    _save(fig, path)
