"""Avaliação de verificação facial (E01–E05, E08).

    python -m facerec evaluate --config configs/experiments/E01_facenet_baseline.yaml

1) LFW — protocolo oficial de 10 folds (pairs.txt, 6000 pares):
   para cada fold i, o threshold é escolhido nos OUTROS 9 folds e aplicado no fold i.
   Reportamos média ± desvio da acurácia, além de ROC-AUC e EER sobre todos os pares.
2) Dataset privado — VALIDAÇÃO (sessões de validação, pares entre sessões diferentes).
   É a métrica usada para ESCOLHER o modelo (o teste privado só é usado uma vez, no E07).
3) Custo: nº de parâmetros e tempo médio de inferência por face.

Tudo é salvo AGREGADO em experiments/results/<ID>/ (metrics.json, roc.png, hist_*.png).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from facerec.core.config import environment_info, load_yaml, pick_device, resolve_path, set_seed
from facerec.data.datasets import list_private, make_verification_pairs, parse_lfw_pairs
from facerec.data.splits import split_private_by_session
from facerec.evaluation import plots
from facerec.evaluation.embedding import inference_cost, pair_scores
from facerec.models import build_embedder
from facerec.recognition.metrics import binary_metrics, compute_eer, roc_auc, verification_report
from facerec.recognition.threshold import select_threshold


def lfw_10fold(scores_folds, labels_folds, strategy: str = "max_accuracy") -> dict:
    accs, f1s, precs, recs, thrs = [], [], [], [], []
    cm = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    for i in range(len(scores_folds)):
        tr_s = np.concatenate([s for j, s in enumerate(scores_folds) if j != i])
        tr_y = np.concatenate([y for j, y in enumerate(labels_folds) if j != i])
        thr = select_threshold(tr_s, tr_y, strategy)
        m = binary_metrics(scores_folds[i], labels_folds[i], thr)
        accs.append(m["accuracy"])
        f1s.append(m["f1"])
        precs.append(m["precision"])
        recs.append(m["recall"])
        thrs.append(thr)
        for k in cm:
            cm[k] += m["confusion_matrix"][k]
    all_s, all_y = np.concatenate(scores_folds), np.concatenate(labels_folds)
    eer, eer_thr = compute_eer(all_s, all_y)
    thr_mean = float(np.mean(thrs))
    m_mean = binary_metrics(all_s, all_y, thr_mean)
    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "precision_mean": float(np.mean(precs)),
        "recall_mean": float(np.mean(recs)),
        "threshold_mean": thr_mean,
        "far_at_threshold_mean": m_mean["far"],
        "frr_at_threshold_mean": m_mean["frr"],
        "roc_auc": roc_auc(all_s, all_y),
        "eer": eer,
        "eer_threshold": eer_thr,
        "confusion_matrix_sum": cm,
        "n_pairs": int(len(all_y)),
    }


def save_verification_plots(scores, labels, auc: float, cm: dict, out_dir: Path, tag: str) -> dict:
    """hist_<tag>.png, roc_<tag>.png e confusion_<tag>.png. Devolve a curva ROC subamostrada
    (vai para o metrics.json e alimenta o gráfico comparativo do report)."""
    plots.save_score_hist(scores, labels, out_dir / f"hist_{tag}.png", f"Similaridades — {tag}")
    plots.save_roc(scores, labels, auc, out_dir / f"roc_{tag}.png", f"ROC — {tag}")
    plots.save_confusion(cm, out_dir / f"confusion_{tag}.png", f"Matriz de confusão — {tag}")
    fpr, tpr = plots.roc_points(scores, labels, n=200)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist()}


def private_validation_pairs(cfg: dict, seed: int) -> list[tuple] | None:
    pcfg = cfg.get("private")
    if not pcfg or not pcfg.get("enabled", False):
        return None
    root = resolve_path(pcfg["aligned_root"])
    if not root.exists():
        print(f"Dataset privado não encontrado em {root}; pulando.")
        return None
    split = split_private_by_session(
        list_private(root),
        pcfg["gallery_sessions"],
        pcfg["val_sessions"],
        pcfg["test_sessions"],
        pcfg.get("n_unknown_val", 1),
        pcfg.get("n_unknown_test", 1),
        seed=seed,
    )
    # verificação na validação: galeria + consultas de validação (nunca o teste)
    return make_verification_pairs(split["gallery"] + split["val"], pcfg.get("n_pairs", 2000), seed)


def evaluate(config_path: str, which: str = "best") -> dict:
    """which = "best" (checkpoint escolhido na validação) ou "last" (última época treinada)."""
    cfg = load_yaml(config_path)
    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = pick_device(cfg.get("device", "auto"))
    model_cfg = dict(cfg["model"])
    if cfg.get("training", {}).get("strategy", "none") != "none":
        ckpt = resolve_path(cfg["output"]["checkpoint"])
        if which == "last":
            ckpt = ckpt.with_name(ckpt.stem + "_last" + ckpt.suffix)
        model_cfg["checkpoint"] = str(ckpt)
    embedder = build_embedder(model_cfg, device)
    out_dir = resolve_path(cfg["output"]["results_dir"])
    if which == "last":
        out_dir = out_dir / "ultima_epoca"
    out_dir.mkdir(parents=True, exist_ok=True)
    ev = cfg.get("evaluation", {})
    results: dict = {"experiment": cfg["experiment"], "model": model_cfg.get("name")}

    lfw = ev.get("lfw")
    if lfw and resolve_path(lfw["pairs"]).exists():
        folds = parse_lfw_pairs(resolve_path(lfw["pairs"]), resolve_path(lfw["root"]), ext=".png")
        s_folds, y_folds = zip(*(pair_scores(embedder, f) for f in folds), strict=True)
        rep = lfw_10fold(list(s_folds), list(y_folds), ev.get("lfw_strategy", "max_accuracy"))
        s, y = np.concatenate(s_folds), np.concatenate(y_folds)
        rep["roc_curve"] = save_verification_plots(
            s, y, rep["roc_auc"], rep["confusion_matrix_sum"], out_dir, "lfw"
        )
        results["lfw"] = rep
    else:
        print("LFW não encontrado; rode scripts/download_lfw.py e o preprocess.")

    pairs = private_validation_pairs(ev, seed)
    if pairs:
        s, y = pair_scores(embedder, pairs)
        thr = select_threshold(s, y, "eer")  # threshold da PRÓPRIA validação, só para referência
        rep = verification_report(s, y, thr)
        rep["roc_curve"] = save_verification_plots(
            s, y, rep["roc_auc"], rep["confusion_matrix"], out_dir, "private_val"
        )
        results["private_val"] = rep

    results["cost"] = inference_cost(embedder)
    results["device"] = device
    results["environment"] = environment_info()
    (out_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    return results
