"""Avaliação do reconhecimento no dataset do grupo (pasta `pessoas/`).

    python -m facerec evaluate-folder                         # pessoas/, modelos disponíveis
    python -m facerec evaluate-folder --pre-aligned           # fotos já recortadas/alinhadas
    python -m facerec evaluate-folder --models facenet mobilefacenet --matchers cosine prototype

Estrutura esperada (uma subpasta por pessoa; pode haver subpastas de sessão dentro):
    pessoas/<pessoa>/*.jpg

Protocolo (sem vazamento):
1. Verificação: todos os pares de fotos -> "mesma pessoa" × "pessoas diferentes"
   -> ROC, AUC, EER e histograma de similaridade.
2. Identificação em conjunto aberto, para cada comparador:
   - consulta CONHECIDA: cada foto é retirada da galeria e consultada contra o resto
     (a pessoa continua cadastrada com as outras fotos) -> deve ser reconhecida;
   - consulta DESCONHECIDA: cada pessoa é retirada INTEIRA da galeria e suas fotos são
     consultadas -> devem dar "Pessoa desconhecida".
3. Threshold por validação cruzada POR PESSOA: as pessoas são divididas em k grupos; o
   threshold usado nas consultas de um grupo é escolhido (EER) só com as consultas dos
   OUTROS grupos. Assim nenhuma consulta influencia o próprio threshold.

Privacidade: roda só localmente; nomes de pastas/arquivos NUNCA aparecem na saída nem nos
arquivos gerados (só contagens e métricas agregadas em experiments/results/dataset_grupo/).
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from facerec.core.config import environment_info, pick_device, resolve_path, set_seed
from facerec.data.datasets import list_people
from facerec.evaluation import plots
from facerec.evaluation.embedding import inference_cost
from facerec.evaluation.open_set import (
    accept_reject_metrics,
    accept_targets,
    choose_threshold,
    decide,
    fit_matcher,
    matcher_label,
    separability,
)
from facerec.models.registry import DOWNLOAD_HINT, MODEL_PRESETS
from facerec.recognition.metrics import (
    UNKNOWN,
    binary_metrics,
    compute_eer,
    open_set_metrics,
    roc_auc,
)
from facerec.vision.pipeline import FaceLoader


# ------------------------------------------------------------------ leitura da pasta
def prepare_faces(items, embedder, pre_aligned: bool, device: str, quality: bool, preview_dir=None):
    """Detecta/alinha cada foto. Retorna (faces, rótulos, contagem por motivo)."""
    loader = FaceLoader(embedder, pre_aligned, device, quality)
    faces, labels, stats = [], [], Counter()
    for path, person in items:
        face, reason = loader(path)
        if reason:
            stats[reason] += 1
        if face is None:
            continue
        faces.append(face)
        labels.append(person)
        stats["ok"] += 1
        if preview_dir is not None:  # recorte que entra na rede, para conferência visual
            d = preview_dir / f"pessoa_{person + 1:02d}"
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(face).save(d / f"foto_{stats['ok']:03d}.png")
    return faces, labels, stats


# ------------------------------------------------------------------ avaliação (testável)
def verification(emb: np.ndarray, labels: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    """Todos os pares i<j: similaridade cosseno e se é a mesma pessoa."""
    sims = emb @ emb.T
    iu = np.triu_indices(len(labels), k=1)
    scores = sims[iu]
    same = (labels[iu[0]] == labels[iu[1]]).astype(int)
    res = {"n_pairs_same": int(same.sum()), "n_pairs_diff": int((1 - same).sum())}
    if 0 < same.sum() < len(same):
        eer, thr = compute_eer(scores, same)
        res.update(roc_auc=roc_auc(scores, same), eer=eer, eer_threshold=thr)
        res.update({k: v for k, v in binary_metrics(scores, same, thr).items() if k != "threshold"})
    return res, scores, same


def open_set_queries(emb: np.ndarray, labels: np.ndarray, matcher_cfg: dict) -> list[dict]:
    """Consultas conhecidas (deixa uma foto de fora) e desconhecidas (deixa a pessoa de fora)."""
    lab = labels.astype(str)
    counts = Counter(labels.tolist())
    queries = []
    for i in range(len(labels)):
        if counts[labels[i]] < 2:
            continue  # sem outra foto da pessoa, ela não estaria cadastrada
        keep = np.arange(len(labels)) != i
        top, score = fit_matcher(matcher_cfg, emb[keep], lab[keep]).best(emb[i])
        queries.append({"person": int(labels[i]), "true": lab[i], "top": top, "score": score})
    for person in np.unique(labels):
        keep = labels != person
        if len(np.unique(labels[keep])) < 1:
            continue
        m = fit_matcher(matcher_cfg, emb[keep], lab[keep])
        for i in np.where(labels == person)[0]:
            top, score = m.best(emb[i])
            queries.append({"person": int(person), "true": UNKNOWN, "top": top, "score": score})
    return queries


def cross_validated_decisions(queries: list[dict], k: int = 5, seed: int = 42) -> list[float]:
    """Threshold de cada consulta escolhido (EER) só com consultas de OUTRAS pessoas."""
    people = sorted({q["person"] for q in queries})
    random.Random(seed).shuffle(people)
    k = max(2, min(k, len(people)))
    fold_of = {p: i % k for i, p in enumerate(people)}
    scores = np.array([q["score"] for q in queries])
    target = accept_targets([q["true"] for q in queries], [q["top"] for q in queries])
    folds = np.array([fold_of[q["person"]] for q in queries])
    # sem as duas classes nos outros grupos: usa o EER global (registrado no relatório)
    fallback = choose_threshold(scores, target, "eer")
    thresholds = np.zeros(len(queries))
    for f in range(k):
        train = folds != f
        thresholds[folds == f] = choose_threshold(
            scores[train], target[train], "eer", default=fallback
        )
    return thresholds.tolist()


def evaluate_open_set(emb, labels, matcher_cfg: dict, k: int = 5, seed: int = 42) -> dict:
    queries = open_set_queries(emb, labels, matcher_cfg)
    if not queries:
        return {"erro": "nenhuma consulta possível (cada pessoa precisa de >= 2 fotos)"}
    thr = cross_validated_decisions(queries, k, seed)
    true, top = [q["true"] for q in queries], [q["top"] for q in queries]
    scores = np.array([q["score"] for q in queries])
    pred = decide(top, scores, thr)
    target = accept_targets(true, top)
    res = open_set_metrics(true, pred)
    res.update(threshold_mean=float(np.mean(thr)), threshold_std=float(np.std(thr)))
    res.update(accept_reject_metrics(target, np.array([p != UNKNOWN for p in pred])))
    res.update(separability(scores, target))
    return res


# ------------------------------------------------------------------ saída
def _pct(v) -> str:
    return "—" if v is None else f"{100 * v:.1f}%"


def results_table(results: dict) -> str:
    lines = [
        "| Modelo | Comparador | Acurácia (open-set) | F1 | EER | Reconheceu conhecidos | "
        "Rejeitou desconhecidos | Confundiu pessoas | ms/face |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for model, r in results.items():
        for mname, m in r["open_set"].items():
            if "erro" in m:
                continue
            fau = m["false_accept_unknown"]
            lines.append(
                f"| {model} | {mname} | {_pct(m['accuracy'])} | {_pct(m['f1'])} | "
                f"{_pct(m.get('eer'))} | {_pct(m['known_accuracy'])} | "
                f"{_pct(None if fau is None else 1 - fau)} | {_pct(m['misidentification'])} | "
                f"{r['cost']['inference_ms_per_face']:.1f} |"
            )
    return "\n".join(lines)


def _plots(model: str, scores, same, res: dict, out_dir: Path) -> tuple | None:
    """Histograma e matrizes de confusão do modelo; devolve a curva ROC para o comparativo."""
    for mname, r in res["open_set"].items():
        if r.get("confusion_accept_reject"):
            plots.save_confusion(
                r["confusion_accept_reject"],
                out_dir / f"confusion_{model}_{mname}.png",
                f"{model} + {mname}",
                xticks=("rejeitou", "aceitou"),
                yticks=("deveria rejeitar", "deveria aceitar"),
            )
    if not (same.sum() and (1 - same).sum()):
        return None
    plots.save_score_hist(
        scores, same, out_dir / f"hist_{model}.png", f"Similaridades — {model}", bins=40
    )
    return (*plots.roc_points(scores, same), res["verification"]["roc_auc"])


# ------------------------------------------------------------------ principal
def run(
    root: Path,
    models: list[str],
    matchers: list[dict],
    pre_aligned: bool,
    quality: bool,
    out_dir: Path,
    device: str = "auto",
    k: int = 5,
    seed: int = 42,
    embedder_factory=None,
    preview_dir: Path | None = None,
) -> dict:
    from facerec.models import build_embedder

    set_seed(seed)
    device = pick_device(device)
    items, n_people, loose = list_people(root)
    if loose:
        print(
            f"Aviso: {loose} foto(s) soltas em {root.name}/ foram ignoradas — "
            "coloque cada pessoa numa subpasta própria."
        )
    if n_people < 2:
        raise SystemExit(
            f"São necessárias pelo menos 2 pessoas (subpastas) em {root}; encontradas: {n_people}."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    curves: dict = {}
    for model in models:
        if embedder_factory is not None:
            embedder = embedder_factory(model)
        else:
            cfg = dict(MODEL_PRESETS[model])
            if "path" in cfg and not resolve_path(cfg["path"]).exists():
                print(f"[{model}] pesos não encontrados — pulei. Baixe com: {DOWNLOAD_HINT[model]}")
                continue
            embedder = build_embedder(cfg, device)
        print(f"\n[{model}] preparando faces...")
        pdir = preview_dir / model if preview_dir is not None else None
        faces, labels, stats = prepare_faces(items, embedder, pre_aligned, device, quality, pdir)
        if pdir is not None:
            print(f"[{model}] recortes salvos em {pdir} (local, ignorado pelo git)")
        labels = np.array(labels)
        print(f"[{model}] fotos: {dict(stats)}")
        if len(np.unique(labels)) < 2:
            print(f"[{model}] menos de 2 pessoas com face válida — pulei.")
            continue
        emb = embedder.embed(faces)
        ver, scores, same = verification(emb, labels)
        per_person = Counter(labels.tolist()).values()
        res = {
            "dataset": {
                "n_pessoas": int(len(np.unique(labels))),
                "n_fotos_validas": int(len(labels)),
                "fotos_por_pessoa_min": int(min(per_person)),
                "fotos_por_pessoa_max": int(max(per_person)),
                "preprocessamento": dict(stats),
                "pre_alinhadas": pre_aligned,
            },
            "verification": ver,
            "open_set": {
                matcher_label(mc): evaluate_open_set(emb, labels, mc, k, seed) for mc in matchers
            },
            "cost": inference_cost(embedder),
        }
        results[model] = res
        curve = _plots(model, scores, same, res, out_dir)
        if curve is not None:
            curves[model] = curve
    if curves:
        plots.save_roc_comparison(
            curves, out_dir / "roc_comparison.png", "ROC — dataset do grupo", auc_digits=3
        )
    summary = {
        "protocolo": "leave-one-out (conhecidos) + leave-one-person-out (desconhecidos); "
        f"threshold por validação cruzada em {k} grupos de pessoas (EER)",
        "results": results,
        "environment": environment_info(),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "results_table.md").write_text(results_table(results) + "\n", encoding="utf-8")
    return summary
