"""Gera a tabela de resultados do README e o gráfico ROC comparativo a partir de
experiments/results/*/metrics.json (somente métricas agregadas).

    python -m facerec report
"""

from __future__ import annotations

import json

from facerec.core.config import resolve_path
from facerec.evaluation import plots

RESULTS = resolve_path("experiments/results")


def _fmt(v, pct: bool = True) -> str:
    if v is None:
        return "—"
    return f"{100 * v:.2f}%" if pct else f"{v:.4f}"


def collect() -> dict[str, dict]:
    out = {}
    for f in sorted(RESULTS.glob("E0*/metrics.json")):
        out[f.parent.name] = json.loads(f.read_text(encoding="utf-8"))
        last = f.parent / "ultima_epoca" / "metrics.json"  # modelo da última época (fine-tuning)
        if last.exists():
            out[f"{f.parent.name} (última época)"] = json.loads(last.read_text(encoding="utf-8"))
    return out


def display_model(r: dict) -> str:
    """Nome legível: o MobileFaceNet roda pelo mesmo carregador ONNX do ArcFace."""
    raw = r.get("model") if isinstance(r.get("model"), str) else r.get("model", {}).get("name")
    exp_name = r.get("experiment", {}).get("name", "")
    if "mobilefacenet" in exp_name:
        return "MobileFaceNet (ArcFace loss)"
    return {"facenet": "FaceNet (VGGFace2)", "arcface": "ArcFace iResNet-50"}.get(raw, raw)


def table(results: dict[str, dict]) -> str:
    lines = [
        "| Experimento | Modelo | Estratégia | Dataset | Accuracy | F1 | EER | Observação |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for exp_id, r in results.items():
        e = r.get("experiment", {})
        model = display_model(r)
        strategy = e.get("strategy", "")
        if "lfw" in r:
            lfw = r["lfw"]
            acc = f"{_fmt(lfw['accuracy_mean'])} ± {_fmt(lfw['accuracy_std'])}"
            lines.append(
                f"| {exp_id} | {model} | {strategy} | LFW (10-fold) | {acc} | "
                f"{_fmt(lfw['f1_mean'])} | {_fmt(lfw['eer'])} | AUC {_fmt(lfw['roc_auc'], False)} |"
            )
        if "private_val" in r:
            pv = r["private_val"]
            lines.append(
                f"| {exp_id} | {model} | {strategy} | Privado (val.) | {_fmt(pv['accuracy'])} | "
                f"{_fmt(pv['f1'])} | {_fmt(pv['eer'])} | AUC {_fmt(pv['roc_auc'], False)} |"
            )
        if "matchers" in r:
            for name, m in r["matchers"].items():
                lines.append(
                    f"| {exp_id} | {model} + {name} | open-set | Privado (val.) | "
                    f"{_fmt(m['accuracy'])} | {_fmt(m['macro_f1'])} | {_fmt(m.get('eer'))} | "
                    f"FAR desc. {_fmt(m['false_accept_unknown'])} |"
                )
        if "test" in r:
            t, tb = r["test"], r["test_binary_accept_reject"]
            lines.append(
                f"| {exp_id} | {model} + {r['matcher']['name']} | "
                f"thr={r['chosen_threshold']:.3f} | "
                f"Privado (TESTE) | {_fmt(t['accuracy'])} | {_fmt(t['macro_f1'])} | "
                f"{_fmt(tb.get('eer'))} | FAR {_fmt(tb['far'])} / FRR {_fmt(tb['frr'])} |"
            )
    return "\n".join(lines)


def roc_comparison(results: dict[str, dict], key: str = "lfw") -> None:
    curves = {
        exp_id: (r[key]["roc_curve"]["fpr"], r[key]["roc_curve"]["tpr"], r[key]["roc_auc"])
        for exp_id, r in results.items()
        if key in r and "roc_curve" in r[key]
    }
    if curves:  # nenhum experimento avaliado nesse conjunto: não gera gráfico vazio
        plots.save_roc_comparison(
            curves, RESULTS / f"roc_comparison_{key}.png", f"ROC comparativa — {key}", log_far=True
        )


def build_report() -> str | None:
    """Grava results_table.md e as ROC comparativas; devolve a tabela (None se não há
    resultados)."""
    results = collect()
    if not results:
        return None
    md = table(results)
    (RESULTS / "results_table.md").write_text(md + "\n", encoding="utf-8")
    for key in ("lfw", "private_val"):
        roc_comparison(results, key)
    return md
