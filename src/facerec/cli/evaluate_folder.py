"""Avaliação do reconhecimento no dataset do grupo (pasta `pessoas/`).

    python -m facerec evaluate-folder                         # pessoas/, modelos disponíveis
    python -m facerec evaluate-folder --pre-aligned           # fotos já recortadas/alinhadas
    python -m facerec evaluate-folder --models facenet mobilefacenet --matchers cosine prototype

Protocolo completo em facerec/evaluation/folder.py. Saída agregada e anônima em
experiments/results/dataset_grupo/.
"""

from __future__ import annotations

import argparse

from facerec.core.config import load_yaml, resolve_path
from facerec.evaluation.folder import results_table, run
from facerec.models.registry import MODEL_PRESETS
from facerec.recognition.matching import MATCHERS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--root", default="pessoas")
    ap.add_argument("--models", nargs="+", default=list(MODEL_PRESETS), choices=list(MODEL_PRESETS))
    ap.add_argument(
        "--matchers", nargs="+", default=["cosine", "prototype", "knn"], choices=list(MATCHERS)
    )
    ap.add_argument(
        "--pre-aligned",
        action="store_true",
        help="as fotos já são recortes do rosto: pula a detecção",
    )
    ap.add_argument(
        "--quality",
        action="store_true",
        help="descarta faces pequenas/desfocadas (mesmos limites da aplicação)",
    )
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="experiments/results/dataset_grupo")
    ap.add_argument(
        "--save-aligned",
        action="store_true",
        help="salva os recortes que entram na rede em data/private/aligned_preview/ (conferência)",
    )
    args = ap.parse_args()

    app_cfg = load_yaml(resolve_path("configs/app.yaml"))
    matchers = [{"name": m, **({"k": 3} if m == "knn" else {})} for m in args.matchers]
    summary = run(
        resolve_path(args.root),
        args.models,
        matchers,
        args.pre_aligned,
        args.quality,
        resolve_path(args.out),
        device=app_cfg.get("device", "auto"),
        k=args.folds,
        preview_dir=resolve_path("data/private/aligned_preview") if args.save_aligned else None,
    )
    print("\n" + results_table(summary["results"]))
    for model, r in summary["results"].items():
        v = r["verification"]
        if "eer" in v:
            print(
                f"[{model}] verificação de pares: AUC {v['roc_auc']:.3f} | "
                f"EER {100 * v['eer']:.1f}% | {v['n_pairs_same']} pares mesma pessoa, "
                f"{v['n_pairs_diff']} diferentes"
            )
    print(f"\nResultados agregados em {resolve_path(args.out)}")


if __name__ == "__main__":
    main()
