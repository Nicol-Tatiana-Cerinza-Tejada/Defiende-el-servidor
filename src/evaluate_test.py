"""Evalúa UNA vez la solución congelada (v2) sobre todo el test publicado.

Descarga el modelo de Hugging Face por el SHA guardado en evidencias/seleccion.json,
comprueba su hash y escribe:
  evidencias/predicciones-test.csv  (row_id, y_true, score_attack, y_pred)
  evidencias/metricas-test.json

Uso: python src/evaluate_test.py
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

import pandas as pd

from common import EVIDENCE, ROOT, binary_metrics, load_params, sha256_file, write_json
from predict import load_bundle, predict_frame


def main() -> None:
    params = load_params()
    sel = json.loads((EVIDENCE / "seleccion.json").read_text(encoding="utf-8"))
    # La decisión debe estar en un commit anterior a esta evaluación.
    freeze_commit = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", "evidencias/seleccion.json"], cwd=ROOT, text=True).strip()
    if not freeze_commit or subprocess.run(["git", "diff", "--quiet", "--", "evidencias/seleccion.json"],
                                           cwd=ROOT).returncode:
        raise SystemExit("seleccion.json debe estar en un commit antes de evaluar test.")
    if (EVIDENCE / "metricas-test.json").exists():
        raise SystemExit("El test ya fue evaluado; no se repite.")

    model, config, digest = load_bundle(sel["huggingface_repo"], sel["huggingface_revision_sha"], None)
    assert digest == sel["model_sha256"] and config["threshold"] == sel["threshold"]

    test_path = ROOT / params["data"]["prepared_dir"] / "test.csv"
    test = pd.read_csv(test_path)
    pred = predict_frame(model, config, test)
    pred.insert(1, "y_true", test[params["data"]["target"]].values)
    pred.to_csv(EVIDENCE / "predicciones-test.csv", index=False, float_format="%.10f")

    # Las métricas se recalculan desde el CSV guardado para que sean reproducibles.
    saved = pd.read_csv(EVIDENCE / "predicciones-test.csv")
    m = binary_metrics(saved["y_true"], saved["y_pred"])
    write_json(EVIDENCE / "metricas-test.json", {
        "evaluated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": "v2",
        "exp_id": sel["exp_id"],
        "huggingface_repo": sel["huggingface_repo"],
        "huggingface_revision_sha": sel["huggingface_revision_sha"],
        "model_sha256": digest,
        "selection_frozen_in_commit": freeze_commit,
        "dataset": {
            "file": "data/prepared/test.csv (todas las filas de UNSW_NB15_testing-set.csv, mismo orden)",
            "sha256_prepared_test_csv": sha256_file(test_path),
            "sha256_original_testing_set": sha256_file(ROOT / params["data"]["raw_dir"] / params["data"]["test_file"]),
            "dvc_prepared_md5": sel["dvc_prepared_md5"],
        },
        "rows_evaluated": int(len(saved)),
        "threshold": sel["threshold"],
        "decision_rule": "y_pred = 1 si score_attack >= threshold",
        "positive_class": "1 = ataque",
        "zero_division": "0.0",
        "confusion_matrix": {"rows_true_cols_pred": [[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
                             "labels": ["normal(0)", "ataque(1)"]},
        "metrics": m,
        "predictions_file": "evidencias/predicciones-test.csv",
        "predictions_sha256": sha256_file(EVIDENCE / "predicciones-test.csv"),
    })
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
