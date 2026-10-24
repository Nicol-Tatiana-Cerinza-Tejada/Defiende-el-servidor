"""Exporta los runs de MLflow a archivos consultables sin el servidor local.

Salidas:
  evidencias/mlflow/runs.json                  (todos los params, métricas y tags)
  evidencias/mlflow/<run_id>/...               (config, métricas, matriz, barrido, MLmodel)
  evidencias/experimentos.csv                  (tabla de comparación)

Uso: python src/export_mlflow.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import mlflow
import pandas as pd
import yaml

from common import EVIDENCE, ROOT, load_experiment, load_params, write_json
from select_model import latest_runs

ARTIFACTS = ["config.yaml", "val_metrics.json", "val_confusion_matrix.json", "val_threshold_sweep.csv"]


def main() -> None:
    params = load_params()
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(params["mlflow"]["experiment_name"])
    runs = latest_runs(client, exp.experiment_id)

    out = EVIDENCE / "mlflow"
    out.mkdir(parents=True, exist_ok=True)
    conclusions = load_conclusions()
    records, rows = [], []
    for exp_id in sorted(runs):
        r = runs[exp_id]
        rid = r.info.run_id
        records.append({
            "run_id": rid, "run_name": r.info.run_name, "experiment_id": r.info.experiment_id,
            "status": r.info.status, "start_time_ms": r.info.start_time, "end_time_ms": r.info.end_time,
            "params": r.data.params, "metrics": r.data.metrics, "tags": {
                k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.log-model")},
        })
        run_dir = out / rid
        run_dir.mkdir(exist_ok=True)
        for a in ARTIFACTS:
            shutil.copy(client.download_artifacts(rid, a), run_dir / a)
        model_dir = Path(client.download_artifacts(rid, "model"))
        for f in ("MLmodel", "requirements.txt", "conda.yaml"):
            if (model_dir / f).exists():
                shutil.copy(model_dir / f, run_dir / f"model_{f}")
        cfg = load_experiment(exp_id)
        m = r.data.metrics
        rows.append({
            "exp_id": exp_id, "hypothesis": cfg["hypothesis"], "change": cfg["change"],
            "family": cfg["family"], "params_path": f"configs/experiments/{exp_id}.yaml",
            "model_type": r.data.params["model_type"], "seed": r.data.params["seed"],
            "git_commit": r.data.tags["git_commit"],
            "dvc_raw_md5": r.data.tags["dvc_raw_md5"], "dvc_prepared_md5": r.data.tags["dvc_prepared_md5"],
            "run_id": rid, "threshold": float(r.data.params["threshold"]),
            "val_recall_attack": m["val_recall_attack"], "val_precision_attack": m["val_precision_attack"],
            "val_fpr": m["val_fpr"], "val_macro_f1": m["val_macro_f1"], "val_accuracy": m["val_accuracy"],
            "val_tp": int(m["val_tp"]), "val_fp": int(m["val_fp"]),
            "val_tn": int(m["val_tn"]), "val_fn": int(m["val_fn"]),
            "fit_seconds": round(m["fit_seconds"], 1),
            "conclusion": conclusions.get(exp_id, ""),
        })
    write_json(out / "runs.json", {"tracking_uri": params["mlflow"]["tracking_uri"],
                                   "experiment": exp.name, "runs": records})
    pd.DataFrame(rows).to_csv(EVIDENCE / "experimentos.csv", index=False)
    print(pd.DataFrame(rows)[["exp_id", "run_id", "val_recall_attack", "val_precision_attack",
                              "val_fpr", "val_macro_f1", "fit_seconds"]].to_string())


def load_conclusions() -> dict:
    """Conclusiones de validación escritas a mano tras revisar cada run."""
    p = ROOT / "configs" / "conclusiones.yaml"
    return yaml.safe_load(open(p, encoding="utf-8")) if p.exists() else {}


if __name__ == "__main__":
    main()
