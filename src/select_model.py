"""Elige el modelo y el umbral de v2 usando exclusivamente validación.

Aplica configs/selection.yaml sobre los runs de MLflow y escribe
evidencias/seleccion_validacion.json (decisión) antes de tocar el test.

Uso: python src/select_model.py
"""

from __future__ import annotations

import io
from datetime import datetime, timezone

import mlflow
import pandas as pd
import yaml

from common import EVIDENCE, ROOT, git_commit, load_params, write_json


def latest_runs(client, experiment_id: str) -> dict:
    runs = client.search_runs([experiment_id], filter_string="attributes.status = 'FINISHED'",
                              order_by=["attributes.start_time DESC"])
    by_exp = {}
    for r in runs:
        by_exp.setdefault(r.data.tags["exp_id"], r)
    return by_exp


def main() -> None:
    params = load_params()
    rule = yaml.safe_load(open(ROOT / "configs" / "selection.yaml", encoding="utf-8"))
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(params["mlflow"]["experiment_name"])
    runs = latest_runs(client, exp.experiment_id)

    ms = rule["model_selection"]
    ranking = sorted(((runs[c].data.metrics[ms["metric"]], c) for c in ms["candidates"]), reverse=True)
    best_exp = ranking[0][1]
    run = runs[best_exp]

    sweep_path = client.download_artifacts(run.info.run_id, "val_threshold_sweep.csv")
    sweep = pd.read_csv(sweep_path)
    ts = rule["threshold_selection"]
    ok = sweep[sweep[ts["constraint"]["metric"]] >= ts["constraint"]["min"]]
    ok = ok.sort_values([ts["objective"], "fpr"], ascending=[False, True])
    chosen = ok.iloc[0]

    decision = {
        "decided_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "decided_at_git_commit": git_commit(),
        "rule": rule,
        "ranking_val_macro_f1_at_0.5": [{"exp_id": c, "val_macro_f1": round(m, 6)} for m, c in ranking],
        "chosen_exp_id": best_exp,
        "chosen_run_id": run.info.run_id,
        "chosen_threshold": float(chosen["threshold"]),
        "val_metrics_at_chosen_threshold": {k: (float(v) if isinstance(v, float) else int(v))
                                             for k, v in chosen.items()},
        "val_metrics_at_0.5": {k.removeprefix("val_"): v for k, v in run.data.metrics.items()
                               if k.startswith("val_")},
        "test_used": False,
    }
    write_json(EVIDENCE / "seleccion_validacion.json", decision)
    print(io.StringIO(sweep.to_string()).getvalue())
    print({k: decision[k] for k in ("chosen_exp_id", "chosen_run_id", "chosen_threshold")})


if __name__ == "__main__":
    main()
