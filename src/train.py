"""Entrena una configuración y la registra en MLflow.

Uso: python src/train.py --exp E1_logreg_base [--allow-dirty]

Ajusta todo el preprocesamiento solo con data/prepared/train.csv y mide en
val.csv. Registra parámetros, semilla, versión DVC, commit Git, métricas de
validación, barrido de umbrales y el pipeline completo (preprocesamiento +
estimador). También guarda una copia local en models/<exp>/ para empaquetar.
"""

from __future__ import annotations

import argparse
import json
import time

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from common import (ROOT, apply_threshold, binary_metrics, dvc_md5, feature_columns,
                    git_commit, git_is_dirty, load_experiment, load_params, sha256_file,
                    write_json)

THRESHOLD_GRID = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]


def build_pipeline(cfg: dict, cat_cols: list[str], num_cols: list[str]) -> Pipeline:
    pre = cfg["preprocessing"]
    if pre["categorical_encoding"] == "onehot":
        cat_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        cat_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", cat_enc)])

    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if pre["numeric_transform"] == "log1p":
        # np.log1p se serializa sin código propio: el modelo carga solo con sklearn.
        num_steps.append(("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
    if pre["scale"]:
        num_steps.append(("scale", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    pre_tf = ColumnTransformer([("cat", cat_pipe, cat_cols), ("num", num_pipe, num_cols)])

    m = cfg["model"]
    seed = cfg["seed"]
    if m["type"] == "logistic_regression":
        est = LogisticRegression(random_state=seed, **m["params"])
    elif m["type"] == "random_forest":
        est = RandomForestClassifier(random_state=seed, **m["params"])
    elif m["type"] == "hist_gradient_boosting":
        est = HistGradientBoostingClassifier(
            random_state=seed, categorical_features=list(range(len(cat_cols))), **m["params"])
    else:
        raise ValueError(m["type"])
    return Pipeline([("preprocess", pre_tf), ("model", est)])


def threshold_sweep(y, scores) -> pd.DataFrame:
    rows = []
    for t in THRESHOLD_GRID:
        r = binary_metrics(y, apply_threshold(scores, t))
        rows.append({"threshold": t, **r})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--allow-dirty", action="store_true",
                    help="permite entrenar con cambios sin commit (no recomendado)")
    args = ap.parse_args()

    if git_is_dirty(["src", "configs", "data", "requirements.txt"]) and not args.allow_dirty:
        raise SystemExit("Hay cambios sin commit en src/configs/data: haz commit para que el run sea trazable.")

    params = load_params()
    cfg = load_experiment(args.exp)
    prepared = ROOT / params["data"]["prepared_dir"]
    train = pd.read_csv(prepared / "train.csv")
    val = pd.read_csv(prepared / "val.csv")
    target = params["data"]["target"]
    feats = feature_columns(train, params)
    cat_cols = params["data"]["categorical_columns"]
    num_cols = [c for c in feats if c not in cat_cols]

    pipe = build_pipeline(cfg, cat_cols, num_cols)

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=cfg["id"]) as run:
        t0 = time.time()
        pipe.fit(train[feats], train[target])
        fit_seconds = time.time() - t0
        scores = pipe.predict_proba(val[feats])[:, 1]
        thr = float(cfg["threshold"])
        metrics = binary_metrics(val[target], apply_threshold(scores, thr))
        sweep = threshold_sweep(val[target], scores)

        data_ref = {
            "dvc_raw_md5": dvc_md5("data/raw.dvc"),
            "dvc_prepared_md5": dvc_md5("data/prepared.dvc"),
        }
        mlflow.set_tags({
            "exp_id": cfg["id"],
            "family": cfg["family"],
            "hypothesis": cfg["hypothesis"],
            "change": cfg["change"],
            "git_commit": git_commit(),
            "git_dirty": str(git_is_dirty()),
            **data_ref,
            "train_rows": len(train),
            "val_rows": len(val),
        })
        flat = {"seed": cfg["seed"], "threshold": thr, "model_type": cfg["model"]["type"],
                **{f"pre_{k}": v for k, v in cfg["preprocessing"].items()},
                **{f"model_{k}": v for k, v in cfg["model"]["params"].items()}}
        mlflow.log_params(flat)
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
        mlflow.log_metric("fit_seconds", fit_seconds)
        mlflow.log_dict(cfg, "config.yaml")
        mlflow.log_dict(metrics, "val_metrics.json")
        mlflow.log_dict({"labels": ["normal(0)", "ataque(1)"],
                         "matrix_rows_true_cols_pred": [[metrics["tn"], metrics["fp"]],
                                                        [metrics["fn"], metrics["tp"]]]},
                        "val_confusion_matrix.json")
        mlflow.log_text(sweep.to_csv(index=False), "val_threshold_sweep.csv")
        mlflow.log_text(pd.DataFrame({"row_id": val["row_id"], "y_true": val[target],
                                      "score_attack": scores}).to_csv(index=False),
                        "val_scores.csv")
        mlflow.sklearn.log_model(pipe, name="model", input_example=val[feats].head(5),
                                 serialization_format="cloudpickle")

        out = ROOT / "models" / cfg["id"]
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, out / "model.joblib")
        write_json(out / "run.json", {
            "run_id": run.info.run_id, "exp_id": cfg["id"], "threshold": thr,
            "model_sha256": sha256_file(out / "model.joblib"),
            "features": feats, "val_metrics": metrics, **data_ref,
            "git_commit": git_commit(),
        })
        print(json.dumps({"run_id": run.info.run_id, "fit_seconds": round(fit_seconds, 1),
                          **{k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)}}))


if __name__ == "__main__":
    main()
