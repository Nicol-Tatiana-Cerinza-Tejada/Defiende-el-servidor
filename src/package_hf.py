"""Empaqueta v1/v2 y los publica en Hugging Face Hub.

  python src/package_hf.py build   --version v1      -> hf_build/v1/
  python src/package_hf.py publish --version v1      -> sube, etiqueta y registra el SHA

Cada revisión contiene model.joblib (preprocesamiento + estimador), config.json
(umbral, variables, hashes y referencias), requirements.txt y README.md (model card).
"""

from __future__ import annotations

import argparse
import json
import shutil

import mlflow
import pandas as pd
import yaml

from common import EVIDENCE, ROOT, load_experiment, load_params, sha256_file, write_json

HF_VERSIONS = EVIDENCE / "hf_versiones.json"


def version_spec(version: str) -> dict:
    spec = yaml.safe_load(open(ROOT / "configs" / "versions.yaml", encoding="utf-8"))[version]
    if spec["threshold"] == "from_selection":
        sel = json.loads((EVIDENCE / "seleccion_validacion.json").read_text(encoding="utf-8"))
        assert sel["chosen_exp_id"] == spec["exp_id"], "versions.yaml no coincide con la selección"
        spec["threshold"] = sel["chosen_threshold"]
    return spec


def val_metrics_at(run_id: str, threshold: float) -> dict:
    client = mlflow.MlflowClient()
    sweep = pd.read_csv(client.download_artifacts(run_id, "val_threshold_sweep.csv"))
    row = sweep.loc[(sweep["threshold"] - threshold).abs() < 1e-9].iloc[0]
    return {k: (int(v) if k in {"tp", "fp", "tn", "fn", "n"} else round(float(v), 6)) for k, v in row.items()}


def build(version: str) -> dict:
    params = load_params()
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    spec = version_spec(version)
    exp_id = spec["exp_id"]
    run_info = json.loads((ROOT / "models" / exp_id / "run.json").read_text(encoding="utf-8"))
    run = mlflow.MlflowClient().get_run(run_info["run_id"])
    cfg = load_experiment(exp_id)
    thr = float(spec["threshold"])
    metrics = val_metrics_at(run_info["run_id"], thr)

    out = ROOT / "hf_build" / version
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy(ROOT / "models" / exp_id / "model.joblib", out / "model.joblib")
    model_sha = sha256_file(out / "model.joblib")
    assert model_sha == run_info["model_sha256"]

    config = {
        "version": version,
        "exp_id": exp_id,
        "family": cfg["family"],
        "model_type": cfg["model"]["type"],
        "model_params": cfg["model"]["params"],
        "preprocessing": cfg["preprocessing"],
        "threshold": thr,
        "decision_rule": "y_pred = 1 (ataque) si score_attack >= threshold; score_attack = predict_proba[:, 1]",
        "positive_class": "1 = ataque",
        "features": run_info["features"],
        "excluded_columns": params["data"]["excluded_columns"],
        "model_file": "model.joblib",
        "model_sha256": model_sha,
        "git_repo": params["github"]["repo_url"],
        "git_commit": run.data.tags["git_commit"],
        "git_tag": spec["git_tag"],
        "dvc_raw_md5": run.data.tags["dvc_raw_md5"],
        "dvc_prepared_md5": run.data.tags["dvc_prepared_md5"],
        "mlflow_run_id": run_info["run_id"],
        "mlflow_experiment": params["mlflow"]["experiment_name"],
        "val_metrics": metrics,
        "change_vs_previous": spec["change"],
    }
    write_json(out / "config.json", config)
    (out / "requirements.txt").write_text(
        "\n".join(line for line in (ROOT / "requirements.txt").read_text().splitlines()
                  if line.split("==")[0] in {"pandas", "numpy", "scikit-learn", "joblib", "huggingface_hub"})
        + "\n", encoding="utf-8")
    (out / "README.md").write_text(model_card(config, spec, params), encoding="utf-8")
    print(f"{out}: model_sha256={model_sha} threshold={thr}")
    return config


def pct(x: float) -> str:
    return f"{100 * x:.2f} %"


def model_card(c: dict, spec: dict, params: dict) -> str:
    m = c["val_metrics"]
    repo = params["huggingface"]["model_repo"]
    test_section = spec.get("test_section", "").strip() or (
        "No evaluado en test." if c["version"] == "v1" else
        "Pendiente: la evaluación única en test se añade después de congelar esta revisión.")
    return f"""---
license: other
library_name: sklearn
tags:
  - tabular-classification
  - intrusion-detection
  - unsw-nb15
  - mlops-course
pipeline_tag: tabular-classification
---

# Defiende el servidor · {c['version']} ({c['exp_id']})

Clasificador binario de conexiones de red: **normal (0) vs. ataque (1)**, entrenado con
las particiones publicadas de UNSW-NB15. Trabajo académico del curso MLOps 2026-II
(Universidad Externado de Colombia). **No es un producto de seguridad.**

## Propósito

Demostrar el versionado y la recuperación de una solución de ML: código (Git), datos (DVC),
experimentos (MLflow) y modelo (esta revisión de Hugging Face).

## Contenido de la revisión

| Archivo | Descripción |
|---|---|
| `model.joblib` | `sklearn.pipeline.Pipeline` completo: imputación, codificación, escalado/transformación y estimador |
| `config.json` | umbral, variables de entrada, hashes y referencias de trazabilidad |
| `requirements.txt` | dependencias fijadas para cargar el modelo |

- Modelo: `{c['model_type']}` (familia {c['family']}), parámetros `{json.dumps(c['model_params'])}`
- Preprocesamiento: `{json.dumps(c['preprocessing'])}`
- **Umbral: {c['threshold']}** → `ataque si score_attack >= {c['threshold']}`
- SHA-256 de `model.joblib`: `{c['model_sha256']}`
- Cambio respecto a la versión anterior: {c['change_vs_previous']}

## Datos

- Fuente: UNSW-NB15 (Moustafa & Slay, 2015), {params['data']['source_official']}.
  Copia usada: mirror fijado en commit (`{params['data']['source_mirror']}`), SHA-256 verificados.
- Entrenamiento: 80 % de `UNSW_NB15_training-set.csv` dividido por grupos de entradas idénticas
  (semilla {params['split']['seed']}); validación: el 20 % restante. Se excluyeron 940 filas con
  entradas idénticas y etiquetas contradictorias. `label`, `attack_cat` e `id` no son entradas.
- Versión DVC: datos originales `{c['dvc_raw_md5']}`, preparados `{c['dvc_prepared_md5']}`.

## Métricas de validación (34 880 filas, umbral {c['threshold']})

| Métrica | Valor |
|---|---|
| Recall de ataque | {pct(m['recall_attack'])} |
| Precision de ataque | {pct(m['precision_attack'])} |
| Tasa de falsos positivos | {pct(m['fpr'])} |
| Macro-F1 | {m['macro_f1']:.4f} |
| Accuracy | {pct(m['accuracy'])} |

Matriz de confusión (filas = real, columnas = predicho):

| | pred. normal | pred. ataque |
|---|---|---|
| **normal** | {m['tn']} | {m['fp']} |
| **ataque** | {m['fn']} | {m['tp']} |

Denominadores cero se reportan como 0.0.

## Evaluación en test

{test_section}

## Limitaciones

- Dataset histórico (2015, tráfico sintético de laboratorio): no demuestra eficacia en una red actual.
- Train y test comparten ~10 % de entradas idénticas; validación comparte entradas con test (1 739 filas).
- El umbral se eligió en validación; las métricas de validación con ese umbral son algo optimistas.
- Las puntuaciones no están calibradas como probabilidades reales.

## Uso

```python
import json, joblib, pandas as pd
from huggingface_hub import hf_hub_download

REV = "<SHA completo de esta revisión>"
path = hf_hub_download("{repo}", "model.joblib", revision=REV)
cfg = json.load(open(hf_hub_download("{repo}", "config.json", revision=REV)))
model = joblib.load(path)

df = pd.read_csv("conexiones.csv")            # columnas de UNSW-NB15
score = model.predict_proba(df[cfg["features"]])[:, 1]
y_pred = (score >= cfg["threshold"]).astype(int)   # 1 = ataque
```

O desde el repositorio de código: `python src/predict.py --revision <SHA> --input X.csv --output Y.csv`.

## Trazabilidad

| Pieza | Referencia |
|---|---|
| Código | {c['git_repo']} · commit `{c['git_commit']}` · tag `{c['git_tag']}` |
| Datos (DVC) | `data/raw.dvc` md5 `{c['dvc_raw_md5']}` · `data/prepared.dvc` md5 `{c['dvc_prepared_md5']}` |
| Experimento (MLflow) | experimento `{c['mlflow_experiment']}` · run `{c['mlflow_run_id']}` (exportado en `evidencias/mlflow/`) |
| Modelo | este repositorio, tag `{c['version']}` |

## Referencias

Moustafa, N., & Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion
detection systems. *MilCIS 2015*.
"""


def publish(version: str) -> None:
    from huggingface_hub import HfApi
    params = load_params()
    repo = params["huggingface"]["model_repo"]
    api = HfApi()
    api.create_repo(repo, repo_type="model", private=params["huggingface"]["private"], exist_ok=True)
    folder = ROOT / "hf_build" / version
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    info = api.upload_folder(repo_id=repo, folder_path=folder, repo_type="model",
                             commit_message=f"{version}: {config['exp_id']} umbral {config['threshold']}",
                             delete_patterns=["*"])
    sha = info.oid
    api.create_tag(repo, tag=version, revision=sha, tag_message=f"Solución {version}", exist_ok=False)
    data = json.loads(HF_VERSIONS.read_text(encoding="utf-8")) if HF_VERSIONS.exists() else {}
    data[version] = {"repo_id": repo, "url": f"https://huggingface.co/{repo}", "revision_sha": sha,
                     "tag": version, "model_sha256": config["model_sha256"],
                     "threshold": config["threshold"], "exp_id": config["exp_id"],
                     "mlflow_run_id": config["mlflow_run_id"], "git_commit": config["git_commit"],
                     "dvc_prepared_md5": config["dvc_prepared_md5"]}
    write_json(HF_VERSIONS, data)
    print(f"{version} publicado: {repo}@{sha}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["build", "publish"])
    ap.add_argument("--version", required=True, choices=["v1", "v2"])
    args = ap.parse_args()
    if args.action == "build":
        mlflow.set_tracking_uri(load_params()["mlflow"]["tracking_uri"])
        build(args.version)
    else:
        publish(args.version)


if __name__ == "__main__":
    main()
