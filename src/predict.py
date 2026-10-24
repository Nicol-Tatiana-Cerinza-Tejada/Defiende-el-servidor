"""Descarga una revisión del modelo desde Hugging Face Hub y predice.

Uso:
  python src/predict.py --revision <SHA completo> --input entrada.csv --output salida.csv
  python src/predict.py --local-dir hf_build/v1 --input entrada.csv --output salida.csv

El CSV de entrada debe traer las columnas predictoras (puede traer además
row_id, id, label y attack_cat, que se ignoran como entradas). La salida tiene
row_id, score_attack y y_pred, con la regla y_pred = 1 si score_attack >= umbral.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_threshold, load_params, read_raw_csv, sha256_file  # noqa: E402


def load_bundle(repo_id: str | None, revision: str | None, local_dir: str | None,
                cache_dir: str | None = None):
    if local_dir:
        base = Path(local_dir)
        model_path, config_path = base / "model.joblib", base / "config.json"
    else:
        from huggingface_hub import hf_hub_download
        kw = {"repo_id": repo_id, "revision": revision, "cache_dir": cache_dir}
        model_path = Path(hf_hub_download(filename="model.joblib", **kw))
        config_path = Path(hf_hub_download(filename="config.json", **kw))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    digest = sha256_file(model_path)
    if digest != config["model_sha256"]:
        raise SystemExit(f"Hash del modelo no coincide: {digest} != {config['model_sha256']}")
    return joblib.load(model_path), config, digest


def predict_frame(model, config: dict, df: pd.DataFrame) -> pd.DataFrame:
    scores = model.predict_proba(df[config["features"]])[:, 1]
    out = pd.DataFrame({
        "row_id": df["row_id"] if "row_id" in df else range(len(df)),
        "score_attack": scores,
        "y_pred": apply_threshold(scores, config["threshold"]),
    })
    return out


def main() -> None:
    params = load_params()
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default=params["huggingface"]["model_repo"])
    ap.add_argument("--revision", help="SHA completo de la revisión en Hugging Face")
    ap.add_argument("--local-dir", help="carpeta local con model.joblib y config.json")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if not (args.revision or args.local_dir):
        ap.error("indica --revision o --local-dir")

    model, config, digest = load_bundle(args.repo_id, args.revision, args.local_dir, args.cache_dir)
    df = read_raw_csv(args.input)
    out = predict_frame(model, config, df)
    out.to_csv(args.output, index=False, float_format="%.10f")
    print(json.dumps({"version": config["version"], "revision": args.revision, "model_sha256": digest,
                      "threshold": config["threshold"], "rows": len(out),
                      "attacks_predicted": int(out["y_pred"].sum())}))


if __name__ == "__main__":
    main()
