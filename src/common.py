"""Utilidades compartidas: rutas, hashes, carga de datos y métricas."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidencias"


def load_params(path: str | Path = "configs/params.yaml") -> dict:
    with open(ROOT / path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_experiment(exp_id: str) -> dict:
    with open(ROOT / "configs" / "experiments" / f"{exp_id}.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=_json_default)
        fh.write("\n")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"No serializable: {type(o)}")


def read_raw_csv(path: str | Path) -> pd.DataFrame:
    # Los CSV publicados traen BOM UTF-8 en la cabecera; utf-8-sig lo elimina.
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def feature_columns(df: pd.DataFrame, params: dict) -> list[str]:
    excluded = set(params["data"]["excluded_columns"])
    return [c for c in df.columns if c not in excluded and c != "row_id" and c != "split"]


def row_hash(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Huella por fila de las variables predictoras (para duplicados y cruces)."""
    return pd.util.hash_pandas_object(df[cols], index=False).astype("uint64")


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def git_is_dirty(paths: list[str] | None = None) -> bool:
    cmd = ["git", "status", "--porcelain", "--untracked-files=no"]
    if paths:
        cmd += ["--", *paths]
    return bool(subprocess.check_output(cmd, cwd=ROOT, text=True).strip())


def dvc_md5(dvc_file: str | Path) -> str:
    """md5 registrado por DVC en un archivo .dvc (identifica la versión de datos)."""
    with open(ROOT / dvc_file, encoding="utf-8") as fh:
        meta = yaml.safe_load(fh)
    return meta["outs"][0]["md5"]


def safe_div(num: float, den: float) -> float:
    """Denominador cero -> 0.0 (convención documentada en el informe)."""
    return float(num) / float(den) if den else 0.0


def binary_metrics(y_true, y_pred) -> dict:
    """Métricas con ataque (1) como clase positiva."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    recall = safe_div(tp, tp + fn)
    precision = safe_div(tp, tp + fp)
    f1_attack = safe_div(2 * precision * recall, precision + recall)
    recall_normal = safe_div(tn, tn + fp)
    precision_normal = safe_div(tn, tn + fn)
    f1_normal = safe_div(2 * precision_normal * recall_normal, precision_normal + recall_normal)
    return {
        "recall_attack": recall,
        "precision_attack": precision,
        "fpr": safe_div(fp, fp + tn),
        "f1_attack": f1_attack,
        "f1_normal": f1_normal,
        "macro_f1": (f1_attack + f1_normal) / 2,
        "accuracy": safe_div(tp + tn, tp + tn + fp + fn),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n": int(len(y_true)),
    }


def apply_threshold(scores, threshold: float) -> np.ndarray:
    """Regla de decisión: ataque si score_attack >= umbral."""
    return (np.asarray(scores) >= threshold).astype(int)
