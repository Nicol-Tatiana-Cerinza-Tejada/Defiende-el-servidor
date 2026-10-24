"""Audita los CSV recibidos y genera los datos preparados con particiones fijas.

Salidas:
  data/prepared/train.csv, val.csv, test.csv  (con row_id para auditoría)
  data/prepared/splits.csv                    (asignación reproducible de filas)
  evidencias/auditoria.json

Uso: python src/audit_prepare.py
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from common import (EVIDENCE, ROOT, feature_columns, load_params, read_raw_csv,
                    row_hash, sha256_file, write_json)


def describe_file(path, df: pd.DataFrame, params: dict, name: str) -> dict:
    target = params["data"]["target"]
    feats = feature_columns(df, params)
    h = row_hash(df, feats)
    dup_rows = int(h.duplicated(keep="first").sum())
    labels_per_group = df.groupby(h)[target].nunique()
    contradictory = labels_per_group[labels_per_group > 1].index
    contradictory_rows = int(h.isin(contradictory).sum())
    sha = sha256_file(path)
    expected = params["data"]["expected_sha256"][name]
    return {
        "file": name,
        "sha256": sha,
        "sha256_expected_mirror": expected,
        "sha256_ok": sha == expected,
        "size_bytes": path.stat().st_size,
        "rows": int(len(df)),
        "rows_expected": params["data"]["expected_rows"][name],
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {c: int(v) for c, v in df.isna().sum().items() if v},
        "missing_total": int(df.isna().sum().sum()),
        "dash_placeholder_counts": {c: int((df[c] == "-").sum()) for c in params["data"]["categorical_columns"]},
        "class_counts": {str(k): int(v) for k, v in df[target].value_counts().sort_index().items()},
        "attack_cat_counts": {str(k): int(v) for k, v in df["attack_cat"].value_counts().items()},
        "id_unique": bool(df["id"].is_unique),
        "duplicates_by_features": {
            "rows_that_repeat_a_previous_input": dup_rows,
            "unique_inputs": int(h.nunique()),
            "groups_with_more_than_one_row": int((h.value_counts() > 1).sum()),
        },
        "contradictory_label_groups": {
            "groups": int(len(contradictory)),
            "rows": contradictory_rows,
        },
    }


def group_split(hashes: pd.Series, frac: float, seed: int) -> pd.Series:
    """Asigna grupos completos de entradas idénticas a train o val.

    Se barajan los grupos con la semilla y se van asignando a validación hasta
    alcanzar `frac` de las filas. Ningún grupo queda repartido.
    """
    sizes = hashes.value_counts().sort_index()  # orden estable antes de barajar
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sizes))
    shuffled = sizes.iloc[order]
    target_rows = frac * len(hashes)
    cum = shuffled.cumsum()
    val_groups = set(shuffled.index[cum <= target_rows])
    return hashes.map(lambda x: "val" if x in val_groups else "train")


def ids_fingerprint(row_ids: pd.Series) -> str:
    joined = "\n".join(sorted(row_ids.astype(str)))
    return hashlib.sha256(joined.encode()).hexdigest()


def main() -> None:
    params = load_params()
    d = params["data"]
    raw_dir = ROOT / d["raw_dir"]
    out_dir = ROOT / d["prepared_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    target = d["target"]

    train_path = raw_dir / d["train_file"]
    test_path = raw_dir / d["test_file"]
    train_raw = read_raw_csv(train_path)
    test_raw = read_raw_csv(test_path)

    audit_train = describe_file(train_path, train_raw, params, d["train_file"])
    audit_test = describe_file(test_path, test_raw, params, d["test_file"])
    for a in (audit_train, audit_test):
        if not a["sha256_ok"] or a["rows"] != a["rows_expected"]:
            raise SystemExit(f"Archivo inesperado: {a['file']}")

    feats = feature_columns(train_raw, params)
    assert feats == feature_columns(test_raw, params), "Esquemas distintos"

    # row_id inequívoco: archivo + posición 0-based de la fila de datos.
    train_raw.insert(0, "row_id", [f"train:{i}" for i in range(len(train_raw))])
    test_raw.insert(0, "row_id", [f"test:{i}" for i in range(len(test_raw))])
    train_raw["feature_hash"] = row_hash(train_raw, feats)
    test_raw["feature_hash"] = row_hash(test_raw, feats)

    # Política: los grupos con etiquetas contradictorias se excluyen de
    # entrenamiento y validación (sus entradas no determinan la etiqueta).
    # El resto de duplicados se conserva porque refleja la frecuencia real.
    labels_per_group = train_raw.groupby("feature_hash")[target].nunique()
    contradictory = set(labels_per_group[labels_per_group > 1].index)
    is_contra = train_raw["feature_hash"].isin(contradictory)
    clean = train_raw.loc[~is_contra].copy()

    clean["split"] = group_split(clean["feature_hash"], params["split"]["validation_fraction"],
                                 params["split"]["seed"])
    train = clean[clean["split"] == "train"].drop(columns="split")
    val = clean[clean["split"] == "val"].drop(columns="split")
    assert not set(train["feature_hash"]) & set(val["feature_hash"]), "Fuga entre train y val"

    # Cruces de test con train/val (solo entradas; no se usan etiquetas de test).
    test_h = test_raw["feature_hash"]
    overlap = {
        "test_rows_with_input_seen_in_train_split": int(test_h.isin(set(train["feature_hash"])).sum()),
        "test_rows_with_input_seen_in_val_split": int(test_h.isin(set(val["feature_hash"])).sum()),
        "test_rows_with_input_seen_in_training_file": int(test_h.isin(set(train_raw["feature_hash"])).sum()),
    }
    overlap["test_rows_with_input_seen_in_training_file_pct"] = round(
        100 * overlap["test_rows_with_input_seen_in_training_file"] / len(test_raw), 2)
    overlap["note"] = ("Los cruces se calculan solo con variables predictoras. "
                       "El test no se filtra: se evalúa completo y las conclusiones se limitan por este solapamiento.")

    cols_out = ["row_id", "id", *feats, "attack_cat", target]
    files = {"train.csv": train[cols_out], "val.csv": val[cols_out], "test.csv": test_raw[cols_out]}
    split_info = {}
    for fname, frame in files.items():
        frame.to_csv(out_dir / fname, index=False)
        split_info[fname] = {
            "rows": int(len(frame)),
            "class_counts": {str(k): int(v) for k, v in frame[target].value_counts().sort_index().items()},
            "sha256": sha256_file(out_dir / fname),
            "row_ids_sha256": ids_fingerprint(frame["row_id"]),
        }
    assignment = pd.concat([
        train[["row_id"]].assign(split="train"),
        val[["row_id"]].assign(split="val"),
        train_raw.loc[is_contra, ["row_id"]].assign(split="excluded_contradictory"),
        test_raw[["row_id"]].assign(split="test"),
    ])
    assignment.to_csv(out_dir / "splits.csv", index=False)
    split_info["splits.csv"] = {"rows": int(len(assignment)), "sha256": sha256_file(out_dir / "splits.csv")}

    audit = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": {
            "official_reference": d["source_official"],
            "download_mirror": d["source_mirror"],
            "download_date_utc": d["download_date_utc"],
            "note": "Mirror de terceros; las huellas coinciden con el anexo del taller, no autentican contra el archivo oficial.",
        },
        "files": {"train": audit_train, "test": audit_test},
        "schema_equal_train_test": list(train_raw.columns) == list(test_raw.columns),
        "variables": {
            "target": target,
            "positive_class": "1 = ataque",
            "excluded": d["excluded_columns"],
            "excluded_reason": {
                "label": "es la respuesta",
                "attack_cat": "revela la respuesta (Normal <=> label 0)",
                "id": "identificador de fila, solo auditoría",
            },
            "used_features": feats,
            "categorical": d["categorical_columns"],
            "numeric": [c for c in feats if c not in d["categorical_columns"]],
        },
        "cleaning_policy": {
            "missing_values": "No hay faltantes; igual el pipeline imputa (mediana / más frecuente) ajustado solo con train.",
            "dash_values": "'-' en service/state se trata como una categoría propia, no como faltante.",
            "exact_duplicates": "Se conservan (reflejan frecuencia real de tráfico); los grupos idénticos nunca se reparten entre train y val.",
            "contradictory_groups": "Se excluyen de train y val todas las filas de grupos de entradas idénticas con etiquetas distintas.",
            "contradictory_rows_excluded": int(is_contra.sum()),
            "contradictory_groups_excluded": len(contradictory),
            "test": "No se elimina ninguna fila de test.",
        },
        "split": {
            "method": "group split por huella de variables predictoras, grupos barajados con semilla",
            "seed": params["split"]["seed"],
            "validation_fraction_target": params["split"]["validation_fraction"],
            "validation_fraction_real": round(len(val) / (len(train) + len(val)), 4),
            "groups_shared_train_val": 0,
            "files": split_info,
        },
        "cross_partition_overlap": overlap,
    }
    write_json(EVIDENCE / "auditoria.json", audit)
    print(f"train={len(train)} val={len(val)} test={len(test_raw)} contradictorias_excluidas={int(is_contra.sum())}")
    print(overlap)


if __name__ == "__main__":
    main()
