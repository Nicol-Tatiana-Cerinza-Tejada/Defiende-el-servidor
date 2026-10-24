"""Análisis descriptivo de errores en test (posterior a la evaluación; no cambia el modelo).

Uso: python src/analyze_test_errors.py  -> evidencias/analisis-errores-test.json
"""

from __future__ import annotations

import pandas as pd

from common import EVIDENCE, ROOT, load_params, write_json


def rate_table(df: pd.DataFrame, by: str, top: int = 8) -> list[dict]:
    g = df.groupby(by)["y_pred"].agg(["mean", "size"]).sort_values("size", ascending=False).head(top)
    return [{by: k, "rows": int(r["size"]), "pred_attack_rate": round(float(r["mean"]), 4)} for k, r in g.iterrows()]


def main() -> None:
    params = load_params()
    prep = ROOT / params["data"]["prepared_dir"]
    pred = pd.read_csv(EVIDENCE / "predicciones-test.csv")
    test = pd.read_csv(prep / "test.csv", usecols=["row_id", "attack_cat", "service", "state"])
    train = pd.read_csv(prep / "train.csv", usecols=["state", "label"])
    d = pred.merge(test, on="row_id")
    normal, attack = d[d.y_true == 0], d[d.y_true == 1]
    write_json(EVIDENCE / "analisis-errores-test.json", {
        "note": "Análisis posterior a la evaluación única; no se usó para cambiar modelo ni umbral.",
        "fpr_by_service_normal_rows": rate_table(normal, "service"),
        "fpr_by_state_normal_rows": rate_table(normal, "state"),
        "recall_by_attack_cat": rate_table(attack, "attack_cat", top=20),
        "normal_state_share_train": {k: round(v, 4) for k, v in
                                     train[train.label == 0].state.value_counts(normalize=True).head(5).items()},
        "normal_state_share_test": {k: round(v, 4) for k, v in
                                    normal.state.value_counts(normalize=True).head(5).items()},
    })


if __name__ == "__main__":
    main()
