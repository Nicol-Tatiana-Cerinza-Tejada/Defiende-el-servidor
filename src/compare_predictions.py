"""Compara salidas originales y recuperadas de una versión.

Uso: python src/compare_predictions.py original.csv recuperada.csv [--tol 1e-9]
Sale con código 1 si alguna etiqueta difiere o |Δscore| > tolerancia.
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("original")
    ap.add_argument("recovered")
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()
    a, b = pd.read_csv(args.original), pd.read_csv(args.recovered)
    same_ids = a["row_id"].tolist() == b["row_id"].tolist()
    max_diff = float((a["score_attack"] - b["score_attack"]).abs().max())
    label_mismatches = int((a["y_pred"] != b["y_pred"]).sum())
    ok = same_ids and label_mismatches == 0 and max_diff <= args.tol
    print(json.dumps({"original": args.original, "recovered": args.recovered, "rows": len(a),
                      "same_row_ids": same_ids, "label_mismatches": label_mismatches,
                      "max_abs_score_diff": max_diff, "tolerance": args.tol,
                      "result": "COINCIDEN" if ok else "NO COINCIDEN"}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
