"""Fija la muestra de validación y guarda las salidas ORIGINALES de v1 y v2.

Se ejecuta en el entorno original (antes de la recuperación) con los modelos
empaquetados en hf_build/. La prueba en entorno nuevo debe reproducirlas.

Uso: python src/recovery_sample.py
"""

from __future__ import annotations

import pandas as pd

from common import EVIDENCE, ROOT, load_params, sha256_file, write_json
from predict import load_bundle, predict_frame

N_PER_CLASS = 15  # 30 filas: 15 normales y 15 ataques


def main() -> None:
    params = load_params()
    out = EVIDENCE / "recuperacion"
    out.mkdir(parents=True, exist_ok=True)
    val = pd.read_csv(ROOT / params["data"]["prepared_dir"] / "val.csv")
    sample = (val.groupby(params["data"]["target"], group_keys=False)
                 .sample(n=N_PER_CLASS, random_state=params["split"]["seed"])
                 .sort_values("row_id"))
    sample.to_csv(out / "entrada_validacion.csv", index=False)
    hashes = {"entrada_validacion.csv": sha256_file(out / "entrada_validacion.csv")}
    for v in ("v1", "v2"):
        model, config, digest = load_bundle(None, None, str(ROOT / "hf_build" / v))
        predict_frame(model, config, sample).to_csv(out / f"salida_{v}_original.csv", index=False,
                                                     float_format="%.10f")
        hashes[f"salida_{v}_original.csv"] = sha256_file(out / f"salida_{v}_original.csv")
        hashes[f"model_{v}_sha256"] = digest
    write_json(out / "original_hashes.json", hashes)
    print(hashes)


if __name__ == "__main__":
    main()
