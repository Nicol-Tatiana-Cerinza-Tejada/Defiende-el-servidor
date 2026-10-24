"""Congela la decisión de v2 en evidencias/seleccion.json ANTES de evaluar test.

Combina la decisión de validación con las referencias Git/DVC/Hugging Face.
Después de ejecutarlo hay que hacer commit: ese commit es la evidencia de
congelación previa al test.

Uso: python src/freeze_selection.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from common import EVIDENCE, dvc_md5, write_json


def main() -> None:
    sel = json.loads((EVIDENCE / "seleccion_validacion.json").read_text(encoding="utf-8"))
    hf = json.loads((EVIDENCE / "hf_versiones.json").read_text(encoding="utf-8"))["v2"]
    assert hf["exp_id"] == sel["chosen_exp_id"] and hf["threshold"] == sel["chosen_threshold"]
    v = sel["val_metrics_at_chosen_threshold"]
    frozen = {
        "version": "v2",
        "exp_id": sel["chosen_exp_id"],
        "mlflow_run_id": sel["chosen_run_id"],
        "threshold": sel["chosen_threshold"],
        "decision_rule": "ataque si score_attack >= threshold",
        "git_commit_training": hf["git_commit"],
        "git_tag_code": "v2-code",
        "dvc_raw_md5": dvc_md5("data/raw.dvc"),
        "dvc_prepared_md5": dvc_md5("data/prepared.dvc"),
        "huggingface_repo": hf["repo_id"],
        "huggingface_revision_sha": hf["revision_sha"],
        "model_sha256": hf["model_sha256"],
        "reason": (
            "E4 obtuvo el mayor macro-F1 de validación con umbral 0.5 entre las cuatro configuraciones "
            f"({sel['ranking_val_macro_f1_at_0.5']}). Su umbral se fijó en {sel['chosen_threshold']}: "
            "mayor macro-F1 de validación entre los umbrales con recall de ataque >= 0.95. "
            "Frente a v1 (E1, umbral 0.5) baja el FPR de validación de 0.170 a "
            f"{v['fpr']:.3f} y sube la precision de 0.921 a {v['precision_attack']:.3f}, "
            f"a cambio de bajar el recall de 0.987 a {v['recall_attack']:.3f}."
        ),
        "val_metrics": v,
        "selection_decided_utc": sel["decided_utc"],
        "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_used_for_selection": False,
        "freeze_evidence": "commit Git que añade este archivo (anterior al commit de metricas-test.json)",
    }
    write_json(EVIDENCE / "seleccion.json", frozen)
    print(json.dumps(frozen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
