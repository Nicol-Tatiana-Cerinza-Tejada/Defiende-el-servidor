#!/usr/bin/env bash
# Publica el almacenamiento DVC en un repositorio de datasets de Hugging Face.
#
# DVC sube a un remote local temporal (staging, solo en .dvc/config.local) y
# luego esa carpeta (files/md5/...) se sube al dataset de HF. El remote por
# defecto del proyecto ("hfstore") es HTTP de solo lectura sobre ese dataset,
# así cualquiera puede hacer `dvc pull` sin credenciales.
#
#   bash scripts/dvc_push_hf.sh            (requiere `hf auth login` con token de escritura)
set -euo pipefail
cd "$(dirname "$0")/.."

REPO=$(python -c "import yaml;print(yaml.safe_load(open('configs/params.yaml'))['huggingface']['dvc_dataset_repo'])")
STAGE="${DVC_STAGE_DIR:-$(mktemp -d)}"

dvc remote add --local --force staging "$STAGE"
dvc push -r staging
hf repos create "$REPO" --type dataset --public --exist-ok
hf upload "$REPO" "$STAGE" . --repo-type dataset \
  --commit-message "Almacenamiento DVC: data/raw.dvc + data/prepared.dvc"
dvc remote remove --local staging
echo "Remote publicado: https://huggingface.co/datasets/$REPO"
