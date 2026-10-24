#!/usr/bin/env bash
# Prueba de recuperación en una carpeta y un entorno NUEVOS.
#
#   bash scripts/prueba_recuperacion.sh [REVISION_GIT] [CARPETA_DESTINO]
#
# No reutiliza cachés: clon nuevo, venv nuevo, caché DVC nueva (dentro del clon)
# y HF_HOME nuevo. Requiere python3 (3.14), git y acceso a internet.
set -euo pipefail

REPO_URL="https://github.com/Nicol-Tatiana-Cerinza-Tejada/Defiende-el-servidor.git"
REV="${1:-main}"
WORK="${2:-$HOME/recuperacion-defiende-$(date +%Y%m%d-%H%M%S)}"
TOL="1e-9"

mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd)"
LOG="$WORK/recuperacion.log"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) · $* ====="; }

step "0. Entorno"
echo "carpeta: $WORK"; uname -a; python3 --version; git --version

step "1. Código: clon nuevo en la revisión $REV"
git clone --quiet "$REPO_URL" "$WORK/repo"
cd "$WORK/repo"
git checkout --quiet "$REV"
echo "commit: $(git rev-parse HEAD)"
git tag --points-at HEAD || true

step "1b. Dependencias fijadas en un venv nuevo"
python3 -m venv "$WORK/venv"
# shellcheck disable=SC1091
source "$WORK/venv/bin/activate"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
python --version
pip freeze | grep -iE "^(scikit-learn|numpy|pandas|joblib|mlflow|dvc|huggingface.hub)=="

step "2. Datos: dvc pull desde el remote entregado (caché nueva)"
export HF_HOME="$WORK/hf_home"            # caché de Hugging Face nueva
dvc remote list
test ! -d .dvc/cache && echo "sin caché DVC previa"
dvc pull
python - <<'PY'
import json, sys
sys.path.insert(0, "src")
from common import sha256_file, load_params
p = load_params(); ok = True
for name, exp in p["data"]["expected_sha256"].items():
    got = sha256_file(f"data/raw/{name}"); ok &= got == exp
    print(f"raw  {name}: {got} {'OK' if got == exp else 'DIFERENTE'}")
audit = json.load(open("evidencias/auditoria.json"))
for name, info in audit["split"]["files"].items():
    got = sha256_file(f"data/prepared/{name}"); ok &= got == info["sha256"]
    print(f"prep {name}: {got} {'OK' if got == info['sha256'] else 'DIFERENTE'}")
sys.exit(0 if ok else 1)
PY

step "3-4. Modelos: descarga por SHA completo y predicción sobre la muestra fija"
OUT="$WORK/salidas"; mkdir -p "$OUT"
IN=evidencias/recuperacion/entrada_validacion.csv
python - "$IN" <<'PY'
import sys, pandas as pd
sys.path.insert(0, "src")
from common import sha256_file
s = pd.read_csv(sys.argv[1]); v = pd.read_csv("data/prepared/val.csv")
assert set(s["row_id"]) <= set(v["row_id"]), "la muestra no pertenece a validación"
print(f"muestra: {len(s)} filas de validación, sha256 {sha256_file(sys.argv[1])}")
PY
for V in v1 v2; do
  SHA=$(python -c "import json;print(json.load(open('evidencias/hf_versiones.json'))['$V']['revision_sha'])")
  echo "--- $V @ $SHA"
  python src/predict.py --revision "$SHA" --input "$IN" --output "$OUT/salida_${V}_recuperada.csv"
  sha256sum "$OUT/salida_${V}_recuperada.csv"
done

step "5. Comparación con las salidas guardadas (tolerancia $TOL)"
STATUS=0
for V in v1 v2; do
  python src/compare_predictions.py "evidencias/recuperacion/salida_${V}_original.csv" \
    "$OUT/salida_${V}_recuperada.csv" --tol "$TOL" || STATUS=1
done

step "Resultado"
if [ "$STATUS" -eq 0 ]; then echo "RECUPERACION OK"; else echo "RECUPERACION FALLIDA"; fi
echo "log: $LOG"
exit "$STATUS"
