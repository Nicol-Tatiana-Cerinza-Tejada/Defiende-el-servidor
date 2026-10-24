# Defiende el servidor

Taller de MLOps (Universidad Externado de Colombia, 2026-II): clasificación binaria
**normal (0) / ataque (1)** con UNSW-NB15, con código, datos, experimentos y modelos
versionados y recuperables.

| Pieza | Herramienta | Dónde |
|---|---|---|
| Código, configuración, documentación, referencias DVC | Git/GitHub | este repositorio; tags `v1-code`, `v2-code` |
| Datos originales y preparados | DVC | `data/raw.dvc`, `data/prepared.dvc`; remote HTTP en un dataset de Hugging Face |
| Experimentos (4 runs) | MLflow | `mlflow.db` local; exportados en `evidencias/mlflow/` y `evidencias/experimentos.csv` |
| Modelos v1/v2 | Hugging Face Hub | ver `evidencias/hf_versiones.json` (SHA completo de cada revisión) |

`label` es el objetivo. `label`, `attack_cat` e `id` nunca son entradas del modelo.
El índice completo de la entrega (identificadores, hashes y ubicación de cada evidencia) está en
[entrega.md](entrega.md).

## Requisitos

- Python **3.14** (probado con 3.14.4), Linux/WSL2, CPU. No se necesita GPU.
- Dependencias fijadas en [requirements.txt](requirements.txt).
- Git e internet (GitHub y Hugging Face). No hacen falta cuentas para leer: los repositorios son públicos.

## Instalación

```bash
git clone https://github.com/Nicol-Tatiana-Cerinza-Tejada/Defiende-el-servidor.git
cd Defiende-el-servidor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Todos los comandos siguientes se ejecutan desde la raíz del repositorio con el venv activo.

## Recuperar los datos (DVC)

```bash
dvc pull                 # remote por defecto "hfstore" (HTTP, sin credenciales)
sha256sum data/raw/*.csv # deben coincidir con configs/params.yaml y el anexo del taller
```

- `data/raw/`: `UNSW_NB15_training-set.csv` y `UNSW_NB15_testing-set.csv` tal como se descargaron.
- `data/prepared/`: `train.csv`, `val.csv`, `test.csv` y `splits.csv` (asignación fila → partición).

Para regenerar los datos preparados desde los originales (resultado idéntico, mismos SHA-256):

```bash
python src/audit_prepare.py   # escribe data/prepared/ y evidencias/auditoria.json
```

## Ejecutar un experimento

```bash
python src/train.py --exp E4_hist_gb          # o E1_logreg_base, E2_logreg_log1p, E3_random_forest
mlflow ui --backend-store-uri sqlite:///mlflow.db   # opcional, para consultar los runs
```

`train.py` exige que `src/`, `configs/` y `data/*.dvc` no tengan cambios sin commit, para que el
run quede ligado a un commit exacto. Cada run registra hipótesis, parámetros, semilla, commit Git,
md5 de DVC, métricas de validación, barrido de umbrales, matriz de confusión y el pipeline completo.

Otros pasos del flujo:

```bash
python src/select_model.py            # elige modelo/umbral de v2 solo con validación
python src/export_mlflow.py           # evidencias/mlflow/ y evidencias/experimentos.csv
python src/package_hf.py build --version v2   # arma hf_build/v2 (modelo, config, deps, model card)
```

## Descargar un modelo y predecir

Cada versión se descarga por el **SHA completo** de su revisión en Hugging Face
(ver `evidencias/hf_versiones.json` o [entrega.md](entrega.md)):

```bash
V1=$(python -c "import json;print(json.load(open('evidencias/hf_versiones.json'))['v1']['revision_sha'])")
V2=$(python -c "import json;print(json.load(open('evidencias/hf_versiones.json'))['v2']['revision_sha'])")

python src/predict.py --revision "$V1" --input evidencias/recuperacion/entrada_validacion.csv --output pred_v1.csv
python src/predict.py --revision "$V2" --input evidencias/recuperacion/entrada_validacion.csv --output pred_v2.csv
```

`predict.py` comprueba que el SHA-256 de `model.joblib` coincide con `config.json` y aplica la regla
`y_pred = 1 si score_attack >= umbral`. La entrada es un CSV con las columnas de UNSW-NB15.

## Prueba de recuperación en entorno nuevo

```bash
bash scripts/prueba_recuperacion.sh entrega /tmp/recuperacion-defiende
```

Clona el repositorio, crea un venv nuevo, hace `dvc pull` con caché nueva, verifica huellas,
descarga v1 y v2 por SHA con un `HF_HOME` nuevo, predice sobre 30 filas fijas de validación y las
compara con las salidas guardadas. El registro de la ejecución entregada está en
`evidencias/recuperacion/`.

## Estructura

```
configs/            parámetros, 4 experimentos, regla de selección, versiones v1/v2
src/                auditoría/preparación, entrenamiento, selección, empaquetado, predicción, test
scripts/            publicación del remote DVC y prueba de recuperación
data/*.dvc          referencias DVC (los CSV no están en Git)
evidencias/         auditoría, experimentos, selección, test, MLflow exportado, recuperación
informe/            informe PDF y su fuente
```
