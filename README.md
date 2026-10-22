# Defiende el servidor

Taller universitario de MLOps para clasificar tráfico de red normal o ataque
con el dataset UNSW-NB15, usando Python, scikit-learn y CPU.

## Objetivo

Construir un flujo reproducible donde:

- Git/GitHub versiona el código y la documentación.
- DVC versiona los datos originales y preparados.
- MLflow registra experimentos y artefactos.
- Hugging Face Hub almacena los modelos recuperables.

El objetivo es `label`: `0` representa tráfico normal y `1` representa ataque.
Las columnas `label`, `attack_cat` e `id` nunca serán entradas del modelo.

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

El desarrollo se realiza en CPU. Las instrucciones completas se irán
documentando en `entrega.md`.

