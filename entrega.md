# Entrega · MLOps — Defiende el servidor — Grupo A — Nicol Cerinza, Natalia Rios Moreno

## Integrantes y repositorio

- **Integrantes:** Nicol Tatiana Cerinza Tejada (grupo A) · Natalia Rios Moreno (grupo A)
- **GitHub:** https://github.com/Nicol-Tatiana-Cerinza-Tejada/Defiende-el-servidor (público)
- **Commit final de la entrega:** el que apunta el tag `entrega` (`git rev-parse entrega`). Su SHA completo
  va en el mensaje de entrega.
- **Modelos (Hugging Face):** https://huggingface.co/NicolTcT/defiende-el-servidor (público)
- **Almacenamiento DVC:** https://huggingface.co/datasets/NicolTcT/defiende-el-servidor-dvc (público)

## Versiones

| | v1 | v2 |
|---|---|---|
| Experimento | `E1_logreg_base` (regresión logística) | `E4_hist_gb` (HistGradientBoosting) |
| Commit de código que la produjo | `7315fc6150b5290a62ee510057842a641ed2ff5f` (tag `v1-code`) | entrenamiento `0412c387fc1433f164876e00c743bdafaa1f221a`; selección y empaquetado `e29151c191d8d8fdd06d00224b2dcdc2fc03a3cd` (tag `v2-code`) |
| Datos DVC | `data/raw.dvc` md5 `fe93051ae209ffb0128a31e7a478c009.dir` · `data/prepared.dvc` md5 `b3533724917eef902710882f831d14e5.dir` | iguales a v1 |
| Run MLflow | `c88acf26956545c8b86da907c31e8a3e` | `83c62ca284954280b3cca861e87dbc9f` |
| Revisión HF (SHA completo) | `2e405585467e01b576cbfd324488fa07174ef39e` (tag `v1`) | `4ea2e9e7b2245d27a672e58b81d27574843615f7` (tag `v2`, revisión evaluada en test) |
| SHA-256 de `model.joblib` | `e288bc0451744d7ba243b3d847c56a0bfcede62a83a86fc6f3856947a162cb13` | `b3c55ced64ce60d2c75f2d0c5fddc3e0e306bea65cd8592c8dbee8806c4c9a8a` |
| Umbral | 0.50 | 0.55 |
| Validación: recall / precision / FPR / macro-F1 | 0.987 / 0.921 / 0.170 / 0.923 | 0.971 / 0.967 / 0.067 / 0.953 |

La revisión HF `a9099d0937ab71bb8052002293fb50415286026d` solo añade el resultado de test a la model card
de v2. Su `model.joblib` es idéntico (mismo SHA-256).

## Ubicación de las evidencias

| Evidencia | Archivo |
|---|---|
| Auditoría de datos, limpieza, splits y cruces | `evidencias/auditoria.json` |
| Tabla de los 4 experimentos | `evidencias/experimentos.csv` |
| Registros MLflow exportados (params, métricas, tags, artefactos) | `evidencias/mlflow/runs.json`, `evidencias/mlflow/<run_id>/` |
| Decisión con validación (ranking, barrido de umbrales) | `evidencias/seleccion_validacion.json` |
| Selección congelada antes de test | `evidencias/seleccion.json` (commit `e378e2005e499e5f48ec153c421c7982e49573cd`) |
| Revisiones de Hugging Face | `evidencias/hf_versiones.json` |
| Métricas de test | `evidencias/metricas-test.json` |
| Predicciones de test por fila | `evidencias/predicciones-test.csv` (`row_id` = `test:<fila 0-based del CSV original>`) |
| Análisis de errores en test (descriptivo) | `evidencias/analisis-errores-test.json` |
| Prueba de recuperación | `evidencias/recuperacion/` (`resultado.json`, `recuperacion.log`, entrada, salidas originales y recuperadas) |
| Informe | `informe/informe.pdf` |
| Revisión independiente de Natalia (resultados, límites y guía de decisiones) | `evidencias/revision-natalia/` |

### Acceso al almacenamiento DVC

El remote por defecto `hfstore` es HTTP de solo lectura sobre el dataset público de HF, así que no
necesita credenciales: `dvc pull`. Para publicar nuevos datos: `bash scripts/dvc_push_hf.sh`, que
requiere `hf auth login` con permiso de escritura.

### Consultar MLflow

- Con el servidor: `mlflow ui --backend-store-uri sqlite:///mlflow.db`. La base local no va en el ZIP.
- Sin servidor: `evidencias/mlflow/runs.json` y los artefactos por `run_id`.

## Comandos

```bash
# Instalación
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
# Recuperación de datos
dvc pull
# Entrenamiento (un experimento)
python src/train.py --exp E4_hist_gb
# Predicción con cada versión
python src/predict.py --revision 2e405585467e01b576cbfd324488fa07174ef39e --input evidencias/recuperacion/entrada_validacion.csv --output pred_v1.csv
python src/predict.py --revision 4ea2e9e7b2245d27a672e58b81d27574843615f7 --input evidencias/recuperacion/entrada_validacion.csv --output pred_v2.csv
# Prueba completa de recuperación en carpeta nueva
bash scripts/prueba_recuperacion.sh entrega /tmp/recuperacion-defiende
```

### Tiempo y recursos medidos

Equipo: WSL2 en Windows, 12 CPU lógicas, 30 GB RAM, sin GPU.

| Paso | Tiempo |
|---|---|
| Auditoría y preparación (`audit_prepare.py`) | 4.5 s |
| E1 / E2 / E3 / E4 (ajuste) | 7.0 / 8.3 / 5.6 / 3.7 s |
| Evaluación de test (82 332 filas) | < 10 s |
| Prueba de recuperación completa (incluye `pip install`) | 3 min 24 s |

Tamaño de los modelos: v1 9.6 KB, v2 2.2 MB. E3 pesa 69 MB y no se publicó.

## Limitaciones conocidas

- UNSW-NB15 es un dataset sintético de laboratorio de 2015. No demuestra eficacia sobre una red actual.
- En test, v2 tiene un FPR de 24.4 % frente al 6.7 % de validación. El tráfico normal de test difiere del de
  entrenamiento, y la validación salió del mismo archivo que el entrenamiento, así que fue optimista.
- El 10.4 % de las filas de test repite entradas del archivo de entrenamiento.
- El umbral se eligió en validación, así que las métricas de validación de v2 son algo optimistas.
- Se excluyeron de entrenamiento y validación 940 filas con etiquetas contradictorias. El test se evaluó completo.

## Uso de IA, código externo y contribuciones

- **IA:** se usó Claude Code (modelo Claude Opus 5.5, Anthropic) para:
  - redactar el código de `src/` y `scripts/`, la documentación y el borrador del informe;
  - ejecutar los pasos de auditoría, entrenamiento, publicación y recuperación.

  Lo que verificamos:
  - las huellas SHA-256;
  - que ningún grupo de entradas idénticas queda repartido entre train y val;
  - que las métricas de test se recalculan desde `predicciones-test.csv`;
  - la recuperación en entorno nuevo (predicciones idénticas y reentrenamiento con el mismo hash).
- **Revisión adicional con IA:** se utilizó Codex (OpenAI) para la revisión independiente en la rama `natalia`, ejecutar verificaciones y corregir documentación; no se modificaron la selección ni la evaluación final.
- **Código externo:** no se copió código de terceros. Se usan las librerías de `requirements.txt` según
  su documentación oficial. Datos: UNSW-NB15 (Moustafa & Slay, 2015), descargados de un mirror fijado.
- **Contribuciones:**
  - Nicol Tatiana Cerinza Tejada: preparación de la solución entregada (código, auditoría, cuatro experimentos, publicación en Hugging Face, evidencias de recuperación e informe), con la asistencia de IA declarada arriba. Revisar esta descripción antes de la entrega final.
  - Natalia Rios Moreno: revisión de la guía frente a entrega.md, informe y evidencias; comprobación independiente de recuperación, huellas, predicciones y reproducibilidad de E4; revisión de las explicaciones del split por grupos, exclusión de contradicciones, selección del umbral y errores en test. Estas verificaciones se realizaron con ayuda de Codex; sus resultados y limitaciones constan en evidencias/revision-natalia/. La comprensión personal para la sustentación corresponde a cada integrante.
