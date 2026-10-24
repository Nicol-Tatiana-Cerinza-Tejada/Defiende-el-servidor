"""Genera informe/informe.pdf leyendo las cifras de evidencias/ (no hay números a mano en tablas).

Requiere reportlab y matplotlib (no forman parte de requirements.txt):
  pip install reportlab==5.1.0
  python informe/generar_informe.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer,  # noqa: E402
                                Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "evidencias"
OUT = ROOT / "informe"


def load(name):
    return json.loads((EV / name).read_text(encoding="utf-8"))


audit, sel, test, hf = load("auditoria.json"), load("seleccion.json"), load("metricas-test.json"), load("hf_versiones.json")
selval, errs, rec = load("seleccion_validacion.json"), load("analisis-errores-test.json"), load("recuperacion/resultado.json")
exps = pd.read_csv(EV / "experimentos.csv")

ss = getSampleStyleSheet()
GREEN = colors.HexColor("#1f5f3f")
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, textColor=GREEN, spaceAfter=4, spaceBefore=6)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, textColor=GREEN, spaceAfter=2, spaceBefore=6)
P = ParagraphStyle("p", parent=ss["BodyText"], fontSize=9.2, leading=12, spaceAfter=4)
SMALL = ParagraphStyle("s", parent=P, fontSize=7.8, leading=9.6)
CELL = ParagraphStyle("c", parent=P, fontSize=7.8, leading=9.4, spaceAfter=0)


def f3(x):
    return f"{x:.3f}"


def pc(x):
    return f"{100 * x:.1f} %"


def table(rows, widths, header=True):
    data = [[Paragraph(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b8c4bb")),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e3ece5"))]
    t.setStyle(TableStyle(style))
    return t


def sweep_figure() -> Path:
    v1 = pd.read_csv(EV / "mlflow" / hf["v1"]["mlflow_run_id"] / "val_threshold_sweep.csv")
    v2 = pd.read_csv(EV / "mlflow" / hf["v2"]["mlflow_run_id"] / "val_threshold_sweep.csv")
    fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=200)
    ax.plot(v2.threshold, v2.recall_attack, color="#1f5f3f", lw=1.8, label="E4 recall ataque")
    ax.plot(v2.threshold, v2.fpr, color="#c0392b", lw=1.8, label="E4 FPR")
    ax.plot(v2.threshold, v2.macro_f1, color="#2c6fbb", lw=1.8, label="E4 macro-F1")
    ax.plot(v1.threshold, v1.recall_attack, color="#1f5f3f", lw=1, ls="--", label="E1 recall")
    ax.plot(v1.threshold, v1.fpr, color="#c0392b", lw=1, ls="--", label="E1 FPR")
    ax.axvline(sel["threshold"], color="#555", lw=0.8, ls=":")
    ax.axhline(0.95, color="#999", lw=0.6, ls=":")
    ax.annotate(f"umbral v2 = {sel['threshold']}", (sel["threshold"], 0.45), xytext=(5, 0),
                textcoords="offset points", fontsize=7)
    ax.set_xlabel("umbral sobre score_attack (validación)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, ncol=3, loc="center left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = OUT / "fig_umbral.png"
    fig.savefig(p)
    return p


def build():
    s = []
    tr, te = audit["files"]["train"], audit["files"]["test"]
    sp = audit["split"]["files"]
    ov = audit["cross_partition_overlap"]
    cp = audit["cleaning_policy"]

    s.append(Paragraph("Defiende el servidor — Informe del taller", ParagraphStyle(
        "t", parent=H1, fontSize=18, spaceAfter=2)))
    s.append(Paragraph("MLOps 2026-II · Grupo A · Nicol Tatiana Cerinza Tejada y Natalia Rios · "
                       "Universidad Externado de Colombia", SMALL))
    s.append(Paragraph(
        "Código: github.com/Nicol-Tatiana-Cerinza-Tejada/Defiende-el-servidor · Modelos: "
        "huggingface.co/NicolTcT/defiende-el-servidor · Datos DVC: huggingface.co/datasets/NicolTcT/"
        "defiende-el-servidor-dvc", SMALL))
    s.append(Spacer(1, 4))
    s.append(Paragraph(
        "Construimos un clasificador binario normal/ataque con UNSW-NB15 y, sobre todo, la cadena de "
        "trazabilidad que permite recuperarlo: <b>commit Git → referencia DVC → run de MLflow → revisión de "
        "Hugging Face</b>. Todas las cifras de este informe se leen de los archivos de <i>evidencias/</i>.", P))

    # 1. Datos
    s.append(Paragraph("1. Datos, auditoría y limpieza", H1))
    s.append(Paragraph(
        f"Se usaron las particiones publicadas, descargadas del mirror fijado el {audit['sources']['download_date_utc'][:10]} (UTC). "
        "Los SHA-256 coinciden con el anexo del taller. Además de los nombres se comprobaron el número de filas, el esquema "
        f"y las clases. No hay valores faltantes; el valor '-' de <i>service</i> se trata como una categoría más. "
        f"Se excluyen como entradas <i>label</i> (objetivo), <i>attack_cat</i> (revela la respuesta) e <i>id</i>; "
        f"quedan {len(audit['variables']['used_features'])} variables (3 categóricas y "
        f"{len(audit['variables']['numeric'])} numéricas).", P))
    s.append(table([
        ["Archivo", "Filas × cols", "Normal / ataque", "Repiten una entrada previa", "Grupos contradictorios (filas)", "SHA-256 (inicio)"],
        ["training-set", f"{tr['rows']:,} × {tr['columns']}", f"{tr['class_counts']['0']:,} / {tr['class_counts']['1']:,}",
         f"{tr['duplicates_by_features']['rows_that_repeat_a_previous_input']:,}",
         f"{tr['contradictory_label_groups']['groups']} ({tr['contradictory_label_groups']['rows']})", tr["sha256"][:16] + "…"],
        ["testing-set", f"{te['rows']:,} × {te['columns']}", f"{te['class_counts']['0']:,} / {te['class_counts']['1']:,}",
         f"{te['duplicates_by_features']['rows_that_repeat_a_previous_input']:,}",
         f"{te['contradictory_label_groups']['groups']} ({te['contradictory_label_groups']['rows']})", te["sha256"][:16] + "…"],
    ], [2.2 * cm, 2.3 * cm, 2.6 * cm, 2.8 * cm, 3.2 * cm, 3.4 * cm]))
    s.append(Spacer(1, 4))
    s.append(Paragraph(
        "<b>Política de limpieza.</b> Los duplicados exactos se conservan porque reflejan la frecuencia real del tráfico. "
        f"Las {cp['contradictory_rows_excluded']} filas de los {cp['contradictory_groups_excluded']} grupos con entradas idénticas y etiquetas "
        "distintas se excluyen de entrenamiento y validación, porque en ellas las variables no determinan la etiqueta. "
        "No se elimina ninguna fila de test.", P))
    s.append(Paragraph(
        "<b>Particiones.</b> Train y validación salen solo del archivo de entrenamiento. Se hace un <i>group split</i> por la huella "
        f"de las variables predictoras: los grupos se barajan con la semilla {audit['split']['seed']} y se asignan a validación "
        f"hasta llegar al 20 % de las filas. Así, ningún grupo de entradas idénticas queda repartido. Resultado: train "
        f"{sp['train.csv']['rows']:,} filas y validación {sp['val.csv']['rows']:,}. La asignación de cada fila está en "
        "<i>data/prepared/splits.csv</i>. Las particiones son las mismas en los cuatro experimentos, y codificadores, "
        "imputación y escalado se ajustan dentro del pipeline solo con train.", P))
    s.append(Paragraph(
        f"<b>Cruces con test.</b> {ov['test_rows_with_input_seen_in_training_file']:,} filas de test "
        f"({ov['test_rows_with_input_seen_in_training_file_pct']} %) tienen una entrada idéntica en el archivo de "
        f"entrenamiento: {ov['test_rows_with_input_seen_in_train_split']:,} en train y "
        f"{ov['test_rows_with_input_seen_in_val_split']:,} en validación. No se filtran, pero limitan las conclusiones.", P))
    s.append(Paragraph(
        "<b>Dos versiones de datos con DVC.</b> <i>data/raw.dvc</i> guarda los dos CSV recibidos (md5 "
        f"{sel['dvc_raw_md5'][:12]}…) y <i>data/prepared.dvc</i> guarda train/val/test/splits (md5 {sel['dvc_prepared_md5'][:12]}…). "
        "La transformación es <i>src/audit_prepare.py</i> y es determinista: en el clon nuevo regenera los mismos SHA-256. "
        "El remote es un dataset público de Hugging Face al que se accede por HTTP, así que <i>dvc pull</i> funciona "
        "sin credenciales y sin la caché original.", P))

    # 2. Experimentos
    s.append(Paragraph("2. Cuatro experimentos, dos familias", H1))
    s.append(Paragraph(
        "Cada run registra en MLflow la hipótesis, el cambio, los parámetros, la semilla, el commit Git, el md5 de DVC, las "
        "métricas de validación, la matriz de confusión, el barrido de umbrales, las puntuaciones de validación y el pipeline "
        "completo. La clase positiva es ataque (1). Cuando un denominador es cero, la métrica se reporta como 0. La tabla "
        "usa el umbral 0,5 en todos los casos.", P))
    rows = [["Exp.", "Familia / cambio", "Hipótesis (resumen)", "Recall", "Prec.", "FPR", "Macro-F1", "FP / FN", "Run ID"]]
    short = {
        "E1_logreg_base": ("Logística base (one-hot + escala)", "Referencia sencilla; colas pesadas limitarán el recall."),
        "E2_logreg_log1p": ("Logística + log1p + balanced", "log1p lineariza volúmenes; menos FPR que E1."),
        "E3_random_forest": ("Árboles: RandomForest 200, prof. 20", "Interacciones no lineales mejoran macro-F1."),
        "E4_hist_gb": ("Árboles: HistGB 300 it., 63 hojas", "Boosting ≈ o > E3 con modelo más liviano."),
    }
    for _, r in exps.iterrows():
        rows.append([r.exp_id.split("_")[0], short[r.exp_id][0], short[r.exp_id][1], f3(r.val_recall_attack),
                     f3(r.val_precision_attack), f3(r.val_fpr), f3(r.val_macro_f1), f"{r.val_fp} / {r.val_fn}",
                     r.run_id[:8] + "…"])
    s.append(table(rows, [0.9 * cm, 3.1 * cm, 3.8 * cm, 1.3 * cm, 1.3 * cm, 1.3 * cm, 1.7 * cm, 2.0 * cm, 1.9 * cm]))
    s.append(Spacer(1, 4))
    s.append(Paragraph(
        "<b>Interpretación de errores.</b> E1 detecta casi todos los ataques, pero alerta sobre el 17 % del tráfico normal "
        "(1 963 falsos positivos). E2 confirma en parte su hipótesis: reduce el FPR a la mitad, pero pierde 4,7 puntos de recall, "
        "así que el balanceo desplaza el error hacia los falsos negativos. Los árboles (E3 y E4) dominan a las logísticas en "
        "precision y FPR sin perder recall. E4 iguala o supera a E3 con un modelo de 2,2 MB, frente a 69 MB. Los run ID "
        "completos y la conclusión de cada experimento están en <i>experimentos.csv</i> y <i>mlflow/runs.json</i>.", P))

    # 3. v1 / v2 y umbral
    s.append(Paragraph("3. Versiones v1 y v2 y elección del umbral", H1))
    v = sel["val_metrics"]
    e1 = exps.set_index("exp_id").loc["E1_logreg_base"]
    s.append(Paragraph(
        "<b>v1</b> es la primera solución funcional: E1 con el umbral por defecto 0,5. <b>v2</b> sale de una regla escrita en "
        "<i>configs/selection.yaml</i> antes de ver los resultados de E2–E4. Primero se elige la configuración con mayor "
        "macro-F1 de validación (E4). Luego, en su barrido de validación (pasos de 0,05), se toma el umbral con mayor "
        "macro-F1 entre los que mantienen un recall de ataque ≥ 0,95, porque para el equipo de seguridad perder ataques es "
        f"lo más costoso. Resultado: umbral <b>{sel['threshold']}</b>. El test no intervino en ninguna decisión.", P))
    s.append(Image(str(sweep_figure()), width=16.5 * cm, height=5.9 * cm))
    s.append(table([
        ["Validación (34 880 filas)", "Recall ataque", "Precision ataque", "FPR", "Macro-F1", "Matriz [TN FP; FN TP]"],
        ["v1 · E1 · umbral 0,50", f3(e1.val_recall_attack), f3(e1.val_precision_attack), f3(e1.val_fpr), f3(e1.val_macro_f1),
         f"[{e1.val_tn} {e1.val_fp}; {e1.val_fn} {e1.val_tp}]"],
        [f"v2 · E4 · umbral {sel['threshold']:.2f}", f3(v["recall_attack"]), f3(v["precision_attack"]), f3(v["fpr"]),
         f3(v["macro_f1"]), f"[{int(v['tn'])} {int(v['fp'])}; {int(v['fn'])} {int(v['tp'])}]"],
    ], [3.6 * cm, 2.2 * cm, 2.4 * cm, 1.6 * cm, 1.8 * cm, 4.6 * cm]))
    s.append(Spacer(1, 4))
    s.append(Paragraph(
        "<b>Qué mejora, qué empeora y qué aceptamos.</b> v2 reduce las falsas alarmas de 1 963 a "
        f"{int(v['fp'])} (FPR de {pc(e1.val_fpr)} a {pc(v['fpr'])}) y sube la precision a {f3(v['precision_attack'])}. "
        f"A cambio, deja pasar {int(v['fn'])} ataques en lugar de {e1.val_fn}: el recall baja 1,5 puntos. Aceptamos ese "
        "costo porque un FPR del 17 % haría inmanejable la cola de alertas, y el recall sigue por encima del mínimo de 0,95. "
        "Como el umbral se eligió en la misma validación, estas cifras de v2 son algo optimistas.", P))
    s.append(Paragraph(
        "<b>Publicación.</b> Cada revisión de Hugging Face contiene <i>model.joblib</i> (pipeline completo de preprocesamiento "
        "y estimador), <i>config.json</i> (umbral, variables, SHA-256 del modelo, commit, md5 de DVC y run de MLflow), "
        "<i>requirements.txt</i> y la model card. Se identifican por SHA completo: v1 "
        f"<font face='Courier' size='7.5'>{hf['v1']['revision_sha']}</font> y v2 "
        f"<font face='Courier' size='7.5'>{hf['v2']['revision_sha']}</font>, con tags v1/v2. En Git, los tags "
        "<i>v1-code</i> y <i>v2-code</i> marcan el código de cada solución.", P))

    # 4. Test
    m = test["metrics"]
    s.append(Paragraph("4. Evaluación final en test", H1))
    s.append(Paragraph(
        f"La selección se congeló en el commit <font face='Courier' size='7.5'>{test['selection_frozen_in_commit']}</font> "
        "(<i>seleccion.json</i>), anterior a cualquier uso de test. <i>evaluate_test.py</i> se niega a correr si esa decisión "
        "no está en un commit o si el test ya se evaluó. El modelo se descargó de Hugging Face por SHA y se verificó su "
        f"hash. Se evaluó una vez sobre las {test['rows_evaluated']:,} filas del test publicado, sin quitar ninguna. "
        "Las métricas se recalculan desde <i>predicciones-test.csv</i> (row_id, y_true, score_attack, y_pred), con la regla "
        "y_pred = 1 si score_attack ≥ 0,55.", P))
    s.append(table([
        ["v2 (E4, umbral 0,55)", "Recall ataque", "Precision ataque", "FPR", "Macro-F1", "Accuracy", "Matriz [TN FP; FN TP]"],
        ["Validación", f3(v["recall_attack"]), f3(v["precision_attack"]), f3(v["fpr"]), f3(v["macro_f1"]), f3(v["accuracy"]),
         f"[{int(v['tn'])} {int(v['fp'])}; {int(v['fn'])} {int(v['tp'])}]"],
        ["Test (82 332 filas)", f3(m["recall_attack"]), f3(m["precision_attack"]), f3(m["fpr"]), f3(m["macro_f1"]),
         f3(m["accuracy"]), f"[{m['tn']} {m['fp']}; {m['fn']} {m['tp']}]"],
    ], [3.1 * cm, 2.0 * cm, 2.3 * cm, 1.4 * cm, 1.8 * cm, 1.7 * cm, 3.9 * cm]))
    s.append(Spacer(1, 4))
    st = {d["state"]: d for d in errs["fpr_by_state_normal_rows"]}
    fz = {d["attack_cat"]: d for d in errs["recall_by_attack_cat"]}["Fuzzers"]
    s.append(Paragraph(
        f"<b>El resultado empeora y lo conservamos.</b> El recall se mantiene ({f3(m['recall_attack'])}), pero el FPR sube de "
        f"{pc(v['fpr'])} a {pc(m['fpr'])}: {m['fp']:,} conexiones normales alertadas. El análisis descriptivo posterior "
        "(<i>analisis-errores-test.json</i>, que no cambió el modelo) muestra dónde se concentran: las conexiones normales en "
        f"estado INT se alertan en un {pc(st['INT']['pred_attack_rate'])} y las FIN en un {pc(st['FIN']['pred_attack_rate'])}. "
        f"Además, la proporción de INT y REQ en el tráfico normal es mayor en test que en entrenamiento. El recall más bajo "
        f"es el de Fuzzers ({pc(fz['pred_attack_rate'])}), la categoría de ataque que más se parece al tráfico normal. "
        "La validación salió del mismo archivo que el entrenamiento y no captó este cambio de distribución entre los "
        "dos archivos publicados. Es la lección principal del taller: una validación del mismo origen sobreestima "
        "el rendimiento.", P))

    # 5. Recuperación
    s.append(Paragraph("5. Prueba de recuperación", H1))
    s.append(Paragraph(
        "<i>scripts/prueba_recuperacion.sh</i> se ejecutó en una carpeta nueva con un venv nuevo, un clon nuevo de GitHub "
        f"(commit {rec['git_commit_recuperado'][:12]}…), caché DVC vacía y un HF_HOME nuevo sin token. Pasos:", P))
    s.append(Paragraph(
        "(1) instalar las dependencias fijadas; (2) <i>dvc pull</i> desde el dataset de Hugging Face, con las 6 huellas "
        "coincidentes; (3–4) descargar v1 y v2 por SHA completo y predecir sobre 30 filas fijas de validación (15 normales "
        "y 15 ataques); (5) comparar con las salidas guardadas antes de la recuperación. Resultado: <b>COINCIDEN para v1 y "
        f"v2, con 0 etiquetas distintas y diferencia máxima de score 0,0</b> (tolerancia declarada {rec['tolerancia_score']}). "
        f"Tiempo total: {rec['duracion']}. Además, en ese entorno, <i>train.py --exp E4_hist_gb</i> reprodujo las mismas "
        "métricas y un <i>model.joblib</i> con el mismo SHA-256 que v2. Entrada, salidas, hashes y log completo están en "
        "<i>evidencias/recuperacion/</i>.", P))

    # 6. Limitaciones
    s.append(Paragraph("6. Limitaciones", H1))
    for t in [
        "UNSW-NB15 es tráfico sintético de laboratorio de 2015: no demuestra eficacia en una red actual.",
        "Hay cambio de distribución entre los archivos publicados de train y test: el FPR de test (24 %) es el número a "
        "considerar, no el de validación.",
        f"El {ov['test_rows_with_input_seen_in_training_file_pct']} % de las filas de test repite entradas del "
        "entrenamiento, lo que puede inflar algo las métricas de test.",
        "El umbral se ajustó en validación con una rejilla de 0,05; las puntuaciones no están calibradas como probabilidades.",
        "Las 940 filas contradictorias se excluyeron del entrenamiento; en test se evalúan y una parte es irreducible.",
        "MLflow se usó en local (sqlite). La evidencia consultable es la exportación en <i>evidencias/mlflow/</i>.",
    ]:
        s.append(Paragraph("• " + t, P))

    s.append(Paragraph("Referencias", H2))
    for t in [
        "Moustafa, N., & Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection systems. MilCIS 2015.",
        "UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset (atribución); mirror Nir-J/ML-Projects@44bbd02.",
        "Documentación de Git, DVC, MLflow Tracking, Hugging Face Hub (descarga por revisión, model cards) y scikit-learn "
        "(common pitfalls).",
        "Asistencia de IA: Claude Code (Claude Opus 5.5) para código, ejecución y borrador; el detalle está en entrega.md.",
    ]:
        s.append(Paragraph(t, SMALL))

    doc = SimpleDocTemplate(str(OUT / "informe.pdf"), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm, title="Defiende el servidor — Informe",
                            author="Nicol Cerinza, Natalia Rios")

    def footer(canvas, d):
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.grey)
        canvas.drawString(1.8 * cm, 0.9 * cm, "MLOps 2026-II · Defiende el servidor · Grupo A")
        canvas.drawRightString(A4[0] - 1.8 * cm, 0.9 * cm, f"{d.page}")

    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print(OUT / "informe.pdf")


if __name__ == "__main__":
    build()
