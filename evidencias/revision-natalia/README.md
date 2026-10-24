# Revisión independiente · Natalia Rios Moreno · Grupo A

Revisión asistida con Codex (OpenAI) del commit de entrega `1e94c07957f8c742b51fb7ce94e6fe50c81fb011`, contra la guía de ocho páginas del 16/10/2026 y el informe de tres páginas. No se cambian modelo, umbral, splits ni resultados finales de test.

- `comprobacion-evidencias.json`: verificación de las 82 332 predicciones y sus métricas recalculadas, cuatro experimentos/run ID, selección y archivos públicos de las revisiones HF.
- `decisiones.md`: guía de las decisiones, costos de errores y límites metodológicos para preparar la sustentación.
- `recuperacion.log`, salidas recuperadas, `datos-verificados.json`, `reentrenamiento.log` y `reentrenamiento.json`: resultados de las ejecuciones reales descritos abajo.

## Observaciones corregidas

- Nombre confirmado por la integrante: Natalia Rios Moreno (en entrega e informe).
- Los árboles reducen falsas alarmas respecto a E1, pero sí pierden algo de recall.
- E2 cambia log1p y balanceo simultáneamente: su comparación no identifica el efecto aislado de cada uno.
- La auditoría describió las etiquetas de test. La congelación precede a la evaluación del modelo; no a toda lectura de esas etiquetas. No encontré uso de métricas de test en la selección.
- E4 a umbral 0,5 tiene macro-F1 0,952462 (0,952 a tres decimales); 0,953138 corresponde al umbral elegido 0,55.
- README usa ahora `entrega` en la prueba completa, consistente con las evidencias disponibles.

## Entorno de esta revisión

Python 3.14.4, Linux sobre WSL1, CPU. El Python local carece de ensurepip: el primer intento de crear un venv falló antes de instalar dependencias. En el segundo intento se creó un venv vacío y se arrancó pip con un bootstrap local de get-pip; después se ejecutó la instalación original de requirements.txt sin modificar sus versiones. Este ajuste del equipo se distingue de un fallo de la entrega. No se reutilizaron entornos ni cachés DVC/Hugging Face originales.

La revisión de archivos y la regeneración del informe usaron un entorno auxiliar Python 3.12; las pruebas de modelos usan el entorno nuevo Python 3.14 y los requisitos fijados del proyecto.

Las contribuciones indican tareas verificables y asistencia utilizada. La comprensión personal y una eventual sustentación deben realizarla las integrantes; un registro automático no la acredita.

## Resultado de las ejecuciones

- `recuperacion.log`: instalación de las siete dependencias fijadas, seis huellas SHA-256 correctas, dos descargas públicas por SHA y salida **RECUPERACION OK**. Ambas versiones coinciden en 30 filas; cero etiquetas distintas y diferencia máxima de puntuación 0,0 (tolerancia 1e-9). Tiempo de la prueba: 5 min 42 s, incluida la instalación y el arranque local de pip.
- `salida_v1_recuperada.csv` y `salida_v2_recuperada.csv`: salidas independientes; la entrada y las salidas originales están en `evidencias/recuperacion/`.
- `datos-verificados.json`: comprobación directa sobre los CSV recuperados, con 229 grupos contradictorios (940 filas) excluidos y cero grupos compartidos entre train y validación. Las etiquetas de test coinciden con las de las predicciones guardadas.
- `reentrenamiento.log` y `reentrenamiento.json`: E4 se ejecutó correctamente en el clon limpio, con un nuevo run MLflow. Sus métricas de validación al umbral original 0,5 coinciden con las entregadas. Se limitaron OpenMP y BLAS a dos hilos para la prueba en este equipo; ajuste medido de 17,7 s. El JSON registra por separado si el hash del artefacto coincide.

La prueba ejecutó el tag `entrega` original. Después de integrar esta rama, Nicol debe mover el tag a la nueva entrega y regenerar el ZIP. El reentrenamiento no sustituye los cuatro runs originales ni modifica la solución v2 congelada.

El reentrenamiento independiente produjo otro hash de archivo, pero sus puntuaciones coinciden exactamente con las del modelo congelado en las 34 880 filas de validación (diferencia máxima 0,0 y cero etiquetas diferentes a umbral 0,55). No se atribuye una causa única al cambio del archivo serializado. La descarga de la revisión publicada sí mantuvo el hash original.
