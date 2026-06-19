# Versión simplificada

Misma funcionalidad y **mismo resultado** que la versión original (95,6% de
exactitud, 0 falsos positivos), pero con el código junto en **3 archivos** en
vez de 7, y con comentarios más cortos y simples.

> La versión original (7 archivos comentados en detalle) sigue intacta en la
> carpeta de arriba, para estudiarla. Esta es la versión "limpia" para usar.

## Los 3 archivos

| Archivo | Junta a... | Qué hace |
|---|---|---|
| `tornillos.py` | pipeline + detectores + clasificador | Todo el núcleo: segmentar, alinear, los 3 detectores y clasificar. Son funciones, no se ejecuta solo. |
| `entrenar.py` | build_template + calibrar | Aprende cómo es un tornillo sano y fija los umbrales. Se corre una vez. |
| `app.py` | main + evaluar | Inspecciona una imagen (con figura) o evalúa todo el dataset. |

## Cómo usarlo

Desde esta carpeta (`simple/`):

```bash
# 1. Entrenar (una sola vez) -> crea la carpeta plantillas/
python entrenar.py

# 2a. Inspeccionar una imagen (muestra la figura del pipeline)
python app.py ../datos/scratch_head/000.png

# 2b. Evaluar todo el dataset (imprime las métricas, crea resultados.csv)
python app.py
```

Usa la misma carpeta `datos/` de la versión original (la busca en `../datos`),
así que no hace falta duplicar las imágenes.

## Equivalencia con la versión original

| Original (7 archivos) | Acá (3 archivos) |
|---|---|
| `pipeline.py` | parte de `tornillos.py` |
| `detectores.py` | parte de `tornillos.py` |
| `clasificador.py` | parte de `tornillos.py` |
| `build_template.py` | parte de `entrenar.py` |
| `calibrar.py` | parte de `entrenar.py` |
| `main.py` | `app.py <imagen>` |
| `evaluar.py` | `app.py` (sin argumentos) |
