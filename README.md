# Detección de Anomalías Industriales en Tornillos mediante PDI

Trabajo Final de **Procesamiento Digital de Imágenes** — UNL · FICH
Alumnos: Glorioso, Agustín; Meier, Adrián

Sistema de inspección visual automática que clasifica tornillos del dataset
[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
(categoría *screw*) como **Normales** o **Anómalos**, indica el **tipo de
defecto** y **señala su ubicación**, usando exclusivamente técnicas clásicas
de procesamiento de imágenes (sin redes neuronales).

## Resultados

Evaluado sobre las 160 imágenes del dataset:

| Categoría          | Detección binaria |
|--------------------|------------------:|
| good (sanos)       | 41/41 (100%) — sin falsos positivos |
| scratch_head       | 24/24 (100%) |
| scratch_neck       | 21/25 (84%)  |
| thread_side        | 21/23 (91%)  |
| thread_top         | 22/23 (96%)  |
| manipulated_front  | 24/24 (100%) |
| **Global**         | **153/160 (95,6%)** |

## Cómo ejecutarlo

Requiere Python 3 con `opencv-python`, `numpy` y `matplotlib`, y la carpeta
`datos/` con las imágenes (no está en el repo por su tamaño).

```bash
# 1. Construir la plantilla estadística con los 41 tornillos sanos
python build_template.py

# 2. Calibrar los umbrales de decisión (regla de las 3 sigmas)
python calibrar.py

# 3a. Inspeccionar una imagen y ver el pipeline completo
python main.py datos/scratch_head/000.png

# 3b. Evaluar el sistema sobre todo el dataset (genera resultados.csv)
python evaluar.py
```

## Arquitectura del pipeline

```
imagen ─> SEGMENTACIÓN ─> ALINEACIÓN ─> 3 DETECTORES ─> SCORES ─> DIAGNÓSTICO
          (Canny +        (momentos +    en paralelo     por      (umbrales
           relleno)        anclaje)                      región    calibrados)
```

1. **Segmentación por bordes** ([pipeline.py](pipeline.py)): Canny +
   dilatación + relleno del contorno mayor + erosión. Se eligió sobre el
   umbralado de Otsu porque las sombras difusas del dataset inflaban la
   máscara hasta un 70%; con bordes, el área de los tornillos sanos es
   constante (±1%).
2. **Alineación espacial** ([pipeline.py](pipeline.py)): orientación del
   eje por momentos de segundo orden (equivalente a PCA), cabeza a la
   izquierda por distribución de masas, anclaje del extremo izquierdo y
   del eje en posiciones fijas. Tras alinear, todas las imágenes son
   comparables píxel a píxel.
3. **Tres detectores complementarios** ([detectores.py](detectores.py)):
   - **Forma**: compara la silueta contra las zonas estadísticas NÚCLEO
     (siempre metal) y EXTERIOR (nunca metal) de la plantilla. Detecta
     puntas rotas/doblada y deformaciones gruesas.
   - **Intensidad**: puntaje z píxel a píxel contra la media y desviación
     de los 41 sanos, con corrección de iluminación por fila en el cuello.
     Detecta arañazos y raspones que no alteran la silueta.
   - **Perfil de rosca**: envolventes de crestas y valles (filtros de
     máximo/mínimo deslizante de un paso de rosca), invariantes a la fase
     del filete. Detecta rebabas y muescas de 8-25 px en las crestas.
4. **Clasificación** ([clasificador.py](clasificador.py)): 10 scores por
   imagen (2 por región + 2 de perfil) contra umbrales calibrados con la
   regla `max(μ+3σ, 1.25·máximo sano, piso)`. La región del score más
   excedido determina el tipo de defecto.

## Archivos

| Archivo | Rol |
|---|---|
| `pipeline.py` | Fases 1-2: carga, segmentación, alineación |
| `detectores.py` | Fases 3-4: los tres detectores + scores por región |
| `clasificador.py` | Fase 5: umbrales, diagnóstico y tipo de defecto |
| `build_template.py` | Construye la plantilla estadística (carpeta `plantillas/`) |
| `calibrar.py` | Calibra los umbrales con las imágenes sanas |
| `evaluar.py` | Métricas sobre todo el dataset + `resultados.csv` |
| `main.py` | Demo visual del pipeline sobre una imagen |
