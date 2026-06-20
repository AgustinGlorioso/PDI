# Versión fácil — 3 detectores (96,9%)

Versión simple del sistema, pensada para **entender y explicar** sin perder
precisión. Mantiene la idea de fondo (comparar contra un tornillo sano
promedio) y usa los 3 detectores, pero con **mucha menos complejidad de
umbrales** que la versión completa.

## Qué cambió respecto de la versión completa

- En vez de **10 scores por región** + una fórmula de calibración con 3 partes,
  hay **3 puntajes** (uno por detector) y **3 umbrales**, calibrados con una
  sola regla simple: *"el umbral es el del peor tornillo sano + un margen"*.
- El diagnóstico es directo: si algún detector supera su umbral, es anómala.
  El tipo de defecto sale de la región donde está la mancha más grande.

Resultado: **96,9%** de exactitud, **0 falsos positivos** (igual o mejor que
la versión completa), con código bastante más fácil de leer.

## Los 3 archivos

| Archivo | Qué hace |
|---|---|
| `tornillos.py` | Todo el núcleo: preprocesamiento + los 3 detectores + diagnóstico |
| `entrenar.py` | Aprende el tornillo sano y fija los 3 umbrales (correr una vez) |
| `app.py` | Inspecciona una imagen (con figura) o evalúa todo el dataset |

## Los 3 detectores (una idea cada uno)

1. **Intensidad** — compara el gris contra el tornillo promedio (en desvíos
   estándar). Detecta **rayones** (cabeza y cuello).
2. **Forma** — compara la silueta contra la plantilla: metal que falta o que
   sobra. Detecta **puntas dobladas/rotas y deformaciones**.
3. **Perfil de rosca** — mide el "ancho" del tornillo columna a columna, de
   forma que no importe cómo estén girados los dientes. Detecta **roscas
   dañadas**.

## Cómo usarlo

```bash
python entrenar.py                          # una vez -> crea plantillas/
python app.py datos/scratch_head/000.png    # inspeccionar (muestra figura)
python app.py                               # evaluar todo el dataset
```
