"""
calibrar.py — Calibración estadística de los umbrales de decisión.

Corre el pipeline completo (segmentación + alineación + detectores) sobre
las 41 imágenes SIN defecto y registra los 8 scores de cada una. Como esas
imágenes son piezas aceptables, sus scores describen la "respuesta normal"
del sistema: ruido residual de los detectores ante variaciones legítimas
(fase de la rosca, brillos especulares del metal, polvo).

El umbral de cada score se fija con la regla estadística clásica:

    umbral_k = max( media_k + 3 * desvio_k,      <- regla de las 3 sigmas
                    1.25 * maximo_good_k,        <- margen sobre el peor caso
                    PISO_MINIMO )                <- piso absoluto

  - media + 3*desvio: si la respuesta normal fuera gaussiana, solo el 0.13%
    de las piezas sanas la superaría (falsos positivos casi nulos).
  - 1.25 * máximo observado: protege cuando la distribución tiene colas más
    pesadas que la gaussiana (el máximo real manda sobre la teoría).
  - PISO_MINIMO: si un score dio siempre 0 en las good (detector "mudo" en
    esa región), los dos criterios anteriores darían umbral 0 y cualquier
    píxel detectado dispararía una alarma; el piso exige al menos una
    fracción mínima razonable de la región afectada.

Uso:
    python calibrar.py
"""

import os

import numpy as np

from pipeline import procesar_imagen
from detectores import inspeccionar
from clasificador import cargar_plantillas, guardar_umbrales

CARPETA_GOOD = os.path.join("datos", "good")

# Pisos mínimos de los umbrales, según las unidades de cada score:
#   - forma_* / int_*  : fracción del área de la región. 0.2% es menos que
#     el ruido esperable de la discretización (rotaciones interpoladas).
#   - perfil_*         : px promedio de exceso de envolvente por columna.
#     0.10 px equivale, p. ej., a una muesca de ~6 px sostenida durante
#     ~5 columnas más allá de la banda de los sanos: el defecto mínimo
#     que vale la pena reportar.
PISO_AREA = 0.002
PISO_PERFIL = 0.10


def calibrar():
    """Procesa todas las 'good', acumula scores y deriva los umbrales."""
    plantillas = cargar_plantillas()

    # ------------------------------------------------------------------
    # 1. Recolectar los scores de cada imagen sana
    # ------------------------------------------------------------------
    todos = []  # lista de dicts {score: valor}
    archivos = sorted(a for a in os.listdir(CARPETA_GOOD) if a.endswith(".png"))

    for nombre in archivos:
        r = procesar_imagen(
            os.path.join(CARPETA_GOOD, nombre),
            mascara_plantilla=plantillas["binaria"],  # corrige flip vertical
        )
        resultado = inspeccionar(
            r["gris_alineada"], r["mascara_alineada"], plantillas
        )
        todos.append(resultado["scores"])
        print(f"  {nombre}: " + "  ".join(
            f"{k}={v:.4f}" for k, v in resultado["scores"].items() if v > 0
        ) or f"  {nombre}: todos los scores en 0")

    # ------------------------------------------------------------------
    # 2. Estadísticas por score y regla de umbral
    # ------------------------------------------------------------------
    claves = todos[0].keys()
    umbrales, estadisticas = {}, {}

    print(f"\n{'score':<15}{'media':>10}{'desvio':>10}{'maximo':>10}{'umbral':>10}")
    for k in claves:
        valores = np.array([t[k] for t in todos])
        media, desvio, maximo = valores.mean(), valores.std(), valores.max()

        piso = PISO_PERFIL if k.startswith("perfil") else PISO_AREA
        umbral = max(media + 3 * desvio, 1.25 * maximo, piso)

        umbrales[k] = float(umbral)
        estadisticas[k] = {
            "media": float(media),
            "desvio": float(desvio),
            "maximo": float(maximo),
            "n": len(valores),
        }
        print(f"{k:<15}{media:>10.5f}{desvio:>10.5f}{maximo:>10.5f}{umbral:>10.5f}")

    # ------------------------------------------------------------------
    # 3. Persistir para evaluar.py y main.py
    # ------------------------------------------------------------------
    guardar_umbrales(umbrales, estadisticas)
    return umbrales


if __name__ == "__main__":
    print(f"Calibrando umbrales con las imágenes de {CARPETA_GOOD} ...")
    calibrar()
