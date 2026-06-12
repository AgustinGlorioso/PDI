"""
main.py — Demostración del pipeline completo sobre una imagen.

Procesa una imagen del dataset con las 5 fases del sistema y muestra una
figura con todos los pasos intermedios y el diagnóstico final:

    1. Imagen original          4. Defectos detectados (superpuestos)
    2. Segmentación (máscara)   5. Diagnóstico: Normal / Anómala + tipo
    3. Alineación + regiones

Uso:
    python main.py                              (imagen de demostración)
    python main.py datos/scratch_head/000.png   (imagen elegida)

Requiere haber ejecutado antes:
    python build_template.py   (construye las plantillas en plantillas/)
    python calibrar.py         (calibra los umbrales de decisión)
"""

import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

from pipeline import procesar_imagen
from detectores import inspeccionar, REGIONES
from clasificador import cargar_plantillas, cargar_umbrales, clasificar

# Imagen procesada si no se pasa ninguna por línea de comandos
IMAGEN_DEMO = "datos/scratch_head/000.png"

# Colores (RGB) para superponer cada tipo de evidencia sobre la imagen
COLOR_FORMA = (255, 60, 60)        # rojo:    defecto de forma (silueta)
COLOR_INTENSIDAD = (255, 220, 0)   # amarillo: defecto de intensidad (gris)
COLOR_PERFIL = (255, 0, 255)       # magenta: violación de envolvente


def construir_overlay(gris_alineada, insp):
    """Pinta las evidencias de los tres detectores sobre la imagen alineada.

    Devuelve una imagen RGB con los defectos coloreados según el detector
    que los encontró (ver constantes COLOR_*).
    """
    # Base: la imagen alineada en gris convertida a RGB
    overlay = cv2.cvtColor(gris_alineada, cv2.COLOR_GRAY2RGB)

    # Defectos de forma (faltante + sobrante) en rojo
    defecto_forma = cv2.bitwise_or(
        insp["forma"]["faltante"], insp["forma"]["sobrante"]
    )
    overlay[defecto_forma > 0] = COLOR_FORMA

    # Defectos de intensidad (brillante + oscuro) en amarillo
    defecto_int = cv2.bitwise_or(
        insp["intensidad"]["brillante"], insp["intensidad"]["oscuro"]
    )
    overlay[defecto_int > 0] = COLOR_INTENSIDAD

    # Violaciones de la envolvente del perfil en magenta
    overlay[insp["perfil"]["marcas"] > 0] = COLOR_PERFIL

    return overlay


def visualizar(ruta, r, insp, veredicto):
    """Arma la figura resumen del pipeline para la imagen procesada."""
    es_anomala = veredicto["diagnostico"] == "Anómala"
    color_titulo = "red" if es_anomala else "green"

    fig, axs = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle(
        f"{ruta}  —  Diagnóstico: {veredicto['diagnostico']}"
        + (f"  ({veredicto['tipo']})" if es_anomala else "")
        + f"  |  Nivel de anomalía: {veredicto['nivel']:.2f}  (límite: 1.00)",
        fontsize=15, fontweight="bold", color=color_titulo,
    )

    axs[0].imshow(r["rgb"])
    axs[0].set_title("1. Imagen original")

    axs[1].imshow(r["mascara"], cmap="gray")
    axs[1].set_title("2. Segmentación (Canny + relleno)")

    # Imagen alineada con las fronteras de regiones dibujadas
    axs[2].imshow(r["gris_alineada"], cmap="gray")
    for nombre, (x0, x1) in REGIONES.items():
        axs[2].axvline(x1, color="cyan", lw=0.8, ls="--")
        axs[2].text((x0 + x1) / 2, 990, nombre, color="cyan",
                    ha="center", fontsize=9)
    axs[2].set_title("3. Alineación + regiones")

    axs[3].imshow(construir_overlay(r["gris_alineada"], insp))
    axs[3].set_title(
        "4. Defectos: forma (rojo), intensidad (amarillo), perfil (magenta)"
    )

    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ruta = sys.argv[1] if len(sys.argv) > 1 else IMAGEN_DEMO
    print(f"Inspeccionando {ruta} ...")

    # Cargar el modelo estadístico (plantillas) y los umbrales calibrados
    plantillas = cargar_plantillas()
    umbrales = cargar_umbrales()

    # Fases 1-2: cargar, segmentar, alinear (incluye corrección de flip)
    r = procesar_imagen(ruta, mascara_plantilla=plantillas["binaria"])

    # Fases 3-4: correr los tres detectores y calcular los scores
    insp = inspeccionar(r["gris_alineada"], r["mascara_alineada"], plantillas)

    # Fase 5: clasificar contra los umbrales calibrados
    veredicto = clasificar(insp["scores"], umbrales)

    # Reporte por consola
    print(f"\nDiagnóstico : {veredicto['diagnostico']}")
    if veredicto["diagnostico"] == "Anómala":
        print(f"Tipo        : {veredicto['tipo']}")
    print(f"Nivel       : {veredicto['nivel']:.2f}  (anómala si supera 1.00)")
    print("\nScores (score / umbral = exceso relativo):")
    for k, v in sorted(veredicto["excesos"].items(), key=lambda kv: -kv[1]):
        marca = "  <-- DISPARÓ" if v > 1.0 else ""
        print(f"  {k:<15} {insp['scores'][k]:.5f} / {umbrales[k]:.5f} "
              f"= {v:5.2f}{marca}")

    visualizar(ruta, r, insp, veredicto)
