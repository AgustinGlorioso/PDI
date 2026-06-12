"""
generar_figuras.py — Produce las figuras del informe LaTeX.

Se ejecuta desde la raíz del repo (necesita las plantillas ya construidas):

    python build_template.py
    python calibrar.py
    python informe/generar_figuras.py

Genera en informe/figuras/:
    pipeline_etapas.png   las 4 etapas del pipeline sobre un scratch_head
    detecciones.png       overlays de defectos en 6 categorías
    plantilla.png         media de máscaras + zonas núcleo/exterior + std gris
    perfil_rosca.png      envolvente de un thread_side fuera de banda
"""

import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Permite importar los módulos del proyecto desde la raíz del repo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline import procesar_imagen
from detectores import (inspeccionar, perfiles_envolvente, REGIONES,
                        COLS_PERFIL, MARGEN_PERFIL, FILA_EJE)
from clasificador import cargar_plantillas, cargar_umbrales, clasificar
from main import construir_overlay

SALIDA = os.path.join(os.path.dirname(__file__), "figuras")
os.makedirs(SALIDA, exist_ok=True)

P = cargar_plantillas()
U = cargar_umbrales()


def fig_pipeline_etapas():
    """Las 4 etapas del pipeline sobre una imagen con arañazo en la cabeza."""
    ruta = "datos/scratch_head/010.png"
    r = procesar_imagen(ruta, mascara_plantilla=P["binaria"])
    insp = inspeccionar(r["gris_alineada"], r["mascara_alineada"], P)
    v = clasificar(insp["scores"], U)

    fig, axs = plt.subplots(1, 4, figsize=(20, 5.2))
    axs[0].imshow(r["rgb"]); axs[0].set_title("(a) Original")
    axs[1].imshow(r["mascara"], cmap="gray")
    axs[1].set_title("(b) Segmentacion")
    axs[2].imshow(r["gris_alineada"], cmap="gray")
    for nombre, (x0, x1) in REGIONES.items():
        axs[2].axvline(x1, color="cyan", lw=0.8, ls="--")
        axs[2].text((x0 + x1) / 2, 1000, nombre, color="cyan", ha="center", fontsize=9)
    axs[2].set_title("(c) Alineacion + regiones")
    axs[3].imshow(construir_overlay(r["gris_alineada"], insp))
    axs[3].set_title(f"(d) Deteccion: {v['diagnostico']} ({v['tipo']})")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(SALIDA, "pipeline_etapas.png"), dpi=110, bbox_inches="tight")
    plt.close()


def fig_detecciones():
    """Overlays de defectos en una imagen de cada categoría."""
    muestras = [
        ("datos/good/020.png", "good"),
        ("datos/scratch_head/004.png", "scratch_head"),
        ("datos/scratch_neck/003.png", "scratch_neck"),
        ("datos/thread_side/013.png", "thread_side"),
        ("datos/thread_top/006.png", "thread_top"),
        ("datos/manipulated_front/003.png", "manipulated_front"),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(18, 7))
    for ax, (ruta, nom) in zip(axs.ravel(), muestras):
        r = procesar_imagen(ruta, mascara_plantilla=P["binaria"])
        insp = inspeccionar(r["gris_alineada"], r["mascara_alineada"], P)
        v = clasificar(insp["scores"], U)
        ax.imshow(construir_overlay(r["gris_alineada"], insp))
        color = "red" if v["diagnostico"] == "Anómala" else "green"
        ax.set_title(f"{nom}: {v['diagnostico']} (nivel={v['nivel']:.1f})",
                     color=color, fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(SALIDA, "detecciones.png"), dpi=95, bbox_inches="tight")
    plt.close()


def fig_plantilla():
    """Modelo estadístico: media de máscaras, zonas y desviación del gris."""
    media = np.load("plantillas/media_mascara.npy")
    nucleo = cv2.imread("plantillas/nucleo.png", cv2.IMREAD_GRAYSCALE)
    exterior = cv2.imread("plantillas/exterior.png", cv2.IMREAD_GRAYSCALE)
    std = np.load("plantillas/std_gris.npy")

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    im0 = axs[0].imshow(media, cmap="viridis")
    axs[0].set_title("(a) Media de mascaras (n=41)")
    plt.colorbar(im0, ax=axs[0], fraction=0.046)
    axs[1].imshow(nucleo, cmap="gray"); axs[1].set_title("(b) Nucleo (>=97%)")
    axs[2].imshow(exterior, cmap="gray"); axs[2].set_title("(c) Exterior (<=3%)")
    im3 = axs[3].imshow(std, cmap="hot"); axs[3].set_title("(d) Desv. estandar del gris")
    plt.colorbar(im3, ax=axs[3], fraction=0.046)
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(SALIDA, "plantilla.png"), dpi=110, bbox_inches="tight")
    plt.close()


def fig_perfil_rosca():
    """Envolvente de crestas de un thread defectuoso vs. la banda de los sanos."""
    ruta = "datos/thread_side/020.png"
    r = procesar_imagen(ruta, mascara_plantilla=P["binaria"])
    perf = perfiles_envolvente(r["mascara_alineada"])

    x0, x1 = COLS_PERFIL
    xs = np.arange(x0, x1)
    bandas = P["bandas_perfil"]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.fill_between(
        xs,
        bandas["cresta_sup_min"][x0:x1] - MARGEN_PERFIL,
        bandas["cresta_sup_max"][x0:x1] + MARGEN_PERFIL,
        color="green", alpha=0.2, label="Banda admisible (41 sanos)",
    )
    ax.plot(xs, perf["cresta_sup"][x0:x1], color="red", lw=1.5,
            label="Envolvente de la pieza")
    ax.axvline(REGIONES["rosca"][1], color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Columna (eje del tornillo)")
    ax.set_ylabel("Radio cresta superior (px)")
    ax.set_title("Detector de perfil: envolvente fuera de banda = rosca danada")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SALIDA, "perfil_rosca.png"), dpi=110, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    fig_pipeline_etapas()
    fig_detecciones()
    fig_plantilla()
    fig_perfil_rosca()
    print(f"Figuras generadas en {SALIDA}")
