"""
build_template.py — Construcción de la plantilla estadística ("golden template").

Procesa las 41 imágenes SIN defecto (carpeta datos/good) con las fases 1-2
del pipeline y acumula sus máscaras alineadas para construir un modelo
estadístico de "cómo es un tornillo sano":

    media_mascara(x, y) = fracción de tornillos sanos que tienen
                          metal en el píxel (x, y)   ∈ [0, 1]

De esa media se derivan las dos zonas que usa el detector de forma:

    NÚCLEO    = píxeles con media >= UMBRAL_NUCLEO  (casi SIEMPRE hay metal)
                -> si a un tornillo le FALTA material acá, es anómalo.
    EXTERIOR  = píxeles con media <= UMBRAL_EXTERIOR (casi NUNCA hay metal)
                -> si a un tornillo le SOBRA material acá, es anómalo.

    La franja intermedia (0.03 < media < 0.97) es la BANDA DE TOLERANCIA:
    allí los tornillos sanos a veces tienen metal y a veces no (los dientes
    de la rosca cambian de fase según cuánto esté girado el tornillo sobre
    su propio eje), así que esa zona NO se usa para decidir.

Además guarda la imagen de gris promedio y el mapa de desviación estándar,
útiles para ilustrar la variabilidad del dataset en el informe.

Salidas (carpeta plantillas/):
    media_mascara.npy   media de las máscaras alineadas (float32 0-1)
    nucleo.png          zona núcleo binaria {0,255}
    exterior.png        zona exterior binaria {0,255}
    mascara_binaria.png plantilla binaria (media > 0.5), para corregir flips
    media_gris.png      promedio de las imágenes de gris alineadas
    std_gris.png        desviación estándar del gris (mapa de variabilidad)
    plantilla_info.json metadatos (cantidad de imágenes, umbrales usados)

Uso:
    python build_template.py
"""

import json
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

from pipeline import procesar_imagen, corregir_flip_vertical
from detectores import perfiles_envolvente

# ----------------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------------
CARPETA_GOOD = os.path.join("datos", "good")
CARPETA_SALIDA = "plantillas"

# Umbrales de consenso para definir las zonas estadísticas. Con 41 imágenes,
# 0.97 equivale a exigir que 40 de 41 tornillos sanos tengan metal en el
# píxel, y 0.03 a que como máximo 1 de 41 lo tenga.
UMBRAL_NUCLEO = 0.97
UMBRAL_EXTERIOR = 0.03


def cargar_good_alineadas():
    """Procesa todas las imágenes 'good' y resuelve el espejo vertical.

    El espejo vertical (¿el tornillo quedó "boca arriba" o "boca abajo"?)
    es ambiguo imagen por imagen, así que se resuelve por consenso en DOS
    pasadas:

      Pasada 1: la primera imagen fija la orientación de referencia y cada
        imagen siguiente se compara contra el PROMEDIO ACUMULADO de las
        máscaras ya aceptadas (versión normal vs. espejada, gana la de
        mayor superposición). El promedio acumulado hace la decisión cada
        vez más estable a medida que se suman imágenes.

      Pasada 2: con la plantilla provisoria COMPLETA, se vuelve a decidir
        el flip de TODAS las imágenes contra ella. Esto es imprescindible
        para la consistencia: durante la inspección (calibrar/evaluar) el
        flip se decide contra la plantilla final, y si acá quedara alguna
        imagen decidida contra un promedio parcial distinto, esa imagen
        "sana" violaría las bandas construidas con ella misma.

    Devuelve:
        (lista_grises, lista_mascaras): imágenes y máscaras alineadas, todas
        con la misma orientación vertical.
    """
    grises, mascaras = [], []
    suma = None  # acumulador float de las máscaras aceptadas

    # --- Pasada 1: orientación por consenso acumulado ----------------------
    archivos = sorted(os.listdir(CARPETA_GOOD))
    for nombre in archivos:
        if not nombre.endswith(".png"):
            continue
        r = procesar_imagen(os.path.join(CARPETA_GOOD, nombre))
        gris, masc = r["gris_alineada"], r["mascara_alineada"]

        if suma is None:
            # La primera imagen define la orientación de referencia
            suma = (masc > 0).astype(np.float32)
        else:
            # Plantilla provisoria = promedio acumulado binarizado al 50%
            promedio = (suma / len(mascaras) > 0.5).astype(np.uint8) * 255
            gris, masc, _ = corregir_flip_vertical(gris, masc, promedio)
            suma += (masc > 0).astype(np.float32)

        grises.append(gris)
        mascaras.append(masc)
        print(f"  {nombre}: angulo={r['angulo']:7.2f} grados")

    # --- Pasada 2: re-decidir todos los flips contra la plantilla final ----
    plantilla_v1 = (
        np.mean([(m > 0) for m in mascaras], axis=0) > 0.5
    ).astype(np.uint8) * 255

    corregidos = 0
    for i in range(len(mascaras)):
        gris, masc, giro = corregir_flip_vertical(
            grises[i], mascaras[i], plantilla_v1
        )
        grises[i], mascaras[i] = gris, masc
        corregidos += int(giro)
    print(f"  Pasada 2: {corregidos} imagen(es) reorientada(s) contra la "
          f"plantilla final")

    return grises, mascaras


def construir_plantillas(grises, mascaras):
    """Acumula las máscaras y deriva las zonas estadísticas.

    Devuelve un diccionario con todos los productos de la plantilla.
    """
    # Pila (N, H, W) de máscaras en {0,1} y de grises
    pila_masc = np.stack([(m > 0).astype(np.float32) for m in mascaras])
    pila_gris = np.stack(grises).astype(np.float32)

    # Media píxel a píxel: fracción de tornillos sanos con metal en cada punto
    media_mascara = pila_masc.mean(axis=0)

    # Zonas de decisión (ver docstring del módulo)
    nucleo = (media_mascara >= UMBRAL_NUCLEO).astype(np.uint8) * 255
    exterior = (media_mascara <= UMBRAL_EXTERIOR).astype(np.uint8) * 255

    # Plantilla binaria al 50%: la silueta "típica", usada para orientar
    # (corregir flips) las imágenes nuevas durante la inspección
    binaria = (media_mascara > 0.5).astype(np.uint8) * 255

    # Estadísticos del gris: promedio (tornillo "ideal") y desviación
    # estándar (dónde varía la apariencia entre tornillos sanos). El
    # detector de intensidad compara cada pieza nueva contra estos dos
    # mapas: las zonas de std alta (brillos del cuello, dientes de rosca
    # que cambian de fase) quedan automáticamente toleradas.
    media_gris_f = pila_gris.mean(axis=0)
    std_gris = pila_gris.std(axis=0)

    # ------------------------------------------------------------------
    # Bandas de los perfiles de envolvente (detector de perfil de rosca):
    # para cada una de las 4 señales (cresta/valle, superior/inferior) se
    # registra el mínimo y el máximo que toma en los tornillos sanos,
    # columna a columna. Una pieza nueva cuyo perfil escape de la banda
    # [min, max] (+ margen) tiene la rosca o la punta deformada.
    # ------------------------------------------------------------------
    perfiles_buenos = [perfiles_envolvente(m) for m in mascaras]
    bandas = {}
    for senal in ("cresta_sup", "valle_sup", "cresta_inf", "valle_inf"):
        pila = np.stack([p[senal] for p in perfiles_buenos])
        bandas[f"{senal}_min"] = pila.min(axis=0).astype(np.float32)
        bandas[f"{senal}_max"] = pila.max(axis=0).astype(np.float32)

    return {
        "media_mascara": media_mascara.astype(np.float32),
        "nucleo": nucleo,
        "exterior": exterior,
        "binaria": binaria,
        "media_gris": media_gris_f.astype(np.uint8),
        "media_gris_f": media_gris_f,
        "std_gris": std_gris,
        "bandas_perfil": bandas,
        "n": len(mascaras),
    }


def guardar_plantillas(p):
    """Persiste las plantillas en la carpeta plantillas/."""
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    np.save(os.path.join(CARPETA_SALIDA, "media_mascara.npy"), p["media_mascara"])
    # Media y desviación del gris en float32 SIN cuantizar: el detector de
    # intensidad las usa para el mapa z y una versión redondeada a 8 bits
    # perdería precisión justo donde la desviación es chica
    np.save(os.path.join(CARPETA_SALIDA, "media_gris.npy"),
            p["media_gris_f"].astype(np.float32))
    np.save(os.path.join(CARPETA_SALIDA, "std_gris.npy"),
            p["std_gris"].astype(np.float32))
    # Bandas del detector de perfil de rosca (8 arrays en un solo .npz)
    np.savez(os.path.join(CARPETA_SALIDA, "bandas_perfil.npz"),
             **p["bandas_perfil"])
    cv2.imwrite(os.path.join(CARPETA_SALIDA, "nucleo.png"), p["nucleo"])
    cv2.imwrite(os.path.join(CARPETA_SALIDA, "exterior.png"), p["exterior"])
    cv2.imwrite(os.path.join(CARPETA_SALIDA, "mascara_binaria.png"), p["binaria"])
    cv2.imwrite(os.path.join(CARPETA_SALIDA, "media_gris.png"), p["media_gris"])
    # La std se reescala a 0-255 solo para poder guardarla como imagen
    std_vis = cv2.normalize(p["std_gris"], None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(CARPETA_SALIDA, "std_gris.png"), std_vis.astype(np.uint8))

    with open(os.path.join(CARPETA_SALIDA, "plantilla_info.json"), "w") as f:
        json.dump(
            {
                "imagenes_usadas": p["n"],
                "umbral_nucleo": UMBRAL_NUCLEO,
                "umbral_exterior": UMBRAL_EXTERIOR,
            },
            f,
            indent=2,
        )
    print(f"Plantillas guardadas en '{CARPETA_SALIDA}/'")


def visualizar(p):
    """Figura resumen de la plantilla construida (se guarda como PNG)."""
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    im0 = axs[0, 0].imshow(p["media_mascara"], cmap="viridis")
    axs[0, 0].set_title(f"Media de máscaras (n={p['n']})")
    plt.colorbar(im0, ax=axs[0, 0], fraction=0.046)

    axs[0, 1].imshow(p["nucleo"], cmap="gray")
    axs[0, 1].set_title(f"Núcleo (media >= {UMBRAL_NUCLEO})")

    axs[0, 2].imshow(p["exterior"], cmap="gray")
    axs[0, 2].set_title(f"Exterior (media <= {UMBRAL_EXTERIOR})")

    axs[1, 0].imshow(p["binaria"], cmap="gray")
    axs[1, 0].set_title("Plantilla binaria (50%)")

    axs[1, 1].imshow(p["media_gris"], cmap="gray")
    axs[1, 1].set_title("Gris promedio (golden template)")

    im5 = axs[1, 2].imshow(p["std_gris"], cmap="hot")
    axs[1, 2].set_title("Desv. estándar del gris")
    plt.colorbar(im5, ax=axs[1, 2], fraction=0.046)

    for fila in axs:
        for ax in fila:
            ax.axis("off")
    plt.tight_layout()
    ruta = os.path.join(CARPETA_SALIDA, "resumen_plantilla.png")
    plt.savefig(ruta, dpi=80)
    print(f"Figura resumen: {ruta}")


if __name__ == "__main__":
    print("Construyendo plantilla a partir de", CARPETA_GOOD)
    grises, mascaras = cargar_good_alineadas()
    plantilla = construir_plantillas(grises, mascaras)
    guardar_plantillas(plantilla)
    visualizar(plantilla)

    # Reporte rápido de consistencia: qué fracción del área típica del
    # tornillo quedó como núcleo. Si fuera baja, la alineación sería mala.
    area_bin = np.count_nonzero(plantilla["binaria"])
    area_nuc = np.count_nonzero(plantilla["nucleo"])
    print(f"Área plantilla binaria: {area_bin} px | núcleo: {area_nuc} px "
          f"({100 * area_nuc / max(area_bin, 1):.1f}% del área típica)")
