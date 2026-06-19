"""
entrenar.py - Prepara el sistema a partir de los tornillos sanos.

Hace dos cosas que en la version original estaban en build_template.py y
calibrar.py por separado:
    1. Construye la plantilla estadistica (como es un tornillo sano).
    2. Calibra los umbrales de decision (cuanto defecto es "demasiado").

Se ejecuta una sola vez, antes de inspeccionar nada:
    python entrenar.py
"""

import os
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt

import tornillos as t

CARPETA_GOOD = os.path.join(t.CARPETA_DATOS, "good")

# Umbral = max(media + 3*desvio, 1.25*peor_caso, piso). El piso evita umbral 0
# cuando un score dio siempre 0 en los sanos. Hay dos pisos segun la unidad.
PISO_AREA = 0.002        # para los scores de forma/intensidad (fraccion de area)
PISO_PERFIL = 0.10       # para los scores de perfil (px de exceso por columna)


# ---------------------------------------------------------------------------
# Parte 1: construir la plantilla
# ---------------------------------------------------------------------------

def cargar_good_alineadas():
    """Alinea las 41 fotos sanas y deja todas con la misma orientacion (2 pasadas)."""
    grises, mascaras = [], []
    suma = None                                          # promedio acumulado de mascaras

    # Pasada 1: cada foto se orienta contra el promedio de las que ya entraron
    for nombre in sorted(os.listdir(CARPETA_GOOD)):
        if not nombre.endswith(".png"):
            continue
        r = t.procesar_imagen(os.path.join(CARPETA_GOOD, nombre))
        gris, masc = r["gris_alineada"], r["mascara_alineada"]
        if suma is None:                                 # la primera fija la referencia
            suma = (masc > 0).astype(np.float32)
        else:
            promedio = (suma / len(mascaras) > 0.5).astype(np.uint8) * 255
            gris, masc, _ = t.corregir_flip_vertical(gris, masc, promedio)
            suma += (masc > 0).astype(np.float32)
        grises.append(gris)
        mascaras.append(masc)
        print(f"  {nombre}: angulo={r['angulo']:7.2f} grados")

    # Pasada 2: con la plantilla ya completa, re-decidimos el flip de TODAS.
    # Hace falta para que sean consistentes con como se inspecciona despues.
    plantilla_v1 = (np.mean([(m > 0) for m in mascaras], axis=0) > 0.5).astype(np.uint8) * 255
    corregidos = 0
    for i in range(len(mascaras)):
        gris, masc, giro = t.corregir_flip_vertical(grises[i], mascaras[i], plantilla_v1)
        grises[i], mascaras[i] = gris, masc
        corregidos += int(giro)
    print(f"  Pasada 2: {corregidos} imagen(es) reorientada(s)")
    return grises, mascaras


def construir_plantillas(grises, mascaras):
    """Calcula todo lo que define al tornillo sano: zonas, gris promedio, bandas."""
    pila_masc = np.stack([(m > 0).astype(np.float32) for m in mascaras])
    pila_gris = np.stack(grises).astype(np.float32)

    # Para cada pixel: que fraccion de los sanos tiene metal ahi
    media_mascara = pila_masc.mean(axis=0)
    nucleo = (media_mascara >= t.UMBRAL_NUCLEO).astype(np.uint8) * 255     # casi siempre metal
    exterior = (media_mascara <= t.UMBRAL_EXTERIOR).astype(np.uint8) * 255 # casi nunca metal
    binaria = (media_mascara > 0.5).astype(np.uint8) * 255                 # silueta tipica

    # Gris promedio y cuanto varia (esto alimenta al detector de intensidad)
    media_gris_f = pila_gris.mean(axis=0)
    std_gris = pila_gris.std(axis=0)

    # Bandas [min, max] de las envolventes de la rosca en los sanos
    perfiles_buenos = [t.perfiles_envolvente(m) for m in mascaras]
    bandas = {}
    for senal in ("cresta_sup", "valle_sup", "cresta_inf", "valle_inf"):
        pila = np.stack([p[senal] for p in perfiles_buenos])
        bandas[f"{senal}_min"] = pila.min(axis=0).astype(np.float32)
        bandas[f"{senal}_max"] = pila.max(axis=0).astype(np.float32)

    return {"media_mascara": media_mascara.astype(np.float32), "nucleo": nucleo,
            "exterior": exterior, "binaria": binaria,
            "media_gris": media_gris_f.astype(np.uint8), "media_gris_f": media_gris_f,
            "std_gris": std_gris, "bandas_perfil": bandas, "n": len(mascaras)}


def guardar_plantillas(p):
    """Guarda todo en la carpeta plantillas/."""
    os.makedirs(t.CARPETA_PLANTILLAS, exist_ok=True)
    d = t.CARPETA_PLANTILLAS
    np.save(os.path.join(d, "media_mascara.npy"), p["media_mascara"])
    np.save(os.path.join(d, "media_gris.npy"), p["media_gris_f"].astype(np.float32))
    np.save(os.path.join(d, "std_gris.npy"), p["std_gris"].astype(np.float32))
    np.savez(os.path.join(d, "bandas_perfil.npz"), **p["bandas_perfil"])
    cv2.imwrite(os.path.join(d, "nucleo.png"), p["nucleo"])
    cv2.imwrite(os.path.join(d, "exterior.png"), p["exterior"])
    cv2.imwrite(os.path.join(d, "mascara_binaria.png"), p["binaria"])
    cv2.imwrite(os.path.join(d, "media_gris.png"), p["media_gris"])
    std_vis = cv2.normalize(p["std_gris"], None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(d, "std_gris.png"), std_vis.astype(np.uint8))
    print(f"Plantillas guardadas en '{d}/'")


# ---------------------------------------------------------------------------
# Parte 2: calibrar los umbrales
# ---------------------------------------------------------------------------

def calibrar(plantillas):
    """Corre las 41 sanas y fija cada umbral con la regla de las 3 sigmas."""
    todos = []                                           # scores de cada foto sana
    for nombre in sorted(a for a in os.listdir(CARPETA_GOOD) if a.endswith(".png")):
        r = t.procesar_imagen(os.path.join(CARPETA_GOOD, nombre),
                              mascara_plantilla=plantillas["binaria"])
        insp = t.inspeccionar(r["gris_alineada"], r["mascara_alineada"], plantillas)
        todos.append(insp["scores"])

    umbrales, estadisticas = {}, {}
    print(f"\n{'score':<15}{'media':>10}{'desvio':>10}{'maximo':>10}{'umbral':>10}")
    for k in todos[0].keys():
        valores = np.array([s[k] for s in todos])
        media, desvio, maximo = valores.mean(), valores.std(), valores.max()
        piso = PISO_PERFIL if k.startswith("perfil") else PISO_AREA
        umbral = max(media + 3 * desvio, 1.25 * maximo, piso)
        umbrales[k] = float(umbral)
        estadisticas[k] = {"media": float(media), "desvio": float(desvio),
                           "maximo": float(maximo), "n": len(valores)}
        print(f"{k:<15}{media:>10.5f}{desvio:>10.5f}{maximo:>10.5f}{umbral:>10.5f}")

    with open(t.RUTA_UMBRALES, "w") as f:
        json.dump({"umbrales": umbrales, "estadisticas": estadisticas}, f, indent=2)
    print(f"Umbrales guardados en {t.RUTA_UMBRALES}")


if __name__ == "__main__":
    print("Construyendo la plantilla con los tornillos sanos...")
    grises, mascaras = cargar_good_alineadas()
    plantillas = construir_plantillas(grises, mascaras)
    guardar_plantillas(plantillas)

    print("\nCalibrando los umbrales...")
    calibrar(plantillas)
    print("\nListo. Ya podes inspeccionar imagenes con: python app.py <imagen>")
