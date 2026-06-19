"""
entrenar.py - Aprende como es un tornillo sano y fija los umbrales.

1. Promedia los 41 tornillos sanos -> plantilla (gris promedio, zonas, bandas).
2. Corre los 3 detectores sobre los sanos y fija cada umbral con una regla
   simple: "mas grande que el peor tornillo sano" (max + un margen chico).

Se corre una sola vez:  python entrenar.py
"""

import os
import json

import cv2
import numpy as np

import tornillos as t

CARPETA_GOOD = os.path.join(t.CARPETA_DATOS, "good")

# Margen que se suma al peor sano para fijar el umbral de cada detector
MARGEN_INT = 20
MARGEN_FORMA = 50
MARGEN_PERFIL = 20


def cargar_sanos_alineados():
    # Alinea los 41 sanos y los deja todos con la misma orientacion (2 pasadas).
    grises, mascaras, suma = [], [], None
    for nombre in sorted(a for a in os.listdir(CARPETA_GOOD) if a.endswith(".png")):
        r = t.procesar_imagen(os.path.join(CARPETA_GOOD, nombre))
        gris, masc = r["gris_alineada"], r["mascara_alineada"]
        if suma is None:
            suma = (masc > 0).astype(np.float32)
        else:
            prov = (suma / len(mascaras) > 0.5).astype(np.uint8) * 255
            gris, masc = t.corregir_flip(gris, masc, prov)
            suma += (masc > 0).astype(np.float32)
        grises.append(gris); mascaras.append(masc)

    # Segunda pasada: re-orientar todos contra la plantilla ya completa
    plantilla = (np.mean([(m > 0) for m in mascaras], axis=0) > 0.5).astype(np.uint8) * 255
    for i in range(len(mascaras)):
        grises[i], mascaras[i] = t.corregir_flip(grises[i], mascaras[i], plantilla)
    return grises, mascaras


def construir_plantilla(grises, mascaras):
    # Calcula todo lo que define al tornillo sano.
    pm = np.stack([(m > 0).astype(np.float32) for m in mascaras])
    pg = np.stack(grises).astype(np.float32)
    media_mascara = pm.mean(axis=0)

    P = {"nucleo": (media_mascara >= 0.97).astype(np.uint8) * 255,     # casi siempre metal
         "exterior": (media_mascara <= 0.03).astype(np.uint8) * 255,   # casi nunca metal
         "binaria": (media_mascara > 0.5).astype(np.uint8) * 255,      # silueta tipica
         "media_gris": pg.mean(axis=0), "std_gris": pg.std(axis=0)}

    # Bandas [min, max] de las envolventes de la rosca en los sanos
    bandas = {}
    perfiles = [_envolventes(m) for m in mascaras]
    for s in ("cresta_sup", "valle_sup", "cresta_inf", "valle_inf"):
        pila = np.stack([p[s] for p in perfiles])
        bandas[f"{s}_min"] = pila.min(axis=0).astype(np.float32)
        bandas[f"{s}_max"] = pila.max(axis=0).astype(np.float32)
    P["bandas"] = bandas
    return P


def _envolventes(masc):
    # Igual que en el detector de perfil: cresta y valle por columna.
    b = masc > 0
    h, w = b.shape
    hay = b.any(axis=0)
    sup = np.where(hay, t.FILA_EJE - b.argmax(axis=0), 0.0).astype(np.float32)
    inf = np.where(hay, (h - 1 - b[::-1].argmax(axis=0)) - t.FILA_EJE, 0.0).astype(np.float32)
    ee = cv2.getStructuringElement(cv2.MORPH_RECT, (63, 1))
    return {"cresta_sup": cv2.dilate(sup.reshape(1, -1), ee).ravel(),
            "valle_sup": cv2.erode(sup.reshape(1, -1), ee).ravel(),
            "cresta_inf": cv2.dilate(inf.reshape(1, -1), ee).ravel(),
            "valle_inf": cv2.erode(inf.reshape(1, -1), ee).ravel()}


def guardar_plantilla(P):
    os.makedirs(t.CARPETA_PLANTILLAS, exist_ok=True)
    d = t.CARPETA_PLANTILLAS
    cv2.imwrite(os.path.join(d, "nucleo.png"), P["nucleo"])
    cv2.imwrite(os.path.join(d, "exterior.png"), P["exterior"])
    cv2.imwrite(os.path.join(d, "binaria.png"), P["binaria"])
    np.save(os.path.join(d, "media_gris.npy"), P["media_gris"].astype(np.float32))
    np.save(os.path.join(d, "std_gris.npy"), P["std_gris"].astype(np.float32))
    np.savez(os.path.join(d, "bandas.npz"), **P["bandas"])


def calibrar(P, mascaras, grises):
    # Corre los detectores sobre los sanos y fija el umbral = peor sano + margen.
    s_int, s_for, s_per = [], [], []
    for gris, masc in zip(grises, mascaras):
        a = t.analizar(gris, masc, P)
        s_int.append(a["scores"]["intensidad"])
        s_for.append(a["scores"]["forma"])
        s_per.append(a["scores"]["perfil"])
    umbrales = {"intensidad": max(s_int) + MARGEN_INT,
                "forma": max(s_for) + MARGEN_FORMA,
                "perfil": max(s_per) + MARGEN_PERFIL}
    print(f"Peor sano -> int={max(s_int)}  forma={max(s_for)}  perfil={max(s_per)}")
    print(f"Umbrales  -> {umbrales}")
    with open(t.RUTA_UMBRALES, "w") as f:
        json.dump(umbrales, f, indent=2)


if __name__ == "__main__":
    print("Alineando los tornillos sanos...")
    grises, mascaras = cargar_sanos_alineados()
    print("Construyendo la plantilla...")
    P = construir_plantilla(grises, mascaras)
    guardar_plantilla(P)
    print("Calibrando umbrales...")
    calibrar(P, mascaras, grises)
    print("\nListo. Proba con: python app.py ../datos/scratch_head/000.png")
