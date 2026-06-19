"""
entrenar.py - Aprende como es un tornillo sano y fija los 2 umbrales.

1. Promedia los 41 sanos -> plantilla (gris promedio, zonas nucleo/exterior).
2. Corre los 2 detectores sobre los sanos y fija cada umbral como
   "mas grande que el peor tornillo sano" (max + margen).

Se corre una sola vez:  python entrenar.py
"""

import os
import json

import cv2
import numpy as np

import tornillos as t

CARPETA_GOOD = os.path.join(t.CARPETA_DATOS, "good")
MARGEN_INT = 20
MARGEN_FORMA = 50


def cargar_sanos_alineados():
    # Alinea los 41 sanos con la misma orientacion (2 pasadas).
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
    plantilla = (np.mean([(m > 0) for m in mascaras], axis=0) > 0.5).astype(np.uint8) * 255
    for i in range(len(mascaras)):
        grises[i], mascaras[i] = t.corregir_flip(grises[i], mascaras[i], plantilla)
    return grises, mascaras


def construir_plantilla(grises, mascaras):
    pm = np.stack([(m > 0).astype(np.float32) for m in mascaras])
    pg = np.stack(grises).astype(np.float32)
    media_mascara = pm.mean(axis=0)
    return {"nucleo": (media_mascara >= 0.97).astype(np.uint8) * 255,
            "exterior": (media_mascara <= 0.03).astype(np.uint8) * 255,
            "binaria": (media_mascara > 0.5).astype(np.uint8) * 255,
            "media_gris": pg.mean(axis=0), "std_gris": pg.std(axis=0)}


def guardar_plantilla(P):
    os.makedirs(t.CARPETA_PLANTILLAS, exist_ok=True)
    d = t.CARPETA_PLANTILLAS
    cv2.imwrite(os.path.join(d, "nucleo.png"), P["nucleo"])
    cv2.imwrite(os.path.join(d, "exterior.png"), P["exterior"])
    cv2.imwrite(os.path.join(d, "binaria.png"), P["binaria"])
    np.save(os.path.join(d, "media_gris.npy"), P["media_gris"].astype(np.float32))
    np.save(os.path.join(d, "std_gris.npy"), P["std_gris"].astype(np.float32))


def calibrar(P, grises, mascaras):
    s_int, s_for = [], []
    for gris, masc in zip(grises, mascaras):
        a = t.analizar(gris, masc, P)
        s_int.append(a["scores"]["intensidad"])
        s_for.append(a["scores"]["forma"])
    umbrales = {"intensidad": max(s_int) + MARGEN_INT, "forma": max(s_for) + MARGEN_FORMA}
    print(f"Peor sano -> int={max(s_int)}  forma={max(s_for)}")
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
    calibrar(P, grises, mascaras)
    print("\nListo. Proba con: python app.py ../datos/scratch_head/000.png")
