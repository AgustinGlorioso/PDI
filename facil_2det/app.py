"""
app.py - Usa el sistema ya entrenado.

    python app.py <imagen>   -> inspecciona una imagen y muestra la figura
    python app.py            -> evalua todo el dataset

Antes:  python entrenar.py
"""

import os
import sys
from collections import defaultdict

import cv2
import numpy as np
import matplotlib.pyplot as plt

import tornillos as t

COLORES = {"intensidad": (255, 220, 0), "forma": (255, 60, 60)}
TIPO_ESPERADO = {"good": "good", "scratch_head": "scratch_head", "scratch_neck": "scratch_neck",
                 "thread_side": "thread", "thread_top": "thread", "manipulated_front": "manipulated_front"}


def inspeccionar(ruta):
    P = t.cargar_plantillas()
    U = t.cargar_umbrales()
    r = t.procesar_imagen(ruta, plantilla=P["binaria"])
    a = t.analizar(r["gris_alineada"], r["mascara_alineada"], P)
    diag = t.diagnosticar(a, U)

    print(f"\nDiagnostico: {diag['diagnostico']}", end="")
    print(f" ({diag['tipo']})" if diag["diagnostico"] != "Normal" else "")
    print(f"Nivel: {diag['nivel']:.2f}  (anomala si supera 1.00)")
    for k in a["scores"]:
        print(f"  {k:<11} area={a['scores'][k]:<6} umbral={U[k]:<6} -> {a['scores'][k]/U[k]:.2f}")

    overlay = cv2.cvtColor(r["gris_alineada"], cv2.COLOR_GRAY2RGB)
    for k, color in COLORES.items():
        overlay[a[k] > 0] = color

    es_anom = diag["diagnostico"] != "Normal"
    fig, axs = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle(f"{os.path.basename(ruta)} -> {diag['diagnostico']}"
                 + (f" ({diag['tipo']})" if es_anom else "") + f"  nivel={diag['nivel']:.2f}",
                 fontsize=14, fontweight="bold", color="red" if es_anom else "green")
    axs[0].imshow(r["rgb"]); axs[0].set_title("Original")
    axs[1].imshow(r["mascara"], cmap="gray"); axs[1].set_title("Segmentacion")
    axs[2].imshow(overlay); axs[2].set_title("Defectos (amarillo=rayon, rojo=forma)")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def evaluar():
    P = t.cargar_plantillas()
    U = t.cargar_umbrales()
    binario = defaultdict(lambda: [0, 0])
    confusion = defaultdict(lambda: defaultdict(int))

    for cat, tipo_esp in TIPO_ESPERADO.items():
        carpeta = os.path.join(t.CARPETA_DATOS, cat)
        for nombre in sorted(a for a in os.listdir(carpeta) if a.endswith(".png")):
            r = t.procesar_imagen(os.path.join(carpeta, nombre), plantilla=P["binaria"])
            a = t.analizar(r["gris_alineada"], r["mascara_alineada"], P)
            diag = t.diagnosticar(a, U)
            acierto = (cat != "good") == (diag["diagnostico"] != "Normal")
            binario[cat][0] += int(acierto); binario[cat][1] += 1
            confusion[tipo_esp][diag["tipo"]] += 1

    print("\n=== EXACTITUD (Normal vs Anomala) ===")
    tot_ok = tot_n = 0
    for cat, (ok, n) in binario.items():
        tot_ok += ok; tot_n += n
        print(f"  {cat:<18} {ok:>2}/{n:<2} ({100*ok/n:5.1f}%)")
    print(f"  GLOBAL {tot_ok}/{tot_n} ({100*tot_ok/tot_n:.1f}%)")

    tipos = ["good", "scratch_head", "scratch_neck", "thread", "manipulated_front"]
    print("\n=== TIPO (filas=real, columnas=predicho) ===")
    print(f"{'':<18}" + "".join(f"{x[:10]:>12}" for x in tipos))
    for esp in tipos:
        print(f"{esp:<18}" + "".join(f"{confusion[esp].get(p, 0):>12}" for p in tipos))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspeccionar(sys.argv[1])
    else:
        print("Evaluando todo el dataset...")
        evaluar()
