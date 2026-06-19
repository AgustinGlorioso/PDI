"""
app.py - Usa el sistema ya entrenado.

Dos modos (junta lo que antes estaba en main.py y evaluar.py):

    python app.py <imagen>     -> inspecciona una imagen y muestra la figura
    python app.py              -> evalua TODO el dataset y arma las metricas

Requiere haber corrido antes:  python entrenar.py
"""

import os
import sys
import csv
from collections import defaultdict

import cv2
import numpy as np
import matplotlib.pyplot as plt

import tornillos as t

# Colores (RGB) para pintar el defecto segun que detector lo encontro
COLOR_FORMA = (255, 60, 60)        # rojo:     defecto de forma (silueta)
COLOR_INTENSIDAD = (255, 220, 0)   # amarillo: defecto de intensidad (gris)
COLOR_PERFIL = (255, 0, 255)       # magenta:  defecto de perfil de rosca

# Que tipo de defecto esperamos segun la carpeta (para evaluar)
TIPO_ESPERADO = {"good": "good", "scratch_head": "scratch_head",
                 "scratch_neck": "scratch_neck", "thread_side": "thread",
                 "thread_top": "thread", "manipulated_front": "manipulated_front"}


# ---------------------------------------------------------------------------
# Modo 1: inspeccionar una imagen
# ---------------------------------------------------------------------------

def construir_overlay(gris_alineada, insp):
    """Pinta los defectos sobre la imagen, con un color por detector."""
    overlay = cv2.cvtColor(gris_alineada, cv2.COLOR_GRAY2RGB)
    defecto_forma = cv2.bitwise_or(insp["forma"]["faltante"], insp["forma"]["sobrante"])
    overlay[defecto_forma > 0] = COLOR_FORMA
    defecto_int = cv2.bitwise_or(insp["intensidad"]["brillante"], insp["intensidad"]["oscuro"])
    overlay[defecto_int > 0] = COLOR_INTENSIDAD
    overlay[insp["perfil"]["marcas"] > 0] = COLOR_PERFIL
    return overlay


def inspeccionar_imagen(ruta):
    """Procesa una imagen, imprime el diagnostico y muestra la figura de 4 etapas."""
    plantillas = t.cargar_plantillas()
    umbrales = t.cargar_umbrales()

    r = t.procesar_imagen(ruta, mascara_plantilla=plantillas["binaria"])
    insp = t.inspeccionar(r["gris_alineada"], r["mascara_alineada"], plantillas)
    veredicto = t.clasificar(insp["scores"], umbrales)

    # Reporte por consola
    print(f"\nDiagnostico : {veredicto['diagnostico']}")
    if veredicto["diagnostico"] == "Anómala":
        print(f"Tipo        : {veredicto['tipo']}")
    print(f"Nivel       : {veredicto['nivel']:.2f}  (anomala si supera 1.00)")
    print("\nScores (score / umbral = cuanto se paso):")
    for k, v in sorted(veredicto["excesos"].items(), key=lambda kv: -kv[1]):
        marca = "  <-- DISPARO" if v > 1.0 else ""
        print(f"  {k:<15} {insp['scores'][k]:.5f} / {umbrales[k]:.5f} = {v:5.2f}{marca}")

    # Figura con las 4 etapas
    es_anomala = veredicto["diagnostico"] == "Anómala"
    fig, axs = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle(f"{ruta}  -  {veredicto['diagnostico']}"
                 + (f" ({veredicto['tipo']})" if es_anomala else "")
                 + f"  |  Nivel: {veredicto['nivel']:.2f}",
                 fontsize=15, fontweight="bold", color="red" if es_anomala else "green")
    axs[0].imshow(r["rgb"]); axs[0].set_title("1. Original")
    axs[1].imshow(r["mascara"], cmap="gray"); axs[1].set_title("2. Segmentacion")
    axs[2].imshow(r["gris_alineada"], cmap="gray")
    for nombre, (x0, x1) in t.REGIONES.items():
        axs[2].axvline(x1, color="cyan", lw=0.8, ls="--")
        axs[2].text((x0 + x1) / 2, 990, nombre, color="cyan", ha="center", fontsize=9)
    axs[2].set_title("3. Alineacion + regiones")
    axs[3].imshow(construir_overlay(r["gris_alineada"], insp))
    axs[3].set_title("4. Defectos (rojo=forma, amarillo=intensidad, magenta=perfil)")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Modo 2: evaluar todo el dataset
# ---------------------------------------------------------------------------

def evaluar():
    """Corre el sistema sobre las 160 imagenes y compara con la verdad conocida."""
    plantillas = t.cargar_plantillas()
    umbrales = t.cargar_umbrales()

    resultados, errores = [], []
    binario = defaultdict(lambda: [0, 0])                # aciertos/total por categoria
    confusion = defaultdict(lambda: defaultdict(int))    # matriz de tipos

    for categoria, tipo_esp in TIPO_ESPERADO.items():
        carpeta = os.path.join(t.CARPETA_DATOS, categoria)
        for nombre in sorted(a for a in os.listdir(carpeta) if a.endswith(".png")):
            ruta = os.path.join(carpeta, nombre)
            r = t.procesar_imagen(ruta, mascara_plantilla=plantillas["binaria"])
            insp = t.inspeccionar(r["gris_alineada"], r["mascara_alineada"], plantillas)
            v = t.clasificar(insp["scores"], umbrales)

            # acerto el binario? (good=normal, el resto=anomala)
            acierto = (categoria != "good") == (v["diagnostico"] == "Anómala")
            binario[categoria][0] += int(acierto)
            binario[categoria][1] += 1
            confusion[tipo_esp][v["tipo"]] += 1
            if not acierto:
                errores.append((ruta, v))
            resultados.append({"imagen": ruta, "categoria": categoria, "esperado": tipo_esp,
                               "diagnostico": v["diagnostico"], "tipo_predicho": v["tipo"],
                               "nivel": round(v["nivel"], 3)})

    # Exactitud binaria
    print("\n=== EXACTITUD BINARIA (Normal vs Anomala) ===")
    tot_ok = tot_n = 0
    for cat, (ok, n) in binario.items():
        tot_ok += ok; tot_n += n
        print(f"  {cat:<20} {ok:>3}/{n:<3}  ({100 * ok / n:5.1f}%)")
    print(f"  {'GLOBAL':<20} {tot_ok:>3}/{tot_n:<3}  ({100 * tot_ok / tot_n:5.1f}%)")

    # Matriz de confusion del tipo
    tipos = ["good", "scratch_head", "scratch_neck", "thread", "manipulated_front"]
    print("\n=== MATRIZ DE CONFUSION DE TIPO (filas=real, columnas=predicho) ===")
    print(f"{'':<20}" + "".join(f"{t_[:12]:>14}" for t_ in tipos))
    for esp in tipos:
        print(f"{esp:<20}" + "".join(f"{confusion[esp].get(p, 0):>14}" for p in tipos))

    # Imagenes mal clasificadas
    if errores:
        print(f"\n=== MAL CLASIFICADAS ({len(errores)}) ===")
        for ruta, v in errores:
            print(f"  {ruta:<45} nivel={v['nivel']:.2f}")

    # CSV con el detalle
    salida_csv = os.path.join(t._AQUI, "resultados.csv")
    with open(salida_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=resultados[0].keys())
        w.writeheader()
        w.writerows(resultados)
    print(f"\nDetalle por imagen en {salida_csv}")


if __name__ == "__main__":
    if len(sys.argv) > 1:                                # paso una imagen -> inspeccionar
        inspeccionar_imagen(sys.argv[1])
    else:                                                # sin argumentos -> evaluar todo
        print("Evaluando todo el dataset...")
        evaluar()
