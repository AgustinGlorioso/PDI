"""
evaluar.py — Evaluación del sistema completo sobre todo el dataset.

Procesa TODAS las imágenes de datos/ (las 6 categorías) con el pipeline
completo y compara el diagnóstico del sistema contra la verdad conocida
(la carpeta de origen de cada imagen):

    carpeta            diagnóstico esperado   tipo esperado
    ------------------------------------------------------------
    good               Normal                 good
    scratch_head       Anómala                scratch_head
    scratch_neck       Anómala                scratch_neck
    thread_side        Anómala                thread
    thread_top         Anómala                thread
    manipulated_front  Anómala                manipulated_front

(thread_side y thread_top se agrupan como "thread": ambas son roscas
dañadas y distinguir la dirección del daño no aporta a la decisión
industrial de aceptar/rechazar la pieza.)

Reporta:
  - Exactitud BINARIA (Normal/Anómala) por categoría y global, que es la
    métrica principal del sistema.
  - Matriz de confusión del TIPO de defecto.
  - Listado de las imágenes mal clasificadas con sus scores, para poder
    inspeccionarlas y ajustar parámetros.
  - resultados.csv con el detalle por imagen.

Uso:
    python evaluar.py
"""

import csv
import os
from collections import defaultdict

from pipeline import procesar_imagen
from detectores import inspeccionar
from clasificador import cargar_plantillas, cargar_umbrales, clasificar

# Verdad de terreno: tipo esperado según la carpeta de origen
TIPO_ESPERADO = {
    "good": "good",
    "scratch_head": "scratch_head",
    "scratch_neck": "scratch_neck",
    "thread_side": "thread",
    "thread_top": "thread",
    "manipulated_front": "manipulated_front",
}


def evaluar():
    """Corre el pipeline sobre todo el dataset y acumula métricas."""
    plantillas = cargar_plantillas()
    umbrales = cargar_umbrales()

    resultados = []          # detalle por imagen (para el CSV)
    errores_binarios = []    # imágenes con diagnóstico Normal/Anómala errado

    # aciertos binarios por categoría: {categoria: [aciertos, total]}
    binario = defaultdict(lambda: [0, 0])
    # matriz de confusión de tipos: {tipo_esperado: {tipo_predicho: n}}
    confusion = defaultdict(lambda: defaultdict(int))

    for categoria, tipo_esp in TIPO_ESPERADO.items():
        carpeta = os.path.join("datos", categoria)
        archivos = sorted(a for a in os.listdir(carpeta) if a.endswith(".png"))

        for nombre in archivos:
            ruta = os.path.join(carpeta, nombre)

            # Pipeline completo: fases 1-2 (+ corrección de flip) y 3-5
            r = procesar_imagen(ruta, mascara_plantilla=plantillas["binaria"])
            insp = inspeccionar(
                r["gris_alineada"], r["mascara_alineada"], plantillas
            )
            veredicto = clasificar(insp["scores"], umbrales)

            # ¿Acertó el diagnóstico binario?
            esperado_anomala = categoria != "good"
            predicho_anomala = veredicto["diagnostico"] == "Anómala"
            acierto = esperado_anomala == predicho_anomala

            binario[categoria][0] += int(acierto)
            binario[categoria][1] += 1
            confusion[tipo_esp][veredicto["tipo"]] += 1

            if not acierto:
                errores_binarios.append((ruta, veredicto))

            resultados.append({
                "imagen": ruta,
                "categoria": categoria,
                "esperado": tipo_esp,
                "diagnostico": veredicto["diagnostico"],
                "tipo_predicho": veredicto["tipo"],
                "nivel": round(veredicto["nivel"], 3),
                **{k: round(v, 5) for k, v in insp["scores"].items()},
            })

    # ------------------------------------------------------------------
    # Reporte: exactitud binaria
    # ------------------------------------------------------------------
    print("\n=== EXACTITUD BINARIA (Normal vs. Anómala) ===")
    total_ok = total_n = 0
    for categoria, (ok, n) in binario.items():
        total_ok += ok
        total_n += n
        print(f"  {categoria:<20} {ok:>3}/{n:<3}  ({100 * ok / n:5.1f}%)")
    print(f"  {'GLOBAL':<20} {total_ok:>3}/{total_n:<3}  ({100 * total_ok / total_n:5.1f}%)")

    # ------------------------------------------------------------------
    # Reporte: matriz de confusión de tipos
    # ------------------------------------------------------------------
    tipos = ["good", "scratch_head", "scratch_neck", "thread", "manipulated_front"]
    print("\n=== MATRIZ DE CONFUSIÓN DE TIPO (filas=esperado, columnas=predicho) ===")
    print(f"{'':<20}" + "".join(f"{t[:12]:>14}" for t in tipos))
    for esp in tipos:
        fila = "".join(f"{confusion[esp].get(pred, 0):>14}" for pred in tipos)
        print(f"{esp:<20}{fila}")

    # ------------------------------------------------------------------
    # Reporte: errores binarios con sus scores (para diagnóstico fino)
    # ------------------------------------------------------------------
    if errores_binarios:
        print(f"\n=== IMÁGENES MAL CLASIFICADAS ({len(errores_binarios)}) ===")
        for ruta, veredicto in errores_binarios:
            peores = sorted(
                veredicto["excesos"].items(), key=lambda kv: -kv[1]
            )[:3]
            detalle = "  ".join(f"{k}={v:.2f}" for k, v in peores)
            print(f"  {ruta:<45} nivel={veredicto['nivel']:.2f}  [{detalle}]")

    # ------------------------------------------------------------------
    # CSV con el detalle completo
    # ------------------------------------------------------------------
    with open("resultados.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=resultados[0].keys())
        w.writeheader()
        w.writerows(resultados)
    print("\nDetalle por imagen guardado en resultados.csv")

    return binario, confusion, errores_binarios


if __name__ == "__main__":
    print("Evaluando el sistema sobre todo el dataset ...")
    evaluar()
