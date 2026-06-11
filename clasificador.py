"""
clasificador.py — Fase 5: Clasificación y diagnóstico.

Toma el vector de 8 scores producido por detectores.calcular_scores() y lo
compara contra los umbrales calibrados estadísticamente con los tornillos
sanos (ver calibrar.py). La decisión es:

    score_k > umbral_k  para algún k   ==>   pieza ANÓMALA

y el TIPO de defecto se deduce de la región del score más excedido,
aprovechando que en este dataset cada tipo de defecto vive en una región
característica del tornillo:

    región del exceso máximo   ->  tipo de defecto
    ------------------------------------------------
    cabeza                     ->  scratch_head  (arañazo en la cabeza)
    cuello                     ->  scratch_neck  (raspón/gubia en el cuello)
    rosca                      ->  thread        (rosca deformada)
    punta                      ->  manipulated_front (punta dañada)

El "exceso" se mide en forma RELATIVA (score / umbral) para que scores de
regiones distintas sean comparables entre sí aunque sus umbrales difieran.

También expone las funciones para cargar/guardar plantillas y umbrales,
compartidas por calibrar.py, evaluar.py y main.py.
"""

import json
import os

import cv2
import numpy as np

CARPETA_PLANTILLAS = "plantillas"
RUTA_UMBRALES = os.path.join(CARPETA_PLANTILLAS, "umbrales.json")

# Mapa región -> etiqueta de tipo de defecto (ver docstring del módulo)
TIPO_POR_REGION = {
    "cabeza": "scratch_head",
    "cuello": "scratch_neck",
    "rosca": "thread",
    "punta": "manipulated_front",
}


def cargar_plantillas():
    """Carga las plantillas generadas por build_template.py.

    Devuelve:
        dict con 'nucleo', 'exterior', 'binaria' (uint8 {0,255}) y
        'media_gris' (uint8), listas para pasar a detectores.inspeccionar().
    """
    def _leer(nombre):
        ruta = os.path.join(CARPETA_PLANTILLAS, nombre)
        img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(
                f"Falta la plantilla '{ruta}'. Ejecutá primero build_template.py"
            )
        return img

    def _leer_npy(nombre):
        ruta = os.path.join(CARPETA_PLANTILLAS, nombre)
        if not os.path.exists(ruta):
            raise FileNotFoundError(
                f"Falta la plantilla '{ruta}'. Ejecutá primero build_template.py"
            )
        return np.load(ruta)

    return {
        "nucleo": _leer("nucleo.png"),
        "exterior": _leer("exterior.png"),
        "binaria": _leer("mascara_binaria.png"),
        "media_gris": _leer("media_gris.png"),
        # Estadísticos en float32 para el detector de intensidad (mapa z)
        "media_gris_f": _leer_npy("media_gris.npy"),
        "std_gris": _leer_npy("std_gris.npy"),
    }


def guardar_umbrales(umbrales, estadisticas):
    """Persiste los umbrales calibrados junto con las estadísticas base.

    Se guardan también la media y el desvío de cada score sobre las "good"
    para poder justificar/auditar los umbrales en el informe.
    """
    with open(RUTA_UMBRALES, "w") as f:
        json.dump({"umbrales": umbrales, "estadisticas": estadisticas}, f, indent=2)
    print(f"Umbrales guardados en {RUTA_UMBRALES}")


def cargar_umbrales():
    """Carga los umbrales calibrados por calibrar.py."""
    if not os.path.exists(RUTA_UMBRALES):
        raise FileNotFoundError(
            f"Falta '{RUTA_UMBRALES}'. Ejecutá primero calibrar.py"
        )
    with open(RUTA_UMBRALES) as f:
        return json.load(f)["umbrales"]


def clasificar(scores, umbrales):
    """Emite el diagnóstico final a partir de los scores y umbrales.

    Parámetros:
        scores:   dict {score: valor} de detectores.calcular_scores().
        umbrales: dict {score: umbral} de cargar_umbrales().

    Devuelve:
        dict con:
            'diagnostico': "Normal" o "Anómala"
            'tipo':        tipo de defecto estimado (o "good")
            'nivel':       nivel de anomalía global = max(score/umbral).
                           1.0 es el límite de decisión: <1 normal, >1
                           anómala, y cuanto mayor, más severo el defecto.
            'excesos':     dict {score: score/umbral} para inspección
    """
    # Exceso relativo de cada score respecto de su umbral calibrado
    excesos = {
        k: scores[k] / umbrales[k] if umbrales[k] > 0 else 0.0
        for k in scores
    }

    # Nivel de anomalía global: el peor exceso relativo
    score_max = max(excesos, key=excesos.get)
    nivel = excesos[score_max]

    if nivel <= 1.0:
        return {
            "diagnostico": "Normal",
            "tipo": "good",
            "nivel": nivel,
            "excesos": excesos,
        }

    # La región del score más excedido determina el tipo de defecto
    # (los nombres de score son "forma_<region>" o "int_<region>")
    region = score_max.split("_", 1)[1]
    tipo = TIPO_POR_REGION[region]

    return {
        "diagnostico": "Anómala",
        "tipo": tipo,
        "nivel": nivel,
        "excesos": excesos,
    }
