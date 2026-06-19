"""
tornillos.py - Deteccion de anomalias en tornillos (version minima, 2 detectores).

La version mas facil de entender: solo DOS ideas.
    1. RAYONES  -> comparar el gris contra el tornillo promedio.
    2. FORMA    -> comparar la silueta contra la plantilla.

No tiene el detector de rosca (el mas complicado), asi que detecta peor las
roscas y puntas (~74%), pero el codigo es muy corto y claro, y no da falsos
positivos en tornillos sanos.

Flujo:  cargar -> segmentar -> alinear -> 2 detectores -> diagnostico
"""

import os
import json

import cv2
import numpy as np

_AQUI = os.path.dirname(os.path.abspath(__file__))
CARPETA_DATOS = os.path.normpath(os.path.join(_AQUI, "..", "datos"))
CARPETA_PLANTILLAS = os.path.join(_AQUI, "plantillas")
RUTA_UMBRALES = os.path.join(CARPETA_PLANTILLAS, "umbrales.json")

MARGEN_X = 50            # la cabeza queda anclada en esta columna
FILA_EJE = 512           # el eje del tornillo queda en esta fila

REGIONES = {"cabeza": (0, 185), "cuello": (185, 490), "rosca": (490, 810), "punta": (810, 1024)}
TIPO_POR_REGION = {"cabeza": "scratch_head", "cuello": "scratch_neck",
                   "rosca": "thread", "punta": "manipulated_front"}


# ===========================================================================
# PREPROCESAMIENTO (igual que la version completa, funciona muy bien)
# ===========================================================================

def cargar_imagen(ruta):
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontro la imagen: {ruta}")
    bgr = cv2.imread(ruta)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def segmentar(img_gris):
    # Separa el tornillo del fondo con bordes (Canny), no por nivel de gris.
    suave = cv2.GaussianBlur(img_gris, (3, 3), 0)
    bordes = cv2.Canny(suave, 40, 90)
    ee = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bordes = cv2.dilate(bordes, ee, iterations=2)
    cont, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mascara = np.zeros_like(img_gris)
    if cont:
        cv2.drawContours(mascara, [max(cont, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return cv2.erode(mascara, ee, iterations=2)


def _angulo_por_momentos(mascara):
    M = cv2.moments(mascara, binaryImage=True)
    if M["m00"] == 0:
        return 0.0, (mascara.shape[1] // 2, mascara.shape[0] // 2)
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    ang = 0.5 * np.arctan2(2 * M["mu11"], M["mu20"] - M["mu02"])
    return np.degrees(ang), (cx, cy)


def alinear(img_gris, mascara):
    # Tornillo horizontal, cabeza a la izquierda, en un lugar fijo.
    h, w = mascara.shape
    angulo, (cx, cy) = _angulo_por_momentos(mascara)
    M_prueba = cv2.getRotationMatrix2D((cx, cy), angulo, 1.0)
    masc_prueba = cv2.warpAffine(mascara, M_prueba, (w, h), flags=cv2.INTER_NEAREST)

    x_r, y_r, w_r, h_r = cv2.boundingRect(masc_prueba)
    mitad = x_r + w_r // 2
    if cv2.countNonZero(masc_prueba[:, mitad:x_r + w_r]) > cv2.countNonZero(masc_prueba[:, x_r:mitad]):
        angulo += 180.0
        M_prueba = cv2.getRotationMatrix2D((cx, cy), angulo, 1.0)
        masc_prueba = cv2.warpAffine(mascara, M_prueba, (w, h), flags=cv2.INTER_NEAREST)
        x_r, y_r, w_r, h_r = cv2.boundingRect(masc_prueba)

    M = cv2.getRotationMatrix2D((cx, cy), angulo, 1.0)
    M[0, 2] += MARGEN_X - x_r
    mom = cv2.moments(masc_prueba, binaryImage=True)
    cy_rot = mom["m01"] / mom["m00"] if mom["m00"] else h / 2
    M[1, 2] += h / 2 - cy_rot
    gris_a = cv2.warpAffine(img_gris, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    masc_a = cv2.warpAffine(mascara, M, (w, h), flags=cv2.INTER_NEAREST)
    return gris_a, masc_a, angulo


def corregir_flip(gris_a, masc_a, plantilla):
    volteada = cv2.flip(masc_a, 0)
    if cv2.countNonZero(cv2.bitwise_and(volteada, plantilla)) > cv2.countNonZero(cv2.bitwise_and(masc_a, plantilla)):
        return cv2.flip(gris_a, 0), volteada
    return gris_a, masc_a


def procesar_imagen(ruta, plantilla=None):
    rgb, gris = cargar_imagen(ruta)
    masc = segmentar(gris)
    gris_a, masc_a, angulo = alinear(gris, masc)
    if plantilla is not None:
        gris_a, masc_a = corregir_flip(gris_a, masc_a, plantilla)
    return {"rgb": rgb, "gris": gris, "mascara": masc,
            "gris_alineada": gris_a, "mascara_alineada": masc_a, "angulo": angulo}


# ===========================================================================
# LOS 2 DETECTORES
# ===========================================================================

def _limpiar(mascara, area_min):
    # Borra las manchas mas chicas que area_min.
    n, lab, st, _ = cv2.connectedComponentsWithStats(mascara, 8)
    out = np.zeros_like(mascara)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= area_min:
            out[lab == i] = 255
    return out


def detectar_intensidad(gris_a, masc_a, media, std):
    """Detector 1 - RAYONES. Marca el gris que se aparta del tornillo promedio.

    Mide en 'desvios estandar': donde los sanos varian mucho, perdona mas.
    """
    ee = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    interna = cv2.erode(masc_a, ee)                          # sacamos el borde

    gris_f = cv2.GaussianBlur(gris_a, (3, 3), 0).astype(np.float32)
    dif = gris_f - media

    # En el cuello, filas enteras se ven mas claras por el reflejo. Restamos la
    # mediana de cada fila para sacar ese brillo sin tocar el rayon (que es local).
    interna_bool = interna > 0
    x0, x1 = REGIONES["cuello"]
    sub, msub = dif[:, x0:x1], interna_bool[:, x0:x1]
    validas = msub.any(axis=1)
    con_nan = np.where(msub, sub, np.nan)
    med = np.zeros(sub.shape[0], dtype=np.float32)
    med[validas] = np.nanmedian(con_nan[validas], axis=1)
    dif[:, x0:x1] = sub - med[:, None]

    z = dif / (std + 6.0)
    defecto = ((np.abs(z) > 2.5) & (np.abs(dif) > 20) & interna_bool).astype(np.uint8) * 255
    defecto = cv2.morphologyEx(defecto, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return _limpiar(defecto, 80)


def detectar_forma(masc_a, nucleo, exterior):
    """Detector 2 - DEFORMACIONES. Compara la silueta contra la plantilla.

    nucleo  = donde SIEMPRE hay metal -> si falta, defecto.
    exterior = donde NUNCA hay metal  -> si sobra, defecto.
    """
    ee = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    nuc, ext = cv2.erode(nucleo, ee), cv2.erode(exterior, ee)
    falta = cv2.bitwise_and(nuc, cv2.bitwise_not(masc_a))
    sobra = cv2.bitwise_and(masc_a, ext)
    defecto = cv2.bitwise_or(falta, sobra)
    defecto = cv2.morphologyEx(defecto, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return _limpiar(defecto, 120)


# ===========================================================================
# ANALISIS Y DIAGNOSTICO
# ===========================================================================

def analizar(gris_a, masc_a, P):
    # Corre los 2 detectores y devuelve sus mascaras y el area de cada uno.
    d_int = detectar_intensidad(gris_a, masc_a, P["media_gris"], P["std_gris"])
    d_for = detectar_forma(masc_a, P["nucleo"], P["exterior"])
    return {"intensidad": d_int, "forma": d_for,
            "scores": {"intensidad": cv2.countNonZero(d_int), "forma": cv2.countNonZero(d_for)}}


def _region_del_defecto(defecto_total):
    n, lab, st, cen = cv2.connectedComponentsWithStats(defecto_total, 8)
    if n <= 1:
        return "rosca"
    mayor = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    cx = cen[mayor][0]
    for nombre, (x0, x1) in REGIONES.items():
        if x0 <= cx < x1:
            return nombre
    return "rosca"


def diagnosticar(analisis, umbrales):
    s = analisis["scores"]
    nivel = max(s[k] / umbrales[k] if umbrales[k] > 0 else 0 for k in s)
    if nivel <= 1.0:
        return {"diagnostico": "Normal", "tipo": "good", "nivel": nivel}
    total = cv2.bitwise_or(analisis["intensidad"], analisis["forma"])
    return {"diagnostico": "Anómala", "tipo": TIPO_POR_REGION[_region_del_defecto(total)], "nivel": nivel}


# ===========================================================================
# Cargar lo que aprendio entrenar.py
# ===========================================================================

def cargar_plantillas():
    d = CARPETA_PLANTILLAS
    def leer(n):
        img = cv2.imread(os.path.join(d, n), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Falta {n}. Ejecuta primero: python entrenar.py")
        return img
    return {"nucleo": leer("nucleo.png"), "exterior": leer("exterior.png"),
            "binaria": leer("binaria.png"), "media_gris": np.load(os.path.join(d, "media_gris.npy")),
            "std_gris": np.load(os.path.join(d, "std_gris.npy"))}


def cargar_umbrales():
    if not os.path.exists(RUTA_UMBRALES):
        raise FileNotFoundError("Falta umbrales.json. Ejecuta primero: python entrenar.py")
    with open(RUTA_UMBRALES) as f:
        return json.load(f)
