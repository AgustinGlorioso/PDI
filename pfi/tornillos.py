"""
tornillos.py - Sistema de deteccion de anomalias en tornillos (version simple).

Misma precision que la version completa (~95%), pero mucho mas facil de seguir:
en vez de 10 umbrales por region, hay UN puntaje por detector y UN umbral por
detector, calibrado con una regla unica: "mas grande que el peor tornillo sano".

Flujo de una imagen:
    cargar -> segmentar -> alinear -> 3 detectores -> 3 puntajes -> diagnostico

La idea de fondo: aprendemos como es un tornillo SANO (promedio) y marcamos
como defecto todo lo que se aparta de el.
"""

import os
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Rutas (relativas a este archivo) ---
_AQUI = os.path.dirname(os.path.abspath(__file__))
CARPETA_DATOS = os.path.normpath(os.path.join(_AQUI, "datos"))
CARPETA_PLANTILLAS = os.path.join(_AQUI, "plantillas")
RUTA_UMBRALES = os.path.join(CARPETA_PLANTILLAS, "umbrales.json")

# El tornillo alineado siempre queda igual
MARGEN_X = 50            # la cabeza queda anclada en esta columna
FILA_EJE = 512           # el eje del tornillo en esta fila

# 4 tramos fijos del tornillo (sirven para decir DONDE esta el defecto)
REGIONES = {"cabeza": (0, 200), "cuello": (200, 450), "rosca": (450, 850), "punta": (850, 1024)}
TIPO_POR_REGION = {"cabeza": "scratch_head", "cuello": "scratch_neck",
                   "rosca": "thread", "punta": "manipulated_front"}


# ===========================================================================
# PREPROCESAMIENTO: cargar, segmentar y alinear
# ===========================================================================

def cargar_imagen(ruta):
    # Devuelve la imagen en gris
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontro la imagen: {ruta}")
    return cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)


def segmentar(img_gris):
    # Separa el tornillo del fondo con bordes (Canny)
    suave = cv2.GaussianBlur(img_gris, (3, 3), 0)
    canny = cv2.Canny(suave, 10, 50)
    ee = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bordes = cv2.dilate(canny, ee, iterations=1)            # cerramos el contorno
    cont, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mascara = np.zeros_like(img_gris)
    if cont:
        cv2.drawContours(mascara, [max(cont, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return cv2.erode(mascara, ee, iterations=1)             # volvemos al tamano real


def angulo_por_momentos(mascara):
    # Angulo del eje largo del tornillo con momentos y su centro.
    M = cv2.moments(mascara, binaryImage=True)
    if M["m00"] == 0:
        return 0.0, (mascara.shape[1] // 2, mascara.shape[0] // 2)
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    ang = 0.5 * np.arctan2(2 * M["mu11"], M["mu20"] - M["mu02"])
    return np.degrees(ang), (cx, cy)


def alinear(img_gris, mascara):
    # Deja el tornillo horizontal, con la cabeza a la izquierda y en un lugar fijo
    h, w = mascara.shape
    angulo, (cx, cy) = angulo_por_momentos(mascara)

    # Matriz de rotación alrededor del centroide
    M_prueba = cv2.getRotationMatrix2D((cx, cy), angulo, 1.0)
    
    # Rotamos la máscara
    masc_prueba = cv2.warpAffine(mascara, M_prueba, (w, h), flags=cv2.INTER_NEAREST)

    # La cabeza es lo mas ancho, si quedo a la derecha, giramos 180
    x_r, y_r, w_r, h_r = cv2.boundingRect(masc_prueba)
    mitad = x_r + w_r // 2 # dos mitades (x_r;w_r/2 | w_r/2;w_r)
    if cv2.countNonZero(masc_prueba[:, mitad:x_r + w_r]) > cv2.countNonZero(masc_prueba[:, x_r:mitad]): # if derecha>izquierda
        angulo += 180.0
        M_prueba = cv2.getRotationMatrix2D((cx, cy), angulo, 1.0)
        masc_prueba = cv2.warpAffine(mascara, M_prueba, (w, h), flags=cv2.INTER_NEAREST)
        x_r, y_r, w_r, h_r = cv2.boundingRect(masc_prueba)

    # Transformacion final: rotar y mover para anclar la pieza siempre igual
    M = cv2.getRotationMatrix2D((cx, cy), angulo, 1.0)
    M[0, 2] += MARGEN_X - x_r # cabeza -> columna fija
    mom = cv2.moments(masc_prueba, binaryImage=True) # momentos de la máscara rotada
    cy_rot = mom["m01"] / mom["m00"] if mom["m00"] else h / 2 # centroide rotado
    M[1, 2] += FILA_EJE - cy_rot # eje -> fila central

    gris_a = cv2.warpAffine(img_gris, M, (w, h))
    masc_a = cv2.warpAffine(mascara, M, (w, h), flags=cv2.INTER_NEAREST)
    return gris_a, masc_a


def corregir_flip(gris_a, masc_a, plantilla):
    # Decide si el tornillo quedo boca arriba o boca abajo (lo compara con la plantilla)
    volteada = cv2.flip(masc_a, 0)
    if cv2.countNonZero(cv2.bitwise_and(volteada, plantilla)) > cv2.countNonZero(cv2.bitwise_and(masc_a, plantilla)):
        return cv2.flip(gris_a, 0), volteada
    return gris_a, masc_a


def procesar_imagen(ruta, plantilla=None):
    # Cargar -> segmentar -> alinear (y corregir el flip si tenemos plantilla)
    gris = cargar_imagen(ruta)
    masc = segmentar(gris)
    gris_a, masc_a = alinear(gris, masc)
    if plantilla is not None:
        gris_a, masc_a = corregir_flip(gris_a, masc_a, plantilla)
    return {"gris": gris, "mascara": masc,
            "gris_alineada": gris_a, "mascara_alineada": masc_a}


# ===========================================================================
# LOS 3 DETECTORES
# ===========================================================================

def _limpiar(mascara, area_min):
    # Borra las manchas mas chicas que area_min (son ruido)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mascara, 8)
    out = np.zeros_like(mascara)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= area_min:
            out[lab == i] = 255
    return out


def detectar_intensidad(gris_a, masc_a, media, std):
    """Detector 1 - RAYONES. Compara el gris contra el tornillo promedio.

    Marca los pixeles que se apartan mucho del promedio, midiendo en 'desvios
    estandar' (asi la tolerancia se adapta: donde los sanos varian, perdona mas)
    """
    # Mascara achicada para no mirar el borde
    ee = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    interna = cv2.erode(masc_a, ee)
    gris_f = cv2.GaussianBlur(gris_a, (3, 3), 0).astype(np.float32)
    dif = gris_f - media

    # En el cuello, filas enteras se ven mas claras segun como gire el tornillo
    # (reflejo). Le restamos la mediana de cada fila para sacar ese brillo global
    # sin tocar el rayon (que es local). Solo en el cuello.
    interna_bool = interna > 0
    x0, x1 = REGIONES["cuello"]
    sub = dif[:, x0:x1]
    msub = interna_bool[:, x0:x1]
    validas = msub.any(axis=1)
    con_nan = np.where(msub, sub, np.nan) # si el pixel no es tornillo -> NaN
    med = np.zeros(sub.shape[0], dtype=np.float32)
    med[validas] = np.nanmedian(con_nan[validas], axis=1)
    dif[:, x0:x1] = sub - med[:, None]

    z = dif / (std + 6.0) # cuantos desvios se aparta
    # defecto: estadisticamente (z) y visualmente (niveles de gris)
    defecto = ((np.abs(z) > 2.5) & (np.abs(dif) > 20) & interna_bool).astype(np.uint8) * 255
    # cierre: unir las lineas finas del rayon
    defecto = cv2.morphologyEx(defecto, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return _limpiar(defecto, 80)


def detectar_forma(masc_a, nucleo):
    """Detector 2 - DEFORMACIONES. Compara la silueta contra la plantilla.

    nucleo  = donde SIEMPRE hay metal en los sanos -> si falta, defecto.
    """
    ee = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    nuc  = cv2.erode(nucleo, ee)
    defecto = cv2.bitwise_and(nuc, cv2.bitwise_not(masc_a))       # falta metal
    defecto = cv2.morphologyEx(defecto, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return _limpiar(defecto, 120)


def detectar_perfil(masc_a, bandas):
    """Detector 3 - ROSCA. Mide el 'ancho' del tornillo columna a columna.

    Las roscas danadas hacen que el ancho se salga de lo normal. Se mide de
    forma que no importe como esten girados los dientes (envolvente de cresta).
    Devuelve una mascara con las columnas problematicas marcadas.
    """
    binaria = masc_a > 0
    h, w = binaria.shape
    hay = binaria.any(axis=0)
    sup = np.where(hay, FILA_EJE - binaria.argmax(axis=0), 0.0).astype(np.float32) # limite superiror
    inf = np.where(hay, (h - 1 - binaria[::-1].argmax(axis=0)) - FILA_EJE, 0.0).astype(np.float32) # limite inferior

    # Maximo/minimo movil de un paso de rosca = dilatar/erosionar en 1D.
    # Asi la cresta y el valle no dependen de donde caiga el diente.
    ee1d = cv2.getStructuringElement(cv2.MORPH_RECT, (63, 1)) # Por qué 63? ############################################################################
    perfiles = {"cresta_sup": cv2.dilate(sup.reshape(1, -1), ee1d).ravel(), # máximo local -> envolvente de las crestas superiores
                "valle_sup": cv2.erode(sup.reshape(1, -1), ee1d).ravel(), # mínimo local -> envolvente de los valles superiores
                "cresta_inf": cv2.dilate(inf.reshape(1, -1), ee1d).ravel(),
                "valle_inf": cv2.erode(inf.reshape(1, -1), ee1d).ravel()}

    # plt.figure(figsize=(12,5))

    # plt.plot(sup, label="sup")
    # plt.plot(perfiles["cresta_sup"], label="cresta_sup")
    # plt.plot(perfiles["cresta_inf"], label="valle_sup")

    # plt.legend()
    # plt.grid(True)
    # plt.show()

    marcas = np.zeros_like(masc_a)
    for nombre, perfil in perfiles.items():
        # banda de los sanos + holgura
        bmin = bandas[f"{nombre}_min"] - 2
        bmax = bandas[f"{nombre}_max"] + 2
        # sanos: bmin <= perfil <= bmax

        fuera = (perfil > bmax) | (perfil < bmin) # columnas anormales
        # restringimos el analisis a rosca + punta
        fuera[:400] = False
        fuera[950:] = False

        for x in np.where(fuera)[0]: # donde hay anomalias
            radio = int(perfil[x]) # distancia al eje
            fila = max(FILA_EJE - radio, 0) if "sup" in nombre else min(FILA_EJE + radio, h - 1)
            marcas[max(fila - 6, 0):fila + 6, x] = 255
    return marcas


# ===========================================================================
# ANALISIS Y DIAGNOSTICO
# ===========================================================================

def analizar(gris_a, masc_a, P):
    # Corre los 3 detectores y devuelve sus mascaras y un puntaje (area) de cada uno.
    d_int = detectar_intensidad(gris_a, masc_a, P["media_gris"], P["std_gris"])
    d_for = detectar_forma(masc_a, P["nucleo"])
    d_per = detectar_perfil(masc_a, P["bandas"])
    scores = {"intensidad": cv2.countNonZero(d_int),
              "forma": cv2.countNonZero(d_for),
              "perfil": cv2.countNonZero(d_per)}
    return {"intensidad": d_int, "forma": d_for, "perfil": d_per, "scores": scores}


def _region_del_defecto(defecto_total):
    # Mira donde esta la mancha mas grande y devuelve a que region pertenece.
    n, lab, st, cen = cv2.connectedComponentsWithStats(defecto_total, 8)
    if n <= 1:
        return "rosca"
    mayor = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    cx = cen[mayor][0]                                       # columna del centro
    for nombre, (x0, x1) in REGIONES.items():
        if x0 <= cx < x1:
            return nombre
    return "rosca"


def diagnosticar(analisis, umbrales):
    # Decide Normal/Anomala comparando cada puntaje con su umbral.
    s = analisis["scores"]
    # nivel = cuanto se paso el detector que mas se paso (1.0 = justo en el limite)
    nivel = max(s[k] / umbrales[k] if umbrales[k] > 0 else 0 for k in s)

    if nivel <= 1.0:
        return {"diagnostico": "Normal", "tipo": "good", "nivel": nivel}

    # tipo = region donde esta el defecto mas grande
    total = cv2.bitwise_or(cv2.bitwise_or(analisis["intensidad"], analisis["forma"]), analisis["perfil"])
    region = _region_del_defecto(total)
    return {"diagnostico": "Anómala", "tipo": TIPO_POR_REGION[region], "nivel": nivel}


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
    with np.load(os.path.join(d, "bandas.npz")) as z:
        bandas = {k: z[k] for k in z.files}
    return {"nucleo": leer("nucleo.png"), "binaria": leer("binaria.png"), 
            "media_gris": np.load(os.path.join(d, "media_gris.npy")),
            "std_gris": np.load(os.path.join(d, "std_gris.npy")), "bandas": bandas}


def cargar_umbrales():
    if not os.path.exists(RUTA_UMBRALES):
        raise FileNotFoundError("Falta umbrales.json. Ejecuta primero: python entrenar.py")
    with open(RUTA_UMBRALES) as f:
        return json.load(f)


if __name__ == "__main__":
    ruta = os.path.join(CARPETA_DATOS, "manipulated_front/012.png")
    r = procesar_imagen(ruta)
    P = cargar_plantillas()
    a = analizar(r["gris_alineada"], r["mascara_alineada"], P)