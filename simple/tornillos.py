"""
tornillos.py - Nucleo del sistema de deteccion de anomalias en tornillos.

Junta en un solo archivo todo lo que en la version original estaba repartido
en pipeline.py + detectores.py + clasificador.py. El algoritmo es identico;
solo se reorganizo y se comento mas simple.

El flujo para una imagen es:
    cargar -> segmentar -> alinear -> 3 detectores -> scores -> diagnostico

Lo usan entrenar.py (para armar la plantilla y calibrar) y app.py (para
inspeccionar o evaluar).
"""

import os
import json

import cv2
import numpy as np

# --- Rutas (relativas a la ubicacion de este archivo, asi corre desde donde sea) ---
_AQUI = os.path.dirname(os.path.abspath(__file__))
CARPETA_DATOS = os.path.normpath(os.path.join(_AQUI, "..", "datos"))   # las fotos
CARPETA_PLANTILLAS = os.path.join(_AQUI, "plantillas")                 # lo que aprende
RUTA_UMBRALES = os.path.join(CARPETA_PLANTILLAS, "umbrales.json")

# --- Parametros generales ---
MARGEN_X = 50            # columna donde queda anclada la cabeza tras alinear
FILA_EJE = 512           # fila central: ahi queda el eje del tornillo

# El tornillo alineado se divide en 4 tramos fijos de columnas. Como todos
# quedan anclados en el mismo lugar, estas fronteras sirven para cualquier foto.
REGIONES = {
    "cabeza": (0, 185),
    "cuello": (185, 490),
    "rosca": (490, 810),
    "punta": (810, 1024),
}

# Cada defecto vive en una region distinta -> de ahi sale el "tipo".
TIPO_POR_REGION = {
    "cabeza": "scratch_head",
    "cuello": "scratch_neck",
    "rosca": "thread",
    "punta": "manipulated_front",
}

# --- Parametros de la plantilla ---
UMBRAL_NUCLEO = 0.97     # un pixel es "nucleo" si >=97% de los sanos tienen metal ahi
UMBRAL_EXTERIOR = 0.03   # es "exterior" si <=3% de los sanos tienen metal ahi

# --- Parametros del detector de forma ---
TOL_FORMA_PX = 5         # cuanto se achican las zonas, como margen de seguridad
AREA_MIN_FORMA = 120     # manchas de defecto mas chicas que esto se descartan

# --- Parametros del detector de perfil de rosca ---
PASO_ROSCA = 63          # separacion entre dientes del filete, en px
COLS_PERFIL = (490, 990) # zona donde se analiza el perfil (rosca + punta)
MARGEN_PERFIL = 2        # holgura sobre la banda de los sanos, en px

# --- Parametros del detector de intensidad ---
EROSION_INTERNA_PX = 8   # cuanto se mete la mascara hacia adentro (evita el borde)
Z_UMBRAL = 2.5           # cuantos desvios estandar para considerar un pixel raro
DIF_MINIMA = 20          # ademas, cuantos niveles de gris de diferencia minima
EPS_STD = 6.0            # evita dividir por una std casi cero
AREA_MIN_INT = 80        # manchas de defecto mas chicas que esto se descartan


# ===========================================================================
# FASE 1-2: cargar, segmentar y alinear
# ===========================================================================

def cargar_imagen(ruta):
    """Lee la imagen y la devuelve en color (para mostrar) y en gris (para trabajar)."""
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontro la imagen en: {ruta}")
    img_bgr = cv2.imread(ruta)                          # OpenCV lee en BGR
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # BGR -> RGB para matplotlib
    img_gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_rgb, img_gris


def segmentar(img_gris):
    """Separa el tornillo del fondo usando bordes (Canny), no umbral de gris.

    Se usa Canny y no Otsu porque la sombra difusa del fondo confunde a Otsu
    e infla la mascara. El borde metal-fondo es nitido y Canny lo agarra bien.
    """
    suavizada = cv2.GaussianBlur(img_gris, (3, 3), 0)   # saca un poco de ruido
    bordes = cv2.Canny(suavizada, 40, 90)               # detecta los bordes nitidos

    # Engordamos los bordes para cerrar el contorno y poder rellenarlo
    ee = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bordes_dilatados = cv2.dilate(bordes, ee, iterations=2)

    # Nos quedamos con el contorno mas grande (el tornillo) y lo pintamos macizo
    contornos, _ = cv2.findContours(bordes_dilatados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mascara = np.zeros_like(img_gris)
    if contornos:
        tornillo = max(contornos, key=cv2.contourArea)
        cv2.drawContours(mascara, [tornillo], -1, 255, thickness=cv2.FILLED)

    # Achicamos lo mismo que habiamos engordado, para volver al tamano real
    mascara = cv2.erode(mascara, ee, iterations=2)
    return mascara


def _angulo_por_momentos(mascara):
    """Calcula el angulo del eje largo del tornillo usando momentos (equivale a PCA)."""
    M = cv2.moments(mascara, binaryImage=True)
    if M["m00"] == 0:                                    # mascara vacia
        return 0.0, (mascara.shape[1] // 2, mascara.shape[0] // 2)
    cx = M["m10"] / M["m00"]                             # centroide en x
    cy = M["m01"] / M["m00"]                             # centroide en y
    theta = 0.5 * np.arctan2(2 * M["mu11"], M["mu20"] - M["mu02"])  # angulo del eje
    return np.degrees(theta), (cx, cy)


def alinear(img_gris, mascara):
    """Pone el tornillo horizontal, con la cabeza a la izquierda y en un lugar fijo."""
    h, w = mascara.shape
    angulo, (cx, cy) = _angulo_por_momentos(mascara)

    # Rotamos de prueba solo la mascara para ver como queda
    M_prueba = cv2.getRotationMatrix2D((cx, cy), angulo, 1.0)
    masc_prueba = cv2.warpAffine(mascara, M_prueba, (w, h), flags=cv2.INTER_NEAREST)

    # La cabeza es la parte mas ancha -> la mitad con mas pixeles es la cabeza
    x_r, y_r, w_r, h_r = cv2.boundingRect(masc_prueba)
    mitad = x_r + w_r // 2
    masa_izq = cv2.countNonZero(masc_prueba[:, x_r:mitad])
    masa_der = cv2.countNonZero(masc_prueba[:, mitad:x_r + w_r])
    if masa_der > masa_izq:                              # cabeza quedo a la derecha
        angulo += 180.0                                  # la giramos 180 grados
        M_prueba = cv2.getRotationMatrix2D((cx, cy), angulo, 1.0)
        masc_prueba = cv2.warpAffine(mascara, M_prueba, (w, h), flags=cv2.INTER_NEAREST)
        x_r, y_r, w_r, h_r = cv2.boundingRect(masc_prueba)

    # Armamos la transformacion final: rotacion + traslacion para anclar la pieza
    M_final = cv2.getRotationMatrix2D((cx, cy), angulo, 1.0)
    M_final[0, 2] += MARGEN_X - x_r                      # cabeza -> columna MARGEN_X
    M_cy = cv2.moments(masc_prueba, binaryImage=True)
    cy_rotado = M_cy["m01"] / M_cy["m00"] if M_cy["m00"] else h / 2
    M_final[1, 2] += h / 2 - cy_rotado                   # eje -> fila central

    # Una sola transformacion sobre la imagen real (mejor que dos seguidas)
    gris_alineada = cv2.warpAffine(img_gris, M_final, (w, h),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    masc_alineada = cv2.warpAffine(mascara, M_final, (w, h), flags=cv2.INTER_NEAREST)
    return gris_alineada, masc_alineada, angulo


def corregir_flip_vertical(gris_a, masc_a, mascara_plantilla):
    """Decide si el tornillo quedo 'boca arriba' o 'boca abajo' comparando con la plantilla."""
    volteada = cv2.flip(masc_a, 0)                       # version espejada
    # Vemos cual de las dos se parece mas a la plantilla (mas pixeles en comun)
    overlap_normal = cv2.countNonZero(cv2.bitwise_and(masc_a, mascara_plantilla))
    overlap_flip = cv2.countNonZero(cv2.bitwise_and(volteada, mascara_plantilla))
    if overlap_flip > overlap_normal:
        return cv2.flip(gris_a, 0), volteada, True
    return gris_a, masc_a, False


def procesar_imagen(ruta, mascara_plantilla=None):
    """Hace todo el pre-procesamiento de una imagen: cargar -> segmentar -> alinear."""
    img_rgb, img_gris = cargar_imagen(ruta)
    mascara = segmentar(img_gris)
    gris_a, masc_a, angulo = alinear(img_gris, mascara)

    se_giro = False
    if mascara_plantilla is not None:                    # si tenemos plantilla, corregimos el flip
        gris_a, masc_a, se_giro = corregir_flip_vertical(gris_a, masc_a, mascara_plantilla)

    return {"rgb": img_rgb, "gris": img_gris, "mascara": mascara,
            "gris_alineada": gris_a, "mascara_alineada": masc_a,
            "angulo": angulo, "flip": se_giro}


# ===========================================================================
# FASE 3-4: los tres detectores
# ===========================================================================

def _filtrar_por_area(mascara_binaria, area_minima):
    """Borra las manchas mas chicas que area_minima (son ruido, no defectos)."""
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(mascara_binaria, connectivity=8)
    salida = np.zeros_like(mascara_binaria)
    for i in range(1, n):                                # la 0 es el fondo
        if stats[i, cv2.CC_STAT_AREA] >= area_minima:
            salida[etiquetas == i] = 255
    return salida


def detectar_forma(mascara_alineada, nucleo, exterior):
    """Compara la silueta contra la plantilla: busca metal que falta o que sobra."""
    # Achicamos las zonas un poco, como margen de seguridad
    ee_tol = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * TOL_FORMA_PX + 1, 2 * TOL_FORMA_PX + 1))
    nucleo_seguro = cv2.erode(nucleo, ee_tol)
    exterior_seguro = cv2.erode(exterior, ee_tol)

    # Falta metal donde siempre deberia haber / Sobra metal donde nunca deberia
    faltante = cv2.bitwise_and(nucleo_seguro, cv2.bitwise_not(mascara_alineada))
    sobrante = cv2.bitwise_and(mascara_alineada, exterior_seguro)

    # Limpieza: apertura (saca pixeles sueltos) + borrar manchas chicas
    ee_chico = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    faltante = cv2.morphologyEx(faltante, cv2.MORPH_OPEN, ee_chico)
    sobrante = cv2.morphologyEx(sobrante, cv2.MORPH_OPEN, ee_chico)
    faltante = _filtrar_por_area(faltante, AREA_MIN_FORMA)
    sobrante = _filtrar_por_area(sobrante, AREA_MIN_FORMA)
    return {"faltante": faltante, "sobrante": sobrante}


def perfiles_envolvente(mascara_alineada):
    """Mide el 'radio' del tornillo columna a columna, de forma que no dependa
    de como esten girados los dientes de la rosca (envolventes de cresta y valle)."""
    binaria = (mascara_alineada > 0)
    h, w = binaria.shape

    # Para cada columna: primera y ultima fila con metal -> radios arriba/abajo
    hay_metal = binaria.any(axis=0)
    fila_sup = binaria.argmax(axis=0).astype(np.float32)
    fila_inf = (h - 1 - binaria[::-1].argmax(axis=0)).astype(np.float32)
    semi_sup = np.where(hay_metal, FILA_EJE - fila_sup, 0.0)
    semi_inf = np.where(hay_metal, fila_inf - FILA_EJE, 0.0)

    # Filtro de maximo/minimo movil de un paso de rosca = dilatar/erosionar en 1D.
    # Asi la cresta y el valle no dependen de donde caiga el diente (invariante a la fase).
    ee_1d = cv2.getStructuringElement(cv2.MORPH_RECT, (PASO_ROSCA, 1))
    def _max_movil(p): return cv2.dilate(p.reshape(1, -1), ee_1d).ravel()
    def _min_movil(p): return cv2.erode(p.reshape(1, -1), ee_1d).ravel()

    return {"cresta_sup": _max_movil(semi_sup), "valle_sup": _min_movil(semi_sup),
            "cresta_inf": _max_movil(semi_inf), "valle_inf": _min_movil(semi_inf)}


def detectar_perfil(mascara_alineada, bandas):
    """Marca donde la envolvente de la pieza se sale de la banda de los sanos."""
    perfiles = perfiles_envolvente(mascara_alineada)
    w = mascara_alineada.shape[1]
    violacion = np.zeros(w, dtype=np.float32)            # exceso por columna
    marcas = np.zeros_like(mascara_alineada)             # para dibujar el defecto

    x0, x1 = COLS_PERFIL
    for nombre, perfil in perfiles.items():
        banda_min = bandas[f"{nombre}_min"] - MARGEN_PERFIL
        banda_max = bandas[f"{nombre}_max"] + MARGEN_PERFIL
        # cuanto se pasa por arriba (sobra) o por abajo (falta) de la banda
        exceso = np.maximum(perfil - banda_max, 0) + np.maximum(banda_min - perfil, 0)
        exceso[:x0] = 0                                  # solo dentro de la zona analizada
        exceso[x1:] = 0
        violacion += exceso

        # Pintamos la columna infractora a la altura del borde, para verlo
        for x in np.where(exceso > 0)[0]:
            radio = int(perfil[x])
            fila = max(FILA_EJE - radio, 0) if "sup" in nombre else min(FILA_EJE + radio, mascara_alineada.shape[0] - 1)
            marcas[max(fila - 6, 0):fila + 6, x] = 255

    return {"violacion": violacion, "marcas": marcas}


def detectar_intensidad(gris_alineada, mascara_alineada, media_gris, std_gris):
    """Compara el gris de cada pixel contra el tornillo promedio (rayones, raspones).

    Marca un pixel si se aparta mucho de lo normal: en desvios estandar (z) Y en
    niveles de gris. La tolerancia se adapta sola: donde los sanos varian, perdona mas.
    """
    # Mascara achicada para no mirar el borde
    ee_int = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * EROSION_INTERNA_PX + 1, 2 * EROSION_INTERNA_PX + 1))
    mascara_interna = cv2.erode(mascara_alineada, ee_int)

    # Suavizamos un toque y restamos el tornillo promedio
    gris_f = cv2.GaussianBlur(gris_alineada, (3, 3), 0).astype(np.float32)
    dif = gris_f - media_gris

    # Correccion de brillo en el cuello: segun como gire el tornillo, filas enteras
    # del cuello se ven mas claras (reflejo). Le restamos la mediana de cada fila:
    # eso saca el brillo global pero deja el defecto (que es local). Solo en el cuello.
    interna_bool = mascara_interna > 0
    x0, x1 = REGIONES["cuello"]
    sub = dif[:, x0:x1]
    msub = interna_bool[:, x0:x1]
    filas_validas = msub.any(axis=1)
    con_nan = np.where(msub, sub, np.nan)
    mediana_fila = np.zeros(sub.shape[0], dtype=np.float32)
    mediana_fila[filas_validas] = np.nanmedian(con_nan[filas_validas], axis=1)
    dif[:, x0:x1] = sub - mediana_fila[:, None]

    # z = cuantos desvios estandar se aparta cada pixel
    z = dif / (std_gris + EPS_STD)

    # Lo marcamos si es raro estadistica Y visualmente, separando claro/oscuro
    brillante = ((z > Z_UMBRAL) & (dif > DIF_MINIMA)).astype(np.uint8) * 255
    oscuro = ((z < -Z_UMBRAL) & (dif < -DIF_MINIMA)).astype(np.uint8) * 255
    brillante = cv2.bitwise_and(brillante, mascara_interna)
    oscuro = cv2.bitwise_and(oscuro, mascara_interna)

    # Cierre (no apertura): suelda las estrias finas del rayon en una mancha compacta
    ee_cierre = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    brillante = cv2.morphologyEx(brillante, cv2.MORPH_CLOSE, ee_cierre)
    oscuro = cv2.morphologyEx(oscuro, cv2.MORPH_CLOSE, ee_cierre)
    brillante = _filtrar_por_area(brillante, AREA_MIN_INT)
    oscuro = _filtrar_por_area(oscuro, AREA_MIN_INT)
    return {"brillante": brillante, "oscuro": oscuro}


def calcular_scores(forma, intensidad, perfil, mascara_plantilla_binaria):
    """Resume las mascaras de defecto en 10 numeros (2 por region + 2 de perfil)."""
    defecto_forma = cv2.bitwise_or(forma["faltante"], forma["sobrante"])
    defecto_int = cv2.bitwise_or(intensidad["brillante"], intensidad["oscuro"])

    scores = {}
    for nombre, (x0, x1) in REGIONES.items():
        # normalizamos por el area de la region para que sean comparables
        area_region = max(cv2.countNonZero(mascara_plantilla_binaria[:, x0:x1]), 1)
        scores[f"forma_{nombre}"] = cv2.countNonZero(defecto_forma[:, x0:x1]) / area_region
        scores[f"int_{nombre}"] = cv2.countNonZero(defecto_int[:, x0:x1]) / area_region

    # scores del perfil: exceso promedio por columna en rosca y en punta
    violacion = perfil["violacion"]
    xa0, xa1 = COLS_PERFIL
    xr0, xr1 = REGIONES["rosca"]
    xp0, xp1 = REGIONES["punta"]
    scores["perfil_rosca"] = float(violacion[max(xr0, xa0):min(xr1, xa1)].mean())
    scores["perfil_punta"] = float(violacion[max(xp0, xa0):min(xp1, xa1)].mean())
    return scores


def inspeccionar(gris_alineada, mascara_alineada, plantillas):
    """Corre los 3 detectores y devuelve las mascaras de defecto + los scores."""
    forma = detectar_forma(mascara_alineada, plantillas["nucleo"], plantillas["exterior"])
    intensidad = detectar_intensidad(gris_alineada, mascara_alineada,
                                     plantillas["media_gris_f"], plantillas["std_gris"])
    perfil = detectar_perfil(mascara_alineada, plantillas["bandas_perfil"])
    scores = calcular_scores(forma, intensidad, perfil, plantillas["binaria"])
    return {"forma": forma, "intensidad": intensidad, "perfil": perfil, "scores": scores}


# ===========================================================================
# FASE 5: clasificacion + carga de plantillas y umbrales
# ===========================================================================

def cargar_plantillas():
    """Carga de disco todo lo que aprendio entrenar.py (la carpeta plantillas/)."""
    def _leer(nombre):
        img = cv2.imread(os.path.join(CARPETA_PLANTILLAS, nombre), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Falta '{nombre}'. Ejecuta primero: python entrenar.py")
        return img

    def _leer_npy(nombre):
        ruta = os.path.join(CARPETA_PLANTILLAS, nombre)
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"Falta '{nombre}'. Ejecuta primero: python entrenar.py")
        return np.load(ruta)

    with np.load(os.path.join(CARPETA_PLANTILLAS, "bandas_perfil.npz")) as npz:
        bandas = {k: npz[k] for k in npz.files}

    return {"nucleo": _leer("nucleo.png"), "exterior": _leer("exterior.png"),
            "binaria": _leer("mascara_binaria.png"), "media_gris": _leer("media_gris.png"),
            "media_gris_f": _leer_npy("media_gris.npy"), "std_gris": _leer_npy("std_gris.npy"),
            "bandas_perfil": bandas}


def cargar_umbrales():
    """Carga los umbrales calibrados por entrenar.py."""
    if not os.path.exists(RUTA_UMBRALES):
        raise FileNotFoundError("Falta umbrales.json. Ejecuta primero: python entrenar.py")
    with open(RUTA_UMBRALES) as f:
        return json.load(f)["umbrales"]


def clasificar(scores, umbrales):
    """Decide Normal/Anomala y el tipo de defecto, a partir de los scores."""
    # cuanto se pasa cada score de su umbral (1.0 = justo en el limite)
    excesos = {k: scores[k] / umbrales[k] if umbrales[k] > 0 else 0.0 for k in scores}
    score_max = max(excesos, key=excesos.get)            # el que mas se paso
    nivel = excesos[score_max]

    if nivel <= 1.0:                                      # ninguno supero su umbral
        return {"diagnostico": "Normal", "tipo": "good", "nivel": nivel, "excesos": excesos}

    # la region del score que mas se paso nos dice el tipo de defecto
    region = score_max.split("_", 1)[1]
    return {"diagnostico": "Anómala", "tipo": TIPO_POR_REGION[region],
            "nivel": nivel, "excesos": excesos}
