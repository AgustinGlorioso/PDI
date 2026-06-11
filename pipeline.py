"""
pipeline.py — Fases 1 y 2: Preprocesamiento, Segmentación y Alineación Espacial.

Este módulo contiene los bloques iniciales del sistema de inspección:

    imagen RGB ──> escala de grises ──> máscara binaria ──> imagen alineada
                   (preprocesado)       (segmentación)      (normalización
                                                             geométrica)

Todas las fases posteriores (detección de defectos, clasificación) trabajan
sobre la salida de este módulo, por lo que la robustez de estas funciones
es crítica para todo el sistema.

Convenciones:
  - Las imágenes en gris son arrays uint8 de 1024x1024 (dataset MVTec AD).
  - Las máscaras son binarias uint8 con valores {0, 255}.
  - Tras la alineación el tornillo queda HORIZONTAL con la CABEZA a la
    IZQUIERDA, con el extremo izquierdo en una columna fija (MARGEN_X) y el
    eje longitudinal sobre la fila central de la imagen. Esto hace que todas
    las imágenes sean comparables píxel a píxel contra una plantilla.
"""

import os

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Constantes de alineación
# ----------------------------------------------------------------------------
# Columna fija donde se ancla el extremo izquierdo (cabeza) del tornillo tras
# alinear. Se ancla por la CABEZA y no por el centroide porque la cabeza es
# estable en todas las categorías de defecto: los arañazos no cambian la
# silueta y los defectos de forma (rosca deformada, punta rota) están del
# lado opuesto. Si ancláramos por el centroide, un tornillo con la punta
# rota tendría el centroide corrido y TODO el tornillo quedaría desplazado
# respecto de la plantilla, generando falsas diferencias en toda la pieza.
MARGEN_X = 50


def cargar_imagen(ruta):
    """Carga una imagen del disco y la devuelve en RGB y en escala de grises.

    Parámetros:
        ruta: ruta al archivo de imagen (PNG del dataset).

    Devuelve:
        (img_rgb, img_gris): la imagen en RGB (para visualizar con
        Matplotlib, que espera RGB y no el BGR nativo de OpenCV) y en
        escala de grises (sobre la que trabaja todo el pipeline).
    """
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró la imagen en: {ruta}")

    img_bgr = cv2.imread(ruta)                          # OpenCV lee en BGR
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # BGR -> RGB (Matplotlib)
    img_gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_rgb, img_gris


def segmentar(img_gris):
    """Fase 1: separa el tornillo (oscuro) del fondo (claro) por BORDES.

    ¿Por qué bordes y no umbralado global (Otsu)?
        Varias imágenes del dataset tienen una sombra DIFUSA alrededor del
        tornillo. Esa sombra alcanza niveles de gris por debajo del umbral
        de Otsu (~145), por lo que la binarización la incluye como objeto y
        la máscara queda "inflada" (medimos áreas de 138k a 169k px según
        la sombra, un error de hasta 70%). En cambio, la transición
        metal-fondo es un borde ABRUPTO (gradiente fuerte) mientras que la
        sombra es un degradé SUAVE (gradiente débil): un detector de bordes
        con umbrales moderados ve el contorno del tornillo e ignora la
        sombra. Con esta estrategia el área de los tornillos sanos resulta
        casi constante (~100.000 px con desvío < 1%), que es exactamente la
        estabilidad que necesita la comparación contra plantilla.

    Estrategia:
      1. Suavizado gaussiano leve 3x3 (un kernel grande difuminaría los
         bordes débiles que justamente queremos detectar).
      2. Canny con histéresis (40, 90): la histéresis conserva bordes
         débiles solo si están conectados a bordes fuertes, dando un
         contorno continuo del metal sin captar el degradé de la sombra.
      3. Dilatación (2 iter., EE elíptico 5x5): "engorda" los bordes para
         soldar los pequeños cortes donde Canny perdió el contorno, dejando
         una curva cerrada.
      4. Contorno externo mayor + FILLED: rellena el interior del contorno
         cerrado, generando la silueta maciza del tornillo.
      5. Erosión (mismas 2 iter. del paso 3): la dilatación había agrandado
         la silueta; la erosión equivalente la devuelve a su tamaño real.
         (Pasos 3+4+5 actúan como un "cierre morfológico dirigido".)

    Devuelve:
        máscara binaria uint8 {0,255} con el tornillo macizo en blanco.
    """
    # 1. Suavizado gaussiano leve para atenuar el ruido del sensor sin
    #    destruir los bordes reales del objeto
    suavizada = cv2.GaussianBlur(img_gris, (3, 3), 0)

    # 2. Detector de Canny: gradiente + supresión de no-máximos + histéresis.
    #    Umbral bajo 40 / alto 90: el borde metal-fondo (salto de ~100
    #    niveles) supera siempre el umbral alto; el degradé de la sombra
    #    (pendiente de pocos niveles por píxel) queda por debajo del bajo.
    bordes = cv2.Canny(suavizada, 40, 90)

    # 3. Dilatación para cerrar cortes en el contorno detectado
    ee = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bordes_dilatados = cv2.dilate(bordes, ee, iterations=2)

    # 4. El contorno externo de mayor área es el tornillo; FILLED lo pinta
    #    macizo (elimina huecos internos por reflejos del metal)
    contornos, _ = cv2.findContours(
        bordes_dilatados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    mascara = np.zeros_like(img_gris)
    if contornos:
        tornillo = max(contornos, key=cv2.contourArea)
        cv2.drawContours(mascara, [tornillo], -1, 255, thickness=cv2.FILLED)

    # 5. Erosión simétrica a la dilatación del paso 3: restaura el tamaño
    #    original del objeto (la silueta había crecido ~2*2 píxeles por lado)
    mascara = cv2.erode(mascara, ee, iterations=2)

    return mascara


def _angulo_por_momentos(mascara):
    """Calcula la orientación del eje principal del objeto con momentos.

    Teoría (Unidad de descriptores / momentos): los momentos centrales de
    segundo orden mu20, mu02 y mu11 describen cómo se distribuye la masa
    del objeto alrededor de su centroide. El eje de mínima inercia (el eje
    "largo" del tornillo) forma con el eje x un ángulo:

        theta = 0.5 * atan2(2*mu11, mu20 - mu02)

    Es equivalente a un análisis de componentes principales (PCA) sobre las
    coordenadas de los píxeles del objeto, pero calculado de forma cerrada.

    Devuelve:
        (angulo_grados, (cx, cy)): orientación del eje y centroide.
    """
    M = cv2.moments(mascara, binaryImage=True)
    if M["m00"] == 0:
        return 0.0, (mascara.shape[1] // 2, mascara.shape[0] // 2)

    # Centroide: primeros momentos normalizados por el área (m00)
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    # Orientación del eje principal a partir de los momentos centrales
    theta = 0.5 * np.arctan2(2 * M["mu11"], M["mu20"] - M["mu02"])
    return np.degrees(theta), (cx, cy)


def alinear(img_gris, mascara):
    """Fase 2: normaliza la pose del tornillo (rotación + traslación).

    Pasos:
      1. Orientación del eje longitudinal por momentos de 2.º orden (PCA).
      2. Rotación de prueba de la máscara para dejar el eje horizontal.
      3. Decisión cabeza-izquierda: la cabeza es la parte más ancha del
         tornillo, así que la mitad que concentra más píxeles blancos es la
         de la cabeza. Si quedó a la derecha, se suman 180° a la rotación.
      4. Transformación afín FINAL única sobre la imagen original:
         rotación + traslación que ancla el extremo izquierdo en MARGEN_X
         y el eje del tornillo en la fila central. Componer todo en una
         sola warpAffine evita degradar la imagen con interpolaciones
         sucesivas.

    Devuelve:
        (gris_alineada, mascara_alineada, angulo_aplicado)
    """
    h, w = mascara.shape
    angulo, (cx, cy) = _angulo_por_momentos(mascara)

    # --- Rotación de prueba (solo la máscara, interpolación nearest) -------
    M_prueba = cv2.getRotationMatrix2D((cx, cy), angulo, 1.0)
    masc_prueba = cv2.warpAffine(
        mascara, M_prueba, (w, h), flags=cv2.INTER_NEAREST
    )

    # --- ¿La cabeza quedó a la izquierda o a la derecha? --------------------
    # La cabeza (cónica, más ancha) concentra más masa que la rosca.
    x_r, y_r, w_r, h_r = cv2.boundingRect(masc_prueba)
    mitad = x_r + w_r // 2
    masa_izq = cv2.countNonZero(masc_prueba[:, x_r:mitad])
    masa_der = cv2.countNonZero(masc_prueba[:, mitad : x_r + w_r])

    if masa_der > masa_izq:           # cabeza a la derecha -> girar 180°
        angulo += 180.0
        M_prueba = cv2.getRotationMatrix2D((cx, cy), angulo, 1.0)
        masc_prueba = cv2.warpAffine(
            mascara, M_prueba, (w, h), flags=cv2.INTER_NEAREST
        )
        x_r, y_r, w_r, h_r = cv2.boundingRect(masc_prueba)

    # --- Traslación de anclaje ----------------------------------------------
    # Componemos la traslación DENTRO de la misma matriz afín 2x3: las dos
    # últimas columnas de M son el desplazamiento, así que basta sumarle el
    # corrimiento deseado (esto equivale a Traslación @ Rotación).
    M_final = cv2.getRotationMatrix2D((cx, cy), angulo, 1.0)
    M_final[0, 2] += MARGEN_X - x_r          # extremo izquierdo -> MARGEN_X

    # Para el eje vertical anclamos el CENTRO DE MASA de la pieza rotada a la
    # fila central: en un tornillo el centroide cae sobre el eje longitudinal.
    M_cy = cv2.moments(masc_prueba, binaryImage=True)
    cy_rotado = M_cy["m01"] / M_cy["m00"] if M_cy["m00"] else h / 2
    M_final[1, 2] += h / 2 - cy_rotado       # eje del tornillo -> fila h/2

    # --- Única transformación afín sobre los datos reales -------------------
    # BORDER_REPLICATE extiende los píxeles del borde (que son fondo claro),
    # evitando esquinas negras artificiales que confundirían a los detectores.
    gris_alineada = cv2.warpAffine(
        img_gris, M_final, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    masc_alineada = cv2.warpAffine(
        mascara, M_final, (w, h), flags=cv2.INTER_NEAREST
    )

    return gris_alineada, masc_alineada, angulo


def corregir_flip_vertical(gris_a, masc_a, mascara_plantilla):
    """Resuelve la ambigüedad de espejo vertical usando la plantilla.

    Tras alinear, el tornillo puede quedar "boca arriba" o "boca abajo"
    (espejado respecto de su eje): la rotación por momentos no distingue
    entre ambas. Como la silueta de la rosca no es perfectamente simétrica
    (los filetes forman dientes de sierra), comparamos la superposición
    (AND lógico) de la máscara contra la plantilla en ambas variantes y nos
    quedamos con la de mayor coincidencia.

    Devuelve:
        (gris, mascara, se_giro): las imágenes (giradas o no) y un bool.
    """
    volteada = cv2.flip(masc_a, 0)  # flip vertical (alrededor del eje x)

    # Superposición = cantidad de píxeles donde ambas máscaras son blancas
    overlap_normal = cv2.countNonZero(cv2.bitwise_and(masc_a, mascara_plantilla))
    overlap_flip = cv2.countNonZero(cv2.bitwise_and(volteada, mascara_plantilla))

    if overlap_flip > overlap_normal:
        return cv2.flip(gris_a, 0), volteada, True
    return gris_a, masc_a, False


def procesar_imagen(ruta, mascara_plantilla=None):
    """Pipeline de las fases 1-2 para una imagen: cargar -> segmentar -> alinear.

    Si se pasa `mascara_plantilla`, además corrige el espejo vertical.
    Es la función de conveniencia que usan calibrar.py, evaluar.py y main.py.

    Devuelve un diccionario con todos los productos intermedios, útil para
    visualizar el pipeline completo paso a paso.
    """
    img_rgb, img_gris = cargar_imagen(ruta)
    mascara = segmentar(img_gris)
    gris_a, masc_a, angulo = alinear(img_gris, mascara)

    se_giro = False
    if mascara_plantilla is not None:
        gris_a, masc_a, se_giro = corregir_flip_vertical(
            gris_a, masc_a, mascara_plantilla
        )

    return {
        "rgb": img_rgb,
        "gris": img_gris,
        "mascara": mascara,
        "gris_alineada": gris_a,
        "mascara_alineada": masc_a,
        "angulo": angulo,
        "flip": se_giro,
    }
