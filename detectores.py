"""
detectores.py — Fases 3 y 4: Extracción de características y refinamiento.

Sobre la imagen ALINEADA (tornillo horizontal, cabeza a la izquierda,
anclado en columnas fijas) se aplican dos detectores COMPLEMENTARIOS,
elegidos tras analizar visualmente todas las categorías del dataset:

  DETECTOR DE FORMA (silueta vs. plantilla estadística)
      Captura defectos que cambian el CONTORNO del tornillo: rosca
      deformada/aplastada (thread_*), punta doblada, machacada o truncada
      (manipulated_front). Compara la máscara alineada contra las zonas
      estadísticas construidas con los 41 tornillos sanos:
        - material FALTANTE: huecos dentro del NÚCLEO (donde >=97% de los
          tornillos sanos tienen metal),
        - material SOBRANTE: metal dentro del EXTERIOR (donde <=3% de los
          tornillos sanos tienen metal).
      La franja intermedia (banda de tolerancia) absorbe la variación
      natural de la fase de la rosca (±10-15 px según el giro del tornillo
      sobre su propio eje) y NO genera detecciones.

  DETECTOR DE INTENSIDAD (comparación estadística contra golden template)
      Captura defectos que NO cambian la silueta: arañazos en la cabeza
      (estrías BRILLANTES, picos de +60..+120 niveles), raspones y gubias
      en el cuello (pueden ser brillantes U oscuros) y facetas fresadas o
      picaduras en la rosca. Cada píxel del tornillo alineado se compara
      contra el modelo estadístico construido con los 41 tornillos sanos:

          z(x,y) = ( gris(x,y) - media_good(x,y) ) / ( std_good(x,y) + eps )

      Es decir, se mide CUÁNTOS DESVÍOS ESTÁNDAR se aparta cada píxel de lo
      esperado en esa posición. La clave es que la tolerancia se ADAPTA
      sola: en zonas donde los tornillos sanos varían mucho (brillos
      especulares del cuello, dientes de rosca que cambian de fase, huella
      de la cabeza) la std es grande y el detector es tolerante; en zonas
      estables, una desviación moderada ya es significativa.

      (Se probó primero la alternativa top-hat/black-hat puramente local,
      pero los operadores hat responden ante las estructuras NATURALES del
      tornillo —brillos del cuello, valles oscuros entre crestas— y la
      respuesta de base en piezas sanas llegaba al 30-70% del área de cada
      región, enterrando los defectos reales. La comparación estadística
      posicional no tiene ese problema porque "sabe" qué apariencia es
      normal en cada lugar.)

      Se evalúa solo DENTRO de una máscara interna erosionada, para que el
      borde del tornillo (transición metal-fondo, sombras pegadas al
      contorno) no dispare falsas detecciones.

Ambos detectores devuelven máscaras binarias de anomalía que luego se
refinan con apertura morfológica (elimina ruido puntual) y filtrado por
área de componentes conexas (elimina detecciones espurias chicas).
"""

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Regiones del tornillo (columnas en el espacio alineado/plantilla)
# ----------------------------------------------------------------------------
# Medidas sobre el perfil de anchos de la plantilla binaria (ver informe):
#   cabeza : ancho 284 px decreciendo hasta ~125 (bisel del avellanado)
#   cuello : vástago liso de ancho constante ~104 px
#   rosca  : ancho oscilante 95-111 px (dientes del filete)
#   punta  : cono final decreciente (incluye la muesca espiral normal de la
#            última vuelta, que se tolera vía la calibración con las "good")
# Como la alineación ancla el tornillo en columnas fijas (x_min = 50,
# largo 910±5 px), estas fronteras valen para TODAS las imágenes alineadas.
REGIONES = {
    "cabeza": (0, 185),
    "cuello": (185, 490),
    "rosca": (490, 810),
    "punta": (810, 1024),
}

# ----------------------------------------------------------------------------
# Parámetros del detector de forma
# ----------------------------------------------------------------------------
# Margen de seguridad adicional sobre las zonas estadísticas: se erosionan
# NÚCLEO y EXTERIOR unos píxeles para exigir que el defecto penetre
# "bien adentro" de la zona y no marcar desbordes de 1-2 px por la
# discretización de la rotación.
TOL_FORMA_PX = 5          # radio de erosión de las zonas (px)
AREA_MIN_FORMA = 120      # área mínima de una componente conexa de defecto

# ----------------------------------------------------------------------------
# Parámetros del detector de intensidad
# ----------------------------------------------------------------------------
EROSION_INTERNA_PX = 8    # radio de erosión de la máscara interna
Z_UMBRAL = 3.0            # desvíos estándar para declarar un píxel anómalo
                          # (regla de las 3 sigmas: ~0.3% de falsos positivos
                          # si la variación normal fuera gaussiana)
DIF_MINIMA = 25           # diferencia absoluta mínima (niveles de gris) que
                          # debe tener el píxel además del criterio z: evita
                          # marcar desviaciones estadísticamente "raras" pero
                          # visualmente insignificantes en zonas de std≈0
EPS_STD = 6.0             # regularización del denominador: impide que una
                          # std casi nula infle el z de diferencias mínimas
AREA_MIN_INT = 150        # área mínima de componente conexa de defecto


def _filtrar_por_area(mascara_binaria, area_minima):
    """Elimina componentes conexas de área menor a `area_minima`.

    Implementa el refinamiento por etiquetado de componentes conexas:
    una detección real (arañazo, rebaba) forma una región compacta de
    decenas/cientos de píxeles; el ruido (brillos puntuales, polvo, borde
    residual) forma manchitas chicas que se descartan acá.
    """
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(
        mascara_binaria, connectivity=8
    )
    salida = np.zeros_like(mascara_binaria)
    for i in range(1, n):  # la etiqueta 0 es el fondo
        if stats[i, cv2.CC_STAT_AREA] >= area_minima:
            salida[etiquetas == i] = 255
    return salida


def detectar_forma(mascara_alineada, nucleo, exterior):
    """Detector de FORMA: compara la silueta contra la plantilla estadística.

    Parámetros:
        mascara_alineada: máscara binaria {0,255} del tornillo a inspeccionar.
        nucleo:   zona "siempre metal" de la plantilla (de build_template).
        exterior: zona "nunca metal" de la plantilla.

    Devuelve:
        dict con las máscaras de defecto:
            'faltante': material ausente donde siempre debería haber metal
                        (rosca aplastada, punta truncada, muescas)
            'sobrante': material presente donde nunca debería haberlo
                        (rebabas, puntas dobladas en gancho)
    """
    # Erosión de seguridad de ambas zonas (ver TOL_FORMA_PX)
    ee_tol = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * TOL_FORMA_PX + 1, 2 * TOL_FORMA_PX + 1)
    )
    nucleo_seguro = cv2.erode(nucleo, ee_tol)
    exterior_seguro = cv2.erode(exterior, ee_tol)

    # FALTANTE = núcleo AND (NOT tornillo): debería haber metal y no hay
    faltante = cv2.bitwise_and(nucleo_seguro, cv2.bitwise_not(mascara_alineada))
    # SOBRANTE = tornillo AND exterior: hay metal donde nunca debería
    sobrante = cv2.bitwise_and(mascara_alineada, exterior_seguro)

    # Refinamiento: apertura (quita píxeles sueltos por discretización de la
    # rotación) y filtrado por área de componentes conexas
    ee_chico = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    faltante = cv2.morphologyEx(faltante, cv2.MORPH_OPEN, ee_chico)
    sobrante = cv2.morphologyEx(sobrante, cv2.MORPH_OPEN, ee_chico)

    faltante = _filtrar_por_area(faltante, AREA_MIN_FORMA)
    sobrante = _filtrar_por_area(sobrante, AREA_MIN_FORMA)

    return {"faltante": faltante, "sobrante": sobrante}


def detectar_intensidad(gris_alineada, mascara_alineada, media_gris, std_gris):
    """Detector de INTENSIDAD: compara el gris contra el modelo estadístico.

    Para cada píxel interior del tornillo se calcula el puntaje z:

        z = (gris - media_good) / (std_good + EPS_STD)

    y se marca como anómalo si se cumplen LAS DOS condiciones:
        |z| > Z_UMBRAL        (desviación estadísticamente significativa)
        |dif| > DIF_MINIMA    (desviación visualmente significativa)

    La doble condición evita los dos modos de falso positivo: el criterio z
    solo se dispararía con diferencias diminutas en zonas de std≈0, y el
    criterio absoluto solo marcaría los brillos naturales del metal que la
    std ya explica.

    El análisis se restringe a una MÁSCARA INTERNA (erosión de la máscara
    del tornillo): cerca del borde la media/std de la plantilla mezclan
    metal con fondo y la comparación no es confiable; además los defectos
    de borde ya los captura el detector de forma.

    Devuelve:
        dict con las máscaras de defecto:
            'brillante': píxeles anómalamente claros (arañazos, facetas)
            'oscuro':    píxeles anómalamente oscuros (gubias, picaduras)
    """
    # Máscara interna: erosión con disco de radio EROSION_INTERNA_PX
    ee_int = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * EROSION_INTERNA_PX + 1, 2 * EROSION_INTERNA_PX + 1),
    )
    mascara_interna = cv2.erode(mascara_alineada, ee_int)

    # Suavizado leve de la pieza a inspeccionar: la media de la plantilla
    # promedia 41 imágenes (es intrínsecamente suave), así que suavizamos
    # también la pieza para comparar señales del mismo "ancho de banda" y
    # tolerar errores de registración de ~1 px
    gris_f = cv2.GaussianBlur(gris_alineada, (5, 5), 0).astype(np.float32)

    # Mapa de diferencias y puntaje z píxel a píxel
    dif = gris_f - media_gris
    z = dif / (std_gris + EPS_STD)

    # Doble condición (estadística Y visual), separada por polaridad
    brillante = ((z > Z_UMBRAL) & (dif > DIF_MINIMA)).astype(np.uint8) * 255
    oscuro = ((z < -Z_UMBRAL) & (dif < -DIF_MINIMA)).astype(np.uint8) * 255

    # Restringir al interior del tornillo
    brillante = cv2.bitwise_and(brillante, mascara_interna)
    oscuro = cv2.bitwise_and(oscuro, mascara_interna)

    # Refinamiento: apertura + filtrado por área (igual que en forma)
    ee_chico = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    brillante = cv2.morphologyEx(brillante, cv2.MORPH_OPEN, ee_chico)
    oscuro = cv2.morphologyEx(oscuro, cv2.MORPH_OPEN, ee_chico)

    brillante = _filtrar_por_area(brillante, AREA_MIN_INT)
    oscuro = _filtrar_por_area(oscuro, AREA_MIN_INT)

    return {"brillante": brillante, "oscuro": oscuro}


def calcular_scores(forma, intensidad, mascara_plantilla_binaria):
    """Convierte las máscaras de defecto en un vector de 8 scores.

    Para cada región (cabeza, cuello, rosca, punta) se calculan dos scores:

        forma_<region> = px de defecto de forma (faltante+sobrante) en la
                         región / área de la plantilla en esa región
        int_<region>   = px de defecto de intensidad (brillante+oscuro) en
                         la región / área de la plantilla en esa región

    Normalizar por el área de la región hace comparables los scores entre
    regiones de distinto tamaño y los vuelve adimensionales (fracción de la
    región afectada), lo que simplifica fijar umbrales.

    Devuelve:
        dict {nombre_score: valor float}
    """
    # Unión de las evidencias de cada detector
    defecto_forma = cv2.bitwise_or(forma["faltante"], forma["sobrante"])
    defecto_int = cv2.bitwise_or(intensidad["brillante"], intensidad["oscuro"])

    scores = {}
    for nombre, (x0, x1) in REGIONES.items():
        # Área de referencia: píxeles de tornillo "típico" en la región
        area_region = max(
            cv2.countNonZero(mascara_plantilla_binaria[:, x0:x1]), 1
        )
        scores[f"forma_{nombre}"] = (
            cv2.countNonZero(defecto_forma[:, x0:x1]) / area_region
        )
        scores[f"int_{nombre}"] = (
            cv2.countNonZero(defecto_int[:, x0:x1]) / area_region
        )

    return scores


def inspeccionar(gris_alineada, mascara_alineada, plantillas):
    """Corre ambos detectores y devuelve máscaras de defecto + scores.

    Parámetros:
        plantillas: dict con 'nucleo', 'exterior', 'binaria', 'media_gris_f'
                    y 'std_gris' (cargado por clasificador.cargar_plantillas()).

    Es la función que usan calibrar.py, evaluar.py y main.py.
    """
    forma = detectar_forma(
        mascara_alineada, plantillas["nucleo"], plantillas["exterior"]
    )
    intensidad = detectar_intensidad(
        gris_alineada, mascara_alineada,
        plantillas["media_gris_f"], plantillas["std_gris"],
    )
    scores = calcular_scores(forma, intensidad, plantillas["binaria"])

    return {"forma": forma, "intensidad": intensidad, "scores": scores}
