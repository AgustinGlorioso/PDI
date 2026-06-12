"""
detectores.py — Fases 3 y 4: Extracción de características y refinamiento.

Sobre la imagen ALINEADA (tornillo horizontal, cabeza a la izquierda,
anclado en columnas fijas) se aplican TRES detectores COMPLEMENTARIOS,
elegidos tras analizar visualmente todas las categorías del dataset.
Cada uno cubre una familia de defectos que los otros no ven:

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

  DETECTOR DE PERFIL DE ROSCA (envolventes invariantes a la fase)
      La banda de tolerancia anterior tiene un costo: los defectos CHICOS
      de las crestas (rebabas de 8-20 px, muescas de 10-25 px) caen justo
      dentro de la franja tolerada y resultan invisibles para el detector
      de forma. Este detector los recupera midiendo, columna a columna,
      las ENVOLVENTES de la silueta (distancia del eje a las crestas y a
      los valles, con filtros de máximo/mínimo deslizante de un paso de
      rosca). Esas envolventes son independientes de la fase del filete,
      así que se pueden comparar contra bandas [min, max] estrechas
      construidas con los tornillos sanos: una cresta aplastada, una
      rebaba o una punta deformada sacan la envolvente de su banda.

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
# Parámetros del detector de perfil de rosca
# ----------------------------------------------------------------------------
PASO_ROSCA = 63           # período del filete en px (medido en la plantilla)
COLS_PERFIL = (490, 990)  # columnas analizadas: zona roscada + punta
MARGEN_PERFIL = 2         # holgura (px) sobre la banda [min, max] de las good
                          # (la banda ya absorbe la variación entre 41 sanos;
                          # con más holgura se escapan muescas de 10-15 px)
FILA_EJE = 512            # fila del eje del tornillo tras la alineación

# ----------------------------------------------------------------------------
# Parámetros del detector de intensidad
# ----------------------------------------------------------------------------
EROSION_INTERNA_PX = 8    # radio de erosión de la máscara interna
Z_UMBRAL = 2.5            # desvíos estándar para declarar un píxel anómalo
                          # (con 2.5 sigmas, <1% de los píxeles sanos lo
                          # superaría si la variación fuera gaussiana; el
                          # filtrado por área elimina esos sueltos)
DIF_MINIMA = 20           # diferencia absoluta mínima (niveles de gris) que
                          # debe tener el píxel además del criterio z: evita
                          # marcar desviaciones estadísticamente "raras" pero
                          # visualmente insignificantes en zonas de std≈0
EPS_STD = 6.0             # regularización del denominador: impide que una
                          # std casi nula infle el z de diferencias mínimas
AREA_MIN_INT = 80         # área mínima de componente conexa de defecto
                          # (las facetas fresadas de thread_top responden
                          # con 100-300 px dispersos; con un mínimo mayor
                          # se perdían tras el cierre morfológico)


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


def perfiles_envolvente(mascara_alineada):
    """Extrae los 4 perfiles invariantes a la fase de la rosca.

    Para cada columna x de la máscara alineada se mide la distancia del eje
    (fila FILA_EJE) al borde superior e inferior de la silueta:

        semi_sup(x) = FILA_EJE - primera fila blanca   (radio hacia arriba)
        semi_inf(x) = última fila blanca - FILA_EJE    (radio hacia abajo)

    Esos perfiles "crudos" oscilan con los dientes del filete, cuya posición
    depende de cuánto esté girado el tornillo sobre su propio eje (fase).
    Para volverlos COMPARABLES entre tornillos se les aplica un filtro de
    máximo y de mínimo deslizante de ancho = 1 paso de rosca:

        cresta(x) = max(semi[x - P/2 : x + P/2])  -> envuelve los dientes
        valle(x)  = min(semi[x - P/2 : x + P/2])  -> sigue el núcleo

    Dentro de cualquier ventana de un paso entra exactamente un diente, así
    que el máximo (la cresta) y el mínimo (el valle) NO dependen de la fase.
    Una cresta aplastada hace caer 'cresta'; una rebaba doblada la hace
    subir; una rosca comida baja 'valle'. Los filtros max/min se implementan
    como dilatación/erosión morfológica 1D del perfil.

    Devuelve:
        dict con 4 arrays float32 de largo W:
        'cresta_sup', 'valle_sup', 'cresta_inf', 'valle_inf'
    """
    binaria = (mascara_alineada > 0)
    h, w = binaria.shape

    # Borde superior: primera fila blanca de cada columna (argmax devuelve
    # el índice del primer True). Columnas vacías quedan en radio 0.
    hay_metal = binaria.any(axis=0)
    fila_sup = binaria.argmax(axis=0).astype(np.float32)
    fila_inf = (h - 1 - binaria[::-1].argmax(axis=0)).astype(np.float32)

    semi_sup = np.where(hay_metal, FILA_EJE - fila_sup, 0.0)
    semi_inf = np.where(hay_metal, fila_inf - FILA_EJE, 0.0)

    # Filtros deslizantes de máximo/mínimo = dilatación/erosión 1D con un
    # elemento estructurante horizontal de un paso de rosca
    ee_1d = cv2.getStructuringElement(cv2.MORPH_RECT, (PASO_ROSCA, 1))

    def _max_movil(perfil):
        return cv2.dilate(perfil.reshape(1, -1), ee_1d).ravel()

    def _min_movil(perfil):
        return cv2.erode(perfil.reshape(1, -1), ee_1d).ravel()

    return {
        "cresta_sup": _max_movil(semi_sup),
        "valle_sup": _min_movil(semi_sup),
        "cresta_inf": _max_movil(semi_inf),
        "valle_inf": _min_movil(semi_inf),
    }


def detectar_perfil(mascara_alineada, bandas):
    """Detector de PERFIL DE ROSCA: envolventes fuera de la banda admisible.

    Compara los 4 perfiles invariantes de la pieza contra las bandas
    [mínimo, máximo] que esos mismos perfiles toman en los 41 tornillos
    sanos (construidas por build_template.py), con una holgura adicional
    de MARGEN_PERFIL px. Todo lo que escapa de la banda es "violación",
    medida en píxeles de exceso por columna.

    Parámetros:
        bandas: dict {señal_min / señal_max: array}, de la plantilla.

    Devuelve:
        dict con:
            'violacion': array float32 (largo W) con el exceso total en px
                         de cada columna (suma de las 4 señales)
            'marcas':    máscara uint8 para visualización, con las columnas
                         infractoras pintadas sobre el borde correspondiente
    """
    perfiles = perfiles_envolvente(mascara_alineada)
    w = mascara_alineada.shape[1]
    violacion = np.zeros(w, dtype=np.float32)
    marcas = np.zeros_like(mascara_alineada)

    x0, x1 = COLS_PERFIL
    for nombre, perfil in perfiles.items():
        banda_min = bandas[f"{nombre}_min"] - MARGEN_PERFIL
        banda_max = bandas[f"{nombre}_max"] + MARGEN_PERFIL

        # Exceso por arriba (material que sobra) y por abajo (que falta)
        exceso = np.maximum(perfil - banda_max, 0) + np.maximum(banda_min - perfil, 0)
        exceso[:x0] = 0   # fuera de la zona analizada no se acumula
        exceso[x1:] = 0
        violacion += exceso

        # Marcar la columna infractora a la altura del borde medido, para
        # que la visualización señale el lugar del defecto
        for x in np.where(exceso > 0)[0]:
            radio = int(perfil[x])
            if "sup" in nombre:
                fila = max(FILA_EJE - radio, 0)
            else:
                fila = min(FILA_EJE + radio, mascara_alineada.shape[0] - 1)
            marcas[max(fila - 6, 0) : fila + 6, x] = 255

    return {"violacion": violacion, "marcas": marcas}


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

    # Suavizado leve de la pieza a inspeccionar: tolera errores de
    # registración de ~1 px sin aplastar las estrías finas (2-5 px) de los
    # arañazos, que son justamente la señal a preservar
    gris_f = cv2.GaussianBlur(gris_alineada, (3, 3), 0).astype(np.float32)

    # Mapa de diferencias contra el tornillo promedio
    dif = gris_f - media_gris

    # ------------------------------------------------------------------
    # Corrección de iluminación por fila (SOLO en el cuello): al rotar el
    # tornillo sobre su propio eje entre foto y foto, cambia qué
    # generatriz del cilindro recibe el reflejo de la luz, y FILAS
    # COMPLETAS del cuello aparecen hasta +100 niveles más claras que el
    # promedio (se observó en piezas sanas). Ese corrimiento es GLOBAL a
    # lo largo de la fila, mientras que un defecto real es LOCAL (un
    # raspón ocupa ~60 de las ~300 columnas del cuello). Restando a cada
    # fila su MEDIANA dentro de la región se elimina el corrimiento sin
    # tocar el defecto: la mediana es robusta porque el defecto nunca
    # cubre más de la mitad de la fila.
    #
    # NO se aplica en cabeza/rosca/punta: allí cada fila interior tiene
    # pocos píxeles de metal (los dientes del filete, el bisel), y un
    # defecto puede dominar la mediana de su propia fila y autocancelarse.
    # ------------------------------------------------------------------
    interna_bool = mascara_interna > 0

    x0, x1 = REGIONES["cuello"]
    sub = dif[:, x0:x1]
    msub = interna_bool[:, x0:x1]
    # mediana de cada fila considerando SOLO píxeles interiores del
    # tornillo; las filas sin metal no se corrigen
    filas_validas = msub.any(axis=1)
    con_nan = np.where(msub, sub, np.nan)
    mediana_fila = np.zeros(sub.shape[0], dtype=np.float32)
    mediana_fila[filas_validas] = np.nanmedian(con_nan[filas_validas], axis=1)
    dif[:, x0:x1] = sub - mediana_fila[:, None]

    # Puntaje z píxel a píxel sobre la diferencia ya corregida
    z = dif / (std_gris + EPS_STD)

    # Doble condición (estadística Y visual), separada por polaridad
    brillante = ((z > Z_UMBRAL) & (dif > DIF_MINIMA)).astype(np.uint8) * 255
    oscuro = ((z < -Z_UMBRAL) & (dif < -DIF_MINIMA)).astype(np.uint8) * 255

    # Restringir al interior del tornillo
    brillante = cv2.bitwise_and(brillante, mascara_interna)
    oscuro = cv2.bitwise_and(oscuro, mascara_interna)

    # Refinamiento: CIERRE (no apertura) + filtrado por área. Los arañazos
    # son estrías paralelas finas (2-5 px) separadas por surcos: una
    # apertura las borraría una por una, mientras que el cierre las SUELDA
    # en un solo parche compacto que sí supera el filtro de área. El ruido
    # disperso, en cambio, no llega a aglutinarse en nada grande.
    ee_cierre = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    brillante = cv2.morphologyEx(brillante, cv2.MORPH_CLOSE, ee_cierre)
    oscuro = cv2.morphologyEx(oscuro, cv2.MORPH_CLOSE, ee_cierre)

    brillante = _filtrar_por_area(brillante, AREA_MIN_INT)
    oscuro = _filtrar_por_area(oscuro, AREA_MIN_INT)

    return {"brillante": brillante, "oscuro": oscuro}


def calcular_scores(forma, intensidad, perfil, mascara_plantilla_binaria):
    """Convierte las salidas de los detectores en un vector de 10 scores.

    Para cada región (cabeza, cuello, rosca, punta) se calculan dos scores
    de área normalizada:

        forma_<region> = px de defecto de forma (faltante+sobrante) en la
                         región / área de la plantilla en esa región
        int_<region>   = px de defecto de intensidad (brillante+oscuro) en
                         la región / área de la plantilla en esa región

    y para las dos regiones que cubre el detector de perfil de rosca:

        perfil_rosca / perfil_punta = promedio del exceso de envolvente
                                      (px por columna) en la región

    Normalizar (por área de región o por cantidad de columnas) hace los
    scores comparables entre regiones de distinto tamaño y simplifica la
    calibración de umbrales.

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

    # Scores del perfil de rosca: exceso promedio por columna en la región.
    # La zona analizada por el perfil (COLS_PERFIL) se reparte entre rosca
    # y punta usando la misma frontera de REGIONES.
    violacion = perfil["violacion"]
    xr0, xr1 = REGIONES["rosca"]
    xp0, xp1 = REGIONES["punta"]
    xa0, xa1 = COLS_PERFIL
    cols_rosca = (max(xr0, xa0), min(xr1, xa1))
    cols_punta = (max(xp0, xa0), min(xp1, xa1))

    scores["perfil_rosca"] = float(
        violacion[cols_rosca[0] : cols_rosca[1]].mean()
    )
    scores["perfil_punta"] = float(
        violacion[cols_punta[0] : cols_punta[1]].mean()
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
    perfil = detectar_perfil(mascara_alineada, plantillas["bandas_perfil"])
    scores = calcular_scores(forma, intensidad, perfil, plantillas["binaria"])

    return {
        "forma": forma,
        "intensidad": intensidad,
        "perfil": perfil,
        "scores": scores,
    }
