# Documentación técnica del sistema — cómo y por qué se desarrolló

Este documento explica, etapa por etapa y función por función, cómo está
pensado el sistema de detección de anomalías en tornillos. La idea es que
puedan **entender el razonamiento detrás de cada decisión**, no solo qué
hace el código. Está pensado para leerse de arriba a abajo: cada sección
construye sobre la anterior.

---

## 0. La idea central (leer esto primero)

El dataset MVTec AD *screw* tiene una propiedad que define toda la estrategia:
**solo tenemos ejemplos de lo que es NORMAL** (41 tornillos sanos), y los
defectos son muy variados (arañazos, roscas aplastadas, puntas dobladas…).
No podemos "aprender" cada tipo de defecto porque hay pocos ejemplos de cada
uno y son todos distintos.

La solución clásica para esto se llama **detección de anomalías por modelo
de referencia**: en lugar de aprender cómo son los defectos, aprendemos
con mucho detalle **cómo es un tornillo sano**, y marcamos como defecto
*todo lo que se aparta de esa referencia*. Es exactamente el enfoque que
pide la consigna del proyecto.

Para que esa comparación funcione, todos los tornillos tienen que estar en
la **misma posición** en la imagen (si uno está rotado 30° y otro 90°,
compararlos píxel a píxel no tiene sentido). Por eso el pipeline tiene dos
grandes mitades:

1. **Normalización** (fases 1-2): llevar cualquier tornillo a una pose
   canónica fija — horizontal, cabeza a la izquierda, siempre en el mismo
   lugar de la imagen.
2. **Comparación** (fases 3-5): medir en qué se diferencia de la referencia
   y decidir si esa diferencia es un defecto.

```
                    NORMALIZACIÓN                    COMPARACIÓN
   imagen ──> segmentar ──> alinear ──> 3 detectores ──> scores ──> diagnóstico
              (fase 1)      (fase 2)    (fases 3-4)               (fase 5)
```

---

## 1. Flujo de datos entre archivos

Antes de entrar al detalle, conviene ver cómo se conectan los 7 archivos:

```
  build_template.py  ──genera──>  plantillas/   (modelo del tornillo sano)
         │                            │
         │ usa                        │ usa
         ▼                            ▼
   pipeline.py  <──importa──  detectores.py  <──importa──  clasificador.py
   (fases 1-2)                (fases 3-4)                   (fase 5)
         ▲                            ▲                            ▲
         └────────────┬───────────────┴────────────────────────────┘
                      │ todos los usan
          ┌───────────┼────────────┐
     calibrar.py   evaluar.py    main.py
     (umbrales)    (métricas)    (demo)
```

- `pipeline.py`, `detectores.py` y `clasificador.py` son **librerías**: solo
  definen funciones, no "hacen" nada por sí solas.
- `build_template.py`, `calibrar.py`, `evaluar.py` y `main.py` son
  **programas ejecutables** que orquestan a las librerías.

**Orden obligatorio de ejecución** (cada uno produce lo que el siguiente
necesita):

```bash
python build_template.py   # 1. aprende cómo es un tornillo sano
python calibrar.py         # 2. fija los umbrales de decisión
python evaluar.py          # 3. (opcional) mide el rendimiento
python main.py <imagen>    # 4. inspecciona una imagen concreta
```

---

## 2. El análisis previo del dataset (de dónde salieron las decisiones)

Antes de escribir el detector, se examinaron ~8 imágenes de cada una de las
6 categorías. Este análisis no es código, pero es **la base de todo el
diseño**. Lo que se encontró:

| Categoría | Dónde está el defecto | Cómo se ve | ¿Cambia la silueta? |
|---|---|---|---|
| `good` | — (referencia) | — | — |
| `scratch_head` | cabeza | estrías **brillantes** (+60..+120 niveles) | no, solo intensidad |
| `scratch_neck` | cuello | raspones **brillantes U oscuros** | no, solo intensidad |
| `thread_side` | rosca (flanco) | rebabas/muescas en las crestas | **sí**, la silueta |
| `thread_top` | rosca | facetas fresadas / desgarros | sí (silueta + intensidad) |
| `manipulated_front` | punta | doblada, machacada o truncada | **sí**, la silueta |

**Tres "trampas" en los tornillos sanos** que hay que tolerar para no tener
falsos positivos:

1. **Sombra difusa** alrededor del tornillo → arruina el umbralado de Otsu.
2. **Bandas especulares en el cuello**: según cómo rotó el tornillo sobre su
   propio eje, una franja del cuello refleja la luz y aparece hasta +100
   niveles más clara. Esto NO es un defecto.
3. **Fase de la rosca**: los dientes del filete caen en posiciones distintas
   en cada foto (el tornillo está girado distinto). Comparar la rosca píxel
   a píxel da diferencias enormes entre dos tornillos *sanos*.

Conclusión que guió todo: **ningún detector único sirve**. Los arañazos
necesitan análisis de intensidad; las roscas y puntas necesitan análisis de
forma; y la forma de la rosca necesita un truco extra (las envolventes) por
culpa de la fase. De ahí los **tres detectores complementarios**.

---

## 3. `pipeline.py` — Fases 1 y 2 (normalización)

### 3.1 `cargar_imagen(ruta)`

Lee el PNG y lo devuelve en dos formatos: **RGB** (para mostrar con
Matplotlib, que espera RGB) y **escala de grises** (sobre la que trabaja
todo el procesamiento). OpenCV lee en BGR, por eso se convierte
explícitamente. La conversión a gris se justifica porque el color no aporta
información en este problema (el tornillo es metálico gris sobre fondo gris):
todo el defecto está en la **geometría** y en los **cambios de intensidad**.

### 3.2 `segmentar(img_gris)` — separar el tornillo del fondo

**El problema:** queremos una máscara binaria (blanco = tornillo, negro =
fondo). El camino "obvio" es el umbralado de **Otsu**: como el fondo es
claro y el metal oscuro, un umbral los separa. **Pero falla**: medimos que
con Otsu el área de los tornillos sanos variaba entre 124.000 y 169.000 px
(un error de hasta 70%), porque la **sombra difusa** alrededor del tornillo
tiene niveles de gris por debajo del umbral y se incluye como si fuera
objeto.

**La solución — segmentar por BORDES en vez de por nivel de gris:**

La clave es notar que la transición **metal→fondo** es un **borde abrupto**
(salto de ~100 niveles en pocos píxeles), mientras que la sombra es un
**degradé suave** (cambia de a pocos niveles por píxel). Un detector de
bordes ve el contorno del tornillo e **ignora** el degradé de la sombra.

Los pasos (y el porqué de cada uno):

1. **`GaussianBlur(3,3)`** — suavizado leve. Atenúa el ruido del sensor.
   Se usa un kernel chico (3×3) a propósito: uno grande difuminaría los
   bordes débiles que justamente queremos detectar.
2. **`Canny(40, 90)`** — el detector de bordes de la Unidad V. Internamente
   hace: gradiente + supresión de no-máximos + **histéresis**. Los dos
   umbrales (40 bajo, 90 alto) son los de la histéresis: el borde metal-fondo
   supera siempre el alto; el degradé de la sombra no llega ni al bajo.
3. **`dilate(elipse 5×5, 2 iter)`** — "engorda" los bordes. Canny a veces
   deja el contorno con pequeños cortes; la dilatación los suelda para que
   la curva quede **cerrada** (condición necesaria para poder rellenarla).
4. **`findContours` + `drawContours(FILLED)`** — toma el contorno externo de
   **mayor área** (el tornillo, descartando motas de polvo) y lo pinta
   macizo. FILLED rellena el interior, eliminando huecos por reflejos del
   metal.
5. **`erode(mismo kernel, 2 iter)`** — deshace la dilatación del paso 3. Como
   habíamos agrandado la silueta ~2 px por lado, la erosión simétrica la
   devuelve a su tamaño real.

**Resultado:** el área de los tornillos sanos queda constante (~100.000 px,
±1%), que es la estabilidad que la comparación contra plantilla necesita.

> Los pasos 3+4+5 juntos son un "cierre morfológico dirigido": cerramos el
> contorno, rellenamos y restauramos el tamaño. Es morfología de la Unidad VI
> aplicada con un propósito concreto.

### 3.3 `_angulo_por_momentos(mascara)` — orientación del eje

Para poner el tornillo horizontal primero hay que saber **cuánto está
rotado**. Se usan los **momentos de segundo orden** de la máscara:

```
theta = 0.5 * atan2(2·mu11, mu20 - mu02)
```

Los momentos centrales `mu20`, `mu02`, `mu11` describen cómo se distribuye
la masa del objeto alrededor de su centroide. Esta fórmula da el ángulo del
**eje de mínima inercia** (el eje "largo" del tornillo). Es matemáticamente
**equivalente a un PCA** sobre las coordenadas de los píxeles, pero en forma
cerrada (sin calcular autovectores). El centroide `(cx, cy)` sale de los
momentos de primer orden normalizados por el área.

### 3.4 `alinear(img_gris, mascara)` — llevar a pose canónica

Esta es la función más importante de la normalización. Pasos:

1. **Orientación** por momentos (la función anterior).
2. **Rotación de prueba** de la máscara para dejar el eje horizontal.
3. **Decisión cabeza-izquierda:** la cabeza es la parte más ancha del
   tornillo, así que concentra más píxeles. Se compara la masa de la mitad
   izquierda contra la derecha de la máscara rotada; si la cabeza quedó a la
   derecha, se suman 180° y se rota de nuevo.
4. **Anclaje (la parte clave):** se construye **una sola** matriz afín que
   combina rotación + traslación, de modo que:
   - el **extremo izquierdo** (la cabeza) caiga siempre en la columna
     `MARGEN_X = 50`,
   - el **eje del tornillo** caiga siempre en la fila central (512).

   **¿Por qué anclar por la cabeza y no por el centroide?** Porque la cabeza
   es estable en *todas* las categorías: los arañazos no la cambian, y los
   defectos de forma (rosca/punta) están del lado opuesto. Si ancláramos por
   el centroide, un tornillo con la punta rota tendría el centroide corrido
   y **todo** el tornillo quedaría desplazado respecto de la plantilla,
   generando diferencias falsas en toda la pieza. Anclando por la cabeza, el
   defecto de la punta queda localizado donde realmente está.

5. **Una única `warpAffine`** sobre la imagen real. Se combina todo en una
   sola transformación a propósito: cada interpolación degrada un poco la
   imagen, así que aplicar rotación y traslación juntas (en vez de en dos
   pasos) preserva la calidad. Se usa `BORDER_REPLICATE` para que las zonas
   nuevas se rellenen con el fondo claro y no con negro (que confundiría a
   los detectores).

**Verificación:** tras alinear, las 41 imágenes sanas quedan con el extremo
izquierdo en x=50 (±0 px) y el eje en la fila 512 (±0,2 px). Esa precisión
es la que hace posible comparar píxel a píxel.

### 3.5 `corregir_flip_vertical(...)` — resolver el espejo

Queda una última ambigüedad: el tornillo puede estar "boca arriba" o "boca
abajo" (espejado respecto de su eje horizontal). La rotación por momentos no
distingue entre ambas. Como la silueta de la rosca no es simétrica (los
dientes forman un sierra), se compara la **superposición** (AND lógico) de
la máscara contra la plantilla en ambas variantes (normal y espejada) y se
elige la de mayor coincidencia.

### 3.6 `procesar_imagen(ruta, mascara_plantilla)`

Función de conveniencia que encadena todo: cargar → segmentar → alinear →
(corregir flip). Devuelve un diccionario con todos los productos intermedios
para poder visualizarlos. Es la que llaman `calibrar.py`, `evaluar.py` y
`main.py`.

---

## 4. `build_template.py` — construir el modelo del tornillo sano

Este programa procesa las 41 imágenes `good` y construye el **"golden
template"**: la descripción estadística de cómo es un tornillo sano. Es
literalmente "lo que el sistema considera normal".

### 4.1 `cargar_good_alineadas()` — dos pasadas de orientación

Procesa las 41 imágenes y resuelve el espejo vertical de cada una. Lo hace en
**dos pasadas**, y el porqué es sutil pero importante:

- **Pasada 1:** la primera imagen fija la orientación de referencia; cada
  siguiente se orienta contra el **promedio acumulado** de las ya aceptadas.
  El promedio se va volviendo más estable a medida que se suman imágenes.
- **Pasada 2:** con la plantilla ya completa, se vuelve a decidir el flip de
  **todas** las imágenes contra ella. Esto es imprescindible: durante la
  inspección, el flip de cada pieza nueva se decide contra la plantilla
  *final*. Si en la pasada 1 alguna imagen sana quedó orientada contra un
  promedio parcial distinto, esa misma imagen "violaría" las bandas
  construidas con ella → falso positivo. La segunda pasada garantiza
  consistencia.

### 4.2 `construir_plantillas(grises, mascaras)` — los productos estadísticos

Apila las 41 máscaras y grises y calcula:

- **`media_mascara`** — para cada píxel, qué **fracción** de los 41 tornillos
  sanos tiene metal ahí. Un valor de 1.0 = los 41 lo tienen; 0.0 = ninguno.
- **`nucleo`** = píxeles con media ≥ 0.97 (≥40 de 41 tornillos tienen metal):
  zona donde **siempre** hay metal. Si a una pieza le **falta** material acá,
  es anómala.
- **`exterior`** = píxeles con media ≤ 0.03 (≤1 de 41): zona donde **nunca**
  hay metal. Si una pieza tiene material acá, le **sobra** → anómala.
- La franja intermedia (0.03–0.97) es la **banda de tolerancia**: ahí los
  sanos a veces tienen metal y a veces no (por la fase de la rosca), así que
  **no se decide nada**.
- **`media_gris`** y **`std_gris`** — el tornillo "promedio" en escala de
  grises y, píxel a píxel, **cuánto varía** la apariencia entre tornillos
  sanos. Estos dos mapas son el corazón del detector de intensidad (sección
  5.3).
- **`bandas_perfil`** — para las 4 envolventes (cresta/valle × sup/inf), el
  mínimo y el máximo que toman en los 41 sanos, columna a columna. Es la
  banda admisible del detector de perfil (sección 5.4).

### 4.3 `guardar_plantillas` / `visualizar`

Persisten todo en la carpeta `plantillas/` (PNG para las máscaras, `.npy`
para los mapas float, `.npz` para las bandas) y generan una figura resumen.
Estos archivos son **artefactos regenerables**, por eso están en `.gitignore`
(se reconstruyen con `build_template.py`).

---

## 5. `detectores.py` — Fases 3 y 4 (los tres detectores)

El núcleo del sistema. Sobre la imagen alineada corren tres detectores, cada
uno especializado en una familia de defectos.

### 5.0 Las regiones

`REGIONES` divide el tornillo en 4 tramos de columnas fijas
(cabeza 0-185, cuello 185-490, rosca 490-810, punta 810-1024). Como la
alineación ancla el tornillo siempre en el mismo lugar, **estas fronteras
valen para todas las imágenes**. Sirven para dos cosas: localizar dónde está
el defecto y, sobre todo, **deducir el tipo** (cabeza→scratch_head, etc.).

### 5.1 `_filtrar_por_area(mascara, area_minima)` — refinamiento

Herramienta común a los tres detectores. Usa **componentes conexas**
(Unidad V/VI): etiqueta cada "manchón" de la máscara y elimina los de área
menor a un umbral. Un defecto real forma una región compacta de decenas o
cientos de píxeles; el ruido (brillos puntuales, polvo, bordes residuales)
forma manchitas chicas que se descartan así.

### 5.2 `detectar_forma(mascara, nucleo, exterior)` — defectos de silueta

Compara la silueta de la pieza contra las zonas estadísticas:

- **`faltante`** = `nucleo AND (NOT pieza)` → debería haber metal y no lo
  hay (rosca aplastada, punta truncada).
- **`sobrante`** = `pieza AND exterior` → hay metal donde nunca debería
  (rebaba, punta doblada en gancho).

Antes se erosionan núcleo y exterior unos píxeles (`TOL_FORMA_PX = 5`) como
margen de seguridad, para no marcar desbordes de 1-2 px por la
discretización de la rotación. Después se refina con apertura morfológica +
filtrado por área. **Es todo álgebra de conjuntos (Unidad VI):** AND, NOT,
erosión, apertura.

### 5.3 `detectar_intensidad(...)` — defectos de gris (el más elaborado)

Detecta arañazos y raspones, que **no cambian la silueta** pero sí el gris
interno. La idea es comparar cada píxel contra el modelo estadístico:

```
z(x,y) = (gris(x,y) - media_good(x,y)) / (std_good(x,y) + eps)
```

`z` mide **cuántos desvíos estándar** se aparta el píxel de lo esperado *en
esa posición exacta*. Lo potente: la tolerancia **se adapta sola**. Donde los
sanos varían mucho (brillos del cuello, dientes de rosca) la `std` es grande
y el detector es permisivo; donde son estables, una desviación chica ya
resalta.

Un píxel se marca como anómalo solo si cumple **las dos** condiciones:
- `|z| > 2.5` — estadísticamente significativo, **y**
- `|dif| > 20` niveles — visualmente significativo.

La doble condición evita los dos modos de falso positivo: el criterio `z`
solo se dispararía con diferencias diminutas en zonas de `std≈0`, y el
criterio absoluto solo marcaría los brillos naturales que la `std` ya
explica. Se separa por **polaridad** (`brillante` con z>0, `oscuro` con z<0)
porque los raspones del cuello pueden ser de cualquier signo.

**Corrección de iluminación por fila (el truco clave del cuello):** medimos
que en piezas *sanas*, filas enteras del cuello aparecían hasta +100 niveles
más claras por el reflejo especular (que cambia según la rotación axial).
Eso disparaba falsos positivos. La solución: a cada fila del cuello se le
resta su **mediana** (calculada solo sobre los píxeles interiores). El
corrimiento de iluminación es **global** a lo largo de la fila, así que la
mediana lo captura y se cancela; un defecto real es **local** (ocupa ~60 de
las ~300 columnas del cuello) y no mueve la mediana. **Solo se aplica en el
cuello**: en cabeza/rosca/punta cada fila tiene pocos píxeles de metal y un
defecto podría dominar su propia mediana y autocancelarse.

**Refinamiento con CIERRE (no apertura):** los arañazos son estrías finas
(2-5 px) separadas por surcos. Una apertura las borraría una por una; el
**cierre** las **suelda** en un parche compacto que sí supera el filtro de
área. El ruido disperso, en cambio, no se aglutina en nada grande.

> Nota de diseño honesta: el primer intento usó top-hat/black-hat puro
> (morfología local). Se descartó porque respondía a las estructuras
> *naturales* del tornillo (brillos, valles entre crestas), con respuesta de
> base del 30-70% del área en piezas sanas. La comparación estadística
> posicional no tiene ese problema porque "sabe" qué es normal en cada lugar.

### 5.4 `perfiles_envolvente(...)` y `detectar_perfil(...)` — la rosca

**El problema:** los defectos chicos de la rosca (rebabas de 8-20 px, muescas
de 10-25 px) **caen dentro de la banda de tolerancia** del detector de forma
(esa banda tiene que ser ancha por la fase de la rosca) → son invisibles. Y
comparar la rosca píxel a píxel no sirve porque la fase corre los dientes.

**La solución — envolventes invariantes a la fase:**

Para cada columna se mide la distancia del eje al borde superior e inferior
de la silueta (los "radios" `semi_sup`, `semi_inf`). Esos perfiles oscilan
con los dientes (dependen de la fase). El truco: aplicarles un **filtro de
máximo deslizante** y uno de **mínimo deslizante** de ancho = un paso de
rosca (63 px):

- `cresta(x) = max(semi en la ventana)` → envuelve las puntas de los dientes.
- `valle(x) = min(semi en la ventana)` → sigue el núcleo entre dientes.

Dentro de cualquier ventana de un paso entra **exactamente un diente**, así
que el máximo y el mínimo **no dependen de dónde** caiga ese diente → son
**invariantes a la fase**. Los filtros max/min se implementan como
**dilatación/erosión 1D** del perfil (morfología sobre una señal, Unidad VI).

`detectar_perfil` compara las 4 envolventes de la pieza contra las bandas
`[min, max]` de los 41 sanos (con holgura `MARGEN_PERFIL = 2 px`). Lo que se
sale de la banda es la "violación", medida en px de exceso por columna. Una
cresta aplastada hace caer la envolvente por debajo de la banda; una rebaba
la hace subir por encima. (Ver `informe/figuras/perfil_rosca.png`: se ve la
envolvente roja saliéndose de la banda verde justo donde está el defecto.)

### 5.5 `calcular_scores(...)` — del píxel al número

Convierte las máscaras de defecto en un **vector de 10 scores**:
- `forma_<region>` e `int_<region>` para las 4 regiones (8 scores): píxeles
  de defecto en la región, **normalizados por el área** de la región.
- `perfil_rosca` y `perfil_punta` (2 scores): exceso promedio de envolvente
  por columna en esas regiones.

Normalizar por área hace los scores **comparables entre regiones** de
distinto tamaño y adimensionales (fracción de la región afectada), lo que
simplifica fijar umbrales.

### 5.6 `inspeccionar(...)`

Corre los tres detectores y junta todo (máscaras + scores) en un diccionario.
Es la función que usan `calibrar.py`, `evaluar.py` y `main.py`.

---

## 6. `clasificador.py` — Fase 5 (decisión)

### 6.1 `cargar_plantillas()` / `cargar_umbrales()`

Cargan desde disco lo que produjeron `build_template.py` y `calibrar.py`.

### 6.2 `clasificar(scores, umbrales)` — el diagnóstico

1. Calcula el **exceso relativo** de cada score: `score / umbral`.
2. El **nivel de anomalía global** es el **mayor** de esos excesos. Si es
   ≤ 1.0, ningún score superó su umbral → **Normal**. Si es > 1.0 →
   **Anómala**. El 1.0 es el límite natural de decisión, y cuanto mayor el
   nivel, más severo el defecto.
3. El **tipo de defecto** lo da la **región del score más excedido**, gracias
   a que cada defecto vive en una región característica:
   - cabeza → `scratch_head`
   - cuello → `scratch_neck`
   - rosca → `thread`
   - punta → `manipulated_front`

Es un clasificador **por reglas, totalmente interpretable**: siempre se puede
explicar *por qué* dijo lo que dijo (qué score, en qué región, cuánto superó
el umbral). Esto es una ventaja enorme frente a una red neuronal "caja negra".

---

## 7. `calibrar.py` — fijar los umbrales (la estadística)

¿Cuánto defecto es "demasiado"? La respuesta sale de los datos sanos, no de
números inventados. Se corren las 41 `good` por el pipeline completo y se
registran sus 10 scores. Esos scores describen la **"respuesta normal"** del
sistema: el ruido residual que los detectores producen ante variaciones
legítimas (fase de rosca, brillos, polvo).

El umbral de cada score es:

```
umbral = max( media + 3·desvío,      ← regla de las 3 sigmas
              1.25 · máximo_sano,     ← margen sobre el peor caso real
              piso_mínimo )           ← piso absoluto
```

- **media + 3σ**: si la respuesta normal fuera gaussiana, solo el 0,13% de
  las piezas sanas la superaría → casi cero falsos positivos. Es la regla
  estadística clásica que pide la consigna.
- **1,25 × máximo observado**: protege cuando la distribución tiene colas más
  pesadas que la gaussiana; el peor caso real manda sobre la teoría.
- **piso mínimo**: si un score dio siempre 0 en las sanas, los dos criterios
  anteriores darían umbral 0 y cualquier píxel detectado sería una alarma. El
  piso exige al menos una fracción mínima razonable. Hay dos pisos según las
  unidades del score (área vs. perfil).

El resultado se guarda en `plantillas/umbrales.json` junto con las
estadísticas (media, desvío, máximo de cada score), para poder auditar y
justificar cada umbral en el informe.

---

## 8. `evaluar.py` — medir el rendimiento

Corre el pipeline completo sobre las **160 imágenes** y compara el
diagnóstico contra la verdad conocida (la carpeta de origen). `thread_side` y
`thread_top` se agrupan como `thread` porque ambas son rosca dañada y
distinguir la dirección no aporta a la decisión de aceptar/rechazar.

Reporta tres cosas:
- **Exactitud binaria** (Normal/Anómala) por categoría y global — la métrica
  principal.
- **Matriz de confusión del tipo** de defecto.
- **Lista de imágenes mal clasificadas** con sus scores, para iterar.

Y vuelca el detalle por imagen en `resultados.csv`.

**Resultados obtenidos:** 95,6% global (153/160), **0 falsos positivos**
(las 41 sanas correctas), con `scratch_head` y `manipulated_front` al 100%.
Los 7 fallos son defectos sub-umbral (estrías de +10 niveles, muescas de
<10 px) que están en el límite de lo detectable sin empezar a rechazar
piezas sanas.

---

## 9. `main.py` — la demostración visual

Procesa **una** imagen y muestra las 4 etapas: original → máscara → alineada
con regiones → defectos coloreados (rojo=forma, amarillo=intensidad,
magenta=perfil), con el diagnóstico y el nivel de anomalía en el título.
También imprime por consola los 10 scores con su exceso relativo, marcando
cuál disparó. Es la herramienta para **inspeccionar casos concretos** y
entender qué vio el sistema. La constante `IMAGEN_DEMO` arriba del archivo
permite elegir qué imagen procesar sin pasar argumentos.

---

## 10. Resumen del mapeo con los contenidos de la materia

| Etapa | Técnica de PDI | Unidad |
|---|---|---|
| Suavizado | Filtro gaussiano (convolución LSI) | II |
| Segmentación | Canny (gradiente, no-máximos, histéresis) | V |
| Relleno de máscara | Contornos + componentes conexas | V |
| Cierre dirigido | Dilatación + erosión | VI |
| Orientación del eje | Momentos de 2.º orden (≡ PCA) | descriptores |
| Alineación | Transformación afín (`warpAffine`) | II / geometría |
| Detector de forma | Operaciones de conjuntos sobre máscaras | VI |
| Detector de intensidad | Comparación estadística (media/σ), z-score | II / IV |
| Detector de perfil | Dilatación/erosión 1D (filtros max/min) | VI |
| Refinamiento | Apertura/cierre + componentes conexas | VI |
| Umbrales | Estadística (media + 3σ) | — |

Todo el sistema usa **exclusivamente técnicas clásicas**, sin redes
neuronales, cumpliendo la restricción del proyecto.
