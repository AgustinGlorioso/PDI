# Explicación de `cargar_sanos_alineados()`

Esta función **alinea todos los 41 tornillos sanos** para que terminen en la misma posición y orientación. Necesita 2 pasadas para hacerlo bien.

## Primera pasada: Alineación inicial (líneas 32-45)

```python
for nombre in sorted(...):
    r = t.procesar_imagen(...)  # Rota, recorta y genera máscara
    gris, masc = r["gris_alineada"], r["mascara_alineada"]
```

Para cada tornillo:
1. `procesar_imagen()` la rota y recorta automáticamente
2. **Primera imagen:** se toma como referencia base (sin correcciones)
3. **Imágenes 2 a 41:** se comparan contra una plantilla provisional

```python
prov = (suma / len(mascaras) > 0.5).astype(np.uint8) * 255
gris, masc = t.corregir_flip(gris, masc, prov)
```

- `prov` = promedio de todas las máscaras vistas hasta ahora
- Si la imagen nueva está **espejada** (rotada 180°) respecto a `prov`, `corregir_flip()` la da vuelta
- Se acumula en `suma` la máscara

**Problema:** Las primeras imágenes se alinearon contra una plantilla pobre (solo 1 o 2 imágenes). Puede haber errores.

## Segunda pasada: Refinamiento (líneas 48-51)

```python
plantilla = (np.mean([(m > 0) for m in mascaras], axis=0) > 0.5).astype(np.uint8) * 255
for i in range(len(mascaras)):
    grises[i], mascaras[i] = t.corregir_flip(grises[i], mascaras[i], plantilla)
```

- Calcula la **plantilla final** promediando todas las 41 máscaras
- Vuelve a pasar todas las imágenes **contra esta plantilla completa**
- Corrige de nuevo los flips que se hayan cometido

**Resultado:** 41 imágenes perfectamente alineadas y con la misma orientación, listas para promediarlas.

-------------------------------------------------------------------

Esta función **crea un modelo de referencia** de un tornillo sano a partir de 41 imágenes de tornillos sin defectos. Sirve para definir qué se considera "normal" en un tornillo.

## Paso 1: Preparar datos
```python
pm = np.stack([(m > 0).astype(np.float32) for m in mascaras])
pg = np.stack(grises).astype(np.float32)
media_mascara = pm.mean(axis=0)
```
- Convierte cada máscara en binaria (metal = 1.0, no-metal = 0.0)
- Apila todas las 41 máscaras en un cubo de 3D
- `media_mascara` calcula, para cada píxel, en qué **porcentaje** de los 41 tornillos había metal (0 a 1)

## Paso 2: Definir zonas características
El diccionario `P` contiene:

| Clave | Significado |
|-------|-------------|
| **nucleo** | Píxeles con metal en ≥97% de sanos → zona que **siempre** es tornillo |
| **exterior** | Píxeles con metal en ≤3% de sanos → zona que **nunca** es tornillo |
| **binaria** | Silueta típica (metal en >50% de sanos) → forma esperada |
| **media_gris** / **std_gris** | Color promedio y su variación natural → qué tonalidad debe tener el tornillo |

## Paso 3: Bandas de la rosca
```python
perfiles = [_envolventes(m) for m in mascaras]
```
Para cada tornillo extrae **4 perfiles** (cresta superior, valle superior, cresta inferior, valle inferior) de la rosca. Luego calcula el **rango permitido** por columna:
- `_min`: el valor más bajo visto en los 41 sanos
- `_max`: el valor más alto visto en los 41 sanos

Esto define un "corredor" donde la rosca puede variar naturalmente. Si sale del rango, es un defecto.

**En resumen:** retorna `P`, una "foto robot" del tornillo sano que usarán los detectores para comparar.