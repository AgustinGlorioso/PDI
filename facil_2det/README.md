# Versión mínima — 2 detectores (74,4%)

La versión **más fácil de entender de todas**: solo **dos ideas**. Pensada
para estudiar el sistema sin la parte más complicada (el detector de rosca).

## Las 2 ideas

1. **Rayones (intensidad)** — comparar el gris de la pieza contra el tornillo
   promedio. Lo que se aparta mucho es un rayón.
2. **Deformaciones (forma)** — comparar la silueta contra la plantilla: si
   falta metal donde siempre debería haber, o sobra donde nunca debería, es
   un defecto de forma.

No tiene el detector de perfil de rosca (el más difícil de entender), así que
**detecta peor las roscas y las puntas**. A cambio, el código es muy corto.

## Resultado

**74,4%** de exactitud, **0 falsos positivos**. Detecta muy bien rayones de
cabeza (100%) y thread_top (91%), pero flojo en thread_side (26%) y puntas
manipuladas (37%), porque esos defectos necesitan el detector de rosca.

> Si querés el mejor resultado, usá la carpeta `facil_3det/` (96,9%). Esta
> versión es para **entender lo esencial** con el mínimo de código.

## Los 3 archivos

| Archivo | Qué hace |
|---|---|
| `tornillos.py` | Preprocesamiento + los 2 detectores + diagnóstico |
| `entrenar.py` | Aprende el tornillo sano y fija los 2 umbrales |
| `app.py` | Inspecciona una imagen o evalúa todo el dataset |

## Cómo usarlo

```bash
python entrenar.py
python app.py ../datos/scratch_head/000.png
python app.py
```
