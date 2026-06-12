# Informe LaTeX

Informe en formato congreso (estilo NeurIPS), máximo 5 páginas.

## Archivos

- `informe.tex` — fuente del informe
- `refs.bib` — bibliografía (BibTeX)
- `figuras/` — figuras generadas desde el código (`generar_figuras.py`)
- `generar_figuras.py` — regenera las figuras del informe

## Cómo compilar en Overleaf

1. Subir `informe.tex`, `refs.bib` y la carpeta `figuras/`.
2. **Importante:** el proyecto necesita el archivo `neurips.sty` (el mismo
   que usa el template de la cátedra). Copiarlo desde el proyecto de
   ejemplo si no está.
3. Compilar con pdfLaTeX. La secuencia de compilación es:
   `pdflatex → bibtex → pdflatex → pdflatex` (Overleaf lo hace automático).

## Regenerar las figuras

Desde la raíz del repositorio, con las plantillas ya construidas:

```bash
python build_template.py
python calibrar.py
python informe/generar_figuras.py
```
