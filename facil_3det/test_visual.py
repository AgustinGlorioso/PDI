import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import tornillos as t

if len(sys.argv) < 2:
    print("Uso: python test_visual.py <ruta_imagen>")
    print("Ejemplo: python test_visual.py ../datos/scratch_head/000.png")
    sys.exit(1)

ruta = sys.argv[1]

COLORES = {'intensidad': (255, 220, 0), 'forma': (255, 60, 60), 'perfil': (255, 0, 255)}

P = t.cargar_plantillas()
U = t.cargar_umbrales()
r = t.procesar_imagen(ruta, plantilla=P['binaria'])
a = t.analizar(r['gris_alineada'], r['mascara_alineada'], P)
diag = t.diagnosticar(a, U)

overlay = cv2.cvtColor(r['gris_alineada'], cv2.COLOR_GRAY2RGB)
for k, color in COLORES.items():
    overlay[a[k] > 0] = color

es_anom = diag['diagnostico'] != 'Normal'
nombre = os.path.basename(ruta)
fig, axs = plt.subplots(1, 3, figsize=(17, 6))
fig.suptitle(f"{nombre} -> {diag['diagnostico']}" +
             (f" ({diag['tipo']})" if es_anom else "") +
             f"  nivel={diag['nivel']:.2f}",
             fontsize=14, fontweight='bold', color='red' if es_anom else 'green')
axs[0].imshow(r['rgb']); axs[0].set_title('Original')
axs[1].imshow(r['mascara'], cmap='gray'); axs[1].set_title('Segmentacion')
axs[2].imshow(overlay); axs[2].set_title('Defectos (amarillo=rayon, rojo=forma, magenta=rosca)')
for ax in axs:
    ax.axis('off')
plt.tight_layout()

salida = 'resultado_test.png'
plt.savefig(salida, dpi=120, bbox_inches='tight')
print(f"Imagen guardada en {salida}")

print(f"\nDiagnostico: {diag['diagnostico']}", end="")
print(f" ({diag['tipo']})" if es_anom else "")
print(f"Nivel: {diag['nivel']:.2f}  (anomala si supera 1.00)")
for k in a['scores']:
    print(f"  {k:<11} area={a['scores'][k]:<6} umbral={U[k]:<6} -> {a['scores'][k]/U[k]:.2f}")
