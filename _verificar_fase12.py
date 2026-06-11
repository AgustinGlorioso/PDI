# Script temporal de verificación visual de las fases 1-2 (no es parte del entregable)
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline import procesar_imagen

muestras = [
    "datos/good/000.png",
    "datos/good/010.png",
    "datos/good/025.png",
    "datos/manipulated_front/000.png",
    "datos/scratch_head/000.png",
    "datos/thread_side/000.png",
]

fig, axs = plt.subplots(len(muestras), 3, figsize=(14, 4 * len(muestras)))
for i, ruta in enumerate(muestras):
    r = procesar_imagen(ruta)
    axs[i, 0].imshow(r["gris"], cmap="gray"); axs[i, 0].set_title(f"{ruta} (orig)")
    axs[i, 1].imshow(r["mascara"], cmap="gray"); axs[i, 1].set_title("mascara")
    axs[i, 2].imshow(r["gris_alineada"], cmap="gray")
    axs[i, 2].set_title(f"alineada (ang={r['angulo']:.1f})")
    # linea central y margen para chequear anclaje
    axs[i, 2].axhline(512, color="r", lw=0.5)
    axs[i, 2].axvline(110, color="r", lw=0.5)
    for ax in axs[i]: ax.axis("off")

plt.tight_layout()
plt.savefig("_verif_fase12.png", dpi=60)
print("OK - figura guardada")

# chequeo numérico de consistencia del anclaje en las good
xs, ys = [], []
for i in range(0, 41, 5):
    r = procesar_imagen(f"datos/good/{i:03d}.png")
    x, y, w, h = cv2.boundingRect(r["mascara_alineada"])
    xs.append((x, x + w))
    m = cv2.moments(r["mascara_alineada"], binaryImage=True)
    ys.append(m["m01"] / m["m00"])
print("x_min de cada good:", [a for a, b in xs])
print("x_max de cada good:", [b for a, b in xs])
print("cy de cada good:", [f"{v:.1f}" for v in ys])
