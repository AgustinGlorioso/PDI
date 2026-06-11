# Debug temporal: ¿por qué la máscara llega a x=1024 en algunas good?
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline import procesar_imagen

for idx in [15, 30, 35]:
    r = procesar_imagen(f"datos/good/{idx:03d}.png")
    m = r["mascara_alineada"]
    # perfil de anchos: cantidad de píxeles blancos por columna
    perfil = np.count_nonzero(m, axis=0)
    cols = np.where(perfil > 0)[0]
    print(f"good/{idx:03d}: x_min={cols[0]} x_max={cols[-1]}")
    # ¿dónde se hace finito el perfil? mostrar las últimas columnas con masa
    print("   perfil en cola:", perfil[940:1024:8])

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(r["gris"], cmap="gray"); axs[0].set_title("original")
    axs[1].imshow(r["mascara"], cmap="gray"); axs[1].set_title("mascara")
    axs[2].imshow(m, cmap="gray"); axs[2].set_title("mascara alineada")
    for ax in axs: ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"_debug_{idx:03d}.png", dpi=70)
print("OK")
