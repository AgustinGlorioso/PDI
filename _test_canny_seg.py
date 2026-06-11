# Test temporal: segmentación por Canny vs Otsu en imágenes con sombra
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def seg_otsu(g):
    b = cv2.GaussianBlur(g, (5, 5), 0)
    _, m = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ee = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ee)
    cont, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    if cont:
        cv2.drawContours(out, [max(cont, key=cv2.contourArea)], -1, 255, -1)
    return out


def seg_canny(g, t1=40, t2=90):
    b = cv2.GaussianBlur(g, (3, 3), 0)
    edges = cv2.Canny(b, t1, t2)
    ee = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dil = cv2.dilate(edges, ee, iterations=2)
    cont, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(g)
    if cont:
        cv2.drawContours(out, [max(cont, key=cv2.contourArea)], -1, 255, -1)
    out = cv2.erode(out, ee, iterations=2)
    return out


def stats(m):
    perfil = np.count_nonzero(m, axis=0)
    cols = np.where(perfil > 0)[0]
    if len(cols) == 0:
        return "VACIA"
    return f"largo={cols[-1]-cols[0]:4d} ancho_max={perfil.max():3d} area={np.count_nonzero(m):6d}"


tests = ["good/000", "good/005", "good/015", "good/030", "good/035",
         "manipulated_front/000", "thread_side/000", "scratch_head/000"]

fig, axs = plt.subplots(len(tests), 3, figsize=(13, 3.2 * len(tests)))
for i, name in enumerate(tests):
    g = cv2.imread(f"datos/{name}.png", cv2.IMREAD_GRAYSCALE)
    mo = seg_otsu(g)
    mc = seg_canny(g)
    print(f"{name:25s} OTSU: {stats(mo)}   CANNY: {stats(mc)}")
    axs[i, 0].imshow(g, cmap="gray"); axs[i, 0].set_title(name, fontsize=8)
    axs[i, 1].imshow(mo, cmap="gray"); axs[i, 1].set_title("otsu", fontsize=8)
    axs[i, 2].imshow(mc, cmap="gray"); axs[i, 2].set_title("canny", fontsize=8)
    for ax in axs[i]: ax.axis("off")
plt.tight_layout()
plt.savefig("_test_canny.png", dpi=55)
print("figura OK")
