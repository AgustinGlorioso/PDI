import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Asumo que estas funciones vienen de tu archivo 'test'
from test import (
    load_image,
    preprocess_and_segment,
    align_spatial
)

GOOD_PATH = "datos/good"

def load_aligned_data():
    aligned_images = []
    aligned_masks = []
    master_mask = None 

    for i in range(41):
        filename = f"{i:03d}.png"
        path = os.path.join(GOOD_PATH, filename)
        if not os.path.exists(path): continue

        img = load_image(path)
        _, mask = preprocess_and_segment(img)
        
        # OJO: align_spatial debe devolver (imagen_alineada, mascara_alineada)
        aligned_img, aligned_mask = align_spatial(img, mask)
        
        curr_mask = (aligned_mask > 0).astype(np.uint8)

        if master_mask is None:
            master_mask = curr_mask
            aligned_images.append(aligned_img)
            aligned_masks.append(curr_mask)
        else:
            # Lógica del Flip para cerrar la "V"
            flipped_mask = cv2.flip(curr_mask, 0)
            overlap_normal = np.sum(cv2.bitwise_and(curr_mask, master_mask))
            overlap_flipped = np.sum(cv2.bitwise_and(flipped_mask, master_mask))

            if overlap_flipped > overlap_normal:
                aligned_images.append(cv2.flip(aligned_img, 0)) # Girar imagen también
                aligned_masks.append(flipped_mask)
            else:
                aligned_images.append(aligned_img)
                aligned_masks.append(curr_mask)

    return aligned_images, aligned_masks

def plot_overlay(masks):
    plt.figure(figsize=(10, 8))
    # Creamos un fondo oscuro para que resalte el blanco
    canvas = np.zeros(masks[0].shape, dtype=float)
    for mask in masks:
        canvas += mask
    
    # Normalizamos para visualización (0 a 1)
    canvas /= len(masks)
    
    plt.imshow(canvas, cmap="gray")
    plt.title("Superposición de máscaras alineadas (Unificadas)")
    plt.axis("off")
    plt.show()

def plot_mean_mask(masks):
    stack = np.stack(masks)
    mean_mask = np.mean(stack, axis=0)

    plt.figure(figsize=(10, 8))
    im = plt.imshow(mean_mask, cmap='viridis') # Viridis ayuda a ver mejor los niveles
    plt.colorbar(im)
    plt.title("Mapa de consistencia de alineación (Unificada)")
    plt.axis("off")
    plt.show()

    return mean_mask

def compute_alignment_score(mean_mask):
    # Usamos un umbral pequeño para definir el área del tornillo total
    screw_pixels = mean_mask > 0
    if not np.any(screw_pixels):
        return 0
        
    score = np.mean(mean_mask[screw_pixels])
    print(f"Alignment Score: {score:.4f}")
    return score

def create_golden_template(images):
    # Convertimos la lista a un array de 4 dimensiones (N, H, W, C)
    stack = np.stack(images).astype(np.float32)
    
    # Promedio (Template Maestro)
    mean_img = np.mean(stack, axis=0).astype(np.uint8)
    
    # Desviación Estándar (Mapa de Tolerancia)
    std_img = np.std(stack, axis=0).astype(np.uint8)
    
    return mean_img, std_img

if __name__ == "__main__":
    # 1. Cargar datos
    images, masks = load_aligned_data()
    print(f"Datos cargados: {len(images)}")

    # 2. Crear templates
    golden_template, std_template = create_golden_template(images)

    # 3. Graficar con subplots
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    ax[0].imshow(cv2.cvtColor(golden_template, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Golden Template (Media)")
    ax[0].axis("off")

    # La desviación estándar se ve mejor en escala de grises o mapa térmico
    if len(std_template.shape) == 3:
        std_viz = cv2.cvtColor(std_template, cv2.COLOR_BGR2GRAY)
    else:
        std_viz = std_template
        
    im = ax[1].imshow(std_viz, cmap='hot')
    ax[1].set_title("Mapa de Variabilidad (STD)")
    ax[1].axis("off")
    plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()