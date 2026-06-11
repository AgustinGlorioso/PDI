import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_image(image_path):
    """Carga la imagen y verifica que exista."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No se encontró la imagen en: {image_path}")
    img = cv2.imread(image_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convertir a RGB para Matplotlib

def preprocess_and_segment(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Binarización invertida
    _, binary_mask = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask_filled = np.zeros_like(binary_mask)
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask_filled, [main_contour], -1, 255, thickness=cv2.FILLED)
    
    # plt.figure(figsize=(12,4))
    # plt.subplot(1,3,1)
    # plt.imshow(blurred)
    # plt.subplot(1,3,2)
    # plt.imshow(binary_mask)
    # plt.subplot(1,3,3)
    # plt.imshow(mask_filled)

    mask = get_solid_metal_mask(image, mask_filled)
    return gray, mask

def get_solid_metal_mask(image, original_mask):
    # 1. Obtener bordes (tu Estrategia 2)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    #blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 100)
    
    # Limpiar bordes fuera de la zona de interés
    closed_edges = cv2.bitwise_and(edges, edges, mask=original_mask)

    # DILATAR mucho para cerrar los huecos de las roscas
    kernel = np.ones((9, 9), np.uint8) # Kernel más grande
    dilated = cv2.dilate(closed_edges, kernel, iterations=2)

    # Rellenar los huecos internos
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(original_mask)
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask_filled, [c], -1, 255, -1)
    
    # EROSIONAR para volver al tamaño original del metal
    final_mask = cv2.erode(mask_filled, kernel, iterations=2)
    
    # plt.subplot(1,2,1)
    # plt.imshow(edges)
    # plt.subplot(1,2,2)
    # plt.imshow(final_mask)
    # plt.show

    return final_mask

def align_spatial(image, mask):
    """
    Rota la imagen para que el eje longitudinal del tornillo quede horizontal.
    """
    # Encontrar contornos en la máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, mask
    
    # Tomar el contorno más grande (el tornillo)
    main_contour = max(contours, key=cv2.contourArea)
    
    # 1. Obtener el rectángulo mínimo y su ángulo
    rect = cv2.minAreaRect(main_contour)
    (center_x, center_y), (width, height), angle = rect
    
    # 2. Corregir el ángulo para que el eje largo sea horizontal
    if width < height:
        angle = angle + 90
        
    # 3. DEFINIR EL DESTINO: Queremos el tornillo en el centro de la imagen
    h, w = image.shape[:2]
    target_cx, target_cy = w // 2, h // 2

    # 4. CREAR MATRIZ DE TRANSFORMACIÓN COMBINADA (Rotación + Traslación)
    # getRotationMatrix2D crea una matriz que rota sobre (cx, cy)
    # Solo si el tamaño varía mucho entre fotos:
    # target_width = 800 # O el promedio de tus tornillos
    # scale = target_width / w
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    
    # Modificamos la parte de traslación de la matriz para que el centro (cx, cy) 
    # del tornillo termine en (target_cx, target_cy)
    M[0, 2] += (target_cx - center_x)
    M[1, 2] += (target_cy - center_y)

    # 5. Aplicar la transformación
    aligned_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    aligned_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)

    # 6. Corregir orientación Cabeza/Punta (Flip 180)
    # Usamos un margen más pequeño para que sea más robusto al ruido
    # La cabeza del tornillo es más ancha, por lo que tiene más píxeles blancos en la máscara.
    left_half = aligned_mask[:, :target_cx]
    right_half = aligned_mask[:, target_cx:]
    
    if np.sum(right_half) > np.sum(left_half):
        # Si hay más masa a la derecha, rotamos 180 grados
        aligned_img = cv2.rotate(aligned_img, cv2.ROTATE_180)
        aligned_mask = cv2.rotate(aligned_mask, cv2.ROTATE_180)
    
    return aligned_img, aligned_mask

def align_to_template(img, mask, template_mask):
    curr_mask = (mask > 0).astype(np.uint8)

    flipped_mask = cv2.flip(curr_mask, 0)

    print("curr_mask:", curr_mask.shape, curr_mask.dtype)
    print("template_mask:", template_mask.shape, template_mask.dtype)
    
    overlap_normal = np.sum(
       cv2.bitwise_and(curr_mask, template_mask)
    )

    overlap_flipped = np.sum(
        cv2.bitwise_and(flipped_mask, template_mask)
    )

    if overlap_flipped > overlap_normal:
        return cv2.flip(img, 0), flipped_mask
    else:
        return img, curr_mask

def split_regions(aligned_gray, aligned_mask):

    col_sums = np.sum(aligned_mask, axis=0)
    valid_cols = np.where(col_sums > 0)[0]

    if len(valid_cols) == 0:
        return {}, {}, []

    x_min, x_max = valid_cols[0], valid_cols[-1]
    length = x_max - x_min

    cut1 = x_min + int(length * 0.20)
    cut2 = x_min + int(length * 0.45)

    secciones = {
        "Cabeza": (x_min, cut1),
        "Cuello": (cut1, cut2),
        "Rosca": (cut2, x_max)
    }

    regions_gray = {}
    regions_mask = {}

    for nombre, (inicio, fin) in secciones.items():

        recorte_gris = aligned_gray[:, inicio:fin].copy()
        recorte_mask = aligned_mask[:, inicio:fin].copy()

        roi_limpio = cv2.bitwise_and(
            recorte_gris,
            recorte_mask
        )

        regions_gray[nombre] = roi_limpio
        regions_mask[nombre] = recorte_mask

    return regions_gray, regions_mask


if __name__ == "__main__":
    try:
        img = load_image("datos/thread_top/011.png") 
        mask_template = cv2.imread("datos/mask_template.png", cv2.IMREAD_GRAYSCALE)

        gray_img, mask = preprocess_and_segment(img)
        aligned_img, aligned_mask = align_spatial(gray_img, mask)
        #aligned_img, aligned_mask = align_to_template(aligned_img, aligned_mask, mask_template)
        
        plt.subplot(1,2,1)
        plt.imshow(aligned_img)
        plt.subplot(1,2,2)
        plt.imshow(aligned_mask)
        plt.show()
        #regions_gray, regions_masks = split_regions(aligned_img, aligned_mask)

        # plt.figure(figsize=(12,4))

        # plt.subplot(1,3,1)
        # plt.imshow(regions_gray["Cabeza"], cmap='gray')
        # plt.title("Cabeza")

        # plt.subplot(1,3,2)
        # plt.imshow(regions_gray["Cuello"], cmap='gray')
        # plt.title("Cuello")

        # plt.subplot(1,3,3)
        # plt.imshow(regions_gray["Rosca"], cmap='gray')
        # plt.title("Rosca")

        plt.show()

    except FileNotFoundError as e:
        print(f"{e} Por favor, coloca una imagen de un tornillo.")