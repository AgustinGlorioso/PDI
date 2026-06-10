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
    _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask_filled = np.zeros_like(binary_mask)
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask_filled, [main_contour], -1, 255, thickness=cv2.FILLED)
    plt.imshow(mask_filled)
    
    return gray, mask_filled

def align_spatial(image, mask):
    """
    Fase 2: Alineación Espacial.
    Rota la imagen para que el eje longitudinal del tornillo quede horizontal.
    """
    # Encontrar contornos en la máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, mask
    
    # Tomar el contorno más grande (el tornillo)
    main_contour = max(contours, key=cv2.contourArea)
    
    # Calcular el rectángulo de área mínima (Bounding box rotado)
    rect = cv2.minAreaRect(main_contour)
    (center_x, center_y), (width, height), angle = rect
    
    # Ajustar el ángulo para que quede perfectamente horizontal
    if width < height:
        angle = angle + 90
        
    # Obtener matriz de rotación y aplicar transformación afín
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    
    aligned_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    aligned_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)

    # La cabeza del tornillo es más ancha, por lo que tiene más píxeles blancos en la máscara.
    # Comparamos la mitad izquierda con la mitad derecha de la máscara.
    left_half = aligned_mask[:, :w//2]
    right_half = aligned_mask[:, w//2:]
    
    if np.sum(right_half) > np.sum(left_half):
        # Si hay más masa a la derecha, rotamos 180 grados
        aligned_img = cv2.rotate(aligned_img, cv2.ROTATE_180)
        aligned_mask = cv2.rotate(aligned_mask, cv2.ROTATE_180)
    
    return aligned_img, aligned_mask

def split_regions(aligned_img):
    """
    Divide la máscara del tornillo en Cabeza, Cuello y Rosca basándose en el largo.
    """
    # Proyección vertical para encontrar dónde empieza y termina el tornillo exactamente
    col_sums = np.sum(aligned_img, axis=0)
    valid_cols = np.where(col_sums > 0)[0]
    
    if len(valid_cols) == 0:
        return {}, []

    x_min, x_max = valid_cols[0], valid_cols[-1]
    length = x_max - x_min

    # Definir los puntos de corte (Ajusta estos porcentajes si es necesario)
    cut1 = x_min + int(length * 0.20) # 20%: Fin de la cabeza, inicio del cuello
    cut2 = x_min + int(length * 0.45) # 50%: Fin del cuello, inicio de la rosca

    regions = {}
    
    # Máscara de la Cabeza
    regions["Cabeza"] = np.zeros_like(aligned_img)
    regions["Cabeza"][:, x_min:cut1] = aligned_img[:, x_min:cut1]
    
    # Máscara del Cuello
    regions["Cuello"] = np.zeros_like(aligned_img)
    regions["Cuello"][:, cut1:cut2] = aligned_img[:, cut1:cut2]
    
    # Máscara de la Rosca
    regions["Rosca"] = np.zeros_like(aligned_img)
    regions["Rosca"][:, cut2:x_max] = aligned_img[:, cut2:x_max]

    # Definir los rangos de las columnas (X) para cada sección
    secciones = {
        "Cabeza": (x_min, cut1),
        "Cuello": (cut1, cut2),
        "Rosca": (cut2, x_max)
    }

    regions_gray = {}
    regions_mask = {}
    
    for nombre, (inicio, fin) in secciones.items():
        # 1. Extraemos físicamente los pedazos
        recorte_gris = aligned_gray[:, inicio:fin].copy()
        recorte_mask = aligned_mask[:, inicio:fin].copy()
        
        # 2. Aplicamos la máscara al recorte gris para eliminar cualquier fondo
        roi_limpio = cv2.bitwise_and(recorte_gris, recorte_gris, mask=recorte_mask)
        
        regions_gray[nombre] = roi_limpio
        regions_mask[nombre] = recorte_mask

    plt.imshow(regions_gray["Cabeza"])
    plt.imshow(regions_gray["Cuello"])
    plt.imshow(regions_gray["Rosca"])        
    
    return regions, [x_min, cut1, cut2, x_max]


def extract_features_and_classify(aligned_gray, aligned_mask, threshold_anomaly=100):
    """
    Fase 3, 4 y 5: Extracción, Segmentación y Clasificación.
    """
    # # 1. Máscara interna agresiva para quitar las sombras de los bordes (9x9)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    inner_mask = cv2.erode(aligned_mask, kernel_erode, iterations=1)
    
    # Operación morfológica Black-Hat.
    # Extrae elementos oscuros (arañazos) que sean más pequeños que el kernel (15x15)
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(aligned_gray, cv2.MORPH_BLACKHAT, kernel_bh)

    # Umbralizamos el resultado del Black-Hat para quedarnos solo con los defectos marcados
    _, anomalies_thresh = cv2.threshold(blackhat, 50, 255, cv2.THRESH_BINARY)
    
    # Aplicamos la máscara interna para ignorar el fondo y el borde exterior
    internal_anomalies = cv2.bitwise_and(anomalies_thresh, anomalies_thresh, mask=inner_mask)
    
    # Limpieza final. Usamos una apertura pequeña para eliminar ruido 
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    internal_anomalies = cv2.morphologyEx(internal_anomalies, cv2.MORPH_OPEN, kernel_clean)
    
    # En lugar de sumar todos los píxeles, buscamos los "manchones" detectados
    contours, _ = cv2.findContours(internal_anomalies, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    final_anomalies_mask = np.zeros_like(internal_anomalies)
    anomaly_score = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Ignorar ruido muy pequeño (< 15 píxeles) 
        # Ignorar sombras gigantes (como la cruz de la cabeza) (> 400 píxeles)
        # ESTOS VALORES PUEDES AJUSTARLOS SEGÚN EL TAMAÑO DE TUS FOTOS
        if 15 < area < 400:
            # Dibujamos solo las anomalías que cumplen el criterio
            cv2.drawContours(final_anomalies_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            anomaly_score += area
            
    # Nueva condición: si el área total de arañazos válidos es mayor a 50, es anómala.
    is_anomalous = anomaly_score > 50
    diagnosis = "Anómala" if is_anomalous else "Normal"
    
    return final_anomalies_mask, anomaly_score, diagnosis

def visualize_pipeline(original, mask, aligned, anomalies, diagnosis, score):
    """Muestra el resultado del pipeline completo."""
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f'Diagnóstico: {diagnosis} (Score: {score})', fontsize=16, fontweight='bold', 
                 color='red' if diagnosis == "Anómala" else 'green')
    
    axs[0].imshow(original)
    axs[0].set_title("1. Imagen Original")
    axs[0].axis('off')
    
    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title("2. Máscara (Segmentación)")
    axs[1].axis('off')
    
    axs[2].imshow(aligned)
    axs[2].set_title("3. Alineación Espacial")
    axs[2].axis('off')
    
    # Superponer anomalías en rojo sobre la imagen alineada
    anomaly_overlay = aligned.copy()
    anomaly_overlay[anomalies > 0] = [255, 0, 0] # Pintar de rojo los defectos
    
    axs[3].imshow(anomaly_overlay)
    axs[3].set_title("4. Detección de Anomalías")
    axs[3].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        img = load_image("datos/scratch_head/000.png") 
        
        gray_img, mask = preprocess_and_segment(img)
        aligned_img, aligned_mask = align_spatial(img, mask)
        
        aligned_gray = cv2.cvtColor(aligned_img, cv2.COLOR_RGB2GRAY)
        regions_masks, cuts = split_regions(aligned_gray)
        #anomalies_mask, score, diagnosis = extract_features_and_classify(aligned_gray, aligned_mask)
        
        #visualize_pipeline(img, mask, aligned_img, anomalies_mask, diagnosis, score)
        
    except FileNotFoundError as e:
        print(f"{e} Por favor, coloca una imagen de un tornillo.")