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
    """
    Fase 1: Preprocesamiento y Segmentación.
    Aísla el tornillo del fondo rellenando su silueta.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Binarización invertida
    _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Esto garantiza que cualquier hueco interno (por brillos) desaparezca.
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask_filled = np.zeros_like(binary_mask)
    if contours:
        # Tomamos el contorno más grande (el tornillo)
        main_contour = max(contours, key=cv2.contourArea)
        # Lo dibujamos relleno (thickness=cv2.FILLED)
        cv2.drawContours(mask_filled, [main_contour], -1, 255, thickness=cv2.FILLED)
    
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
    
    return aligned_img, aligned_mask

def extract_features_and_classify(aligned_gray, aligned_mask, threshold_anomaly=100):
    """
    Fase 3, 4 y 5: Extracción, Segmentación y Clasificación.
    Usa filtro Black-Hat para detectar arañazos oscuros.
    """
    # FIX 2.1: Reducimos la erosión a (5,5) para no borrar los bordes de la cabeza, 
    # que es donde suele estar el defecto "scratch_head".
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inner_mask = cv2.erode(aligned_mask, kernel_erode, iterations=1)
    
    # FIX 2.2: Usamos la operación morfológica Black-Hat.
    # Extrae elementos oscuros (arañazos) que sean más pequeños que el kernel (15x15)
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(aligned_gray, cv2.MORPH_BLACKHAT, kernel_bh)
    
    # Umbralizamos el resultado del Black-Hat para quedarnos solo con los defectos marcados
    # (Puedes ajustar el valor '40' dependiendo de la iluminación de tus fotos)
    _, anomalies_thresh = cv2.threshold(blackhat, 40, 255, cv2.THRESH_BINARY)
    
    # Aplicamos la máscara interna para ignorar el fondo y el borde exterior
    internal_anomalies = cv2.bitwise_and(anomalies_thresh, anomalies_thresh, mask=inner_mask)
    
    # FIX 2.3: Limpieza final. Usamos una apertura pequeña para eliminar ruido 
    # (puntitos aislados) y dejar solo las manchas/líneas reales.
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    internal_anomalies = cv2.morphologyEx(internal_anomalies, cv2.MORPH_OPEN, kernel_clean)
    
    # Cuantificar hallazgos
    anomaly_score = np.sum(internal_anomalies > 0)
    
    # Clasificación
    is_anomalous = anomaly_score > threshold_anomaly
    diagnosis = "Anómala" if is_anomalous else "Normal"
    
    return internal_anomalies, anomaly_score, diagnosis

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
    # ==========================================
    # EJEMPLO DE USO (Reemplaza con tu ruta real)
    # ==========================================
    # Para probar el MVP, crea una imagen de prueba o descarga una de MVTec AD
    # test_image_path = "data/anomaly/scratch/000.png" 
    
    # Simulación para que el script no falle si no tienes la imagen aún:
    print("Iniciando Pipeline de Inspección de Tornillos...")
    try:
        # Intenta cargar una imagen real
        img = load_image("datos/scratch_head/000.png") 
        
        # Ejecutar Pipeline
        gray_img, mask = preprocess_and_segment(img)
        aligned_img, aligned_mask = align_spatial(img, mask)
        
        # Convertir imagen alineada a grises para el análisis de características
        aligned_gray = cv2.cvtColor(aligned_img, cv2.COLOR_RGB2GRAY)
        anomalies_mask, score, diagnosis = extract_features_and_classify(aligned_gray, aligned_mask)
        
        # Visualizar
        visualize_pipeline(img, mask, aligned_img, anomalies_mask, diagnosis, score)
        
    except FileNotFoundError as e:
        print(f"⚠️ {e}")
        print("Por favor, coloca una imagen de un tornillo.")