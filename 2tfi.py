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
    Fase 1: Preprocesamiento y Segmentación (Canny Mejorado).
    Aísla el tornillo forzando el cierre de los contornos sobre los reflejos.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 1. Suavizado leve (Kernel 3x3 para no difuminar el borde débil del reflejo)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 2. Canny hipersensible
    # Bajamos los umbrales (15, 50) para capturar gradientes muy suaves.
    edges = cv2.Canny(blurred, 15, 50)
    
    # 3. Dilatación (Cerrando la jaula)
    # Hinchamos los bordes de Canny para que cualquier línea rota se conecte.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # 4. Extracción de Contornos y Relleno
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask_filled = np.zeros_like(gray)
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        # Ahora que el contorno está 100% cerrado, FILLED pintará todo el interior de blanco
        cv2.drawContours(mask_filled, [main_contour], -1, 255, thickness=cv2.FILLED)
        
    # 5. Compensación geométrica (Restaurar tamaño)
    # Al dilatar 2 veces en el paso 3, el tornillo creció. 
    # Erosionamos 2 veces la máscara sólida para devolverla a su tamaño original exacto.
    mask_filled = cv2.erode(mask_filled, kernel, iterations=2)
        
    return gray, mask_filled

def align_spatial(image, mask):
    """
    Fase 2: Alineación Espacial de Precisión (FitLine).
    Calcula el eje mediante mínimos cuadrados, centra la pieza y forja una diagonal de 45º.
    Devuelve la imagen alineada, la máscara y una imagen de "Debug" con el eje dibujado.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, mask, image
        
    main_contour = max(contours, key=cv2.contourArea)
    
    # 1. Calcular el Centroide Real mediante Momentos Espaciales
    M = cv2.moments(main_contour)
    if M['m00'] == 0: 
        return image, mask, image
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    
    # 2. Ajuste de Línea (FitLine - PCA)
    linea = cv2.fitLine(main_contour, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy = linea[0][0], linea[1][0]
    x0, y0 = linea[2][0], linea[3][0]
    
    angulo_vector = np.arctan2(vy, vx) * 180.0 / np.pi
    h_img, w_img = image.shape[:2]
    
    # --- CREAR IMAGEN DE DEBUG VISUAL ---
    img_debug = image.copy()
    mult = max(h_img, w_img) 
    pt1 = (int(x0 - mult * vx), int(y0 - mult * vy))
    pt2 = (int(x0 + mult * vx), int(y0 + mult * vy))
    cv2.line(img_debug, pt1, pt2, (255, 0, 0), 2)
    cv2.circle(img_debug, (cx, cy), 6, (0, 255, 0), -1)
    # ------------------------------------
    
    # 3. Prueba de Orientación (Búsqueda de la cabeza)
    M_prueba = cv2.getRotationMatrix2D((cx, cy), angulo_vector, 1.0)
    mask_prueba = cv2.warpAffine(mask, M_prueba, (w_img, h_img), flags=cv2.INTER_NEAREST)
    
    # 4. Detectar Distribución de Masas
    x_r, y_r, w_r, h_r = cv2.boundingRect(mask_prueba)
    mitad_x = x_r + (w_r // 2)
    
    area_izq = cv2.countNonZero(mask_prueba[y_r:y_r+h_r, x_r:mitad_x])
    area_der = cv2.countNonZero(mask_prueba[y_r:y_r+h_r, mitad_x:x_r+w_r])
    
    angulo_final = angulo_vector
    
    # Forzar cabeza a la izquierda
    if area_der > area_izq:
        angulo_final += 180
        
    # 5. Forzar la diagonal (45 grados horario)
    angulo_final -= 45
    
    # --- 6. Transformación Final Definitiva (Rotación + Centrado Absoluto) ---
    M_final = cv2.getRotationMatrix2D((cx, cy), angulo_final, 1.0)
    
    # Modificamos la columna de traslación de la matriz afín para centrar la pieza
    centro_imagen_x = w_img / 2.0
    centro_imagen_y = h_img / 2.0
    
    M_final[0, 2] += (centro_imagen_x - cx)  # Desplazamiento en X
    M_final[1, 2] += (centro_imagen_y - cy)  # Desplazamiento en Y
    
    # Aplicamos la transformación modificada
    aligned_img = cv2.warpAffine(image, M_final, (w_img, h_img), flags=cv2.INTER_LINEAR, borderValue=255)
    aligned_mask = cv2.warpAffine(mask, M_final, (w_img, h_img), flags=cv2.INTER_NEAREST, borderValue=0)
    
    return aligned_img, aligned_mask, img_debug

def extract_features_and_classify(aligned_gray, aligned_mask, threshold_anomaly=100):
    """
    Fase 3, 4 y 5: Extracción, Segmentación y Clasificación.
    Usa filtro Black-Hat para detectar arañazos oscuros.
    """
    # 1. Reducimos la erosión a (5,5) para no borrar los bordes de la cabeza
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inner_mask = cv2.erode(aligned_mask, kernel_erode, iterations=1)
    
    # 2. Operación morfológica Black-Hat (Extrae elementos oscuros)
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(aligned_gray, cv2.MORPH_BLACKHAT, kernel_bh)
    
    # 3. Umbralizamos el resultado del Black-Hat
    _, anomalies_thresh = cv2.threshold(blackhat, 40, 255, cv2.THRESH_BINARY)
    
    # 4. Aplicamos la máscara interna para ignorar el fondo y el borde exterior
    internal_anomalies = cv2.bitwise_and(anomalies_thresh, anomalies_thresh, mask=inner_mask)
    
    # 5. Limpieza final (Apertura)
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    internal_anomalies = cv2.morphologyEx(internal_anomalies, cv2.MORPH_OPEN, kernel_clean)
    
    # Cuantificar hallazgos
    anomaly_score = np.sum(internal_anomalies > 0)
    
    # Clasificación
    is_anomalous = anomaly_score > threshold_anomaly
    diagnosis = "Anómala" if is_anomalous else "Normal"
    
    return internal_anomalies, anomaly_score, diagnosis

def visualize_pipeline(original, mask, img_debug, aligned, anomalies, diagnosis, score):
    """Muestra el resultado del pipeline completo incluyendo el análisis de línea."""
    fig, axs = plt.subplots(1, 5, figsize=(24, 5))
    fig.suptitle(f'Diagnóstico: {diagnosis} (Score: {score})', fontsize=16, fontweight='bold', 
                 color='red' if diagnosis == "Anómala" else 'green')
    
    axs[0].imshow(original)
    axs[0].set_title("1. Imagen Original")
    axs[0].axis('off')
    
    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title("2. Segmentación Canny")
    axs[1].axis('off')
    
    axs[2].imshow(img_debug)
    axs[2].set_title("3. PCA / FitLine")
    axs[2].axis('off')
    
    axs[3].imshow(aligned)
    axs[3].set_title("4. Alineación Diagonal (45º)")
    axs[3].axis('off')
    
    anomaly_overlay = aligned.copy()
    anomaly_overlay[anomalies > 0] = [255, 0, 0] 
    
    axs[4].imshow(anomaly_overlay)
    axs[4].set_title("5. Análisis de Defectos")
    axs[4].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Iniciando Pipeline de Inspección de Tornillos...")
    try:
        # Intenta cargar una imagen real. (Asegúrate de ajustar la ruta)
        img = load_image("datos/scratch_head/010.png") 
        
        # Ejecutar Pipeline
        gray_img, mask = preprocess_and_segment(img)
        aligned_img, aligned_mask, img_debug = align_spatial(img, mask)
        
        aligned_gray = cv2.cvtColor(aligned_img, cv2.COLOR_RGB2GRAY)
        anomalies_mask, score, diagnosis = extract_features_and_classify(aligned_gray, aligned_mask)
        
        # Visualizar
        visualize_pipeline(img, mask, img_debug, aligned_img, anomalies_mask, diagnosis, score)
        
    except FileNotFoundError as e:
        print(f"⚠️ {e}")
        print("Por favor, coloca una imagen de un tornillo válida en la ruta.")