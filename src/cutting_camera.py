import cv2

# 1. Cargar la imagen original
img = cv2.imread('D:/Dev_Space/python/Gloabal_proyect/seminario1.jpg')

# 2. Redimensionar para que, tras el recorte, quede en 682x384
# Necesitamos un ancho de 682 + 31 + 31 = 744
# El alto se mantiene en 384
target_width = 744
target_height = 384
img_resized = cv2.resize(img, (target_width, target_height))

# 3. Recortar 31 píxeles de la izquierda y 31 de la derecha
# y_start:y_end, x_start:x_end
# x_start = 31
# x_end = 744 - 31 = 713
img_final = img_resized[0:384, 31:713]

# 4. Verificar que las dimensiones sean 682 x 384
h, w = img_final.shape[:2]
print(f"Dimensiones finales: {w} ancho x {h} alto")

# 5. Guardar la imagen lista para tu Pipeline de SfM
cv2.imwrite('seminario1_fixed.jpg', img_final)