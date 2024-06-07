import cv2 as cv
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline   
import os

# Ruta de la carpeta que contiene las imágenes de monedas colombianas
ruta_base_datos = 'C:/Users/mario/OneDrive/Escritorio/PYTHON/Agentes-inteligentes/monedascontornoConIA/BD_monedas'

# Diccionario de denominaciones de monedas con sus valores y diámetros
denominations_info = {
    '50$': {'value': 50, 'diameter': 20},
    '100$': {'value': 100, 'diameter': 25},
    '200$': {'value': 200, 'diameter': 30},
    '500$': {'value': 500, 'diameter': 35},
    '1000$': {'value': 1000, 'diameter': 40}
}

# Función para preprocesar la imagen
def preprocess_image(imagen):
    imagen = cv.resize(imagen, (100, 100))  # Cambiar el tamaño de la imagen a 100x100 píxeles
    imagen_gris = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)  # Convertir a escala de grises
    _, imagen_umbral = cv.threshold(imagen_gris, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)  # Umbralización
    return imagen_umbral

# Leer las imágenes de la base de datos y preprocesarlas
X_train = []
y_train = []
for denominacion, info in denominations_info.items():
    ruta_denominacion = os.path.join(ruta_base_datos, denominacion)
    print(f'Ruta de la denominación "{denominacion}": {ruta_denominacion}')
    for imagen_file in os.listdir(ruta_denominacion):
        imagen = cv.imread(os.path.join(ruta_denominacion, imagen_file))
        imagen_preprocesada = preprocess_image(imagen)
        X_train.append(imagen_preprocesada.flatten())  # Aplanar la imagen preprocesada
        y_train.append(info['diameter'])  # Usar el diámetro como etiqueta para el modelo

# Convertir a numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Entrenar el modelo SVM con el diámetro como etiqueta
model = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1.0, gamma='scale'))
model.fit(X_train, y_train)

# Capturar video desde la cámara del teléfono
capturavideo = cv.VideoCapture(1)  # Índice 0 para la cámara predeterminada

# Diccionario para almacenar las monedas detectadas
monedas_detectadas = {}

while True:
    ret, frame = capturavideo.read()
    if not ret:
        break

    # Procesar la imagen
    imagen_preprocesada = preprocess_image(frame)

    # Encontrar contornos en la imagen preprocesada
    contornos, _ = cv.findContours(imagen_preprocesada.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Limpiar el diccionario de monedas detectadas en cada iteración
    monedas_detectadas.clear()

    # Variable para asignar un identificador único a cada moneda detectada
    id_moneda = 1

    for contorno in contornos:
        # Calcular el área del contorno para determinar si es una moneda
        area = cv.contourArea(contorno)
        if area > 500:  # Ajustar el valor según el tamaño de las monedas en la imagen
            # Calcular el centro del contorno
            M = cv.moments(contorno)
            centro_x = int(M["m10"] / M["m00"])
            centro_y = int(M["m01"] / M["m00"])

            # Recortar la imagen alrededor del centro y redimensionar si es necesario
            imagen_moneda = imagen_preprocesada[max(0, centro_y - 300):min(300, centro_y + 300), max(0, centro_x - 300):min(300, centro_x + 300)]
            imagen_moneda_resized = cv.resize(imagen_moneda, (100, 100))  # Redimensionar a 100x100 píxeles

            # Clasificar la moneda si contiene información suficiente
            if imagen_moneda_resized.any():
                imagen_moneda_flatten = imagen_moneda_resized.flatten().reshape(1, -1)
                diameter = model.predict(imagen_moneda_flatten)[0]
                for denom, info in denominations_info.items():
                    if info['diameter'] == diameter:
                        moneda_id = f"{denom}_{id_moneda}"  # Generar un identificador único para cada moneda
                        monedas_detectadas[moneda_id] = monedas_detectadas.get(moneda_id, 0) + 1
                        id_moneda += 1

    # Calcular el total en COP sumando el valor de cada moneda detectada
    total_valor_monedas = sum(denominations_info[moneda.split('_')[0]]['value'] * cantidad for moneda, cantidad in monedas_detectadas.items())

    print("Monedas detectadas:", monedas_detectadas)
    print("Total en $:", total_valor_monedas)

    # Mostrar información sobre las monedas
    y_offset = 50  
    for moneda, cantidad in monedas_detectadas.items():
        denom, moneda_id = moneda.split('_')
        valor_moneda = denominations_info[denom]['value']
        text = f"{cantidad} x {denom}$ - Valor: $ {valor_moneda * cantidad}"
        cv.putText(frame, text, (50, y_offset), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        y_offset += 50 

    total_text = f"Total: $ {total_valor_monedas}"
    cv.putText(frame, total_text, (50, y_offset), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Mostrar el frame con la información de las monedas
    cv.imshow("Monedas", frame)

    if cv.waitKey(1) == ord('q'):
        break

capturavideo.release()
cv.destroyAllWindows()