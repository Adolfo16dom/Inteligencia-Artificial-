# Clasificar colores y marcar el centro de cada color

Utilizamos OpenCV para detectar un color específico dentro de una imagen y marcar el centro del área donde aparece ese color. Primero, cargamos la imagen original figura.png y la convierte al espacio de color HSV, que es mucho más adecuado para trabajar con detección de colores que el formato RGB. A partir de ahí, define una función llamada detectar_color() que recibe la imagen, su versión en HSV y el color que se desea detectar rojo, verde, azul o amarillo. Dentro de esa función, se establecen los umbrales de color (rangos en HSV) para crear una máscara binaria, que resalta solo los píxeles que pertenecen al color indicado.

Luego, usando esa máscara, el programa aplica una operación bitwise_and para aislar visualmente las regiones del color detectado. Posteriormente, encuentra los contornos de esas zonas y calcula su centroide mediante los momentos cv.moments. Si el área del contorno es suficientemente grande (mayor a 500 píxeles, para evitar ruido), dibuja un pequeño círculo rojo en el centro y escribe el texto “Centro [color]” sobre la imagen. Finalmente, el programa entra en un bucle donde pide al usuario elegir un color o salir. Cuando se selecciona un color, se muestran tres ventanas: la imagen original, la máscara binaria y el resultado con el color detectado y marcado.

```python
import cv2 as cv
import numpy as np

# Cargar la imagen
img_original = cv.imread('Imagenes/figura.png', 1)
img_rgb = cv.cvtColor(img_original, cv.COLOR_BGR2RGB)
img_hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)

def detectar_color(img, hsv_img, color):
    """
    Detecta el color especificado en la imagen y devuelve la imagen resultante
    con el color aislado y el centro marcado.
    """
    # Definir los umbrales según el color
    if color == 'rojo':
        umbralBajo1 = (0, 80, 80)
        umbralAlto1 = (10, 255, 255)
        umbralBajo2 = (170, 80, 80)
        umbralAlto2 = (180, 255, 255)
        mascara1 = cv.inRange(hsv_img, umbralBajo1, umbralAlto1)
        mascara2 = cv.inRange(hsv_img, umbralBajo2, umbralAlto2)
        mascara = mascara1 + mascara2

    elif color == 'verde':
        umbralBajo = (35, 80, 80)
        umbralAlto = (85, 255, 255)
        mascara = cv.inRange(hsv_img, umbralBajo, umbralAlto)

    elif color == 'azul':
        umbralBajo = (90, 80, 80)
        umbralAlto = (130, 255, 255)
        mascara = cv.inRange(hsv_img, umbralBajo, umbralAlto)

    elif color == 'amarillo':
        umbralBajo = (20, 100, 100)
        umbralAlto = (30, 255, 255)
        mascara = cv.inRange(hsv_img, umbralBajo, umbralAlto)

    else:
        print("Color no reconocido.")
        return img, None

    # Aplicar máscara
    resultado = cv.bitwise_and(img, img, mask=mascara)

    # Encontrar contornos para localizar el centro
    contornos, _ = cv.findContours(mascara, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contorno in contornos:
        area = cv.contourArea(contorno)
        if area > 500:  # Filtrar áreas pequeñas para evitar ruido
            M = cv.moments(contorno)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv.circle(resultado, (cx, cy), 5, (0, 0, 255), -1)
                cv.putText(resultado, f"Centro {color}", (cx - 40, cy - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return resultado, mascara


while True:
    print("\nColores disponibles para detectar:")
    print("1. Rojo")
    print("2. Verde")
    print("3. Azul")
    print("4. Amarillo")
    print("5. Salir")
    opcion = input("Elige un color (escribe el nombre o el número): ").lower()

    if opcion in ['5', 'salir']:
        print("Saliendo del programa...")
        break

    # Determinar el color elegido
    if opcion in ['1', 'rojo']:
        color = 'rojo'
    elif opcion in ['2', 'verde']:
        color = 'verde'
    elif opcion in ['3', 'azul']:
        color = 'azul'
    elif opcion in ['4', 'amarillo']:
        color = 'amarillo'
    else:
        print("Opción no válida, intenta de nuevo.")
        continue

    # Procesar detección
    resultado, mascara = detectar_color(img_original, img_hsv, color)

    if resultado is not None:
        cv.imshow('Imagen Original', img_original)
        cv.imshow('Mascara', mascara)
        cv.imshow(f'Resultado - {color.capitalize()}', resultado)
        print("Presiona cualquier tecla en la ventana para continuar...")
        cv.waitKey(0)
        cv.destroyAllWindows()
```

# Reconocimiento de rostros con tres personas de ejemplo y dos videos 

## Obtener fotogramas de los rostros 

Este primer código utiliza OpenCV para capturar y guardar automáticamente imágenes de rostros detectados por la cámara. El programa crea una carpeta con el nombre de la persona y guarda allí las imágenes del rostro recortado, ajustado a 100x100 píxeles. Use un clasificador haarcascade_frontalface_alt.xml para identificar rostros en tiempo real. Además, muestra el número de fotos tomadas y permite detener el proceso al presionar la tecla ESC o cuando se alcanza el número máximo de capturas configurado.

```python 
import cv2 as cv
import os

# === CONFIGURACIÓN ===
nombre_persona = 'adolfo'  #  Cambia este nombre por el de cada persona
ruta_base = 'Clasificar Rostros'
ruta_guardado = os.path.join(ruta_base, 'Rostro de ' + nombre_persona + ' con iluminacion')
max_fotos = 3000  #  Número de imágenes que quieres capturar

# === CREA LA CARPETA SI NO EXISTE ===
if not os.path.exists(ruta_guardado):
    os.makedirs(ruta_guardado)
    print('Carpeta creada:', ruta_guardado)

# === CARGA EL CLASIFICADOR DE ROSTROS ===
rostro_cascade = cv.CascadeClassifier('Clasificar Rostros/haarcascade_frontalface_alt.xml')

# === INICIA LA CÁMARA ===
cap = cv.VideoCapture(0)  # 0 = cámara predeterminada
i = 0

print("Presiona ESC para detener manualmente...\n")
print(f"Capturando rostros para: {nombre_persona}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara.")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rostros = rostro_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in rostros:
        # Recorta el rostro
        rostro = frame[y:y+h, x:x+w]
        rostro = cv.resize(rostro, (100, 100), interpolation=cv.INTER_AREA)

        # Guarda la imagen del rostro
        cv.imwrite(f'{ruta_guardado}/{nombre_persona}_{i}.jpg', rostro)
        i += 1

        # Dibuja el rectángulo y muestra conteo
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.putText(frame, f'Fotos: {i}/{max_fotos}', (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Muestra el rostro recortado
        cv.imshow('Rostro', rostro)

    # Muestra la cámara con el contador
    cv.imshow('Captura de rostros', frame)

    # Finaliza si se llega al límite de fotos o presionas ESC
    if i >= max_fotos:
        print(f"\n Se capturaron {i} imágenes de {nombre_persona}")
        break
    if cv.waitKey(1) == 27:
        print("\n Captura interrumpida por el usuario.")
        break

# === FINALIZA ===
cap.release()
cv.destroyAllWindows()
```

Este código realiza una tarea similar, pero en lugar de usar una cámara en vivo, toma los fotogramas de un video (video del mariana.mp4). Detecta rostros en cada cuadro, los recorta, los redimensiona a 100x100 píxeles y los guarda en una carpeta específica. Esto permite generar un conjunto de datos de rostros a partir de videos. El programa muestra en pantalla tanto los fotogramas del video como los rostros detectados y recortados, deteniéndose cuando se presiona la tecla ESC.
```python
import numpy as np
import cv2 as cv
import os; 

rostro = cv.CascadeClassifier('Clasificar Rostros/haarcascade_frontalface_alt.xml')
cap = cv.VideoCapture('Clasificar Rostros/video del mariana.mp4')
ruta_base = 'Clasificar Rostros'
ruta_guardado = os.path.join(ruta_base, 'Rostro del  mariana')

if not os.path.exists(ruta_guardado):
    os.makedirs(ruta_guardado)
    print('Carpeta creada:', ruta_guardado)
    
i = 0  
while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rostros = rostro.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in rostros:
       #frame = cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
       frame2 = frame[ y:y+h, x:x+w]
       #frame3 = frame[x+30:x+w-30, y+30:y+h-30]
       
       frame2 = cv.resize(frame2, (100, 100), interpolation=cv.INTER_AREA)
       
        
       if(i%1==0):
           cv.imwrite(f'{ruta_guardado}/mariana_{i}.jpg', frame2)

           cv.imshow('rostror', frame2)
    cv.imshow('rostros', frame)
    i = i+1
    k = cv.waitKey(1)
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()
```

## Entrenamiento del modelo

Una vez tengamos el dataset generado con los rostros de las personas ahora si podemos entrenar un modelo el cual posteriormente lo podemos guardar para despues hacer las predicciones en tiempo real.

```python
import cv2 as cv 
import numpy as np 
import os
dataSet = 'Clasificar Rostros/dataset'
faces  = os.listdir(dataSet)
print(faces)

labels = []
facesData = []
label = 0 
for face in faces:
    facePath = dataSet+'/'+face
    for faceName in os.listdir(facePath):
        labels.append(label)
        facesData.append(cv.imread(facePath+'/'+faceName,0))
    label = label + 1
print(np.count_nonzero(np.array(labels)==0)) 

faceRecognizer = cv.face.EigenFaceRecognizer_create()
faceRecognizer.train(facesData, np.array(labels))
faceRecognizer.write('Clasificar Rostros/Eigenface.xml')
```

## Modelo entrenado 

Una vez el modelo este entrenado ahora si podemos mandar llamar al modelo que se genero para hacer el reconocimiento de las personas con el que se entreno. Es importante recalcar que en el orden que se se le pasan las clases al modelo para entrenar debe ser el mismo al momento de hacer las predicciones.

```python
import cv2 as cv
import os 

faceRecognizer = cv.face.EigenFaceRecognizer_create()
faceRecognizer.read('Clasificar Rostros/Eigenface.xml')
faces =['Rostro de adolfo con iluminacion', 'Rostro del  mariana']
cap = cv.VideoCapture(0)
rostro = cv.CascadeClassifier('Clasificar Rostros/haarcascade_frontalface_alt.xml')
while True:
    ret, frame = cap.read()
    if ret == False: break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cpGray = gray.copy()
    rostros = rostro.detectMultiScale(gray, 1.3, 3)
    for(x, y, w, h) in rostros:
        frame2 = cpGray[y:y+h, x:x+w]
        frame2 = cv.resize(frame2,  (100,100), interpolation=cv.INTER_CUBIC)
        result = faceRecognizer.predict(frame2)
        cv.putText(frame, '{}'.format(result), (x,y-20), 1,3.3, (0,0,255), 1, 3)
        if result[1] > 2800:
            #cv.putText(frame,'{}'.format(faces[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv.LINE_AA)
            cv.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        else:
            #cv.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv.LINE_AA)
            cv.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
    cv.imshow('frame', frame)
    k = cv.waitKey(1)
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()
```

# Programa para detectar las emociones de las personas en tiempo real con un CNN

## Conseguir un dataset 
Conseguimos un dataset de la pagina kaggle con muchas clases de emociones descargamos el archivo con emociones de (alegria, tristeza, enojo y sorpresa).

## Entrenamiento del CNN
Bueno entrenamos una red neuronal convolucional (CNN) para clasificar emociones a partir de imágenes. Primero, usa ImageDataGenerator para cargar y preprocesar el dataset desde la carpeta indicada, normalizando los valores de los píxeles y aplicando aumentos de datos (rotaciones, desplazamientos y espejos) que ayudan a mejorar la generalización del modelo. Luego, se dividen las imágenes en un 80 % para entrenamiento y 20 % para validación, asegurando una evaluación adecuada del rendimiento durante el aprendizaje.

El modelo CNN está compuesto por varias capas convolucionales y de pooling que extraen características visuales relevantes, seguidas de capas densas que realizan la clasificación final mediante una función softmax. Se entrena durante 20 épocas usando el optimizador Adam y la función de pérdida categorical crossentropy, adecuada para problemas de clasificación multiclase. Al finalizar el entrenamiento, el modelo resultante se guarda en el archivo modelo_emociones.h5 para su uso posterior en la detección de emociones en tiempo real.

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Ruta del dataset ---
data_dir = 'Detectar Emociones/dataset'  # Ajusta a tu ruta real

# --- Preprocesamiento de imágenes ---
# Escala los píxeles (0-255 → 0-1) y genera más datos con rotaciones y espejos
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80% entrenamiento, 20% validación
)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# --- Modelo CNN ---
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # reduce sobreajuste
    layers.Dense(train_data.num_classes, activation='softmax')
])

# --- Compilar el modelo ---
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- Entrenamiento ---
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20
)

# --- Guardar modelo entrenado ---
model.save('Detectar Emociones/modelo_emociones.h5')

print("Entrenamiento completado")

```
## Modelo CNN entreando 

Una vez entrenado el modelo en el siguiente codigo hacemos la detección de las emociones en tiempo real.

```python
import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model

# --- Cargar modelo entrenado ---
model = load_model('Detectar Emociones/modelo_emociones.h5')

# --- Clases de emociones según tu dataset ---
emociones = ['enojado', 'feliz', 'sorpresa', 'triste']  # Ajusta según tus carpetas

# --- Cargar el detector de rostros Haarcascade ---
rostro_cascade = cv.CascadeClassifier('Detectar Emociones/haarcascade_frontalface_alt.xml')

# --- Abrir la cámara ---
cap = cv.VideoCapture(0)  # 0 = cámara principal

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a escala de grises solo para detección de rostro
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rostros = rostro_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in rostros:
        # Recortar rostro en color (para mantener los 3 canales RGB)
        rostro_img = frame[y:y+h, x:x+w]

        # Redimensionar al tamaño que espera el modelo
        rostro_img = cv.resize(rostro_img, (64, 64))

        # Normalizar valores entre 0 y 1
        rostro_img = rostro_img.astype('float32') / 255.0

        # Añadir dimensión batch (1 imagen)
        rostro_img = np.expand_dims(rostro_img, axis=0)

        # --- Predicción ---
        prediccion = model.predict(rostro_img, verbose=0)
        indice = np.argmax(prediccion)
        emocion = emociones[indice]
        confianza = prediccion[0][indice]

        # --- Dibujar resultados en la imagen ---
        color = (0,255,0) if confianza > 0.5 else (0,0,255)
        cv.rectangle(frame, (x,y), (x+w, y+h), color, 2)
        cv.putText(frame, f"{emocion} ({confianza:.2f})", (x, y-10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Mostrar resultado
    cv.imshow('Deteccion de Emociones', frame)

    # Salir con ESC
    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()

```