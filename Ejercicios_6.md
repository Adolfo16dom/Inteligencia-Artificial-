# Detectar emociones pero ahora con face_mesh y hacer una comparación con el modelo fisherface

Aqui utilizamos MediaPipe Face Mesh junto con OpenCV para detectar y analizar expresiones faciales en tiempo real a través de la cámara. Primero inicializa el detector de puntos faciales FaceMesh y configura un dibujador para mostrar los landmarks sobre la imagen. Captura frames del video, los convierte a RGB y procesa cada frame para detectar los landmarks del rostro, que son puntos clave como ojos, boca y cejas. Luego calcula distancias entre estos puntos y normaliza esas medidas usando el ancho de la cara como referencia, asegurando que las proporciones sean consistentes independientemente del tamaño del rostro en la cámara.

Con esas proporciones, el código implementa reglas simples para clasificar la emoción: si la boca está muy abierta y ancha, se interpreta como “Feliz”; si los ojos están muy abiertos y la boca ligeramente abierta, como “Sorprendido”; si las cejas están juntas y los ojos entrecerrados, como “Enojado”; y si la boca está muy cerrada, como “Triste”. Cada frame se dibuja con los landmarks y la emoción detectada, que se muestra en la pantalla, y el programa continúa en tiempo real hasta que se presiona la tecla "q" para salir. Esto permite observar la emoción percibida de manera instantánea y visual.

```python
import cv2
import mediapipe as mp
import math

# Inicializar Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Dibujador
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

# Captura de video
cap = cv2.VideoCapture(0)

# Función para calcular distancia entre dos puntos
def distancia(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    emocion = "Neutral"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Dibujar landmarks
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, drawing_spec, drawing_spec
            )

            # Puntos clave
            boca_izq = face_landmarks.landmark[61]
            boca_der = face_landmarks.landmark[291]
            boca_sup = face_landmarks.landmark[13]
            boca_inf = face_landmarks.landmark[14]
            ceja_izq = face_landmarks.landmark[105]
            ceja_der = face_landmarks.landmark[334]
            ojo_izq_sup = face_landmarks.landmark[159]
            ojo_izq_inf = face_landmarks.landmark[145]
            ojo_der_sup = face_landmarks.landmark[386]
            ojo_der_inf = face_landmarks.landmark[374]
            ojo_izq = face_landmarks.landmark[33]
            ojo_der = face_landmarks.landmark[263]

            # Calcular ancho de cara (referencia para normalizar)
            ancho_cara = distancia(ojo_izq, ojo_der)

            if ancho_cara > 0:  # Evita división por cero
                # Calcular medidas normalizadas
                ancho_boca = distancia(boca_izq, boca_der) / ancho_cara
                altura_boca = distancia(boca_sup, boca_inf) / ancho_cara
                altura_ojo_izq = distancia(ojo_izq_sup, ojo_izq_inf) / ancho_cara
                altura_ojo_der = distancia(ojo_der_sup, ojo_der_inf) / ancho_cara
                altura_ojo = (altura_ojo_izq + altura_ojo_der) / 2
                distancia_cejas = distancia(ceja_izq, ceja_der) / ancho_cara

                # Clasificación basada en proporciones
                if ancho_boca > 0.62 and altura_boca > 0.08:
                    emocion = "Feliz"
                elif altura_boca > 0.25 and altura_ojo > 0.13:
                    emocion = "Sorprendido"
                elif distancia_cejas < 0.92 and altura_ojo < 0.12:
                    emocion = "Enojado"
                elif altura_boca < 0.03:
                    emocion = "Triste"

                #print(f"Ancho Boca: {ancho_boca:.4f}, Altura Boca: {altura_boca:.4f}, Altura Ojo: {altura_ojo:.4f}, Cejas: {distancia_cejas:.4f}, Emoción: {emocion}")

    # Mostrar emoción en pantalla
    cv2.putText(frame, emocion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    cv2.imshow('Emociones (Normalizado)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

```

## Comparación entre face mesh y fisherface 

Face Mesh no clasifica emociones por sí mismo; primero detecta un conjunto muy denso de landmarks faciales (468 puntos) en tiempo real. A partir de estos puntos se pueden calcular distancias y proporciones entre ojos, boca, cejas, etc., que luego se usan como features para inferir emociones mediante reglas o un modelo de Machine Learning. Es un enfoque basado en geometría del rostro y es robusto a cambios de iluminación y pose ligera.

En cambio Fisherfaces es un método clásico de análisis de componentes lineales discriminantes (LDA) aplicado a imágenes de rostro. Toma directamente las imágenes en escala de grises, reduce su dimensionalidad y encuentra las proyecciones que maximizan la separabilidad entre clases. La entrada son pixeles completos, no landmarks específicos. Es un enfoque estadístico y global, sensible a cambios de iluminación, escala y orientación del rostro.


Face Mesh es más moderno, flexible y robusto para video y condiciones del mundo real, pero requiere un poco más de trabajo para traducir landmarks a emociones mientras que Fisherfaces es más simple y directo para datasets de fotos controladas, pero más sensible a variaciones de iluminación, pose y ruido.

# Detección de emociones con un decisionTreeClassifier

##  Entrenamiento del modelo 
Primero, definimos la función extract_features para extraer características de cada imagen: redimensiona la imagen a 64x64 píxeles, calcula el promedio de color, convierte la imagen a escala de grises, obtiene un histograma de intensidad en 16 bins, detecta bordes con el método Canny y cuenta la cantidad de píxeles de borde, y calcula la desviación estándar del gris. Todas estas medidas se combinan en un vector de características que representa cada imagen.

Luego, recorre el dataset organizado por carpetas de emociones, extrae las features de cada imagen y crea los arreglos "X" (features) e "y" (etiquetas de emociones). También guarda opcionalmente estas features en un archivo CSV para análisis o reutilización. Después se divide los datos en conjuntos de entrenamiento y prueba, entrena un Decision Tree sobre las features y predice las emociones en el conjunto de prueba. Finalmente, se evalúa el modelo mostrando accuracy y un reporte de clasificación, y guarda el modelo entrenado en un archivo .pkl para usarlo posteriormente sin tener que reentrenar.

```python
import cv2
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


#Función para extraer features
def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    
    # Promedio de color
    avg_color = cv2.mean(img)[:3]
    
    # Gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Histograma
    hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
    hist = hist.flatten()
    
    # Bordes
    edges = cv2.Canny(gray, 100, 200)
    edge_count = np.sum(edges > 0)
    
    # Desviación estándar
    std_gray = np.std(gray)
    
    features = np.hstack([avg_color, hist, edge_count, std_gray])
    return features


#Cargar dataset y extraer features
dataset_dir = "Detectar Emociones/dataset"  # Cambia esta ruta a tu carpeta de imágenes
X = []
y = []

for emotion in os.listdir(dataset_dir):
    emotion_dir = os.path.join(dataset_dir, emotion)
    if os.path.isdir(emotion_dir):
        for filename in os.listdir(emotion_dir):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                path = os.path.join(emotion_dir, filename)
                feats = extract_features(path)
                X.append(feats)
                y.append(emotion)

X = np.array(X)
y = np.array(y)

print("Features shape:", X.shape)
print("Labels shape:", y.shape)


#Guardar features en CSV (opcional)
data = pd.DataFrame(X)
data['label'] = y
data.to_csv('Detectar Emociones con un DecisionTreeClassifier/features_emociones.csv', index=False)
print("CSV guardado como 'features_emociones.csv'")


#Entrenar Decision Tree
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


#Evaluar modelo
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))


#Guardar modelo entrenado
joblib.dump(clf, 'Detectar Emociones con un DecisionTreeClassifier/modelo_emociones.pkl')
print("Modelo guardado como 'modelo_emociones.pkl'")
```

## Modelo entrenado listo para usar 

```python
import cv2
import numpy as np
import joblib


clf = joblib.load('Detectar Emociones con un DecisionTreeClassifier/modelo_emociones.pkl')
print("Modelo cargado correctamente")


#Función para extraer features del frame
def extract_features(frame):
    img = cv2.resize(frame, (64, 64))
    
    # Promedio de color
    avg_color = cv2.mean(img)[:3]
    
    # Gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Histograma
    hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
    hist = hist.flatten()
    
    # Bordes
    edges = cv2.Canny(gray, 100, 200)
    edge_count = np.sum(edges > 0)
    
    # Desviación estándar
    std_gray = np.std(gray)
    
    features = np.hstack([avg_color, hist, edge_count, std_gray])
    return features.reshape(1, -1)

#Iniciar cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    feats = extract_features(frame)
    pred = clf.predict(feats)[0]
    
    # Mostrar emoción en pantalla
    cv2.putText(frame, f"Emocion: {pred}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Deteccion de Emociones", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

# Escalar y rotar un rectangulo con los dedos usando mediapipe 
Primero inicializa el detector de manos con un máximo de dos manos y una confianza mínima de 0.5, y define la función draw_rect_with_dual_rotation que toma dos puntos generalmente los dedos índice de dos manos o los dedos índice y pulgar de una sola mano para calcular un rectángulo centrado, escalado y rotado según dos ángulos: uno horizontal y otro vertical (aproximación perpendicular). Este rectángulo se dibuja sobre la imagen usando líneas de OpenCV, permitiendo visualizar de manera geométrica la orientación relativa de los dedos.

Durante la captura de video, cada frame se convierte a RGB y se procesa con MediaPipe para obtener los landmarks de la mano. Si se detectan dos manos, se usan los índices de ambas manos como puntos de referencia para dibujar un rectángulo verde; si hay una sola mano, se usan el índice y el pulgar para dibujar un rectángulo rojo. Además, se dibujan los landmarks y las conexiones de la mano sobre el frame. 

```python
import cv2
import mediapipe as mp
import numpy as np
import math

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Función para dibujar rectángulo rotado en 2D con dos ángulos
def draw_rect_with_dual_rotation(img, pt1, pt2, scale=1.0, color=(0,255,0), thickness=2):
    # Calcular centro
    cx = (pt1[0] + pt2[0]) / 2
    cy = (pt1[1] + pt2[1]) / 2
    
    # Tamaño base del rectángulo
    width = abs(pt2[0] - pt1[0]) * scale
    height = abs(pt2[1] - pt1[1]) * scale
    
    # Ángulo horizontal (línea entre los dedos)
    angle_h = math.degrees(math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))
    # Ángulo vertical (tipo “abrir la tapa”)
    angle_v = math.degrees(math.atan2(pt2[0] - pt1[0], pt2[1] - pt1[1]))  # perpendicular
    
    # Combinar ambos ángulos (simple aproximación)
    angle = angle_h + angle_v/2
    
    # Crear rectángulo centrado en (0,0)
    rect = np.array([
        [-width/2, -height/2],
        [ width/2, -height/2],
        [ width/2,  height/2],
        [-width/2,  height/2]
    ])
    
    # Rotación
    theta = np.radians(angle)
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    rotated = rect @ rot_matrix.T
    
    # Trasladar al centro
    points = rotated + np.array([cx, cy])
    points = points.astype(int)
    
    # Dibujar rectángulo
    for i in range(4):
        cv2.line(img, tuple(points[i]), tuple(points[(i+1)%4]), color, thickness)

# Captura de video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    h, w, _ = frame.shape

    if results.multi_hand_landmarks:
        num_hands = len(results.multi_hand_landmarks)

        if num_hands >= 2:
            # Dos índices de diferentes manos
            l1 = results.multi_hand_landmarks[0].landmark[8]
            l2 = results.multi_hand_landmarks[1].landmark[8]
            pt1 = (l1.x*w, l1.y*h)
            pt2 = (l2.x*w, l2.y*h)
            draw_rect_with_dual_rotation(frame, pt1, pt2, color=(0,255,0), thickness=2)
            
            # Dibujar landmarks
            mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[1], mp_hands.HAND_CONNECTIONS)
        
        elif num_hands == 1:
            hand_landmarks = results.multi_hand_landmarks[0]
            pt1 = (hand_landmarks.landmark[8].x*w, hand_landmarks.landmark[8].y*h)
            pt2 = (hand_landmarks.landmark[4].x*w, hand_landmarks.landmark[4].y*h)
            draw_rect_with_dual_rotation(frame, pt1, pt2, color=(0,0,255), thickness=2)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Salida", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

```