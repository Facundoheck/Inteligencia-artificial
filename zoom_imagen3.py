import cv2
import mediapipe as mp
import math

# Inicializa la detección de manos de mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Inicializa la captura de video
cap = cv2.VideoCapture(0)

while True:
    # Captura el fotograma de la cámara
    ret, frame = cap.read()

    # Convierte la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convierte la imagen a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detecta las manos en la imagen
    results = hands.process(rgb_frame)

    # Si hay manos detectadas, calcula la distancia entre ellas
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        x1, y1 = landmarks[8].x, landmarks[8].y  # Punto del dedo índice
        x2, y2 = landmarks[4].x, landmarks[4].y  # Punto del pulgar

        # Calcula la distancia entre los puntos usando el teorema de Pitágoras
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Aplica el zoom en base a la distancia entre los puntos
        zoom = 1 + distance * 5

        # Cambia el tamaño del fotograma con el zoom
        frame = cv2.resize(frame, None, fx=zoom, fy=zoom)

    # Muestra el fotograma
    cv2.imshow("nn.jpg", frame)

    # Espera a que se pulse una tecla
    k = cv2.waitKey(1)

    # Sale del bucle si se pulsa la tecla "q"
    if k == ord("q"):
        break

# Libera los recursos
cap.release()
cv2.destroyAllWindows()
