import pygame
import cv2
import mediapipe as mp
import math

# Inicializa la detección de manos de mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Inicializa Pygame
pygame.init()

# Tamaño de la ventana
window_size = (1000, 1000)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Zoom con Manos")

# Carga la imagen inicial
imagen_path = "nn.jpg"
imagen_original = pygame.image.load(imagen_path)
imagen_rect = imagen_original.get_rect(center=(window_size[0] // 2, window_size[1] // 2))

# Inicializa la captura de video
cap = cv2.VideoCapture(0)

while True:
    # Captura el fotograma de la cámara
    ret, frame = cap.read()

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

        # Cambia el tamaño de la imagen con el zoom
        imagen_ancho = int(imagen_original.get_width() * zoom)
        imagen_alto = int(imagen_original.get_height() * zoom)
        imagen_redimensionada = pygame.transform.scale(imagen_original, (imagen_ancho, imagen_alto))

        # Obtén el rectángulo de la imagen redimensionada
        imagen_rect = imagen_redimensionada.get_rect(center=imagen_rect.center)

    # Dibuja la imagen en la ventana de Pygame
    screen.fill((255, 255, 255))  # Fondo blanco
    screen.blit(imagen_redimensionada, imagen_rect.topleft)

    # Actualiza la pantalla
    pygame.display.flip()

    # Maneja eventos de Pygame
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            cv2.destroyAllWindows()
            quit()

# Libera los recursos
cap.release()
cv2.destroyAllWindows()
