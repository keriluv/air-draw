import cv2
import mediapipe as mp
import numpy as np
import os
import time
import urllib.request
import qrcode
import requests
import random
from collections import deque

MODEL_PATH = "hand_landmarker.task"
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

MEDIAPIPE_WIDTH = 640
MEDIAPIPE_HEIGHT = 360
MODEL_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/"
    "hand_landmarker.task"
)
DRAWINGS_FOLDER = "dibujos"
GALLERY_FOLDER = "galeria"
UPLOAD_URL = "https://tempfile.org/api/upload/local"
EXPIRATION_HOURS = 24
MEMORY_SHOW_TIME = 3
MEMORY_SHAPES = [
    "CIRCULO",
    "CUADRADO",
    "TRIANGULO",
    "ESTRELLA"
]


INTERFAZ_MENU_PATH = os.path.join("interfaz", "menu.png")
INTERFAZ_LIBRE_PATH = os.path.join("interfaz", "interfaz_modolibre2.png")
INTERFAZ_MEMORIA_PATH = os.path.join("interfaz", "interfaz_modomemoria.png")
INTERFAZ_MEMORIA_FIGURA_PATH = os.path.join("interfaz", "interfaz_modomemoria_figura.png")
INTERFAZ_GUARDAR_PATH = os.path.join("interfaz", "interfaz_guardar_dibujo.png")


INTERFACE_WIDTH = 1920
INTERFACE_HEIGHT = 1080

MEMORY_FIGURE_WIDTH = 662
MEMORY_FIGURE_HEIGHT = 646
MEMORY_FIGURE_X = (INTERFACE_WIDTH - MEMORY_FIGURE_WIDTH) // 2
MEMORY_FIGURE_Y = (INTERFACE_HEIGHT - MEMORY_FIGURE_HEIGHT) // 2


CAMERA_WIDTH = 1704
CAMERA_HEIGHT = 875
CAMERA_X = (INTERFACE_WIDTH - CAMERA_WIDTH) // 2
CAMERA_Y = (INTERFACE_HEIGHT - CAMERA_HEIGHT) // 2
MEDIAPIPE_WIDTH = 640
MEDIAPIPE_HEIGHT = 360

os.makedirs(DRAWINGS_FOLDER, exist_ok=True)
os.makedirs(GALLERY_FOLDER, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print()
    print("========================================")
    print("       DESCARGANDO MODELO DE MEDIAPIPE")
    print("========================================")
    print()
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Modelo descargado correctamente.")
    except Exception as e:
        print("No se pudo descargar el modelo.")
        print("Error:", e)
        exit()

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

COLORS = [
    (255, 80, 80),
    (80, 255, 120),
    (255, 100, 255),
    (0, 220, 255),
    (255, 255, 255)
]

color_index = 0
brush_color = COLORS[color_index]
brush_size = 12

SMOOTHING_POINTS = 4
DEADZONE = 2
MAX_JUMP = 100
point_history = deque(maxlen=SMOOTHING_POINTS)


def finger_is_up(landmarks, tip, pip):
    margin = 0.015
    return landmarks[tip].y < landmarks[pip].y - margin


def get_gesture(landmarks):
    index_up = finger_is_up(landmarks, 8, 6)
    middle_up = finger_is_up(landmarks, 12, 10)
    ring_up = finger_is_up(landmarks, 16, 14)
    pinky_up = finger_is_up(landmarks, 20, 18)

    if index_up and not middle_up and not ring_up and not pinky_up:
        return "DIBUJAR"

    if index_up and middle_up and not ring_up and not pinky_up:
        return "COLOR"

    if index_up and middle_up and ring_up and pinky_up:
        return "BORRAR"

    if not index_up and not middle_up and not ring_up and not pinky_up:
        return "PAUSA"

    return "PAUSA"


def smooth_point(x, y):
    point_history.append((x, y))
    avg_x = int(sum(point[0] for point in point_history) / len(point_history))
    avg_y = int(sum(point[1] for point in point_history) / len(point_history))
    return (avg_x, avg_y)


def erase_at(x, y, radius=35):
    global canvas
    if canvas is None:
        return
    mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (x, y), radius, 255, -1)
    mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    canvas[mask_3d == 255] = 0


def create_transparent_canvas():
    if canvas is None:
        return None
    drawing = canvas.copy()
    mask = np.any(drawing != 0, axis=2)
    transparent = np.zeros((drawing.shape[0], drawing.shape[1], 4), dtype=np.uint8)
    transparent[:, :, :3] = drawing
    transparent[:, :, 3] = mask.astype(np.uint8) * 255
    return transparent


def create_photo_with_drawing(frame):
    result = frame.copy()
    if canvas is None:
        return result
    mask = np.any(canvas != 0, axis=2).astype(np.uint8) * 255
    alpha = (mask.astype(np.float32) / 255.0)
    alpha = alpha[:, :, None]
    drawing_float = canvas.astype(np.float32)
    frame_float = frame.astype(np.float32)
    combined = drawing_float * alpha + frame_float * (1 - alpha)
    result = np.clip(combined, 0, 255).astype(np.uint8)
    return result


def upload_image_to_internet(filepath):
    print()
    print("========================================")
    print("       SUBIENDO IMAGEN A INTERNET")
    print("========================================")
    print()
    print("Espera unos segundos...")
    print()
    try:
        if not os.path.exists(filepath):
            print("No existe el archivo:", filepath)
            return None
        with open(filepath, "rb") as image_file:
            files = {
                "files": (os.path.basename(filepath), image_file, "image/png")
            }
            data = {"expiryHours": str(EXPIRATION_HOURS)}
            response = requests.post(UPLOAD_URL, files=files, data=data, timeout=60)
        if response.status_code != 200:
            print()
            print("ERROR AL SUBIR LA IMAGEN")
            print("Código:", response.status_code)
            print("Respuesta:", response.text)
            return None
        try:
            result = response.json()
        except ValueError:
            print("El servidor no devolvió una respuesta JSON válida.")
            print(response.text)
            return None
        if not result.get("success"):
            print()
            print("El servidor rechazó la imagen.")
            print(result)
            return None
        uploaded_files = result.get("files", [])
        if not uploaded_files:
            print("El servidor no devolvió información del archivo.")
            return None
        uploaded_file = uploaded_files[0]
        public_url = uploaded_file.get("url")
        if not public_url:
            print("No se recibió una URL pública para el archivo.")
            return None
        if public_url.startswith("/"):
            public_url = "https://tempfile.org" + public_url
        if not public_url.startswith("https://"):
            print("La URL recibida no es una URL HTTPS pública:")
            print(public_url)
            return None
        print()
        print("========================================")
        print("          ¡IMAGEN SUBIDA!")
        print("========================================")
        print()
        print("URL pública del QR:")
        print(public_url)
        print()
        print("Esta URL es accesible desde cualquier conexión a Internet.")
        print("Disponible durante", EXPIRATION_HOURS, "horas.")
        print()
        return public_url
    except requests.exceptions.Timeout:
        print()
        print("Tiempo de espera agotado al subir la imagen.")
        print("Revisa que el computador tenga conexión a Internet.")
        print()
        return None
    except requests.exceptions.ConnectionError:
        print()
        print("No se pudo conectar con el servidor de Internet.")
        print("Revisa la conexión del computador.")
        print()
        return None
    except Exception as e:
        print()
        print("========================================")
        print("          ERROR AL SUBIR")
        print("========================================")
        print()
        print("Error:", e)
        print()
        return None


def generate_qr(public_url, filename):
    qr = qrcode.make(public_url)
    qr_filename = f"QR_{filename}"
    qr_path = os.path.join(GALLERY_FOLDER, qr_filename)
    qr.save(qr_path)
    return qr_path


def show_qr(qr_path, title):
    qr = cv2.imread(qr_path)
    if qr is None:
        print("No se pudo cargar el QR.")
        return

    qr = cv2.resize(qr, (500, 500), interpolation=cv2.INTER_NEAREST)

    height, width = qr.shape[:2]

    screen = np.zeros((height + 190, width, 3), dtype=np.uint8)
    screen[:] = (25, 15, 30)

    cv2.putText(screen, "AIR DRAW", (150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(screen, title, (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, brush_color, 2)

    screen[100:100 + height, 0:width] = qr

    cv2.putText(screen, "ESCANEA EL QR CON TU CELULAR", (45, 625), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(screen, "La imagen estara disponible durante 24 horas", (25, 655), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1)
    cv2.putText(screen, "Presiona Q o ESC para continuar", (105, 685), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

    cv2.imshow("AIR DRAW - QR", screen)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    cv2.destroyWindow("AIR DRAW - QR")


def save_canvas_qr():
    filename = f"lienzo_{int(time.time())}.png"
    path = os.path.join(GALLERY_FOLDER, filename)

    transparent = create_transparent_canvas()

    if transparent is None:
        print("No existe ningún lienzo.")
        return None

    success = cv2.imwrite(path, transparent)

    if not success:
        print("No se pudo guardar el lienzo.")
        return None

    public_url = upload_image_to_internet(path)

    if public_url is None:
        return None

    qr_path = generate_qr(public_url, filename)

    return (path, qr_path, public_url)


def save_photo_qr(frame):
    filename = f"foto_{int(time.time())}.png"
    path = os.path.join(GALLERY_FOLDER, filename)

    photo = create_photo_with_drawing(frame)

    success = cv2.imwrite(path, photo)

    if not success:
        print("No se pudo guardar la foto.")
        return None

    public_url = upload_image_to_internet(path)

    if public_url is None:
        return None

    qr_path = generate_qr(public_url, filename)

    return (path, qr_path, public_url)


def cargar_interfaz(ruta):
    interfaz = cv2.imread(ruta)

    if interfaz is None:
        print("      ERROR: NO SE ENCONTRO EL PNG")
        print("Se esperaba encontrar:")
        print(os.path.abspath(ruta))
        print()
        return None

    return cv2.resize(
        interfaz,
        (INTERFACE_WIDTH, INTERFACE_HEIGHT),
        interpolation=cv2.INTER_AREA
    )


def save_menu(frame):
    print()
    print("========================================")
    print("   [DEBUG] Se presiono 's' - abriendo menu de guardar")
    print("   [DEBUG] Buscando imagen en:")
    print("   [DEBUG]", os.path.abspath(INTERFAZ_GUARDAR_PATH))
    print("========================================")

    menu = cargar_interfaz(INTERFAZ_GUARDAR_PATH)

    if menu is not None:
        print("[DEBUG] Imagen cargada correctamente. Tamano:", menu.shape)
    else:
        print("[DEBUG] cargar_interfaz devolvio None -> se usara el respaldo negro")

    if menu is None:
        menu = np.zeros((INTERFACE_HEIGHT, INTERFACE_WIDTH, 3), dtype=np.uint8)
        menu[:] = (25, 15, 30)

        cv2.putText(menu, "GUARDAR DIBUJO", (700, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
        cv2.putText(menu, "[1] Lienzo + QR", (700, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(menu, "[2] Foto + dibujo + QR", (700, 530), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(menu, "[ESC] Cancelar", (700, 610), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(menu, "Las imagenes se eliminan despues de 24 horas", (700, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(WINDOW_NAME, menu)

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord("1"):
            result = save_canvas_qr()

            if result is not None:
                _, qr_path, _ = result
                show_qr(qr_path, "LIENZO + QR")
            break

        elif key == ord("2"):
            result = save_photo_qr(frame)

            if result is not None:
                _, qr_path, _ = result
                show_qr(qr_path, "FOTO + DIBUJO + QR")
            break

        elif key == 27:
            break


def crear_figura_memoria(shape, width, height):
    referencia = np.zeros((height, width, 3), dtype=np.uint8)

    cx = width // 2
    cy = height // 2
    color = (255, 255, 255)
    thickness = 12

    if shape == "CIRCULO":
        cv2.circle(referencia, (cx, cy), 180, color, thickness, cv2.LINE_AA)

    elif shape == "CUADRADO":
        size = 320
        x1 = cx - size // 2
        y1 = cy - size // 2
        x2 = cx + size // 2
        y2 = cy + size // 2

        cv2.rectangle(referencia, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    elif shape == "TRIANGULO":
        pts = np.array([
            [cx, cy - 190],
            [cx - 200, cy + 160],
            [cx + 200, cy + 160]
        ], dtype=np.int32)

        cv2.polylines(referencia, [pts], True, color, thickness, cv2.LINE_AA)

    elif shape == "ESTRELLA":
        pts = []

        for i in range(10):
            angle = -np.pi / 2 + i * np.pi / 5
            radius = 210 if i % 2 == 0 else 90
            x = int(cx + radius * np.cos(angle))
            y = int(cy + radius * np.sin(angle))
            pts.append([x, y])

        pts = np.array(pts, dtype=np.int32)

        cv2.polylines(referencia, [pts], True, color, thickness, cv2.LINE_AA)

    return referencia


def mostrar_figura_memoria(referencia):
    pantalla = cv2.imread(INTERFAZ_MEMORIA_FIGURA_PATH)
    if pantalla is None:
        print()
        print("========================================")
        print(" ERROR: NO SE ENCONTRO LA INTERFAZ")
        print("========================================")
        print()
        print("Se esperaba encontrar:")
        print(os.path.abspath(INTERFAZ_MEMORIA_FIGURA_PATH))
        print()
        return False

    pantalla = cv2.resize(
        pantalla,
        (INTERFACE_WIDTH, INTERFACE_HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    figura = cv2.resize(
        referencia,
        (MEMORY_FIGURE_WIDTH, MEMORY_FIGURE_HEIGHT),
        interpolation=cv2.INTER_LINEAR
    )

    pantalla[
        MEMORY_FIGURE_Y:MEMORY_FIGURE_Y + MEMORY_FIGURE_HEIGHT,
        MEMORY_FIGURE_X:MEMORY_FIGURE_X + MEMORY_FIGURE_WIDTH
    ] = figura

    inicio = time.time()

    while True:
        elapsed = time.time() - inicio

        if elapsed >= MEMORY_SHOW_TIME:
            break

        restante = max(0, MEMORY_SHOW_TIME - elapsed)

        display = pantalla.copy()

        cv2.putText(
            display, f"{restante:.1f}", (900, 1000),
            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4, cv2.LINE_AA
        )

        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            return False

    return True


def analizar_dibujo_memoria(drawing, reference, elapsed_time, strokes, memory_points):

    drawing_height, drawing_width = drawing.shape[:2]

    if reference.shape[:2] != drawing.shape[:2]:
        reference = cv2.resize(
            reference,
            (drawing_width, drawing_height),
            interpolation=cv2.INTER_LINEAR
        )

    drawing_mask = np.any(
        drawing != 0,
        axis=2
    ).astype(np.uint8) * 255

    reference_mask = np.any(
        reference != 0,
        axis=2
    ).astype(np.uint8) * 255

    kernel = np.ones((21, 21), np.uint8)

    drawing_tolerant = cv2.dilate(
        drawing_mask,
        kernel,
        iterations=1
    )

    overlap = cv2.bitwise_and(
        reference_mask,
        drawing_tolerant
    )

    reference_pixels = cv2.countNonZero(reference_mask)

    if reference_pixels == 0:
        precision = 0
    else:
        covered_pixels = cv2.countNonZero(overlap)

        precision = (
            covered_pixels / reference_pixels
        ) * 100

    precision = max(
        0,
        min(100, precision)
    )

    if elapsed_time <= 10:
        speed = 100
    elif elapsed_time <= 20:
        speed = 85
    elif elapsed_time <= 30:
        speed = 70
    elif elapsed_time <= 45:
        speed = 55
    else:
        speed = 40


    score = (
        precision * 0.70 +
        speed * 0.30
    )

    score = max(
        0,
        min(100, score)
    )

    return {
        "precision": round(precision, 1),
        "speed": round(speed, 1),
        "score": round(score)
    }


def mostrar_resultados_memoria(nombre_figura, reference, drawing, analysis):

    screen = np.zeros(
        (INTERFACE_HEIGHT, INTERFACE_WIDTH, 3),
        dtype=np.uint8
    )

    screen[:] = (25, 15, 30)


    cv2.putText(
        screen,
        "RESULTADOS",
        (760, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (255, 255, 255),
        3,
        cv2.LINE_AA
    )

    cv2.putText(
        screen,
        f"Figura: {nombre_figura}",
        (790, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2,
        cv2.LINE_AA
    )


    reference_display = cv2.resize(
        reference,
        (360, 240),
        interpolation=cv2.INTER_LINEAR
    )

    screen[200:440, 150:510] = reference_display

    cv2.putText(
        screen,
        "FIGURA ORIGINAL",
        (190, 480),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    drawing_display = cv2.resize(
        drawing,
        (360, 240),
        interpolation=cv2.INTER_LINEAR
    )

    screen[200:440, 1410:1770] = drawing_display

    cv2.putText(
        screen,
        "TU REPRODUCCION",
        (1450, 480),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        screen,
        f"PRECISION: {analysis['precision']} %",
        (650, 330),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        screen,
        f"VELOCIDAD: {analysis['speed']} / 100",
        (650, 430),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        screen,
        f"PUNTAJE: {analysis['score']} / 100",
        (650, 570),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.15,
        (255, 255, 255),
        3,
        cv2.LINE_AA
    )


    if analysis["score"] >= 90:
        message = "EXCELENTE!"
    elif analysis["score"] >= 75:
        message = "MUY BIEN!"
    elif analysis["score"] >= 60:
        message = "BUEN TRABAJO!"
    else:
        message = "SIGUE PRACTICANDO!"

    cv2.putText(
        screen,
        message,
        (790, 700),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        3,
        cv2.LINE_AA
    )


    cv2.putText(
        screen,
        "ENTER = NUEVO RETO",
        (600, 900),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        screen,
        "R = MENU",
        (830, 950),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        screen,
        "ESC = SALIR",
        (1030, 950),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.imshow(WINDOW_NAME, screen)


    while True:

        key = cv2.waitKey(30) & 0xFF

        if key in (13, 10):
            return "NUEVO_RETO"

        if key in (ord("r"), ord("R")):
            return "MENU"

        if key == 27:
            return "SALIR"

def crear_pantalla_interfaz(interfaz, frame, canvas):
    if interfaz is None:
        return None

    pantalla = interfaz.copy()

    camera = cv2.resize(
        frame,
        (CAMERA_WIDTH, CAMERA_HEIGHT),
        interpolation=cv2.INTER_LINEAR
    )

    if canvas is not None:
        canvas_resized = cv2.resize(
            canvas,
            (CAMERA_WIDTH, CAMERA_HEIGHT),
            interpolation=cv2.INTER_LINEAR
        )

        gris = cv2.cvtColor(canvas_resized, cv2.COLOR_BGR2GRAY)

        _, mask = cv2.threshold(
            gris,
            1,
            255,
            cv2.THRESH_BINARY
        )

        mask_invertida = cv2.bitwise_not(mask)

        fondo = cv2.bitwise_and(
            camera,
            camera,
            mask=mask_invertida
        )

        dibujo = cv2.bitwise_and(
            canvas_resized,
            canvas_resized,
            mask=mask
        )

        camera = cv2.add(fondo, dibujo)

    pantalla[
        CAMERA_Y:CAMERA_Y + CAMERA_HEIGHT,
        CAMERA_X:CAMERA_X + CAMERA_WIDTH
    ] = camera

    return pantalla


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

canvas = None
previous_point = None
last_color_change = 0
timestamp = 0

memory_shape_name = None
memory_reference = None

memory_start_time = None
memory_strokes = 0
memory_points = []
def mostrar_menu_principal():

    MENU_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "interfaz",
        "menu.png"
    )
    

    menu = cv2.imread(MENU_PATH)

    if menu is None:
        print("No se pudo cargar:", MENU_PATH)
        return None

    menu = cv2.resize(menu, (1280, 720))

    cv2.namedWindow("AIR DRAW - MENU", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        "AIR DRAW - MENU",
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )

    while True:
        cv2.imshow("AIR DRAW - MENU", menu)

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord("1"):
            cv2.destroyWindow("AIR DRAW - MENU")
            return "LIBRE"

        elif tecla == ord("2"):
            cv2.destroyWindow("AIR DRAW - MENU")
            return "MEMORIA"

        elif tecla == ord("q") or tecla == 27:
            cv2.destroyWindow("AIR DRAW - MENU")
            return None
GAME_MODE = mostrar_menu_principal()

if GAME_MODE is None:
    cap.release()
    cv2.destroyAllWindows()
    exit()

WINDOW_NAME = "AIR DRAW - Dibuja en el aire"

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)


with HandLandmarker.create_from_options(options) as detector:
    while GAME_MODE is not None:

        canvas = None
        previous_point = None
        point_history.clear()
        return_to_menu = False
        memory_strokes = 0
        memory_points = []

        if GAME_MODE == "LIBRE":
            interfaz_actual = cargar_interfaz(INTERFAZ_LIBRE_PATH)

        elif GAME_MODE == "MEMORIA":
            interfaz_actual = cargar_interfaz(INTERFAZ_MEMORIA_PATH)

        else:
            interfaz_actual = None

        if interfaz_actual is None:
            GAME_MODE = None
            break

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        if GAME_MODE == "MEMORIA":
            memory_shape_name = random.choice(MEMORY_SHAPES)

            memory_reference = crear_figura_memoria(
                memory_shape_name, MEMORY_FIGURE_WIDTH, MEMORY_FIGURE_HEIGHT
            )

            mostrar_ok = mostrar_figura_memoria(memory_reference)

            if not mostrar_ok:
                GAME_MODE = None
                break

            memory_start_time = time.time()
            memory_strokes = 0
            memory_points = []

        while True:
            success, frame = cap.read()

            if not success:
                print("No se pudo leer la cámara.")
                break

            frame = cv2.flip(frame, 1)

            height, width, _ = frame.shape

            if canvas is None:
                canvas = np.zeros_like(frame)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp += 33

            result = detector.detect_for_video(mp_image, timestamp)

            gesture = "SIN MANO"

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]

                gesture = get_gesture(landmarks)

                index_tip = landmarks[8]

                raw_x = int(index_tip.x * width)
                raw_y = int(index_tip.y * height)

                raw_x = max(0, min(width - 1, raw_x))
                raw_y = max(0, min(height - 1, raw_y))

                x, y = smooth_point(raw_x, raw_y)

                if gesture == "DIBUJAR":
                    current_point = (x, y)

                    6
                    if GAME_MODE == "MEMORIA" and previous_point is None:
                        memory_strokes += 1
                        memory_points.append(current_point)

                    if previous_point is not None:
                        distance = np.sqrt(
                            (x - previous_point[0]) ** 2 + (y - previous_point[1]) ** 2
                        )

                        if distance >= DEADZONE:
                            if distance <= MAX_JUMP:
                                cv2.line(
                                    canvas, previous_point, current_point,
                                    brush_color, brush_size, cv2.LINE_AA
                                )

                                if GAME_MODE == "MEMORIA":
                                    memory_points.append(current_point)

                    previous_point = current_point

                else:
                    previous_point = None
                    point_history.clear()

                if gesture == "COLOR":
                    current_time = time.time()

                    if current_time - last_color_change > 1:
                        color_index = (color_index + 1) % len(COLORS)
                        brush_color = COLORS[color_index]
                        last_color_change = current_time

                if gesture == "BORRAR":
                    erase_radius = 35
                    erase_at(x, y, erase_radius)

            else:
                previous_point = None
                point_history.clear()

            if gesture == "COLOR":
                color_feedback = "CAMBIANDO COLOR"
            elif gesture == "BORRAR":
                color_feedback = "BORRANDO"
            elif gesture == "DIBUJAR":
                color_feedback = "DIBUJANDO"
            elif gesture == "PAUSA":
                color_feedback = "PAUSA"
            else:
                color_feedback = "SIN MANO"

            display = crear_pantalla_interfaz(interfaz_actual, frame, canvas)

            if display is None:
                break

            if result.hand_landmarks:
                interface_x = CAMERA_X + int(x * CAMERA_WIDTH / width)
                interface_y = CAMERA_Y + int(y * CAMERA_HEIGHT / height)

                interface_x = max(CAMERA_X, min(CAMERA_X + CAMERA_WIDTH - 1, interface_x))
                interface_y = max(CAMERA_Y, min(CAMERA_Y + CAMERA_HEIGHT - 1, interface_y))

                if gesture != "BORRAR":
                    cv2.circle(display, (interface_x, interface_y), brush_size, brush_color, 2, cv2.LINE_AA)
                    cv2.circle(display, (interface_x, interface_y), 3, brush_color, -1, cv2.LINE_AA)
                else:
                    cv2.circle(display, (interface_x, interface_y), 35, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.circle(display, (interface_x, interface_y), 3, (255, 255, 255), -1, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                save_menu(frame)

                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            elif GAME_MODE == "MEMORIA" and key in (13, 10):
                elapsed_time = time.time() - memory_start_time

                analysis = analizar_dibujo_memoria(
                    canvas, memory_reference, elapsed_time, memory_strokes, memory_points
                )

                result_action = mostrar_resultados_memoria(
                    memory_shape_name, memory_reference, canvas, analysis
                )

                if result_action == "NUEVO_RETO":
                    break

                elif result_action == "MENU":
                    return_to_menu = True
                    break

                elif result_action == "SALIR":
                    GAME_MODE = None
                    break

            elif key in (ord("r"), ord("R")):
                return_to_menu = True
                break

            elif key == 27:
                return_to_menu = False
                GAME_MODE = None
                break

        if GAME_MODE is None:
            break

        if return_to_menu:
            cv2.destroyWindow(WINDOW_NAME)
            GAME_MODE = mostrar_menu_principal()

cap.release()
cv2.destroyAllWindows()

print()
print("========================================")
print("           AIR DRAW CERRADO")
print("========================================")
print()
