# AIR DRAW

Dibuja en el aire con la mano frente a la webcam. La cámara sigue tu mano con
**MediaPipe** y el dedo índice pinta sobre un lienzo transparente que se mezcla
con el video en tiempo real.

Proyecto de visión por computador hecho con **Python**, **OpenCV** y
**MediaPipe Hand Landmarker**.

## Qué hace

- **Modo libre:** dibujas lo que quieras en el aire.
- **Modo memoria:** la app te muestra una figura (círculo, cuadrado, triángulo o
  estrella) durante unos segundos; luego la dibujas de memoria y te da un
  puntaje de precisión, tiempo, trazos y distancia recorrida.
- **Cambio de color** y **borrador** sin tocar el teclado: solo con gestos.
- **Guardar y compartir:** toma el dibujo o una foto con el dibujo encima y
  genera un **código QR** para descargarlo en el celular.
- **Galería** local con lo que vas guardando.
- Las partidas del modo memoria se anotan en `air_draw_data.csv`.

## Gestos

| Gesto | Acción |
| --- | --- |
| ☝️ Índice levantado | Dibujar |
| ✌️ Índice y medio | Cambiar de color |
| 🖐️ Cuatro dedos | Borrar (borrador) |
| ✊ Mano cerrada | Pausa |

## Teclas

| Dónde | Tecla | Acción |
| --- | --- | --- |
| Menú | `1` | Modo libre |
| Menú | `2` | Modo memoria |
| Menú | `Q` o `Esc` | Salir |
| Jugando | `S` | Guardar y mostrar el QR |
| Jugando | `R` | Volver al menú |
| Jugando | `Q` o `Esc` | Salir |

## Requisitos

- Python 3.10 o superior (64 bits)
- Cámara web
- Conexión a internet la primera vez: si falta `hand_landmarker.task`, el
  programa lo descarga solo desde los servidores de MediaPipe.

## Instalación

```powershell
cd AIR_DRAW
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.text
```

En Linux o macOS el entorno se activa con `source .venv/bin/activate`.

## Uso

```powershell
python main.py
```

Se abre el menú a pantalla completa: elige `1` o `2` y ponte frente a la cámara.

## Estructura

| Archivo / carpeta | Para qué sirve |
| --- | --- |
| `main.py` | Toda la app: cámara, gestos, modos, QR y galería |
| `interfaz/` | Pantallas del menú y de cada modo (PNG) |
| `dibujos/` | Dibujos guardados (se crea sola) |
| `galeria/` | Fotos, dibujos y códigos QR generados (se crea sola) |
| `air_draw_data.csv` | Resultados del modo memoria (se crea al jugar) |
| `AIR_DRAW.spec` | Configuración de PyInstaller para armar el `.exe` |
| `requirements.text` | Dependencias (`opencv-python`, `mediapipe`, `numpy`, `qrcode`, `flask`, `pillow`) |

## Generar el ejecutable

```powershell
pyinstaller AIR_DRAW.spec
```

El resultado queda en `dist/AIR_DRAW/`.

## Cómo funciona

1. OpenCV abre la cámara y entrega los cuadros a MediaPipe.
2. `HandLandmarker` devuelve 21 puntos de la mano; con la posición de la punta
   del índice (punto 8) y de las falanges se deduce el gesto.
3. El punto se suaviza con un promedio móvil (`deque`) para que la línea no
   tiemble.
4. Lo que se dibuja va a un lienzo aparte; al final se compone con el video
   usando una máscara, para que el trazo quede encima de la imagen.
5. Al guardar, la imagen se sube a un servicio temporal y se genera un QR con
   `qrcode` que apunta a esa URL (disponible 24 horas).

## Créditos

- [MediaPipe](https://developers.google.com/mediapipe) — detección de manos.
- [OpenCV](https://opencv.org) — cámara, dibujo y ventanas.
- [qrcode](https://github.com/lincolnloop/python-qrcode) — códigos QR.
