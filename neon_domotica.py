import cv2
import numpy as np
import json
import csv
import os
import time
import threading
from datetime import datetime
from flask import Flask, jsonify, render_template


# ==========================================================
# CONFIGURACIÓN GENERAL
# ==========================================================

app = Flask(__name__)

DWELL_TIME = 1.5

# Filtros de detección
MIN_AREA_RATIO = 0.002
MIN_SOLIDITY = 0.70

# Azul: rango HSV
BLUE_LOWER = np.array([100, 100, 60])
BLUE_UPPER = np.array([130, 255, 255])

# Rojo: dos rangos HSV
RED_LOWER_1 = np.array([0, 100, 70])
RED_UPPER_1 = np.array([10, 255, 255])

RED_LOWER_2 = np.array([170, 100, 70])
RED_UPPER_2 = np.array([180, 255, 255])


# ==========================================================
# ESTADO COMPARTIDO CON HTML
# ==========================================================

state_lock = threading.Lock()

system_state = {
    "connected": False,
    "worn": False,

    "gaze_x": None,
    "gaze_y": None,

    "detected_color": "Ninguno",
    "dwell_time": 0.0,
    "progress": 0,

    "lamp_on": False,
    "lamp_state": "OFF",

    "timestamp": None,
    "gaze_data": {},

    "message": "Esperando conexión con Neon"
}


# ==========================================================
# REGISTROS
# ==========================================================

session_name = datetime.now().strftime("sesion_%Y%m%d_%H%M%S")

session_dir = os.path.join("registros", session_name)

os.makedirs(session_dir, exist_ok=True)

jsonl_path = os.path.join(session_dir, "gaze_completo.jsonl")
csv_path = os.path.join(session_dir, "gaze_completo.csv")
events_path = os.path.join(session_dir, "eventos.jsonl")

csv_fields = [
    "timestamp",
    "gaze_x",
    "gaze_y",
    "worn",
    "detected_color",
    "dwell_time",
    "progress",
    "lamp_on",
    "gaze_data_json"
]

csv_file = open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8-sig"
)

csv_writer = csv.DictWriter(
    csv_file,
    fieldnames=csv_fields
)

csv_writer.writeheader()
csv_file.flush()


def save_record(record):

    with open(jsonl_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    csv_writer.writerow({
        "timestamp": record.get("timestamp"),
        "gaze_x": record.get("gaze_x"),
        "gaze_y": record.get("gaze_y"),
        "worn": record.get("worn"),
        "detected_color": record.get("detected_color"),
        "dwell_time": record.get("dwell_time"),
        "progress": record.get("progress"),
        "lamp_on": record.get("lamp_on"),
        "gaze_data_json": json.dumps(
            record.get("gaze_data", {}),
            ensure_ascii=False,
            default=str
        )
    })

    csv_file.flush()


def save_event(message):

    event = {
        "timestamp": datetime.now().isoformat(),
        "event": message
    }

    with open(events_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")

    print(message)


# ==========================================================
# FUNCIONES DE DETECCIÓN DE COLOR
# ==========================================================

def create_color_masks(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detectar exclusivamente azul
    blue_mask = cv2.inRange(
        hsv,
        BLUE_LOWER,
        BLUE_UPPER
    )

    # Detectar exclusivamente rojo
    red_mask_1 = cv2.inRange(
        hsv,
        RED_LOWER_1,
        RED_UPPER_1
    )

    red_mask_2 = cv2.inRange(
        hsv,
        RED_LOWER_2,
        RED_UPPER_2
    )

    red_mask = cv2.bitwise_or(
        red_mask_1,
        red_mask_2
    )

    kernel = np.ones((5, 5), np.uint8)

    blue_mask = cv2.morphologyEx(
        blue_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    blue_mask = cv2.morphologyEx(
        blue_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    red_mask = cv2.morphologyEx(
        red_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    red_mask = cv2.morphologyEx(
        red_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return blue_mask, red_mask


def get_valid_contours(mask, frame_area):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = []

    min_area = frame_area * MIN_AREA_RATIO

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if h == 0:
            continue

        aspect_ratio = w / float(h)

        # Evitar objetos extremadamente alargados
        if not (0.45 <= aspect_ratio <= 2.2):
            continue

        hull = cv2.convexHull(contour)

        hull_area = cv2.contourArea(hull)

        if hull_area == 0:
            continue

        solidity = area / hull_area

        if solidity < MIN_SOLIDITY:
            continue

        valid_contours.append(contour)

    return valid_contours


# ==========================================================
# CONVERSIÓN DE COORDENADAS DE MIRADA
# ==========================================================

def convert_gaze_coordinates(gaze, width, height):

    x = float(gaze.x)
    y = float(gaze.y)

    # El SDK puede entregar coordenadas normalizadas (0 a 1)
    # o coordenadas expresadas en píxeles.
    if 0 <= x <= 1.5 and 0 <= y <= 1.5:

        x = int(x * width)
        y = int(y * height)

    else:

        x = int(x)
        y = int(y)

    # Limitar coordenadas al tamaño del frame
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))

    return x, y


# ==========================================================
# IDENTIFICAR SI LA MIRADA ESTÁ SOBRE UN COLOR
# ==========================================================

def detect_gaze_color(
    gaze_x,
    gaze_y,
    blue_contours,
    red_contours
):

    # Primero comprobar azul
    for contour in blue_contours:

        result = cv2.pointPolygonTest(
            contour,
            (float(gaze_x), float(gaze_y)),
            False
        )

        if result >= 0:
            return "Azul", contour

    # Después comprobar rojo
    for contour in red_contours:

        result = cv2.pointPolygonTest(
            contour,
            (float(gaze_x), float(gaze_y)),
            False
        )

        if result >= 0:
            return "Rojo", contour

    return "Ninguno", None


# ==========================================================
# ACTUALIZAR LÁMPARA
# ==========================================================

def change_lamp(color):

    with state_lock:

        if color == "Azul":

            system_state["lamp_on"] = True
            system_state["lamp_state"] = "ON"

            message = "LÁMPARA ENCENDIDA mediante mirada al azul"

        elif color == "Rojo":

            system_state["lamp_on"] = False
            system_state["lamp_state"] = "OFF"

            message = "LÁMPARA APAGADA mediante mirada al rojo"

        else:
            return

    save_event(message)


# ==========================================================
# FLASK: COMUNICACIÓN CON HTML
# ==========================================================

@app.route("/")
def index():

    return render_template("index.html")


@app.route("/api/state")
def get_state():

    with state_lock:

        current_state = system_state.copy()

    return jsonify(current_state)


@app.route("/api/health")
def health():

    return jsonify({
        "server": "online",
        "connected": system_state["connected"]
    })


# ==========================================================
# PROCESAMIENTO PRINCIPAL DE NEON
# ==========================================================

def run_neon():

    from pupil_labs.realtime_api.simple import discover_one_device

    print("\nBuscando lentes Pupil Labs Neon...")

    while True:

        try:

            device = discover_one_device()

            if device is None:

                print("No se encontraron lentes. Reintentando...")
                time.sleep(3)
                continue

            print("Neon conectado correctamente.")

            with state_lock:

                system_state["connected"] = True
                system_state["message"] = "Neon conectado"

            save_event("Lentes Neon conectados")

            dwell_start = None
            active_color = "Ninguno"
            triggered_color = None

            while True:

                frame, gaze = (
                    device.receive_matched_scene_video_frame_and_gaze()
                )

                image = frame.bgr_pixels.copy()

                height, width = image.shape[:2]

                # Datos de mirada
                gaze_x, gaze_y = convert_gaze_coordinates(
                    gaze,
                    width,
                    height
                )

                worn = bool(gaze.worn)

                # Datos adicionales expuestos por el objeto gaze
                if hasattr(gaze, "_asdict"):

                    gaze_data = gaze._asdict()

                elif hasattr(gaze, "__dict__"):

                    gaze_data = gaze.__dict__

                else:

                    gaze_data = {}

                # Detectar colores
                blue_mask, red_mask = create_color_masks(image)

                frame_area = width * height

                blue_contours = get_valid_contours(
                    blue_mask,
                    frame_area
                )

                red_contours = get_valid_contours(
                    red_mask,
                    frame_area
                )

                detected_color, selected_contour = detect_gaze_color(
                    gaze_x,
                    gaze_y,
                    blue_contours,
                    red_contours
                )

                # Permanencia de mirada
                current_time = time.time()

                if not worn:

                    detected_color = "Ninguno"

                if detected_color != "Ninguno":

                    if detected_color != active_color:

                        active_color = detected_color
                        dwell_start = current_time
                        triggered_color = None

                    if dwell_start is None:

                        dwell_start = current_time

                    dwell_time = current_time - dwell_start

                else:

                    active_color = "Ninguno"
                    dwell_start = None
                    triggered_color = None
                    dwell_time = 0.0

                # Evitar activar repetidamente mientras
                # la mirada permanece sobre el mismo color
                if (
                    detected_color != "Ninguno"
                    and dwell_time >= DWELL_TIME
                    and triggered_color != detected_color
                ):

                    change_lamp(detected_color)

                    triggered_color = detected_color

                # Progreso de selección
                progress = min(
                    100,
                    int((dwell_time / DWELL_TIME) * 100)
                )

                # Actualizar estado para HTML
                with state_lock:

                    system_state.update({

                        "connected": True,
                        "worn": worn,

                        "gaze_x": gaze_x,
                        "gaze_y": gaze_y,

                        "detected_color": detected_color,

                        "dwell_time": round(dwell_time, 2),

                        "progress": progress,

                        "timestamp": datetime.now().isoformat(),

                        "gaze_data": gaze_data,

                        "message": "Neon funcionando"

                    })

                    record = system_state.copy()

                # Guardar registros
                save_record(record)

                # Dibujar contornos
                for contour in blue_contours:

                    cv2.drawContours(
                        image,
                        [contour],
                        -1,
                        (255, 0, 0),
                        2
                    )

                for contour in red_contours:

                    cv2.drawContours(
                        image,
                        [contour],
                        -1,
                        (0, 0, 255),
                        2
                    )

                # Dibujar punto de mirada
                cv2.circle(
                    image,
                    (gaze_x, gaze_y),
                    10,
                    (0, 255, 0),
                    -1
                )

                # Información en pantalla
                cv2.putText(
                    image,
                    f"Color: {detected_color}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    image,
                    f"Dwell: {dwell_time:.1f}s / 1.5s",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    image,
                    f"Lamp: {'ON' if system_state['lamp_on'] else 'OFF'}",
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                if selected_contour is not None:

                    cv2.drawContours(
                        image,
                        [selected_contour],
                        -1,
                        (0, 255, 0),
                        4
                    )

                cv2.imshow(
                    "Neon Smart Home - Eye Tracking",
                    image
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):

                    save_event("Sistema detenido por el usuario")
                    return

        except Exception as error:

            print("\nError de conexión o procesamiento:")
            print(error)

            with state_lock:

                system_state["connected"] = False
                system_state["worn"] = False
                system_state["message"] = str(error)

            save_event(f"Error: {error}")

            cv2.destroyAllWindows()

            time.sleep(3)


# ==========================================================
# INICIO DEL SISTEMA
# ==========================================================

if __name__ == "__main__":

    print("---------------------------------------")
    print("       NEON SMART HOME MVP")
    print("---------------------------------------")

    print("Iniciando servidor HTML...")

    flask_thread = threading.Thread(
        target=lambda: app.run(
            host="127.0.0.1",
            port=5000,
            debug=False,
            use_reloader=False,
            threaded=True
        ),
        daemon=True
    )

    flask_thread.start()

    time.sleep(1)

    print("\nDashboard disponible en:")
    print("http://127.0.0.1:5000")

    print("\nIniciando conexión con Neon...")

    try:

        run_neon()

    except KeyboardInterrupt:

        print("\nSistema detenido.")

    finally:

        with state_lock:

            system_state["connected"] = False

        csv_file.close()

        cv2.destroyAllWindows()