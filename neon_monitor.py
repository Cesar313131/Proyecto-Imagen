#Prueba 3
import cv2
import numpy as np
import time

from pupil_labs.realtime_api.simple import discover_one_device


# ==========================================
# 1. CONECTAR PUPIL LABS
# ==========================================

print("Buscando Pupil Labs Neon...")

device = discover_one_device(max_search_duration_seconds=10)

if device is None:
    print("No se encontraron los lentes.")
    exit()

print("Pupil Labs conectado correctamente.")


# ==========================================
# 2. CONFIGURAR COLORES
# ==========================================

colores = {

    "VERDE": {
        "inferior": np.array([35, 60, 60]),
        "superior": np.array([85, 255, 255]),
        "color": (0, 255, 0),
        "accion": "ENCENDER LAMPARA"
    },

    "AZUL": {
        "inferior": np.array([90, 80, 50]),
        "superior": np.array([130, 255, 255]),
        "color": (255, 0, 0),
        "accion": "APAGAR LAMPARA"
    }

}


# ==========================================
# 3. CONFIGURAR MONITOR
# ==========================================

ultimo_print = 0

try:

    while True:

        frame, gaze = device.receive_matched_scene_video_frame_and_gaze()

        imagen = frame.bgr_pixels.copy()

        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

        # ==================================
        # 4. DETECTAR HOJAS DE COLORES
        # ==================================

        for nombre, datos in colores.items():

            mascara = cv2.inRange(
                hsv,
                datos["inferior"],
                datos["superior"]
            )

            contornos, _ = cv2.findContours(
                mascara,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for contorno in contornos:

                area = cv2.contourArea(contorno)

                if area > 5000:

                    x, y, w, h = cv2.boundingRect(contorno)

                    cv2.rectangle(
                        imagen,
                        (x, y),
                        (x+w, y+h),
                        datos["color"],
                        3
                    )

                    cv2.putText(
                        imagen,
                        nombre,
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        datos["color"],
                        2
                    )

        # ==================================
        # 5. OBTENER DATOS DE MIRADA
        # ==================================

        datos_gaze = gaze._asdict()

        if gaze.worn:

            gx = int(gaze.x)
            gy = int(gaze.y)

            if 0 <= gx < imagen.shape[1] and 0 <= gy < imagen.shape[0]:

                cv2.circle(
                    imagen,
                    (gx, gy),
                    15,
                    (0, 0, 255),
                    -1
                )

        # ==================================
        # 6. MOSTRAR DATOS EN CONSOLA
        # ==================================

        if time.time() - ultimo_print >= 0.2:

            print("\n========== NEON DATA ==========")

            for nombre, valor in datos_gaze.items():

                print(f"{nombre}: {valor}")

            print("===============================")

            ultimo_print = time.time()

        # ==================================
        # 7. MOSTRAR INFORMACION EN VIDEO
        # ==================================

        cv2.putText(
            imagen,
            f"GAZE X: {gaze.x:.2f}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            imagen,
            f"GAZE Y: {gaze.y:.2f}",
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            imagen,
            f"WORN: {gaze.worn}",
            (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("NEON - EYE TRACKING MONITOR", imagen)

        # ESC para salir
        if cv2.waitKey(1) & 0xFF == 27:
            break


except KeyboardInterrupt:
    pass

finally:

    device.close()
    cv2.destroyAllWindows()

    print("Conexion finalizada.")