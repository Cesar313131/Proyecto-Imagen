import cv2
import numpy as np
import time

from pupil_labs.realtime_api.simple import discover_one_device


# ==========================================
# 1. CONEXION
# ==========================================

print("Buscando Pupil Labs Neon...")

device = discover_one_device(max_search_duration_seconds=10)

if device is None:
    print("No se encontraron los lentes.")
    exit()

print("Neon conectado correctamente.")


# ==========================================
# 2. CONFIGURACION
# ==========================================

TIEMPO_SELECCION = 3

estado_lampara = False

color_actual = None
tiempo_inicio = None

seleccion_realizada = False


# ==========================================
# 3. RANGOS HSV
# ==========================================

azul_bajo = np.array([90, 80, 50])
azul_alto = np.array([130, 255, 255])

rojo_bajo1 = np.array([0, 100, 100])
rojo_alto1 = np.array([10, 255, 255])

rojo_bajo2 = np.array([170, 100, 100])
rojo_alto2 = np.array([180, 255, 255])


# ==========================================
# 4. FUNCION DE DETECCION
# ==========================================

def detectar_hoja(mascara, imagen, nombre, color):

    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    resultado = None

    for contorno in contornos:

        area = cv2.contourArea(contorno)

        if area > 5000:

            x, y, w, h = cv2.boundingRect(contorno)

            cv2.rectangle(
                imagen,
                (x, y),
                (x+w, y+h),
                color,
                3
            )

            cv2.putText(
                imagen,
                nombre,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

            resultado = (x, y, w, h)

    return resultado


# ==========================================
# 5. BUCLE PRINCIPAL
# ==========================================

try:

    while True:

        frame, gaze = device.receive_matched_scene_video_frame_and_gaze()

        imagen = frame.bgr_pixels.copy()

        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)


        # Detectar azul
        mascara_azul = cv2.inRange(
            hsv,
            azul_bajo,
            azul_alto
        )


        # Detectar rojo
        mascara_rojo1 = cv2.inRange(
            hsv,
            rojo_bajo1,
            rojo_alto1
        )

        mascara_rojo2 = cv2.inRange(
            hsv,
            rojo_bajo2,
            rojo_alto2
        )

        mascara_rojo = cv2.bitwise_or(
            mascara_rojo1,
            mascara_rojo2
        )


        # Obtener rectangulos
        hoja_azul = detectar_hoja(
            mascara_azul,
            imagen,
            "AZUL - ENCENDER",
            (255, 0, 0)
        )

        hoja_roja = detectar_hoja(
            mascara_rojo,
            imagen,
            "ROJO - APAGAR",
            (0, 0, 255)
        )


        # ==================================
        # 6. DETECTAR POSICION DE MIRADA
        # ==================================

        seleccion = None

        if gaze.worn:

            gx = int(gaze.x)
            gy = int(gaze.y)

            if 0 <= gx < imagen.shape[1] and 0 <= gy < imagen.shape[0]:

                cv2.circle(
                    imagen,
                    (gx, gy),
                    15,
                    (0, 255, 0),
                    -1
                )

                # Comprobar si mira la hoja azul
                if hoja_azul:

                    x, y, w, h = hoja_azul

                    if x <= gx <= x+w and y <= gy <= y+h:
                        seleccion = "AZUL"


                # Comprobar si mira la hoja roja
                if hoja_roja:

                    x, y, w, h = hoja_roja

                    if x <= gx <= x+w and y <= gy <= y+h:
                        seleccion = "ROJO"


        # ==================================
        # 7. TIEMPO DE PERMANENCIA
        # ==================================

        if seleccion is not None:

            if seleccion != color_actual:

                color_actual = seleccion
                tiempo_inicio = time.monotonic()
                seleccion_realizada = False

            tiempo_mirada = time.monotonic() - tiempo_inicio

            porcentaje = min(
                tiempo_mirada / TIEMPO_SELECCION,
                1
            )

            cv2.putText(
                imagen,
                f"Mirando: {seleccion}",
                (30, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                imagen,
                f"Seleccion: {porcentaje*100:.0f}%",
                (30, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )


            # Ejecutar accion
            if tiempo_mirada >= TIEMPO_SELECCION and not seleccion_realizada:

                if seleccion == "AZUL":

                    estado_lampara = True

                    print("ORDEN: ENCENDER LAMPARA")

                elif seleccion == "ROJO":

                    estado_lampara = False

                    print("ORDEN: APAGAR LAMPARA")


                seleccion_realizada = True


        else:

            color_actual = None
            tiempo_inicio = None
            seleccion_realizada = False


        # ==================================
        # 8. ESTADO DE LA LAMPARA
        # ==================================

        estado_texto = "ENCENDIDA" if estado_lampara else "APAGADA"

        cv2.putText(
            imagen,
            f"LAMPARA: {estado_texto}",
            (30, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0) if estado_lampara else (0, 0, 255),
            2
        )


        # ==================================
        # 9. MOSTRAR INFORMACION
        # ==================================

        cv2.putText(
            imagen,
            f"X: {gaze.x:.1f} Y: {gaze.y:.1f}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            imagen,
            f"WORN: {gaze.worn}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        cv2.imshow("NEON DOMOTICA - CONTROL VISUAL", imagen)


        if cv2.waitKey(1) & 0xFF == 27:
            break


except KeyboardInterrupt:
    pass


finally:

    device.close()
    cv2.destroyAllWindows()

    print("Sistema finalizado.")