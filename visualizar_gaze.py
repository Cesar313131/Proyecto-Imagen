import cv2

from pupil_labs.realtime_api.simple import discover_one_device

print("Buscando Pupil Labs Neon...")

device = discover_one_device(max_search_duration_seconds=10)

if device is None:
    print("No se encontraron los lentes.")
    exit()

print("Conectado correctamente.")
print("Presiona ESC para salir.")

try:

    while True:

        # Recibir video y datos de mirada sincronizados
        frame, gaze = device.receive_matched_scene_video_frame_and_gaze()

        # Obtener imagen de la cámara frontal
        imagen = frame.bgr_pixels.copy()

        # Obtener coordenadas
        x = int(gaze.x)
        y = int(gaze.y)

        # Comprobar que los lentes están puestos
        if gaze.worn:

            # Dibujar punto rojo
            cv2.circle(
                imagen,
                (x, y),
                20,
                (0, 0, 255),
                -1
            )

            # Mostrar coordenadas
            cv2.putText(
                imagen,
                f"X: {x} | Y: {y}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        # Mostrar imagen
        cv2.imshow("Pupil Labs Neon - Eye Tracking", imagen)

        # Salir con ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    pass

finally:

    device.close()
    cv2.destroyAllWindows()