from pupil_labs.realtime_api.simple import discover_one_device

print("Buscando Pupil Labs Neon...")

device = discover_one_device(max_search_duration_seconds=10)

if device is None:
    print("No se encontraron los lentes.")
else:
    print("¡Pupil Labs Neon conectado correctamente!")

    print("Información del dispositivo:")
    print(device)

    device.close()