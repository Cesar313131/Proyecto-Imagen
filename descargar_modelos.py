"""Descarga 2 modelos YA ENTRENADOS POR TERCEROS.

Puerta: perseusdg/doorway-door-people-window-seg, YOLO26n-seg (AGPL-3.0).
Laptop: Ultralytics YOLOv8n, clase COCO laptop.
Los modelos NO son entrenados por el autor de este proyecto.
"""
from __future__ import annotations
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
MODELS=ROOT/'modelos'
DOOR=MODELS/'doorway_seg_yolo26n.pt'
LAPTOP=MODELS/'yolov8n.pt'
REPO='perseusdg/doorway-door-people-window-seg'


def main():
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO
    MODELS.mkdir(parents=True,exist_ok=True)
    if not DOOR.is_file():
        print('[1/2] Descargando puertas YA ENTRENADO por un tercero...',flush=True)
        src=hf_hub_download(repo_id=REPO, filename=DOOR.name, local_dir=str(MODELS))
        if Path(src).resolve()!=DOOR.resolve():
            import shutil
            shutil.copy2(src,DOOR)
    else:
        print('[1/2] Puertas: pesos encontrados.')
    if not LAPTOP.is_file():
        print('[2/2] Descargando YOLOv8n COCO para laptops...',flush=True)
        old=Path.cwd()
        try:
            os.chdir(MODELS)
            YOLO('yolov8n.pt')
        finally:
            os.chdir(old)
    else:
        print('[2/2] Laptops: pesos encontrados.')
    if not DOOR.is_file() or not LAPTOP.is_file():
        raise FileNotFoundError('No se descargaron ambos archivos .pt')
    door_model=YOLO(str(DOOR))
    laptop_model=YOLO(str(LAPTOP))
    from vision_core import normalize_door_class
    labels={normalize_door_class(x) for x in door_model.names.values()}
    if not {'door_open','door_closed'}.issubset(labels):
        raise ValueError(f'Pesos puertas incompatibles: {door_model.names}')
    if 'laptop' not in set(map(str.lower,laptop_model.names.values())):
        raise ValueError('El modelo YOLO de laptops no tiene la clase laptop.')
    print('\nMODELOS LISTOS PARA PRUEBAS CON NEON:')
    print('Puertas:',DOOR,'(doorway~abierta, door~cerrada; estimacion imperfecta)')
    print('Laptop:',LAPTOP)
    return 0


if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as err:
        print(f'ERROR: {type(err).__name__}: {err}',file=sys.stderr)
        print('Necesitas acceso a Internet. Verifica conexiones a Hugging Face y GitHub.\n'
              'No son pesos propios: provienen de modelos publicos preentrenados.',file=sys.stderr)
        raise SystemExit(1)
