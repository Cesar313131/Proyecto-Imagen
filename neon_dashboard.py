"""Neon + YOLO puerta abierta/cerrada + laptop + dashboard.

Usar con lentes reales. Las acciones de puerta y lampara son SOLO VIRTUALES.
Los pesos de puerta, si se descargan, son de terceros; no se entrenan aqui.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import threading
import traceback
import time
import webbrowser
from collections import deque
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template

from vision_core import Detection, Dwell, find_looked_at, normalize_door_class, point_from_gaze, select_color

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(ROOT/'templates'))
STATE_LOCK = threading.Lock()
EVENTS = deque(maxlen=35)
STATE = {
    'connected': False, 'worn': False, 'gaze_x': None, 'gaze_y': None,
    'timestamp': None, 'gaze_data': {}, 'message': 'Buscando lentes Neon...',
    'detected_color': 'Ninguno', 'lamp_on': False, 'lamp_state': 'OFF',
    'dwell_time': 0., 'dwell_required': 1.5, 'progress': 0,
    'looked_at': None, 'detected_objects': [], 'door_detected_state': 'sin deteccion',
    'door_detection_confidence': None,
    'door_virtual_open': False, 'door_event_seq': 0, 'last_door_action': 'Sin acciones',
    'laptop_selections': 0, 'selection_count': 0,
    'confirmation_seq': 0, 'last_confirmation': 'Ninguna',
    'last_confirmation_kind': None, 'selection_confirmed': False,
    'fps': 0.0, 'processing_ms': 0.0,
    'model_sources': {'doors':'No cargado','laptops':'No cargado'},
    'events': []
}
CSV_PATH = ROOT / 'resultados' / 'selecciones.csv'


def add_event(text, kind='info'):
    item = {'seq': int(time.time_ns()), 'time':datetime.now().strftime('%H:%M:%S'), 'text':text, 'kind':kind}
    with STATE_LOCK:
        EVENTS.appendleft(item)
        STATE['events'] = list(EVENTS)
    print(f"[{item['time']}] {text}", flush=True)


def write_selection(action, kind, conf, point, box, seconds):
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    new = not CSV_PATH.exists()
    with CSV_PATH.open('a', encoding='utf-8-sig', newline='') as file:
        writer = csv.writer(file)
        if new:
            writer.writerow(['fecha','accion_virtual','clase_detectada','confianza','mirada_x','mirada_y','bbox','permanencia_s'])
        writer.writerow([datetime.now().isoformat(timespec='seconds'),action,kind,round(conf,3),
                         point[0] if point else '',point[1] if point else '',
                         json.dumps(box) if box else '',round(seconds,2)])


@app.get('/')
def page():
    return render_template('index.html')


@app.get('/api/state')
def api_state():
    with STATE_LOCK:
        return jsonify(dict(STATE))


@app.get('/api/health')
def health():
    with STATE_LOCK:
        return jsonify({'server':'online','connected':STATE['connected']})


# El video solo se muestra en la ventana OpenCV.
# El dashboard consume estados JSON sin transmitir ni codificar frames JPEG.


def fetch_objects(result, aliases: dict[int, str]):
    output=[]
    if result.boxes is None: return output
    for box in result.boxes:
        cls = int(box.cls.item())
        if cls not in aliases: continue
        coords = tuple(int(round(float(i))) for i in box.xyxy[0].tolist())
        if coords[2] <= coords[0] or coords[3] <= coords[1]:
            continue
        output.append(Detection(aliases[cls], float(box.conf.item()), coords))
    return output


def paint(image, objects, pointed, point, dwell, required, cv2):
    palette={'door_open':(85,213,116),'door_closed':(222,166,84),'laptop':(69,211,236)}
    labels={'door_open':'PUERTA ABIERTA *','door_closed':'PUERTA CERRADA *','laptop':'LAPTOP'}
    for obj in objects:
        x1,y1,x2,y2=obj.box
        is_target = obj is pointed
        col = (95,255,125) if is_target else palette[obj.kind]
        cv2.rectangle(image,(x1,y1),(x2,y2),col,3 if is_target else 2)
        cv2.putText(image,f'{labels[obj.kind]} {obj.confidence:.2f}',(max(0,x1),max(26,y1-7)),
                    cv2.FONT_HERSHEY_SIMPLEX,.6,col,2,cv2.LINE_AA)
    if point:
        cv2.circle(image,point,12,(255,255,255),2)
        cv2.circle(image,point,5,(20,55,245),-1)
    H,W=image.shape[:2]
    overlay=image.copy()
    cv2.rectangle(overlay,(0,0),(W,79),(12,18,30),-1)
    cv2.addWeighted(overlay,.7,image,.3,0,image)
    obj_name=(pointed.kind.upper() if pointed else 'NINGUNO')
    cv2.putText(image,f'MIRADA: {obj_name}',(14,28),cv2.FONT_HERSHEY_SIMPLEX,.73,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(image,f'PERMANENCIA: {min(100,int(dwell/required*100))}%  |  puerta virtual, no fisica',
                (14,61),cv2.FONT_HERSHEY_SIMPLEX,.59,(185,223,255),2,cv2.LINE_AA)
    return image


def main():
    p=argparse.ArgumentParser(description='Neon real + puerta abierta/cerrada + laptop + dashboard')
    p.add_argument('--port',type=int,default=5000)
    p.add_argument('--conf',type=float,default=.40)
    p.add_argument('--imgsz',type=int,default=320)
    p.add_argument('--dwell',type=float,default=1.5)
    p.add_argument('--no-window',action='store_true',help='Desactivar la ventana de OpenCV (uso avanzado)')
    p.add_argument('--no-browser',action='store_true',help='No abrir el navegador automaticamente')
    p.add_argument('--door-weights',type=Path,default=ROOT/'modelos'/'doorway_seg_yolo26n.pt')
    p.add_argument('--laptop-weights',type=Path,default=ROOT/'modelos'/'yolov8n.pt')
    args=p.parse_args()
    if args.port<1024 or args.port>65535 or not 0<args.conf<1 or args.dwell<=0 or args.imgsz<128:
        p.error('Parametros fuera de rango')
    if not args.door_weights.is_file() or not args.laptop_weights.is_file():
        print('FALTAN PESOS DE YOLO: ejecuta DESCARGAR_MODELOS.cmd antes de iniciar.\n'
              'Los modelos se descargan de terceros, no son entrenamiento propio.')
        return 2

    import cv2
    from ultralytics import YOLO
    from pupil_labs.realtime_api.simple import discover_one_device

    print('Cargando detector de puertas y detector de laptops...')
    door_model=YOLO(str(args.door_weights))
    laptop_model=YOLO(str(args.laptop_weights))
    door_aliases={int(k): normalize_door_class(v) for k,v in door_model.names.items()}
    door_aliases={k:v for k,v in door_aliases.items() if v}
    laptop_aliases={int(k):'laptop' for k,v in laptop_model.names.items() if str(v).strip().lower()=='laptop'}
    if not {'door_open','door_closed'}.issubset(set(door_aliases.values())):
        print('Modelo de puerta incompatible. Debe tener clases door+doorway o door_open+door_closed.')
        return 2
    if not laptop_aliases:
        print('Modelo de laptop incompatible: clase laptop no encontrada.')
        return 2
    with STATE_LOCK:
        STATE.update({'dwell_required':args.dwell,'model_sources':{
            'doors':args.door_weights.name+' (modelo externo, estimacion)','laptops':args.laptop_weights.name+' (COCO)'}})
    add_event('Dos modelos cargados. Puertas: clasificacion aproximada; acciones: virtuales.')

    # Detectar puerto ocupado ANTES de iniciar hilos.
    import socket
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as sock:
        try: sock.bind(('127.0.0.1',args.port))
        except OSError:
            print(f'Puerto {args.port} ocupado. Cierra neon_domotica.py o usa --port 5050.')
            return 2

    thread=threading.Thread(target=lambda: app.run(host='127.0.0.1',port=args.port,
                                   threaded=True,debug=False,use_reloader=False),daemon=True)
    thread.start()
    url=f'http://127.0.0.1:{args.port}'
    print('DASHBOARD:',url,flush=True)
    if not args.no_browser:
        webbrowser.open(url)

    dwell_object=Dwell(args.dwell,grace=.30)
    dwell_color=Dwell(args.dwell,grace=.30)
    last_frame_at=None
    fps_ema=0.0
    device=None
    title='Neon Smart Home | YOLO + mirada (Q/ESC salir)'
    # No simular resultados: la conexion real es indispensable.
    try:
        while True:
            try:
                print('Buscando lentes Neon (con Neon Companion)...',flush=True)
                with STATE_LOCK:
                    STATE['message']='Buscando lentes Neon...'
                device=discover_one_device(max_search_duration_seconds=15)
                if device is None:
                    print('No encontrados. Reintentando en 3 segundos...')
                    time.sleep(3)
                    continue
                add_event('Lentes Neon conectados')
                with STATE_LOCK:
                    STATE['connected']=True
                    STATE['message']='Conectado. Esperando imagen y mirada...'
                dwell_object.reset(); dwell_color.reset()
                while True:
                    frame,gaze=device.receive_matched_scene_video_frame_and_gaze()
                    if frame is None or gaze is None:
                        continue
                    start_processing=time.perf_counter()
                    image=frame.bgr_pixels.copy()
                    h,w=image.shape[:2]
                    point=point_from_gaze(gaze,w,h)
                    now=time.monotonic()
                    if last_frame_at is not None:
                        instant_fps=1.0/max(now-last_frame_at,1e-3)
                        fps_ema=instant_fps if fps_ema==0 else (fps_ema*.80+instant_fps*.20)
                    last_frame_at=now
                    door_result=door_model.predict(image,conf=min(args.conf,.25),imgsz=args.imgsz,verbose=False)[0]
                    laptop_result=laptop_model.predict(image,classes=list(laptop_aliases),conf=args.conf,
                                                       imgsz=args.imgsz,verbose=False)[0]
                    objects=fetch_objects(door_result,door_aliases)+fetch_objects(laptop_result,laptop_aliases)
                    # Los modelos externos pueden confundir puertas, apertura y reflejos.
                    objects=[obj for obj in objects if obj.confidence >= args.conf]
                    target=find_looked_at(objects,point)
                    object_elapsed,object_fired=dwell_object.update(target,now)
                    color = select_color(image,point) if point is not None else None
                    # Prioridad al objeto para evitar dos acciones simultaneas.
                    color_target = Detection('color_'+color,1.0,(0,0,w,h)) if color and target is None else None
                    color_elapsed,color_fired = dwell_color.update(color_target,now)
                    active_elapsed=object_elapsed if target else color_elapsed
                    active_label=target.kind if target else (color or None)
                    with STATE_LOCK:
                        # El estado visual es independiente del estado real detectado.
                        simulated_open=STATE['door_virtual_open']
                        lamp_on=STATE['lamp_on']
                    if object_fired and target is not None:
                        if target.family=='door':
                            simulated_open=not simulated_open
                            action='abrir' if simulated_open else 'cerrar'
                            with STATE_LOCK:
                                STATE['door_virtual_open']=simulated_open
                                STATE['door_event_seq']+=1
                                STATE['last_door_action']='Puerta virtual ABIERTA' if simulated_open else 'Puerta virtual CERRADA'
                                STATE['selection_count']+=1
                                STATE['confirmation_seq']+=1
                                STATE['last_confirmation']='Puerta virtual: '+('abrir' if simulated_open else 'cerrar')
                                STATE['last_confirmation_kind']='door'
                            add_event(f'Puerta VIRTUAL: {action} tras {args.dwell:.1f}s mirando la puerta','door')
                            write_selection(action+'_virtual',target.kind,target.confidence,point,target.box,object_elapsed)
                        elif target.kind=='laptop':
                            with STATE_LOCK:
                                STATE['laptop_selections']+=1
                                STATE['selection_count']+=1
                                STATE['confirmation_seq']+=1
                                STATE['last_confirmation']='Laptop seleccionada'
                                STATE['last_confirmation_kind']='laptop'
                            add_event('Laptop seleccionada con la mirada (sin accion fisica)','laptop')
                            write_selection('laptop_seleccionada',target.kind,target.confidence,point,target.box,object_elapsed)
                    if color_fired and color_target is not None:
                        lamp_on = color=='Azul'
                        with STATE_LOCK:
                            STATE['lamp_on']=lamp_on
                            STATE['lamp_state']='ON' if lamp_on else 'OFF'
                            STATE['selection_count']+=1
                            STATE['confirmation_seq']+=1
                            STATE['last_confirmation']='Lámpara '+('encendida' if lamp_on else 'apagada')
                            STATE['last_confirmation_kind']='lamp'
                        add_event('Lampara VIRTUAL '+('encendida' if lamp_on else 'apagada'),'lamp')
                        write_selection('lampara_virtual_'+('on' if lamp_on else 'off'),color_target.kind,1.0,point,None,color_elapsed)
                    # Reportar estado percibido SOLO para la puerta a la que se mira.
                    observed = target if target and target.family=='door' else None
                    if observed is None:
                        detected_state='sin deteccion'
                        detected_conf=None
                    else:
                        detected_state='abierta (estimada)' if observed.kind=='door_open' else 'cerrada (estimada)'
                        detected_conf=round(observed.confidence,3)
                    with STATE_LOCK:
                        STATE.update({
                            'connected':True,'worn':bool(getattr(gaze,'worn',False)),
                            'gaze_x':point[0] if point else None,
                            'gaze_y':point[1] if point else None,
                            'timestamp':datetime.now().isoformat(timespec='milliseconds'),
                            'gaze_data':{'x_raw':str(getattr(gaze,'x','--')),
                                         'y_raw':str(getattr(gaze,'y','--')),
                                         'worn':bool(getattr(gaze,'worn',False)),
                                         'frame_size':[w,h]},
                            'detected_color':color or 'Ninguno',
                            'dwell_time':round(active_elapsed,2),
                            'progress':min(100,int(100*active_elapsed/args.dwell)),
                            'looked_at':active_label,
                            'detected_objects':[obj.to_json() for obj in objects],
                            'door_detected_state':detected_state,
                            'door_detection_confidence':detected_conf,
                            'message':('Mirando '+active_label if active_label else 'Neon activo: buscando seleccion'),
                            'selection_confirmed':bool((target and dwell_object.triggered) or
                                                       (color_target and dwell_color.triggered)),
                            'fps':round(fps_ema,1),
                            'processing_ms':round((time.perf_counter()-start_processing)*1000,1)
                        })
                    annotated=paint(image,objects,target,point,active_elapsed,args.dwell,cv2)
                    if not args.no_window:
                        cv2.imshow(title,annotated)
                        key=cv2.waitKey(1) & 0xff
                        if key in (ord('q'),27):
                            add_event('Aplicacion cerrada desde video')
                            return 0
            except KeyboardInterrupt:
                raise
            except Exception as error:
                print('ERROR DETALLADO (copiar todo si se repite):', flush=True)
                traceback.print_exc()
                add_event(f'Error de dispositivo o procesamiento: {error}','error')
                with STATE_LOCK:
                    STATE.update({'connected':False,'worn':False,'gaze_x':None,'gaze_y':None,
                                  'progress':0,'dwell_time':0.,'looked_at':None,'detected_color':'Ninguno',
                                  'detected_objects':[],'door_detected_state':'sin deteccion',
                                  'door_detection_confidence':None,'selection_confirmed':False,
                                  'fps':0.0,'processing_ms':0.0,'message':str(error)})
                dwell_object.reset(); dwell_color.reset()
                last_frame_at=None; fps_ema=0.0
                if not args.no_window:
                    cv2.destroyAllWindows()
                time.sleep(3)
            finally:
                if device is not None:
                    try: device.close()
                    except Exception: pass
                    device=None
    except KeyboardInterrupt:
        print('Programa detenido con Ctrl+C')
        return 0
    finally:
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__=='__main__':
    raise SystemExit(main())
