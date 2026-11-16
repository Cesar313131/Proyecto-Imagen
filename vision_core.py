"""Logica pura de seleccion ocular. No necesita lentes para importarse.

La demo visual no controla una puerta fisica. No altera modelos entrenados.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Detection:
    kind: str  # door_open, door_closed, laptop
    confidence: float
    box: tuple[int, int, int, int]

    @property
    def family(self) -> str:
        return 'door' if self.kind in ('door_open', 'door_closed') else self.kind

    def to_json(self):
        return {'kind': self.kind, 'confidence': round(self.confidence, 3), 'box': list(self.box)}


def point_from_gaze(gaze, image_width: int, image_height: int):
    """SDK Neon emite gaze x/y en pixeles de la camara de escena.

    No normalizar ni fijar al borde coordenadas incorrectas.
    """
    if gaze is None or not bool(getattr(gaze, 'worn', False)):
        return None
    try:
        x = float(gaze.x)
        y = float(gaze.y)
    except (TypeError, ValueError, AttributeError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    if x < 0 or y < 0 or x >= image_width or y >= image_height:
        return None
    return int(x), int(y)


def intersection_over_union(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    aa = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    bb = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    area = aa + bb - inter
    return inter / area if area else 0.0


def find_looked_at(detections: list[Detection], point):
    if point is None:
        return None
    x, y = point
    hits = [d for d in detections if d.box[0] <= x <= d.box[2] and d.box[1] <= y <= d.box[3]]
    # Si hay cajas solapadas, preferir la mas especifica (area menor).
    if not hits:
        return None
    return min(hits, key=lambda d: ((d.box[2]-d.box[0])*(d.box[3]-d.box[1]), -d.confidence))


class Dwell:
    """Dispara una vez al permanecer sobre un mismo objeto.

    Rearme solo cuando se deja de mirar el objeto durante > grace segundos.
    Tolera movimiento pequeno de la caja entre fotogramas (IoU).
    """
    def __init__(self, seconds: float = 1.5, grace: float = .30, min_iou: float = .20):
        if seconds <= 0: raise ValueError('seconds debe ser positivo')
        self.seconds = seconds
        self.grace = grace
        self.min_iou = min_iou
        self.reset()

    def reset(self):
        self.active: Detection | None = None
        self.started: float | None = None
        self.last_seen: float | None = None
        self.triggered = False

    def update(self, found: Detection | None, now: float) -> tuple[float, bool]:
        if found is None:
            if self.last_seen is None or now-self.last_seen > self.grace:
                self.reset()
                return 0.0, False
            # No sumar tiempo mientras la mirada no esta sobre el objeto.
            return max(0.0, self.last_seen-self.started), False
        same = (
            self.active is not None and self.last_seen is not None
            and found.family == self.active.family
            and intersection_over_union(found.box, self.active.box) >= self.min_iou
        )
        if not same:
            self.started = now
            self.triggered = False
        self.active = found
        self.last_seen = now
        elapsed = max(0.0, now-self.started)
        fired = elapsed >= self.seconds and not self.triggered
        if fired:
            self.triggered = True
        return elapsed, fired


def select_color(frame, point):
    """Conserva el control de colores original, solo si se ve el color en la mirada.

    Usa un area minima y limpieza de mascara para limitar el ruido.
    """
    if point is None:
        return None
    import cv2
    import numpy as np
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    opts = {
        'Azul': cv2.inRange(hsv,np.array([100,100,60]),np.array([130,255,255])),
        'Rojo': cv2.bitwise_or(cv2.inRange(hsv,np.array([0,100,70]),np.array([10,255,255])),
                               cv2.inRange(hsv,np.array([170,100,70]),np.array([180,255,255])))
    }
    kernel = np.ones((5,5),np.uint8)
    for label, mask in opts.items():
        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < frame.shape[0]*frame.shape[1]*.002:
                continue
            if cv2.pointPolygonTest(contour,(float(point[0]),float(point[1])),False) >= 0:
                return label
    return None


def normalize_door_class(name: str) -> str | None:
    k = str(name).strip().lower().replace(' ','_').replace('-','_')
    if k in ('doorway','door_open','open_door','puerta_abierta','open'):
        return 'door_open'
    if k in ('door','door_closed','closed_door','puerta_cerrada','closed'):
        return 'door_closed'
    return None
