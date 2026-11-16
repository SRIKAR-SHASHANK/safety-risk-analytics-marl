"""Optional adapter for running the pipeline on *real* CCTV footage with a deep detector.

The rest of the system only consumes :class:`~safetyvision.vision.detector.Detection`
objects plus per-person ``(helmet, vest)`` booleans, so any detector that can
produce those plugs straight into the tracker, risk engine and MARL stages.

Example (requires ``pip install ultralytics opencv-python`` and a PPE model,
e.g. one fine-tuned on the public "Construction Site Safety" / SH17 datasets)::

    from safetyvision.vision.yolo_adapter import YoloPPEDetector
    det = YoloPPEDetector("ppe_yolov8n.pt")
    for frame in video_frames:
        detections, ppe = det(frame)
        tracks = tracker.update(detections, ppe)
        risk_engine.update(frame_idx, tracks)
"""
from __future__ import annotations

from .detector import Detection


class YoloPPEDetector:
    PERSON = {"person", "worker"}
    VEHICLE = {"forklift", "vehicle", "machinery", "truck"}
    NO_HELMET = {"no-hardhat", "no_helmet", "head"}
    NO_VEST = {"no-safety vest", "no_vest"}

    def __init__(self, weights: str, conf: float = 0.35):
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("pip install ultralytics to use YoloPPEDetector") from e
        self.model = YOLO(weights)
        self.conf = conf

    def __call__(self, frame):  # pragma: no cover - needs model weights
        res = self.model(frame, conf=self.conf, verbose=False)[0]
        names = res.names
        people, vehicles, viol = [], [], []
        for box, c in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist()):
            x0, y0, x1, y1 = box
            label = names[int(c)].lower()
            cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
            d = Detection("person" if label in self.PERSON else "forklift", cy, cx,
                          int((x1 - x0) * (y1 - y0)), (int(y0), int(x0), int(y1), int(x1)))
            if label in self.PERSON:
                people.append(d)
            elif label in self.VEHICLE:
                vehicles.append(d)
            elif label in self.NO_HELMET | self.NO_VEST:
                viol.append((label, cy, cx))
        ppe = []
        for p in people:
            y0, x0, y1, x1 = p.bbox
            inside = [lab for lab, cy, cx in viol if y0 <= cy <= y1 and x0 <= cx <= x1]
            ppe.append((not any(l in self.NO_HELMET for l in inside),
                        not any(l in self.NO_VEST for l in inside)))
        return people + vehicles, ppe + [(True, True)] * len(vehicles)
