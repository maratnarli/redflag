from functools import lru_cache
from pathlib import Path

import cv2
from ultralytics import YOLO


# Жобаға қажетті адамдарды, көліктерді және жануарларды ғана анықтаймыз.
TARGET_LABELS = {
    "person", "bicycle", "car", "motorcycle", "bus",
    "truck", "dog", "horse", "sheep", "cow",
}


@lru_cache(maxsize=1)
def get_model():
    # Модель бірінші шақыруда жүктеледі, кейінгі кадрлар үшін қайта қолданылады.
    # Алғашқы іске қосуда модель салмақтарын жүктеу үшін интернет қажет.
    project_dir = Path(__file__).resolve().parents[1]
    return YOLO(str(project_dir / "yolov8n.pt"))


def enhance(frame):
    """LAB кескінінің жарықтық арнасын ғана жақсартып, BGR пішіміне қайтарады."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    improved = clahe.apply(lightness)

    merged = cv2.merge((improved, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def detect(frame, conf=0.25, use_clahe=False):
    """BGR кескінін қабылдап, анықталған нысандар тізімін қайтарады."""
    if frame is None or frame.size == 0:
        raise ValueError("Кескін бос. Файл жолын немесе бейнеден кадрдың оқылғанын тексеріңіз.")

    if not 0 <= conf <= 1:
        raise ValueError("conf мәні 0 мен 1 аралығында болуы керек.")

    image = enhance(frame) if use_clahe else frame
    model = get_model()

    class_ids = [
        class_id
        for class_id, name in model.names.items()
        if name in TARGET_LABELS
    ]

    result = model.predict(
        source=image,
        conf=conf,
        classes=class_ids,
        device="cpu",
        imgsz=640,
        iou=0.7,
        verbose=False,
    )[0]

    detections = []

    for box in result.boxes:
        class_id = int(box.cls.item())
        coordinates = tuple(
            int(value) for value in box.xyxy[0].tolist()
        )

        detections.append({
            "label": model.names[class_id],
            "conf": float(box.conf.item()),
            "box": coordinates,
        })

    return detections
