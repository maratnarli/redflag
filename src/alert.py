"""Ескерту логикасы: модель тапқан нысандар бойынша қауіп бар-жоғын шешеді.

Логика предупреждения. Модель (src/detector.py) только находит объекты,
а здесь решается, опасно ли это для водителя.

Три шага:
1. Убираем капот своей машины (модель видит его как "car" внизу кадра).
2. Оставляем только объекты на своей полосе (трапеция по центру кадра).
3. Решаем, близко ли объект:
   - человек или животное на полосе -> опасно почти сразу;
   - машина/грузовик/автобус на полосе -> опасно, только если она близко
     (рамка высокая, то есть объект крупный в кадре).

Все числа ниже подобраны на кадрах из data/tune/ и после подбора
не меняются, чтобы честно проверить их на data/test/.
"""

import cv2
import numpy as np


# --- Настройки (подобраны на data/tune/) ---------------------------------

# Своя полоса: трапеция в долях кадра (x от 0 до 1 слева направо,
# y от 0 до 1 сверху вниз). Внизу широкая, к горизонту сужается.
LANE_ZONE = [
    (0.20, 1.00),  # низ слева
    (0.45, 0.45),  # верх слева
    (0.55, 0.45),  # верх справа
    (0.80, 1.00),  # низ справа
]

# Капот: рамка касается нижнего края кадра и очень широкая.
HOOD_BOTTOM = 0.95   # нижний край рамки ниже 95% высоты кадра
HOOD_WIDTH = 0.50    # и рамка шире половины кадра

# Люди и животные: они не держат дистанцию и могут выйти на дорогу.
VULNERABLE = {"person", "bicycle", "dog", "horse", "sheep", "cow"}
# Транспорт: впереди идущая машина на нормальной дистанции не опасна.
VEHICLES = {"car", "truck", "bus", "motorcycle"}

# Насколько крупным (по высоте рамки, в доле высоты кадра)
# должен быть объект, чтобы считать его близким.
MIN_HEIGHT_VULNERABLE = 0.03
MIN_HEIGHT_VEHICLE = 0.20


# --- Вспомогательные функции ---------------------------------------------

def lane_polygon(frame_shape):
    """Трапеция полосы в пикселях для кадра размера frame_shape."""
    height, width = frame_shape[:2]
    points = [(int(x * width), int(y * height)) for x, y in LANE_ZONE]
    return np.array(points, dtype=np.int32)


def is_hood(box, frame_shape):
    """True, если рамка похожа на капот своей машины."""
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = box
    touches_bottom = y2 >= HOOD_BOTTOM * height
    very_wide = (x2 - x1) >= HOOD_WIDTH * width
    return touches_bottom and very_wide


def in_lane(box, frame_shape):
    """True, если объект стоит на своей полосе.

    Берём точку посередине нижнего края рамки: там объект касается дороги.
    """
    x1, y1, x2, y2 = box
    foot = (float((x1 + x2) / 2), float(y2))
    return cv2.pointPolygonTest(lane_polygon(frame_shape), foot, False) >= 0


# --- Главные функции -----------------------------------------------------

def check_frame(detections, frame_shape):
    """Разбирает каждую детекцию и объясняет, почему она опасна или нет.

    Возвращает словарь:
      danger  -- True / False
      items   -- список детекций, у каждой добавлены поля
                 "status" ("threat", "in_lane", "outside", "hood")
                 и "reason" (короткое объяснение по-казахски)
    """
    height = frame_shape[0]
    items = []

    for det in detections:
        item = dict(det)
        label = det["label"]
        x1, y1, x2, y2 = det["box"]
        box_height = (y2 - y1) / height

        if is_hood(det["box"], frame_shape):
            item["status"], item["reason"] = "hood", "өз көлігінің капоты"
        elif not in_lane(det["box"], frame_shape):
            item["status"], item["reason"] = "outside", "жолақтан тыс"
        elif label in VULNERABLE and box_height >= MIN_HEIGHT_VULNERABLE:
            item["status"], item["reason"] = "threat", "жолақта адам/жануар"
        elif label in VEHICLES and box_height >= MIN_HEIGHT_VEHICLE:
            item["status"], item["reason"] = "threat", "жолақта жақын көлік"
        else:
            item["status"], item["reason"] = "in_lane", "жолақта, бірақ алыс"

        items.append(item)

    danger = any(item["status"] == "threat" for item in items)
    return {"danger": danger, "items": items}


def is_danger(detections, frame_shape):
    """Келісілген интерфейс: True — қауіп бар, False — қауіп жоқ."""
    return check_frame(detections, frame_shape)["danger"]


# --- Рисование для приложения --------------------------------------------

COLORS = {  # BGR
    "threat": (0, 0, 255),       # красный
    "in_lane": (0, 200, 255),    # жёлтый
    "outside": (180, 180, 180),  # серый
    "hood": (120, 120, 120),     # тёмно-серый
}


def draw_result(frame, result):
    """Рисует на копии кадра полосу, рамки и итоговую надпись."""
    image = frame.copy()
    height, width = image.shape[:2]
    thick = max(2, width // 600)
    font_scale = max(0.6, min(width, height) / 1200)

    # Полупрозрачная полоса
    overlay = image.copy()
    zone_color = (0, 0, 255) if result["danger"] else (0, 200, 0)
    cv2.fillPoly(overlay, [lane_polygon(image.shape)], zone_color)
    image = cv2.addWeighted(overlay, 0.18, image, 0.82, 0)

    # Рамки
    for item in result["items"]:
        x1, y1, x2, y2 = item["box"]
        color = COLORS[item["status"]]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thick)
        text = f'{item["label"]} {item["conf"]:.2f}'
        cv2.putText(image, text, (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thick)

    # Итоговая надпись сверху
    banner = "DANGER" if result["danger"] else "CLEAR"
    color = (0, 0, 255) if result["danger"] else (0, 160, 0)
    cv2.rectangle(image, (0, 0), (width, int(height * 0.08)), color, -1)
    cv2.putText(image, banner, (int(width * 0.02), int(height * 0.06)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale * 2, (255, 255, 255),
                thick * 2)
    return image
