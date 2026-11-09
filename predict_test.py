"""Прогон модели + логики предупреждения по data/test/ для роли 4 (оценка).

Запуск из папки проекта:
    python predict_test.py

Результат: results/test_predictions.csv — для каждой картинки и каждого
варианта (CLAHE выкл/вкл) ответ is_danger и найденные угрозы.
Метрики (precision, recall) считает роль 4, сравнивая этот файл
со своей ручной разметкой.
"""

import csv
from pathlib import Path

import cv2

from src.alert import check_frame
from src.detector import detect

CONF = 0.25
TEST_DIR = Path("data/test")
OUT = Path("results/test_predictions.csv")


def main():
    images = sorted(p for p in TEST_DIR.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    OUT.parent.mkdir(exist_ok=True)

    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "condition", "use_clahe", "danger", "threats"])
        for path in images:
            frame = cv2.imread(str(path))
            if frame is None:
                print("Оқылмады:", path)
                continue
            for use_clahe in (False, True):
                result = check_frame(detect(frame, conf=CONF, use_clahe=use_clahe), frame.shape)
                threats = [i["label"] for i in result["items"] if i["status"] == "threat"]
                writer.writerow([path.name, path.parent.name, use_clahe,
                                 int(result["danger"]), " ".join(threats)])
            print(path.name, "дайын")

    print("Сақталды:", OUT)


if __name__ == "__main__":
    main()
