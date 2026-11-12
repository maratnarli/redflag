"""Екі модельді салыстыру: YOLOv8n (біздің негізгі модель) және YOLOv8s.

Сравнение двух моделей на ТЕХ ЖЕ тестовых кадрах и с ТОЙ ЖЕ логикой
предупреждения. Меняется только модель, всё остальное как в основном тесте:
  - кадры: data/test/, разметка data/test_labels.csv (37 размеченных кадров,
    8 «неясных» ночных кадров пропускаются, как и в results/participant_4/);
  - conf=0.25, imgsz=640, iou=0.7, device="cpu", CLAHE выключен;
  - решение «опасно / нет» принимает src/alert.py -> check_frame().

Метрики считаются на уровне кадра (ескерту бар/жоқ vs danger_gt):
TP, FP, FN, TN, precision, recall, F1, accuracy и среднее время
обработки одного кадра на CPU.

Запуск из папки проекта:
    python compare_models.py

Первый запуск скачает yolov8s.pt (~22 МБ), нужен интернет.

Результаты: results/model_comparison/
    metrics.csv      — метрики по моделям и условиям
    predictions.csv  — ответ каждой модели для каждого кадра
    README.md        — короткая сводная таблица
"""

import csv
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.alert import check_frame
from src.detector import TARGET_LABELS

ROOT = Path(__file__).resolve().parent
TEST_DIR = ROOT / "data/test"
LABELS = ROOT / "data/test_labels.csv"
OUTPUT = ROOT / "results/model_comparison"

MODELS = ["yolov8n.pt", "yolov8s.pt"]
CONF = 0.25
CONDITIONS = ["day", "night", "fog", "all"]


def load_labels():
    """Тек нақты белгіленген кадрлар (danger_gt = 0 немесе 1)."""
    with LABELS.open(encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["danger_gt"].strip() in {"0", "1"}]
    return [(r["image"], r["condition"], int(r["danger_gt"])) for r in rows]


def detect_with(model, frame):
    """src/detector.detect() сияқты, тек модельді сырттан береміз."""
    class_ids = [cid for cid, name in model.names.items() if name in TARGET_LABELS]
    result = model.predict(
        source=frame, conf=CONF, classes=class_ids,
        device="cpu", imgsz=640, iou=0.7, verbose=False,
    )[0]
    return [
        {
            "label": model.names[int(box.cls.item())],
            "conf": float(box.conf.item()),
            "box": tuple(int(v) for v in box.xyxy[0].tolist()),
        }
        for box in result.boxes
    ]


def safe_div(a, b):
    return a / b if b else None


def metrics(rows):
    tp = sum(1 for r in rows if r["pred"] == 1 and r["gt"] == 1)
    fp = sum(1 for r in rows if r["pred"] == 1 and r["gt"] == 0)
    fn = sum(1 for r in rows if r["pred"] == 0 and r["gt"] == 1)
    tn = sum(1 for r in rows if r["pred"] == 0 and r["gt"] == 0)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = (safe_div(2 * precision * recall, precision + recall)
          if precision is not None and recall is not None else None)
    return {
        "n": len(rows), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": safe_div(tp + tn, len(rows)),
        "ms_per_frame": safe_div(sum(r["ms"] for r in rows), len(rows)),
    }


def fmt(value, percent=True):
    if value is None:
        return "есептелмейді"
    return f"{value * 100:.1f}%" if percent else f"{value:.0f}"


def main():
    labels = load_labels()
    if not labels:
        raise RuntimeError("data/test_labels.csv ішінде белгіленген кадр жоқ.")

    frames = {}
    for name, condition, _ in labels:
        path = TEST_DIR / condition / name
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Кескін оқылмады: {path}")
        frames[name] = frame

    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions, metric_rows = [], []

    for model_file in MODELS:
        print(f"\n=== {model_file} ===", flush=True)
        model = YOLO(str(ROOT / model_file) if (ROOT / model_file).exists() else model_file)

        # Бірінші кадр әрқашан баяу, сондықтан уақытты өлшемес бұрын «қыздырамыз».
        detect_with(model, next(iter(frames.values())))

        rows = []
        for name, condition, gt in labels:
            frame = frames[name]
            start = time.perf_counter()
            detections = detect_with(model, frame)
            ms = (time.perf_counter() - start) * 1000
            result = check_frame(detections, frame.shape)
            threats = [i["label"] for i in result["items"] if i["status"] == "threat"]
            row = {"model": model_file, "image": name, "condition": condition,
                   "gt": gt, "pred": int(result["danger"]),
                   "threats": " ".join(threats), "detections": len(detections),
                   "ms": ms}
            rows.append(row)
            print(f"  {condition}/{name}: pred={row['pred']} gt={gt} ({ms:.0f} ms)", flush=True)
        predictions += rows

        for condition in CONDITIONS:
            subset = rows if condition == "all" else [r for r in rows if r["condition"] == condition]
            metric_rows.append({"model": model_file, "condition": condition, **metrics(subset)})

    # --- CSV ------------------------------------------------------------
    with (OUTPUT / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["model", "image", "condition", "gt", "pred", "threats", "detections", "ms"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in predictions:
            writer.writerow({**r, "ms": round(r["ms"], 1)})

    with (OUTPUT / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["model", "condition", "n", "TP", "FP", "FN", "TN",
                  "precision", "recall", "f1", "accuracy", "ms_per_frame"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metric_rows)

    # --- README ---------------------------------------------------------
    lines = [
        "# Модельдерді салыстыру: YOLOv8n және YOLOv8s",
        "",
        "Екі модель бірдей 37 белгіленген тест кадрында, бірдей ескерту логикасымен "
        "(`src/alert.py`) тексерілді. Параметрлер: `conf=0.25`, `imgsz=640`, `iou=0.7`, "
        "CPU, CLAHE өшірулі. Метрикалар кадр деңгейінде есептелді. "
        "Скрипт: `compare_models.py`.",
        "",
        "| Модель | Жағдай | Кадр | TP / FP / FN / TN | Precision | Recall | F1 | Бір кадр, мс |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for m in metric_rows:
        lines.append(
            f"| {m['model'].replace('.pt', '')} | {m['condition']} | {m['n']} | "
            f"{m['TP']} / {m['FP']} / {m['FN']} / {m['TN']} | {fmt(m['precision'])} | "
            f"{fmt(m['recall'])} | {fmt(m['f1'])} | {fmt(m['ms_per_frame'], percent=False)} |"
        )
    lines += [
        "",
        "Түн: белгіленген 7 түнгі кадрдың бәрі қауіпсіз, сондықтан precision мен recall есептелмейді.",
        "Уақыт осы компьютердің CPU-ына байланысты, оны тек екі модельді өзара салыстыру үшін қолданыңыз.",
    ]
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- Қысқаша қорытынды экранға ---------------------------------------
    print("\nБарлық кадрлар бойынша:")
    for m in metric_rows:
        if m["condition"] == "all":
            print(f"  {m['model']:<11} precision={fmt(m['precision'])}  recall={fmt(m['recall'])}  "
                  f"F1={fmt(m['f1'])}  {fmt(m['ms_per_frame'], percent=False)} мс/кадр")

    base = next(m for m in metric_rows if m["model"] == "yolov8n.pt" and m["condition"] == "all")
    if (base["TP"], base["FP"], base["FN"], base["TN"]) != (8, 6, 2, 21):
        print("\nЕскерту: yolov8n нәтижесі results/participant_4/metrics.csv-тен "
              "(8/6/2/21) өзгеше шықты. Параметрлерді тексеріңіз.")
    else:
        print("\nyolov8n нәтижесі негізгі тестпен сәйкес (8/6/2/21) — салыстыру дұрыс.")

    print(f"\nНәтижелер: {OUTPUT}")


if __name__ == "__main__":
    main()
