from pathlib import Path
import csv

import cv2

from src.detector import detect


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data/tune/frames"
OUTPUT = ROOT / "results/threshold_comparison"
THRESHOLDS = [0.25, 0.4, 0.5]


def main():
    images = sorted(INPUT.rglob("*.jpg"))
    if not images:
        raise RuntimeError("Баптауға арналған кескіндер табылмады.")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = []
    predictions = []

    for image_path in images:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Кескін оқылмады: {image_path}")

        relative = image_path.relative_to(INPUT)
        print(f"Өңделуде: {relative}", flush=True)

        for threshold in THRESHOLDS:
            # Алдымен тек сенімділік шегін салыстырамыз.
            detections = detect(
                frame, conf=threshold, use_clahe=False
            )
            preview = frame.copy()

            summary.append({
                "image": relative.as_posix(),
                "threshold": threshold,
                "count": len(detections),
            })

            for item in detections:
                x1, y1, x2, y2 = item["box"]
                predictions.append({
                    "image": relative.as_posix(),
                    "threshold": threshold,
                    "label": item["label"],
                    "confidence": item["conf"],
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2,
                })

                cv2.rectangle(
                    preview, (x1, y1), (x2, y2), (0, 255, 0), 2
                )
                label = f'{item["label"]} {item["conf"]:.2f}'
                cv2.putText(
                    preview, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2,
                )

            destination = OUTPUT / str(threshold) / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(destination), preview):
                raise RuntimeError(f"Сақтау қатесі: {destination}")

    # Нысан табылмаған кескіндер де summary.csv ішінде тіркеледі.
    tables = [
        ("summary.csv", summary, ["image", "threshold", "count"]),
        ("predictions.csv", predictions, [
            "image", "threshold", "label", "confidence",
            "x1", "y1", "x2", "y2",
        ]),
    ]

    for filename, rows, fields in tables:
        with (OUTPUT / filename).open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nДайын! {len(images)} кескін, 3 шек тексерілді.")
    print(f"Нәтижелер: {OUTPUT}")


if __name__ == "__main__":
    main()
