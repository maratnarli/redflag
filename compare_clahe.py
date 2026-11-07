from pathlib import Path
import csv

import cv2

from src.detector import detect, enhance


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data/tune/frames"
OUTPUT = ROOT / "results/clahe_comparison"
CONF = 0.25


def main():
    images = sorted(INPUT.rglob("*.jpg"))
    if not images:
        raise RuntimeError("Баптау кескіндері табылмады.")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = []
    predictions = []

    for image_path in images:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Кескін оқылмады: {image_path}")

        relative = image_path.relative_to(INPUT)
        print(f"Өңделуде: {relative}", flush=True)

        for mode, use_clahe in [("plain", False), ("clahe", True)]:
            # Екі нұсқада да бірдей сенімділік шегі қолданылады.
            detections = detect(
                frame, conf=CONF, use_clahe=use_clahe
            )

            # CLAHE нәтижесін жақсартылған кескінге саламыз.
            preview = enhance(frame) if use_clahe else frame.copy()

            summary.append({
                "image": relative.as_posix(),
                "mode": mode,
                "threshold": CONF,
                "count": len(detections),
            })

            for item in detections:
                x1, y1, x2, y2 = item["box"]

                predictions.append({
                    "image": relative.as_posix(),
                    "mode": mode,
                    "threshold": CONF,
                    "label": item["label"],
                    "confidence": item["conf"],
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2,
                })

                cv2.rectangle(
                    preview, (x1, y1), (x2, y2),
                    (0, 255, 0), 2,
                )
                text = f'{item["label"]} {item["conf"]:.2f}'
                cv2.putText(
                    preview, text, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2,
                )

            destination = OUTPUT / mode / relative
            destination.parent.mkdir(parents=True, exist_ok=True)

            if not cv2.imwrite(str(destination), preview):
                raise RuntimeError(f"Сақтау қатесі: {destination}")

    tables = [
        ("summary.csv", summary, [
            "image", "mode", "threshold", "count",
        ]),
        ("predictions.csv", predictions, [
            "image", "mode", "threshold", "label", "confidence",
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

    print(f"\nДайын! {len(images)} кескін, 2 нұсқа тексерілді.")
    print(f"Нәтижелер: {OUTPUT}")


if __name__ == "__main__":
    main()
