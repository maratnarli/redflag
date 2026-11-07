from pathlib import Path

import cv2

from src.detector import detect


# Барлық жолдар жобаның негізгі бумасынан басталады.
ROOT = Path(__file__).resolve().parent
video_path = ROOT / "data/tune/day/day_03_ugroza.mp4"
output_dir = ROOT / "results/smoke_check"
output_dir.mkdir(parents=True, exist_ok=True)

# Бейненің ортасынан бір кадрды оқимыз.
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise RuntimeError(f"Бейнені ашу мүмкін болмады: {video_path}")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if frame_count > 0:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)

ok, frame = cap.read()
cap.release()

if not ok:
    raise RuntimeError("Кадрды оқу мүмкін болмады. Бейненің дұрыс ойнатылатынын тексеріңіз.")

if not cv2.imwrite(str(output_dir / "original.jpg"), frame):
    raise RuntimeError("Бастапқы кескінді сақтау мүмкін болмады.")

# Бір кадрды жақсартусыз және CLAHE арқылы жақсартып анықтаймыз.
for use_clahe, name in [(False, "plain"), (True, "clahe")]:
    detections = detect(frame, conf=0.4, use_clahe=use_clahe)
    preview = frame.copy()

    print(f"\n{name}: {len(detections)} нысан анықталды")

    for item in detections:
        print(item)
        x1, y1, x2, y2 = item["box"]

        cv2.rectangle(
            preview, (x1, y1), (x2, y2), (0, 255, 0), 2
        )
        text = f'{item["label"]} {item["conf"]:.2f}'
        cv2.putText(
            preview, text, (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    # Жақтаулардың орнын салыстыру үшін екі нәтиже де бастапқы кескінге салынады.
    output_path = output_dir / f"{name}.jpg"
    if not cv2.imwrite(str(output_path), preview):
        raise RuntimeError(f"Нәтижені сақтау мүмкін болмады: {output_path}")

print(f"\nДайын! Кескіндерді көру үшін осы буманы ашыңыз:\n{output_dir}")
