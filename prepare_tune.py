from pathlib import Path
import csv

import cv2


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "data/tune"
OUTPUT_DIR = INPUT_DIR / "frames"

# Әр бейненің ұзақтығы бойынша біркелкі орналасқан 5 кадр.
POSITIONS = [0.1, 0.3, 0.5, 0.7, 0.9]


def main():
    rows = []

    for condition in ["day", "night", "fog"]:
        videos = sorted((INPUT_DIR / condition).glob("*.mp4"))
        if not videos:
            raise RuntimeError(f"Бейне табылмады: {condition}")

        for video in videos:
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                raise RuntimeError(f"Бейнені ашу мүмкін болмады: {video}")

            try:
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                if count < 5 or fps <= 0:
                    raise RuntimeError(
                        f"Кадр саны немесе FPS жарамсыз: {video}"
                    )

                folder = OUTPUT_DIR / condition
                folder.mkdir(parents=True, exist_ok=True)

                for number, position in enumerate(POSITIONS, start=1):
                    frame_index = round((count - 1) * position)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ok, frame = cap.read()

                    if not ok:
                        raise RuntimeError(
                            f"Кадр оқылмады: {video}, {frame_index}"
                        )

                    output = folder / f"{video.stem}_{number:02d}.jpg"
                    if not cv2.imwrite(str(output), frame):
                        raise RuntimeError(f"Сақтау қатесі: {output}")

                    # Әр кескіннің қай бейнеден алынғанын тіркейміз.
                    rows.append({
                        "image": output.relative_to(ROOT).as_posix(),
                        "source_video": video.relative_to(ROOT).as_posix(),
                        "condition": condition,
                        "frame_index": frame_index,
                        "time_seconds": round(frame_index / fps, 3),
                    })
                    print(f"Сақталды: {output.name}")
            finally:
                cap.release()

    manifest = OUTPUT_DIR / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "image", "source_video", "condition",
            "frame_index", "time_seconds",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nДайын! Барлығы {len(rows)} кескін сақталды.")
    print(f"Бума: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
    