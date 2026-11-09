"""redflag — жолдағы кедергілер туралы ескерту прототипі (Streamlit).

Запуск из папки проекта:
    streamlit run app.py

Что делает страница:
  1. Пользователь загружает фото или видео дороги.
  2. Модель Динары (detect) находит объекты.
  3. Логика из src/alert.py решает, есть ли опасность.
  4. Показываем кадр с полосой, рамками и надписью DANGER / CLEAR.
"""

import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.alert import check_frame, draw_result
from src.detector import detect

# Порог выбран ролью 2 на data/tune (см. docs/model_handoff.md).
CONF = 0.25

STATUS_TEXT = {
    "threat": "🔴 қауіп",
    "in_lane": "🟡 жолақта, алыс",
    "outside": "⚪ жолақтан тыс",
    "hood": "⚫ капот",
}


def analyse(frame, use_clahe):
    """Один кадр: модель -> логика предупреждения."""
    detections = detect(frame, conf=CONF, use_clahe=use_clahe)
    return check_frame(detections, frame.shape)


def show_verdict(result):
    if result["danger"]:
        threats = [i["label"] for i in result["items"] if i["status"] == "threat"]
        st.error(f"⚠️ ҚАУІП! Жолақта: {', '.join(threats)}")
    else:
        st.success("✅ Жол бос")


def show_table(result):
    if not result["items"]:
        st.caption("Модель ешқандай нысан таппады.")
        return
    rows = [
        {
            "Нысан": i["label"],
            "Сенімділік": round(i["conf"], 2),
            "Күйі": STATUS_TEXT[i["status"]],
            "Себебі": i["reason"],
        }
        for i in result["items"]
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True)


# --- Страница ------------------------------------------------------------

st.set_page_config(page_title="redflag", page_icon="🚩", layout="wide")
st.title("🚩 redflag — жолдағы кедергілер туралы ескерту")
st.caption(
    "Оқу прототипі. Көлік жүргізу кезінде пайдалануға арналмаған. "
    f"Модель: YOLOv8n, сенімділік шегі {CONF}."
)

with st.sidebar:
    st.header("Баптаулар")
    use_clahe = st.checkbox(
        "CLAHE (түн/тұман үшін контраст)", value=False,
        help="Баптау кезінде CLAHE сапаны жақсартпады, сондықтан әдепкі бойынша өшірулі.",
    )
    st.markdown(
        "**Түстер:**\n\n"
        "🔴 қызыл — қауіп\n\n"
        "🟡 сары — жолақта, бірақ алыс\n\n"
        "⚪ сұр — жолақтан тыс немесе капот"
    )

photo_tab, video_tab = st.tabs(["📷 Фото", "🎬 Бейне"])

# --- Фото ---
with photo_tab:
    uploaded = st.file_uploader("Жол суретін жүктеңіз", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        data = np.frombuffer(uploaded.read(), dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            st.error("Суретті оқу мүмкін болмады.")
        else:
            with st.spinner("Модель жұмыс істеп жатыр..."):
                result = analyse(frame, use_clahe)
            show_verdict(result)
            st.image(draw_result(frame, result), channels="BGR")
            show_table(result)

# --- Видео ---
with video_tab:
    uploaded_video = st.file_uploader("Жол бейнесін жүктеңіз", type=["mp4", "mov", "avi", "mkv"])
    step_seconds = st.slider("Қанша секунд сайын кадр тексеру", 0.2, 2.0, 0.5, 0.1)

    if uploaded_video is not None and st.button("▶️ Талдауды бастау"):
        # OpenCV читает видео только с диска, поэтому сохраняем во временный файл
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(uploaded_video.read())
            video_path = tmp.name

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        step = max(1, int(round(fps * step_seconds)))

        verdict_box = st.empty()
        frame_box = st.empty()
        progress = st.progress(0.0)
        timeline = []

        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % step == 0:
                result = analyse(frame, use_clahe)
                seconds = index / fps
                threats = [i["label"] for i in result["items"] if i["status"] == "threat"]
                timeline.append({
                    "Уақыт, с": round(seconds, 1),
                    "Қауіп": 1 if result["danger"] else 0,
                    "Нысандар": ", ".join(threats),
                })
                with verdict_box.container():
                    show_verdict(result)
                frame_box.image(draw_result(frame, result), channels="BGR",
                                caption=f"{seconds:.1f} с")
                progress.progress(min(index / total, 1.0))
            index += 1

        cap.release()
        progress.progress(1.0)

        if timeline:
            df = pd.DataFrame(timeline)
            dangerous = df[df["Қауіп"] == 1]
            st.subheader("Нәтиже")
            col1, col2 = st.columns(2)
            col1.metric("Тексерілген кадрлар", len(df))
            col2.metric("Қауіпті кадрлар", len(dangerous))
            st.line_chart(df.set_index("Уақыт, с")["Қауіп"])
            if len(dangerous):
                st.dataframe(dangerous, hide_index=True)
        else:
            st.error("Бейнеден кадр оқылмады.")
