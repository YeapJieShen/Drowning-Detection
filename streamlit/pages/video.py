import streamlit as st
from pathlib import Path
import cv2
import tempfile
import os
import io
import pandas as pd
from services import detect_drowning

PAGE_STATES = [
    "video_uploaded_file", "video_uploaded",
    "video_processed_file", "video_processed",
    "video_drowning_log", "video_drowning_log_no_roi",
]


def init():
    st.session_state["cleanup_function"] = cleanup

    if "video_uploaded" not in st.session_state:
        st.session_state["video_uploaded"] = False

    if "video_processed" not in st.session_state:
        st.session_state["video_processed"] = False

    if "video_drowning_log" not in st.session_state:
        st.session_state["video_drowning_log"] = []

    if "video_drowning_log_no_roi" not in st.session_state:
        st.session_state["video_drowning_log_no_roi"] = []


def cleanup():
    for state in PAGE_STATES:
        if state in st.session_state:
            del st.session_state[state]


def log_drowning_info(log):
    nth_second, obj_id, drowning_prob, roi = log
    drowning_prob = float(drowning_prob)

    log_entry = {
        "nth_second": nth_second,
        "object_id": obj_id,
        "drowning_probability": drowning_prob,
        "roi": roi
    }

    log_entry_no_roi = [nth_second, obj_id, drowning_prob]

    st.session_state["video_drowning_log"].append(log_entry)
    st.session_state["video_drowning_log_no_roi"].append(log_entry_no_roi)


def upload_video_onchange():
    st.session_state["video_uploaded"] = True if st.session_state["video_uploaded_file"] is not None else False
    st.session_state["video_processed"] = False


def process_video():
    from ultralytics import YOLO
    from src.classify import TorchClassifier

    uploaded_file = st.session_state["video_uploaded_file"]
    suffix = Path(uploaded_file.name).suffix

    assets_path = Path(__file__).parent.parent / "assets"
    models_path = assets_path / "models"

    yolo_path = models_path / "detection" / "YOLO" / "jrom_yolo11n.pt"
    detection_model= YOLO(yolo_path)

    classification_path = models_path / "classification" / "CNN" / "test_new.pt"
    classification_model = TorchClassifier(
        model="CNNClassifier", model_path=classification_path)

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_infile:
        temp_infile.write(uploaded_file.read())
        temp_infile.flush()

    cap = cv2.VideoCapture(temp_infile.name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if suffix in ['.mp4', '.mpeg4']:
        codec = 'H264'  # H.264 video codec for MP4 files
    elif suffix == '.avi':
        codec = 'XVID'  # Common codec for AVI files
    elif suffix == '.mov':
        codec = 'avc1'  # Common codec for MOV files (H.264)
    else:
        raise ValueError(f"Unsupported video extension: {suffix}")

    temp_outfile = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temp_outfile.close()
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(temp_outfile.name, fourcc, fps, (width, height))

    if not cap.isOpened():
        st.error("Error opening video file")
        return

    progress_bar = st.progress(0, text="Beginning video processing...")
    processed_frames = 0

    obj_confs_info = {}
    class_confs_info = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, _, drowning_info_logs = detect_drowning(
            detection_model,
            classification_model,
            obj_confs_info,
            class_confs_info,
            frame,
            st.session_state["video_sensitivity"]
        )

        out.write(annotated_frame)

        for log in drowning_info_logs:
            log = (processed_frames / fps, log[1], log[2], log[3])
            log_drowning_info(log)

        processed_frames += 1
        proportion_done = min((processed_frames + 1) / total_frames, 1)
        progress_bar.progress(proportion_done, text=f"Processing video... {proportion_done * 100: .2f}%")

    cap.release()
    out.release()

    st.session_state["video_processed"] = True
    st.session_state["video_processed_file"] = io.BytesIO(
        open(temp_outfile.name, 'rb').read())

    progress_bar.empty()

    os.remove(temp_infile.name)
    os.remove(temp_outfile.name)


def video_upload_tab():
    st.subheader("📷 Video Feed")

    with st.expander("Upload Video File", expanded=not st.session_state["video_uploaded"]):
        st.file_uploader("", type=[
                         "mp4", "avi", "mov"], key="video_uploaded_file", on_change=upload_video_onchange)

    with st.expander("Detection Settings", expanded=False):
        st.slider(
            "Sensitivity", min_value=0.01, max_value=1.0, value=0.5, step=0.01, key="video_sensitivity")

    st.button("Process Video", key="video_process_video",
              disabled=not st.session_state["video_uploaded"], use_container_width=True)

    if st.session_state["video_uploaded"]:
        with st.expander("Original Video", expanded=not st.session_state["video_process_video"] and not st.session_state["video_processed"] and st.session_state["video_uploaded"]):
            st.video(st.session_state["video_uploaded_file"])

    if st.session_state["video_process_video"]:
        process_video()

    if st.session_state["video_processed"]:
        st.success("Video processed successfully!")

        with st.expander("Processed Video", expanded=True):
            cols = st.columns(2)
            with cols[0]:
                uploaded_file = st.session_state["video_uploaded_file"]
                st.download_button(
                    "Download Processed Video",
                    data=st.session_state["video_processed_file"],
                    file_name=f"processed_{uploaded_file.name}." + uploaded_file.type.split('/')[1],
                    mime=uploaded_file.type,
                    use_container_width=True)
            with cols[1]:
                df_log = pd.DataFrame(
                    st.session_state["video_drowning_log_no_roi"],
                    columns=["timestamp", "object_id", "drowning_probability"]
                )
                
                st.download_button(
                    "Download Log",
                    key="video_upload_download_log",
                    data=df_log.to_csv().encode("utf-8"),
                    file_name=f"drowning_log_{Path(st.session_state['video_uploaded_file'].name).stem}.csv",
                    mime="text/csv",
                    disabled=not st.session_state["video_drowning_log_no_roi"],
                    use_container_width=True
                )

            st.video(st.session_state['video_processed_file'])


def drowning_logs_tab():
    st.subheader("📄 Drowning Log")

    log_entries = st.session_state["video_drowning_log"]

    df_log = pd.DataFrame(
        st.session_state["video_drowning_log_no_roi"],
        columns=["nth_second", "object_id", "drowning_probability"]
    )
    
    st.download_button(
        "Download Log",
        key="video_log_download_log",
        data=df_log.to_csv().encode("utf-8"),
        file_name=f"drowning_log_{Path(st.session_state['video_uploaded_file'].name).stem if st.session_state['video_uploaded_file'] else 'default'}.csv",
        mime="text/csv",
        disabled=not st.session_state["video_drowning_log_no_roi"],
        use_container_width=True
    )
    
    if log_entries:
        for entry in log_entries:
            cols = st.columns([3, 1])
            with cols[0]:
                for title, value in [("Nth second", 'nth_second'), ("Object ID", 'object_id'), ("Drowning Probability", 'drowning_probability')]:
                    sub_cols = st.columns([1, 1])
                    with sub_cols[0]:
                        st.markdown(f"**{title}**")
                    with sub_cols[1]:
                        if value == 'drowning_probability':
                            st.write(f": {entry[value] * 100:.2f}%")
                        elif value == 'nth_second':
                            st.write(f": {entry[value]:.2f}")
                        else:
                            st.write(f": {entry[value]}")
            with cols[1]:
                st.image(entry["roi"], channels="BGR")
            st.write("---")
    else:
        st.info("No drowning log entries available.")


def video():
    st.title("📷 Video Drowning Detection")

    tabs = st.tabs(["Video Upload", "Drowning Logs"])

    with tabs[0]:
        video_upload_tab()
    with tabs[1]:
        drowning_logs_tab()


if __name__ == "__main__":
    init()
    video()
