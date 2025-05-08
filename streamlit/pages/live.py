import streamlit as st
import cv2
from datetime import datetime
from pathlib import Path
import pandas as pd
from services import detect_drowning

PAGE_STATES = [
    "live_camera_on", "live_VideoCapture",
    "live_detection_model", "live_classification_model",
    "live_siren", "live_activate_siren"
    "live_obj_confs_info", "live_class_confs_info",
    "live_drowning_log", "live_drowning_log_no_roi", 
    "live_last_frame", "live_available_cameras"
]


def init():
    from ultralytics import YOLO
    from src.classify import TorchClassifier
    from pygame import mixer

    st.session_state["cleanup_function"] = cleanup

    if "live_camera_on" not in st.session_state:
        st.session_state["live_camera_on"] = False

    if "live_available_cameras" not in st.session_state:
        available_cameras = []
        for index in range(5):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap.isOpened():
                available_cameras.append(index)
                cap.release()

        st.session_state["live_available_cameras"] = available_cameras

    assets_path = Path(__file__).parent.parent / "assets"
    models_path = assets_path / "models"

    if "live_detection_model" not in st.session_state:
        yolo_path = models_path / "detection" / "YOLO" / "yolo11n_best.pt"
        st.session_state["live_detection_model"] = YOLO(yolo_path)

    if "live_classification_model" not in st.session_state:
        classification_path = models_path / "classification" / "CNN" / "cnn_best.pt"
        st.session_state["live_classification_model"] = TorchClassifier(
            model="CNNClassifier", model_path=classification_path)

    if "live_siren" not in st.session_state:
        siren_path = assets_path / "siren" / "siren.wav"

        mixer.init()
        mixer.music.load(siren_path)
        st.session_state["live_siren"] = mixer.music

    if "live_drowning_log" not in st.session_state:
        st.session_state["live_drowning_log"] = []

    if "live_drowning_log_no_roi" not in st.session_state:
        st.session_state["live_drowning_log_no_roi"] = []

    if "live_last_frame" not in st.session_state:
        st.session_state["live_last_frame"] = None


def cleanup():
    live_camera_on = st.session_state["live_camera_on"]

    if live_camera_on: # Stop the live camera if on
        off_live_camera()

    for state in PAGE_STATES:
        if state in st.session_state:
            del st.session_state[state]


def off_live_camera():
    st.session_state["live_VideoCapture"].release()

    siren = st.session_state["live_siren"]
    if siren.get_busy():
        siren.fadeout(1000)  # Fade out the siren sound over 1 second

def toggle_live_camera():
    live_camera_on = st.session_state["live_camera_on"]

    if live_camera_on: # Stop the live camera if on
        off_live_camera()
    else:
        selected_cam = st.session_state["live_selected_cam"]
        st.session_state["live_VideoCapture"] = cv2.VideoCapture(selected_cam, cv2.CAP_DSHOW)
        st.session_state["live_obj_confs_info"] = {}
        st.session_state["live_class_confs_info"] = {}

    st.session_state["live_camera_on"] = not live_camera_on

def reset_capture():
    live_camera_on = st.session_state["live_camera_on"]

    if "live_VideoCapture" in st.session_state:
        st.session_state["live_VideoCapture"].release()

    if live_camera_on:
        selected_cam = st.session_state["live_selected_cam"]
        st.session_state["live_VideoCapture"] = cv2.VideoCapture(selected_cam, cv2.CAP_DSHOW)

def load_another_yolo_model():
    from ultralytics import YOLO

    MODEL_NAME_MAPPING = {
        "Pretrained YOLO": "yolo11n.pt",
        "Fine-Tuned YOLO": "yolo11n_best.pt",
    }

    selected_model = st.session_state["live_yolo_model"]
    
    assets_path = Path(__file__).parent.parent / "assets"
    models_path = assets_path / "models"

    yolo_path = models_path / "detection" / "YOLO" / MODEL_NAME_MAPPING[selected_model]
    st.session_state["live_detection_model"] = YOLO(yolo_path)

def log_drowning_info(log):
    timestamp, obj_id, drowning_prob, roi = log
    drowning_prob = float(drowning_prob) 

    log_entry = {
        "timestamp": timestamp,
        "object_id": obj_id,
        "drowning_probability": drowning_prob,
        "roi": roi
    }

    log_entry_no_roi = [timestamp, obj_id, drowning_prob]

    st.session_state["live_drowning_log"].append(log_entry)
    st.session_state["live_drowning_log_no_roi"].append(log_entry_no_roi)

def live_camera_tab():
    st.subheader("📹 Live Camera Feed")

    alert_placeholder = st.empty()

    with st.expander("Detection Settings", expanded=False):
        sensitivity = st.slider(
            "Sensitivity", min_value=0.01, max_value=1.0, value=0.5, step=0.01,
            help="Controls how quickly the system reacts to changes. Higher values make the alert respond faster to sudden movements (more sensitive), while lower values make it smoother and more stable."
        )

        selected_cam = st.selectbox(
            "Select Camera",
            options=st.session_state["live_available_cameras"],
            on_change=reset_capture,
            key="live_selected_cam"
        )

        yolo_model = st.selectbox(
            "YOLO Model",
            options=["Fine-Tuned YOLO", "Pretrained YOLO"],
            on_change=load_another_yolo_model,
            key="live_yolo_model"
        )

    live_camera_on = st.session_state["live_camera_on"]

    st.button(f"{'Stop' if live_camera_on else 'Start'} Live Camera",
              key="live_camera_button", on_click=toggle_live_camera, use_container_width=True)

    frame_placeholder = st.empty()

    if live_camera_on:
        cap = st.session_state["live_VideoCapture"]
        siren = st.session_state["live_siren"]

        if not cap.isOpened():
            st.error("Error: Unable to access the camera.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Unable to read from the camera.")
                break

            annotated_frame, activate_siren, drowning_info_logs = detect_drowning(
                st.session_state["live_detection_model"],
                st.session_state["live_classification_model"],
                st.session_state["live_obj_confs_info"],
                st.session_state["live_class_confs_info"],
                frame,
                sensitivity
            )

            st.session_state["live_last_frame"] = annotated_frame
            for log in drowning_info_logs:
                log_drowning_info(log)

            frame_placeholder.image(
                annotated_frame, channels="BGR", use_container_width=True)

            if activate_siren:
                alert_placeholder.error("Drowning detected!")
                if not siren.get_busy():
                    siren.play()
            else:
                alert_placeholder.success("No drowning detected.")
                if siren.get_busy():
                    siren.fadeout(1000)

        cap.release()
    elif st.session_state["live_last_frame"] is not None:
        frame_placeholder.image(
            st.session_state["live_last_frame"], channels="BGR", use_container_width=True)


def drowning_logs_tab():
    st.subheader("📄 Drowning Log")

    log_entries = st.session_state["live_drowning_log"]

    df_log = pd.DataFrame(
        st.session_state["live_drowning_log_no_roi"],
        columns=["timestamp", "object_id", "drowning_probability"]
    )
    
    st.download_button(
        "Download Log",
        key="download_log",
        data=df_log.to_csv().encode("utf-8"),
        file_name=f"drowning_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv",
        mime="text/csv",
        disabled=not st.session_state["live_drowning_log_no_roi"],
        use_container_width=True
    )
    
    if log_entries:
        for entry in log_entries:
            cols = st.columns([3, 1])
            with cols[0]:
                for title, value in [("Timestamp", 'timestamp'), ("Object ID", 'object_id'), ("Drowning Probability", 'drowning_probability')]:
                    sub_cols = st.columns([1, 1])
                    with sub_cols[0]:
                        st.write(f"**{title}**")
                    with sub_cols[1]:
                        if value == 'drowning_probability':
                            st.write(f": {entry[value] * 100:.2f}%")
                        else:
                            st.write(f": {entry[value]}")
            with cols[1]:
                st.image(entry["roi"], channels="BGR")
            st.write("---")
    else:
        st.info("No drowning log entries available.")


def live():
    st.title("📹 Live Drowning Detection")

    tabs = st.tabs(["Live Camera", "Drowning Logs"])

    with tabs[0]:
        live_camera_tab()
    with tabs[1]:
        drowning_logs_tab()


if __name__ == "__main__":
    init()
    live()
