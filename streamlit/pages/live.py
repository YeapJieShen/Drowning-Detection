import streamlit as st
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import deque
import json

PAGE_STATES = [
    "live_camera_on", "live_VideoCapture", "live_detection_model",
    "live_classification_model", "live_activate_siren", "live_siren",
    "live_drowning_log", "obj_confs_info", "class_confs_info",
    "predicted_classes_info", "live_cam_start_datetime"
]


def init():
    from ultralytics import YOLO
    from src.classify import TorchClassifier
    from pygame import mixer

    st.session_state["cleanup_function"] = cleanup

    if "live_camera_on" not in st.session_state:
        st.session_state["live_camera_on"] = False

    assets_path = Path(__file__).parent.parent / "assets"
    models_path = assets_path / "models"

    if "live_detection_model" not in st.session_state:
        yolo_path = models_path / "detection" / "YOLO" / "yolo11n.pt"
        st.session_state["live_detection_model"] = YOLO(yolo_path)

    if "live_classification_model" not in st.session_state:
        classification_path = models_path / "classification" / "CNN" / "test.pt"
        st.session_state["live_classification_model"] = TorchClassifier(
            model="CNNClassifier", model_path=classification_path)

    if "live_siren" not in st.session_state:
        siren_path = assets_path / "siren" / "siren.wav"

        mixer.init()
        mixer.music.load(siren_path)
        st.session_state["live_siren"] = mixer.music

    if "live_drowning_log" not in st.session_state:
        st.session_state["live_drowning_log"] = []


def cleanup():
    toggle_live_camera()

    if st.session_state["live_drowning_log"]:
        logs_dir = Path(__file__).parent.parent / "logs"
        logs_dir.mkdir(exist_ok=True)

        base_filename = f"drowning_log_{datetime.now().strftime('%Y_%m_%d')}"
        drowning_log_path = logs_dir / f"{base_filename}.json"

        # Handle duplicate file names by adding _1, _2, etc.
        counter = 1
        while drowning_log_path.exists():
            drowning_log_path = logs_dir / f"{base_filename} ({counter}).json"
            counter += 1

        # Save the log
        with open(drowning_log_path, "w") as log_file:
            json.dump(st.session_state["live_drowning_log"], log_file, indent=2) 

    for state in PAGE_STATES:
        if state in st.session_state:
            del st.session_state[state]


def toggle_live_camera():
    live_camera_on = st.session_state["live_camera_on"]

    if live_camera_on:
        st.session_state["live_VideoCapture"].release()

        siren = st.session_state["live_siren"]
        if siren.get_busy():
            siren.fadeout(1000)  # Fade out the siren sound over 1 second

    st.session_state["live_camera_on"] = not live_camera_on


def classify_activity(roi):
    import torch
    from torchvision import transforms
    from PIL import Image

    class RGBToHSV:
        def __call__(self, img):
            # Ensure the image is in PIL format before converting
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)  # Convert tensor to PIL image

            # Convert the image to HSV using PIL
            img_hsv = img.convert("HSV")

            return img_hsv

    classification_model = st.session_state["live_classification_model"]

    roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        RGBToHSV(),
        transforms.ToTensor()
    ])

    prediction = classification_model(
        img=roi_pil,
        transform=transform,
        prob=True
    )

    return prediction.cpu().numpy()


def log_drowning_info(obj_id, drowning_prob, roi):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    img_filename = f"drowning_roi_{timestamp}_id{obj_id}.jpg"
    img_path = Path(__file__).parent.parent / "logs" / "snaps" / img_filename
    img_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(img_path), roi)

    log_entry = {
        "timestamp": timestamp,
        "object_id": int(obj_id),
        "drowning_probability": float(drowning_prob),
        "roi_image_path": str(img_path.relative_to(Path(__file__).parent.parent)),
    }

    st.session_state["live_drowning_log"].append(log_entry)


def detect_drowning(frame, sensitivity, lookback):
    CLASS_IDX_TO_NAME = {
        0: 'drowning',
        1: 'swimming',
        2: 'treadwater'
    }

    CLASS_NAME_TO_IDX = {
        value: key
        for key, value in CLASS_IDX_TO_NAME.items()
    }

    annotated_frame = frame.copy()

    detection_model = st.session_state["live_detection_model"]

    obj_confs_info = st.session_state["obj_confs_info"]
    class_confs_info = st.session_state["class_confs_info"]
    predicted_classes_info = st.session_state["predicted_classes_info"]

    siren = st.session_state["live_siren"]

    result = detection_model.track(
        frame, persist=True, tracker="botsort.yaml", verbose=False)[0]

    if result.boxes and result.boxes.id is not None:
        obj_boxes = result.boxes.xyxy.cpu().numpy()
        obj_ids = result.boxes.id.cpu().numpy()
        obj_confs = result.boxes.conf.cpu().numpy()

        for box, obj_id, obj_conf in zip(obj_boxes, obj_ids, obj_confs):
            if obj_id not in obj_confs_info or obj_id not in class_confs_info or obj_id not in predicted_classes_info:
                current_obj_conf = None
                current_class_vec = None
                predicted_class_history = deque(maxlen=lookback)
            else:
                current_obj_conf = obj_confs_info[obj_id]
                current_class_vec = class_confs_info[obj_id]
                predicted_class_history = predicted_classes_info[obj_id]

            new_obj_conf = obj_confs_info[obj_id] = (
                (1 - sensitivity) * current_obj_conf + sensitivity * obj_conf
                if current_obj_conf is not None else
                obj_conf
            )

            if new_obj_conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box)
            roi = frame[y1:y2, x1:x2]
            class_vec = classify_activity(roi)

            new_class_vec = class_confs_info[obj_id] = (
                (1 - sensitivity) * current_class_vec + sensitivity * class_vec
                if current_class_vec is not None else
                class_vec
            )

            predicted_class_idx = int(np.argmax(new_class_vec))
            predicted_class_history.append(predicted_class_idx)
            predicted_classes_info[obj_id] = predicted_class_history

            if len(predicted_class_history) < lookback:
                continue

            predicted_class_name = CLASS_IDX_TO_NAME[max(
                set(predicted_class_history), key=predicted_class_history.count)]
            # TODO: For testing only
            predicted_class_name = "drowning" if predicted_class_name == "treadwater" else predicted_class_name
            # Normalize to 0–1
            drowning_prob = min(max(new_class_vec[CLASS_NAME_TO_IDX["drowning"]], 0), 100)

            # Interpolate: red = (255, 0, 0), green = (0, 255, 0)
            bgr = 0, int(255 * (1 - drowning_prob)), int(255 * drowning_prob)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(annotated_frame, f'{int(obj_id)} {predicted_class_name} {new_class_vec[predicted_class_idx] * 100:.2f}%', (
                x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if obj_conf < 0.5:
                continue

            if predicted_class_name == "drowning":
                st.session_state["live_activate_siren"] = True

                if st.session_state["live_activate_siren"] and not siren.get_busy():
                    siren.play()

                log_drowning_info(obj_id, drowning_prob, roi)

    if not st.session_state["live_activate_siren"] and siren.get_busy():
        siren.fadeout(1000)

    cv2.putText(annotated_frame, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return annotated_frame


def live_camera_tab():
    st.subheader("📹 Live Camera Feed")
    st.write("This tab displays the live camera feed and performs drowning detection.")

    live_camera_on = st.session_state["live_camera_on"]

    with st.expander("Detection Settings", expanded=False):
        sensitivity = st.slider(
            "Sensitivity", min_value=0.01, max_value=1.0, value=0.5, step=0.01)
        lookback = st.slider("Lookback", min_value=1,
                            max_value=30, value=10, step=1)

    st.button(f"{'Stop' if live_camera_on else 'Start'} Live Camera",
              key="live_camera_button", on_click=toggle_live_camera, use_container_width=True)
    
    if live_camera_on:
        cap = st.session_state["live_VideoCapture"] = cv2.VideoCapture(0)
        st.session_state["live_cam_start_datetime"] = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        frame_placeholder = st.empty()
        processing_placeholder = st.empty()

        if not cap.isOpened():
            st.error("Error: Unable to access the camera.")
            return

        st.session_state["obj_confs_info"] = {}
        st.session_state["class_confs_info"] = {}
        st.session_state["predicted_classes_info"] = {}

        while live_camera_on:
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Unable to read from the camera.")
                break

            st.session_state["live_activate_siren"] = False

            processing_placeholder.write(
                f"Processing frame... {datetime.now()}")

            annotated_frame = detect_drowning(frame, sensitivity, lookback)

            frame_placeholder.image(
                annotated_frame, channels="BGR", use_container_width=True)

        cap.release()
    else:
        st.write("Live camera is OFF.")


def drowning_logs_tab():
    st.subheader("📄 Drowning Log")

    logs_dir = Path(__file__).parent.parent / "logs"

    log_files = sorted(logs_dir.glob("drowning_log_*.json"), reverse=True)  # Sorted by date, most recent first

    # If no logs are available in the directory, inform the user
    if not log_files and "live_drowning_log" not in st.session_state:
        st.info("No log files available.")
        return
    
    log_choices = [log_file.name for log_file in log_files]
    selected_log = st.selectbox("Select a log file", log_choices, index=None, disabled=not log_choices)

    if selected_log is None:
        log_entries = st.session_state["live_drowning_log"]
    else:
        selected_log_path = next(log_file for log_file in log_files if log_file.name == selected_log)

        with open(selected_log_path, "r") as log_file:
            log_entries = json.load(log_file)

    if log_entries:
        for entry in log_entries:
            col1, col2 = st.columns([1, 3])
            with col1:
                img_path = Path(__file__).parent.parent / entry["roi_image_path"]

                st.image(str(img_path), caption=f"ID {entry['object_id']}", width=150)
            with col2:
                st.markdown(f"""
                **Timestamp**: {entry['timestamp']}  
                **Object ID**: {entry['object_id']}  
                **Drowning Probability**: {entry['drowning_probability']:.2f}%
                """)
            st.markdown("---")
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
