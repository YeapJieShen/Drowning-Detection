import streamlit as st
import cv2
import time
import os
from ultralytics import YOLO
import pygame
from collections import deque
import numpy as np

PAGE_NAME = 'live'

SIREN_PATH = os.path.join(os.getenv('DATA_DIR'), 'audio', 'siren.mp3')
PERSON_CLASS_ID = 0
DROWNING_CLASS_ID = 2
NUM_CLASSES = 3
HISTORY_LENGTH = 30
DROWNING_THRESHOLD = 15
SENSITIVITY = 0.2  # Sensitivity factor for smoothing, can be dynamically adjusted

def create_class_vector(confidence, class_id, num_classes=80):
    class_vector = np.zeros(num_classes)
    class_vector[class_id] = confidence
    return class_vector


def apply_temporal_smoothing(existing_vector, new_vector, sensitivity=SENSITIVITY):
    if existing_vector is None:
        # If there's no previous value, simply return the new vector
        return new_vector
    else:
        # Apply Exponential Moving Average (EMA) to stabilize
        return sensitivity * existing_vector + (1 - sensitivity) * new_vector


def get_stabilized_object_label(existing_label_vec, confidence, cls, sensitivity=SENSITIVITY, num_classes=80):
    new_label_vec = create_class_vector(confidence, cls, num_classes)

    # Apply temporal smoothing to stabilize the label
    return apply_temporal_smoothing(existing_label_vec, new_label_vec, sensitivity)


def get_class_from_label(label_vec):
    return int(np.argmax(label_vec))


def get_stabilized_class_predictions(existing_label_vec, class_confs, sensitivity=SENSITIVITY):
    # Convert class confidence list to a numpy array
    new_label_vec = np.array(class_confs)

    # Apply temporal smoothing to stabilize the class predictions
    return apply_temporal_smoothing(existing_label_vec, new_label_vec, sensitivity)


def is_activate_drowning_alert(tracked_info):
    for id, info in tracked_info.items():
        if len(info['frame_history']) < HISTORY_LENGTH:
            continue

        most_common = max(set(info['frame_history']),
                          key=info['frame_history'].count)
        is_drowning = (
            most_common == DROWNING_CLASS_ID and
            info['frame_history'].count(DROWNING_CLASS_ID) >= DROWNING_THRESHOLD and
            get_class_from_label(info['obj_label_vec']) == PERSON_CLASS_ID
        )

        if is_drowning:
            return True

    return False


def simulate_action_class_prediction():
    randint = np.random.randint(1, NUM_CLASSES)
    return [1 if i == randint else 0 for i in range(NUM_CLASSES)]


def play_siren():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play()


def stop_siren():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

# _____________________________________


def init():
    if PAGE_NAME + 'status' not in st.session_state:
        st.session_state[PAGE_NAME + 'status'] = 'stopped'
    if PAGE_NAME + 'YOLO' not in st.session_state:
        st.session_state[PAGE_NAME +
                         'YOLO'] = YOLO(os.path.join(os.getenv('YOLO_DIR'), 'yolo11n.pt'))
    if PAGE_NAME + 'PygameMixerInitialised' not in st.session_state:
        pygame.mixer.init()
        pygame.mixer.music.load(SIREN_PATH)
        st.session_state[PAGE_NAME + 'PygameMixer'] = True


def stream_btn_callback():
    if st.session_state[PAGE_NAME + 'status'] == 'stopped':
        st.session_state[PAGE_NAME + 'status'] = 'running'
    else:
        st.session_state[PAGE_NAME + 'status'] = 'stopped'


def live():
    st.title("📹 Live Drowning Detection Stream")

    fps = 120
    frame_interval = 1 / fps

    st.button("Start" if st.session_state[PAGE_NAME + 'status'] == "stopped" else "Stop", key=(
        PAGE_NAME + "btnStream"), on_click=stream_btn_callback, use_container_width=True)
    error_placeholder = st.empty()
    frame_placeholder = st.empty()

    if st.session_state[PAGE_NAME + 'status'] == 'running':
        cap = cv2.VideoCapture(0)
        tracked_info = {}
        yolo_model = st.session_state[PAGE_NAME + 'YOLO']

        while st.session_state[PAGE_NAME + 'status'] == 'running':
            drowning_alert = False
            ret, frame = cap.read()
            start_time = time.time()

            if not ret:
                st.error("Failed to capture video")
                break

            result = yolo_model.track(
                frame, persist=True, tracker="botsort.yaml", verbose=False)[0]

            if result.boxes and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                obj_clses = result.boxes.cls.cpu().numpy()
                obj_confs = result.boxes.conf.cpu().numpy()
                ids = result.boxes.id.cpu().numpy()

                for box, cls, conf, id in zip(boxes, obj_clses, obj_confs, ids):
                    if id not in tracked_info:
                        existing_obj_label_vec = None
                        existing_class_prediction_vec = None
                        frame_history = deque(maxlen=HISTORY_LENGTH)
                    else:
                        existing_obj_label_vec = tracked_info[id]['obj_label_vec']
                        existing_class_prediction_vec = tracked_info[id]['class_prediction_vec']
                        frame_history = tracked_info[id]['frame_history']

                    obj_label_vec = get_stabilized_object_label(existing_obj_label_vec, conf, int(cls))
                    obj_label = get_class_from_label(obj_label_vec)

                    if obj_label == PERSON_CLASS_ID:
                        class_confs = simulate_action_class_prediction() # Simulate class prediction for demonstration purposes
                        class_prediction_vec = get_stabilized_class_predictions(existing_class_prediction_vec, class_confs, SENSITIVITY)
                        class_prediction = get_class_from_label(class_prediction_vec)

                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'Person {id}, Predicted Class: {class_prediction}, Conf: {class_prediction_vec[class_prediction]}, obj_conf: {obj_label_vec[obj_label]}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        frame_history.append(class_prediction)
                    else:
                        # print(f"Skipping object with ID {id} and label {obj_label} due to low confidence or not person, obj_conf{obj_label_vec[obj_label]}.")
                        class_prediction_vec = existing_class_prediction_vec
                        class_prediction = get_class_from_label(class_prediction_vec)

                    tracked_info[id] = {
                        'obj_label_vec': obj_label_vec,
                        'class_prediction_vec': class_prediction_vec,
                        'frame_history': frame_history
                    }

                drowning_alert = is_activate_drowning_alert(tracked_info)

                if drowning_alert:
                    error_placeholder.error("Drowning alert!")
                    play_siren()

            fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.putText(frame, f'FPS: {fps:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if not drowning_alert:
                error_placeholder.empty()
                stop_siren()

            frame_placeholder.image(
                frame, channels="BGR", use_container_width=True)
            st.session_state[PAGE_NAME + 'lastFrame'] = frame

        cap.release()
        stop_siren()
    else:
        if PAGE_NAME + 'lastFrame' in st.session_state:
            frame_placeholder.image(
                st.session_state[PAGE_NAME + 'lastFrame'], channels="BGR", use_container_width=True)


if __name__ == "__main__":
    init()
    live()
