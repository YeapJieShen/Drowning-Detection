import cv2
import numpy as np
from datetime import datetime

ACTIVITY_IDX_TO_NAME = {
    0: 'drowning',
    1: 'swimming',
    2: 'treadwater'
}

def classify_activity(model, roi):
    import torch
    from torchvision import transforms
    from PIL import Image

    class RGBToHSV:
        def __call__(self, img):
            # Ensure the image is in PIL format before converting
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)  # Convert tensor to PIL image

            return img.convert("HSV") # Convert to HSV

    roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        RGBToHSV(),
        transforms.ToTensor()
    ])

    prediction = model(
        img=roi_pil,
        transform=transform,
        prob=True
    )

    return prediction.cpu().numpy() # Convert to numpy array

def detect_drowning(detection_model, classification_model, obj_confs_info, class_confs_info, frame, sensitivity):
    drowning_idx = [key for key, value in ACTIVITY_IDX_TO_NAME.items() if value == "drowning"][0]

    activate_siren = False
    drowning_info_logs = []
    annotated_frame = frame.copy()

    result = detection_model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)[0]

    if result.boxes and result.boxes.id is not None:
        obj_boxes = result.boxes.xyxy.cpu().numpy()
        obj_ids = map(int, result.boxes.id.cpu().numpy())
        obj_confs = result.boxes.conf.cpu().numpy()

        for box, obj_id, obj_conf in zip(obj_boxes, obj_ids, obj_confs):
            if obj_id not in obj_confs_info or obj_id not in class_confs_info:
                current_obj_conf_count = None
                current_activity_vec = None
            else:
                current_obj_conf_count = obj_confs_info[obj_id]
                current_activity_vec = class_confs_info[obj_id]

            # Update objectnesss confidence
            new_obj_conf = obj_confs_info[obj_id] = (
                ((1 - sensitivity) *
                 current_obj_conf_count[0] + sensitivity * obj_conf, current_obj_conf_count[1] + 1)
                if current_obj_conf_count is not None else
                (obj_conf, 1)
            )

            if new_obj_conf[0] < 0.5:  # Object confidence threshold
                continue

            x1, y1, x2, y2 = map(int, box)
            roi = frame[y1:y2, x1:x2]

            # Classify the activity in the ROI
            activity_vec = classify_activity(classification_model, roi)
            new_activity_vec = class_confs_info[obj_id] = (
                (1 - sensitivity) *
                current_activity_vec[0] + sensitivity * activity_vec
                if current_activity_vec is not None else
                activity_vec
            )

            if new_obj_conf[1] < 20:  # Number of frames required before drawing detection
                continue

            # Get the predicted activity and corresponding probability
            predicted_activity_idx = int(np.argmax(new_activity_vec))
            predicted_activity_name = ACTIVITY_IDX_TO_NAME[predicted_activity_idx]
            drowning_prob = float(np.clip(new_activity_vec[drowning_idx], 0, 1)) # Clamp the drowning probability to [0, 1]

            bgr = 0, int(255 * (1 - drowning_prob)), int(255 * drowning_prob) # Interpolate: red = (0, 0, 255), green = (0, 255, 0)

            # Draw the bounding box and label
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(annotated_frame, f'{obj_id} {predicted_activity_name} {drowning_prob * 100:.2f}%', (
                x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1)

            if predicted_activity_name == "drowning":
                activate_siren = True
                drowning_info_logs.append((datetime.now().strftime('%Y-%m-%d %H:%M:%S'), obj_id, drowning_prob, roi))

    cv2.putText(annotated_frame, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return annotated_frame, activate_siren, drowning_info_logs