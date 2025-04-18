import os
import random
import sys

import cv2
from ultralytics import YOLO

from tracker import Tracker


video_path = os.path.join('models', 'human_detection_tracker', 'test_videos', 'manyswimmers.mp4')
video_out_path = os.path.join(os.path.dirname(video_path), 'out.mp4')
print("Video Path:", video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

if not ret or frame is None:
    print("Error: Unable to read video file.")
    cap.release()
    sys.exit()

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

model = YOLO("C:\\Users\\jrom\\DataspellProjects\\Drowning-Detection\\models\\human_detection_yolo11s.pt")

tracker = Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

detection_threshold = 0.3

# Create a named window for display
cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Object Detection", 1280, 720)  # Adjust size as needed

while ret:
    results = model(frame)

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > detection_threshold:
                detections.append([x1, y1, x2, y2, score])

        tracker.update(frame, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
            # Add ID text to the bounding box
            cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[track_id % len(colors)], 2)

    # Display the frame
    cv2.imshow("Object Detection", frame)

    # Write to output video
    cap_out.write(frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()
#%%
