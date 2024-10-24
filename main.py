import time
import cv2
import numpy as np
import os
import json
from ultralytics import YOLO

trapezoids = []
filename = "trapezoids.json"
video_path = "./video.mp4"
detection_status = []
detection_timers = {}
debounce_time = 1
release_time_threshold = 3
long_occupation_threshold = 15
log_filename = "parking_spot_log.json"

model = YOLO('yolov8s.pt')

if not os.path.exists(video_path):
    print(f"Error: Video file '{video_path}' not found.")
    exit(1)

cap = cv2.VideoCapture(video_path)


def load_trapezoids():
    global trapezoids
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, "r") as f:
            trapezoids = json.load(f)
    else:
        trapezoids = []


def save_trapezoids():
    with open(filename, "w") as f:
        json.dump(trapezoids, f)


def point_in_trapezoid(point, trapezoid):
    pts = np.array(trapezoid, dtype=np.int32)
    return cv2.pointPolygonTest(pts, point, False) >= 0


def log_parking_spots(spot_index, action, occupied_time):
    logs = []
    if os.path.exists(log_filename):
        with open(log_filename, "r") as log_file:
            logs = json.load(log_file)

    log_entry = {
        "spot_index": spot_index,
        "action": action,
        "time": occupied_time
    }
    logs.append(log_entry)

    with open(log_filename, "w") as log_file:
        json.dump(logs, log_file, indent=4)

    print(f"Parking spot number {spot_index} {
          action} for {occupied_time:.2f} seconds.")


def check_cars(frame):
    global detection_status
    results = model(frame)

    detection_status = [False] * len(trapezoids)

    for result in results:
        for box in result.boxes:
            if box.conf > 0.35:

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                center_point = ((x1 + x2) // 2, (y1 + y2) // 2)

                for i, trapezoid in enumerate(trapezoids):
                    if point_in_trapezoid(center_point, trapezoid):
                        detection_status[i] = True
                        if i not in detection_timers:
                            detection_timers[i] = time.time()
                        break

    for i in range(len(detection_status)):
        if detection_status[i] and i in detection_timers:
            occupied_time = time.time() - detection_timers[i]

            if occupied_time >= release_time_threshold:
                log_parking_spots(i, "occupied", occupied_time)
        elif not detection_status[i] and i in detection_timers:
            occupied_time = time.time() - detection_timers[i]
            if occupied_time >= release_time_threshold:
                log_parking_spots(i, "released", occupied_time)
                del detection_timers[i]


def draw_trapezoids(frame):
    for i, trapezoid in enumerate(trapezoids):
        pts = np.array(trapezoid, dtype=np.int32)
        color = (0, 255, 0) if not detection_status[i] else (
            (0, 0, 255) if i in detection_timers and time.time() -
            detection_timers[i] >= long_occupation_threshold else (0, 255, 255)
        )
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

        if detection_status[i]:
            occupied_time = int(time.time() - detection_timers[i])

            cv2.rectangle(frame, (pts[0][0], pts[0][1] - 25),
                          (pts[0][0] + 70, pts[0][1]), (0, 0, 0), -1)
            cv2.putText(frame, f"Spot {i}: {occupied_time}s", (pts[0][0], pts[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    load_trapezoids()
    frame_skip = 10

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            check_cars(frame)
            draw_trapezoids(frame)

            cv2.imshow('Parking Spot Detection', frame)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
