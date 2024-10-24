import cv2
import numpy as np
import os
import json
from ultralytics import YOLO

# Initialize global variables
trapezoids = []  # To store trapezoid points
current_trapezoid = []  # To store points for the current trapezoid being defined
dragging_index = None  # Index of the corner currently being dragged
editing_index = -1  # Index of the trapezoid currently being edited
filename = "trapezoids.json"  # Output file for saving trapezoid coordinates
video_path = "./video.mp4"  # Path to video file

# Load YOLO model
model = YOLO('yolov8s.pt')

# Load video file
if not os.path.exists(video_path):
    print(f"Error: Video file '{video_path}' not found.")
    exit(1)

cap = cv2.VideoCapture(video_path)

# Load existing trapezoids from file


def load_trapezoids():
    global trapezoids
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, "r") as f:
            trapezoids = json.load(f)
    else:
        trapezoids = []  # Initialize as empty if file is empty

# Save trapezoids to file


def save_trapezoids():
    with open(filename, "w") as f:
        json.dump(trapezoids, f)

# Function to check if a point is inside a trapezoid


def point_in_trapezoid(point, trapezoid):
    pts = np.array(trapezoid, dtype=np.int32)
    return cv2.pointPolygonTest(pts, point, False) >= 0

# Mouse callback function to handle clicks and dragging


def mouse_callback(event, x, y, flags, param):
    global current_trapezoid, dragging_index, editing_index

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)

        # Check if any existing trapezoid is clicked to start dragging
        for i, trapezoid in enumerate(trapezoids):
            if point_in_trapezoid(clicked_point, trapezoid):
                editing_index = i  # Set index of trapezoid to edit
                # Find which corner is being dragged
                for j in range(4):
                    # Increased size
                    if np.linalg.norm(np.array(clicked_point) - np.array(trapezoids[editing_index][j])) < 15:
                        dragging_index = j
                        print(f"Dragging corner: {
                              trapezoids[editing_index][j]}")
                        break
                break

        # If not dragging any corner, check for new trapezoid creation
        if dragging_index is None and len(current_trapezoid) < 4:
            current_trapezoid.append(clicked_point)
            print(f"Point added: {clicked_point}")

    elif event == cv2.EVENT_LBUTTONUP:
        # If we were dragging a point, stop dragging
        dragging_index = None

        # If the current trapezoid is complete, add it to the list
        if len(current_trapezoid) == 4:
            trapezoids.append(current_trapezoid.copy())
            print(f"Trapezoid added: {current_trapezoid}")
            current_trapezoid.clear()  # Clear current trapezoid for new input

    elif event == cv2.EVENT_MOUSEMOVE and dragging_index is not None:
        # Update the position of the dragged corner
        trapezoids[editing_index][dragging_index] = (x, y)

# Function to draw trapezoids and points


def draw_trapezoids(frame):
    overlay = frame.copy()  # Create a copy of the frame for overlay

    # Draw all existing trapezoids
    for index, trapezoid in enumerate(trapezoids):
        pts = np.array(trapezoid, dtype=np.int32).reshape((-1, 1, 2))
        # Draw trapezoid outline
        cv2.polylines(overlay, [pts], isClosed=True,
                      color=(0, 255, 0), thickness=2)

        # Draw points at corners
        for (x, y) in trapezoid:
            # Larger blue circles for corners
            # Increased radius
            cv2.circle(overlay, (x, y), 10, (255, 0, 0), -1)
            cv2.putText(overlay, str(index + 1), (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw the current trapezoid being defined, if it exists
    if len(current_trapezoid) > 0:
        current_pts = np.array(
            current_trapezoid, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [current_pts], isClosed=False,
                      color=(0, 255, 255), thickness=2)
        for (x, y) in current_trapezoid:
            cv2.circle(overlay, (x, y), 10, (255, 0, 0), -1)  # Blue circles

    return overlay


# Main loop to display video and handle user input
load_trapezoids()

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop the video
        continue

    # Draw trapezoids and current trapezoid on the frame
    overlay = draw_trapezoids(frame)
    cv2.imshow("Video", overlay)

    # Set the mouse callback
    cv2.setMouseCallback("Video", mouse_callback)

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # Save trapezoids
        save_trapezoids()
        print("Trapezoids saved.")
    elif key == ord('q'):  # Quit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
