import cv2
import numpy as np
import os
import json

# Initialize global variables
trapezoids = []  # To store trapezoid points
dragging_index = None  # Index of the trapezoid currently being edited
filename = "trapezoids.json"  # Output file for saving trapezoid coordinates

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
    global dragging_index

    # If left mouse button is clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if clicked inside any trapezoid to start editing
        for i, trapezoid in enumerate(trapezoids):
            for j, point in enumerate(trapezoid):
                # Check if the mouse click is close to a point (for editing)
                if np.linalg.norm(np.array(point) - np.array((x, y))) < 10:  # 10 pixels threshold
                    # Save trapezoid index and point index
                    dragging_index = (i, j)
                    break

    # If the mouse is moved while dragging
    elif event == cv2.EVENT_MOUSEMOVE and dragging_index is not None:
        i, j = dragging_index
        trapezoids[i][j] = (x, y)  # Update the selected point's position

    # If the left mouse button is released
    elif event == cv2.EVENT_LBUTTONUP:
        if dragging_index is not None:
            dragging_index = None  # Stop dragging

# Draw trapezoids and points


def draw_trapezoids(frame):
    for trapezoid in trapezoids:
        pts = np.array(trapezoid, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True,
                      color=(255, 0, 0), thickness=2)
        for point in trapezoid:
            cv2.circle(frame, point, 6, (255, 255, 255), -1)  # White dots

# Main function to run the editor


def main():
    global trapezoids

    load_trapezoids()  # Load existing trapezoids

    cv2.namedWindow("Trapezoid Editor")
    cv2.setMouseCallback("Trapezoid Editor", mouse_callback)

    while True:
        # Create a black frame
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        draw_trapezoids(frame)  # Draw trapezoids on the frame

        cv2.imshow("Trapezoid Editor", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Press 's' to save
            save_trapezoids()
            print("Trapezoids saved!")
        elif key == ord('q'):  # Press 'q' to exit
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
