import argparse
from pathlib import Path

import cv2
import numpy as np
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets

# from fsutils import Dir, Video


def parse_args() -> argparse.Namespace:
    """Function to parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Video processing script")
    parser.add_argument("VIDEO", type=str, help="Path to the video file")
    args = parser.parse_args()
    return args


def detect_name(frame: np.ndarray) -> bool:
    """Function to detect a specific text in an image or video frame.

    Parameters:
    ---------
        frame (numpy.ndarray) : A single frame from the video file.

    """
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Define the text detection parameters
    text_scale = 1.5
    text_thickness = 2
    text_color = (0, 0, 255)  # Red

    # Define the text to detect (your name)
    text_to_detect = "MrHoffman_"

    # Detect text in the frame
    text_detected = cv2.putText(
        gray,
        text_to_detect,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        text_color,
        text_thickness,
    )

    # Return True if text is detected, False otherwise
    return text_detected is not None


def main(video: str | Path) -> None:
    # Load the video clip
    cap = cv2.VideoCapture("clip.mp4")
    # Create a GUI window to display the video
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the video frame
        # cv2.imshow("Video", frame)

        # Handle keyboard events (e.g., space to pause, 'q' to quit)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()

    # Iterate through the video frames
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

    # Create a GUI window to display the timeline
    timeline_window = QtWidgets.QWidget()
    timeline_window.setWindowTitle("Timeline")

    # Create a horizontal slider to represent the timeline
    timeline_slider = QtWidgets.QSlider(QtCore.Qt.Alignment)
    timeline_slider.setRange(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    # Create a label to display the current frame number
    current_frame_label = QtWidgets.QLabel("Frame 0")

    # Create a layout to hold the slider and label
    layout = QtWidgets.QVBoxLayout()
    layout.addWidget(timeline_slider)
    layout.addWidget(current_frame_label)

    # Set the layout for the timeline window
    timeline_window.setLayout(layout)

    # Show the timeline window
    timeline_window.show()
    # Detect your name in the frame
    if detect_name(frame):
        # Update the timeline display to show a flag or marker
        timeline_slider.setValue(frame_count)
        current_frame_label.setText(f"Frame {frame_count}")

    frame_count += 1

    # Add buttons to mark the start and end points of the clip
    start_button = QtWidgets.QPushButton("Mark Start")
    end_button = QtWidgets.QPushButton("Mark End")

    # Add the buttons to the layout
    layout.addWidget(start_button)
    layout.addWidget(end_button)

    # Connect the buttons to functions that update the clip range
    start_button.clicked.connect(lambda: update_clip_range("start"))
    end_button.clicked.connect(lambda: update_clip_range("end"))

    # Define the update_clip_range function
    def update_clip_range(point: str) -> None:
        # Update the clip range based on the selected point
        if point == "start":
            start_frame = timeline_slider.value()
        elif point == "end":
            end_frame = timeline_slider.value()


if __name__ == "__main__":
    # args = parse_args()
    #    main(args.VIDEO)

    VIDEO = "../assets/PUBG_COMPRESSED.mp4"
    main(VIDEO)
