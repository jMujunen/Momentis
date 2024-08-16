#!/usr/bin/env python3

import cv2
import numpy as np
import pytesseract

def name_in_killfeed(frame: np.ndarray) -> bool:
    """Determines if the kill feed is visible in a given frame of video.

    Steps:
    ------
    1. Converts the input frame to grayscale
    2. Applies thresholding for better OCR performance
    3. Run pytesseract for (OCR) to extract text from the image.
    4. Check if any of a predefined list of keywords are present in the extracted text.

    """
    # Convert to grayscale for better OCR performance
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to enhance text visibility
    _, threshold = cv2.threshold(gray_frame, 150, 255, cv2.THRESH_BINARY)

    # Use pytesseract to perform OCR on the frame
    text = pytesseract.image_to_string(threshold)

    # Define keywords related to kills
    kill_keywords = [
        "Hoffman",
        "itsgroovybabe",
        "MrHoffman",
        "MrHoffman_",
        "Mesofunny",
        "Bartarded",
    ]  # Add more keywords as needed

    # Check if any kill-related keyword is present in the extracted text
    return any(keyword.lower() in text.lower() for keyword in kill_keywords)


# 3. Initialize variables to store the start and end frames of the desired clip
def main(path: str):
    # 1. Load the video clips
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print("Error: Unable to load video file.")
        exit()
    else:
        print("Video file loaded successfully.")

    # Get the video properties (e.g., frame rate, width, height)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties:\nFPS: {fps}\nWidth: {width} px\nHeight: {height} px")  # DEBUG

    start_frame = 0
    end_frame = frame_count - 1
    # Iterate through the video frames
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            # Call the detect_name function every 10 frames
        if frame_count % 10 == 0 and detect_name(frame):
            print(
                "Name detected at frame ",
                frame_count,
            )
            # Your name was detected, capture 4 seconds before and after
            start_frame = max(0, frame_count - int(4 * fps))
            end_frame = frame_count + int(4 * fps)

        frame_count += 1

    cap.release()
    trimmed_clip = []
    for i in range(start_frame, end_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            trimmed_clip.append(frame)

    # # 4. Trim the original video clip
    # trimmed_clip = []
    # for i in range(start_frame or 1, end_frame or frame_count):
    #     ret, frame = cap.read()
    #     if ret:
    #         trimmed_clip.append(frame)

    # Join the trimmed clips
    final_clip = []
    for clip in trimmed_clip:
        final_clip.extend(clip)
    print(final_clip)
    print(len(final_clip))
    # Save the final clip to a new video file
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))
    for frame in final_clip:
        out.write(frame)
    out.release()
    cap.release()


if __name__ == "__main__":
    main("/home/joona/python/Projects/Momentis/assets/PUBG_COMPRESSED.mp4")
from cv2