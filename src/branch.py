#!/usr/bin/env python3
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import pytesseract
from moviepy.video.io.VideoFileClip import VideoFileClip
from ProgressBar import ProgressBar


def name_in_killfeed(frame: np.ndarray) -> bool:
    """
    Determines if the kill feed (text related to kills, deaths or events in game)
    is visible in a given frame of video.

    Parameters:
        frame (np.ndarray): A single frame from a video capture object.

    Returns:
        bool: True if the kill feed is visible based on keywords; False otherwise.
    The function converts the input frame to grayscale, applies thresholding for better OCR performance,
    and then uses pytesseract for Optical Character Recognition (OCR) to extract text from the image.
    It checks if any of a predefined list of keywords related to kills is present in the extracted text.

    Note: The function assumes that the input frame is already preprocessed and ready for OCR, such as being grayscale or thresholded.
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
        "Mesofunny",
        "Bartarded",
    ]  # Add more keywords as needed

    # Check if any kill-related keyword is present in the extracted text
    return any(keyword.lower() in text.lower() for keyword in kill_keywords)


def trim_video(mp4_input_path, mp4_output_path, wav_path):
    cap = cv2.VideoCapture(mp4_input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        mp4_output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4)))
    )

    # audio, sr = librosa.load(wav_path, sr=None)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if name_in_killfeed(frame):
            out.write(frame)
        """
        if is_kill_feed_visible(frame) or has_gunfire(audio):
            out.write(frame)
        """

    cap.release()
    out.release()
