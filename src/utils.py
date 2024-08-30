import os
from datetime import timedelta

import cv2
import numpy as np


def format_timedelta(td: timedelta) -> str:
    """Utility function to format timedelta objects in a cool way (e.g 00:00:20.05)
    omitting microseconds and retaining milliseconds"""
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return (result + ".00").replace(":", "-")
    ms = int(ms)
    ms = round(ms / 1e4)
    return f"{result}.{ms:02}".replace(":", "-")


def video_writer(frame_list: list[np.ndarray], video_path: str, output_path: str):
    # Get the total number of frames in the original video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frame_list:
        out.write(frame)


def frame2img(frame: np.ndarray, frame_duration: float, filename: str, output_folder: str):
    frame_duration_formatted = format_timedelta(timedelta(seconds=frame_duration))
    cv2.imwrite(os.path.join(output_folder, f"frame{frame_duration_formatted}.jpg"), frame)
