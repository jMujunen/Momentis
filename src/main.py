#!/usr/bin/env python3
"""Create a new new video containing only relevant frames determined by the killfeed."""

import argparse
import os
import re

import cv2
import pytesseract
from Color import cprint, fg
from fsutils import Dir
from numpy import ndarray
from tqdm import tqdm

from FrameBuffer import FrameBuffer

# Constants
INTERVAL = 60
WRITER_FPS = 60
BUFFER = 240
# List of keywords related to kill feeds
KEYWORDS = {
    "hoff": [
        "sofunny",
        "meso",
        "solunny",
        "hoff",
        "ffman" "bartard",
        "dankniss",
        "vermeme",
        "nissev",
    ],
    "muru": ["groovy", "itsgro", "ybabe"],
}
# Region of interest size (width, height) in pixels
ROI_W, ROI_H = (600, 200)
DATE_EXTRACTOR = re.compile(
    r"(\d{4})\.(\d{2})\.(\d{2})\s-\s(\d{2})\.(\d{2})\.(\d{2})\.(\d{2}).*\.(.{3}$)"
)


def file_sanatizer(filename: str) -> str:
    """Sanitize a filename to remove illegal characters."""
    matches = DATE_EXTRACTOR.findall(filename)
    if matches and len(matches[0]) == 9:
        prefix, year, month, day, hour, minute, second, ms, ext = matches[0]
        if "PLAYER" in prefix:
            prefix = "PUBG"
        return f"{prefix} {'-'.join([year,month,day])} {':'.join([hour, minute, second])}.{ext}"
    return re.sub(r"[\']", "", filename)


def name_in_killfeed(img: ndarray, rio: tuple, keywords: list[str]) -> tuple[bool, str]:
    """Check if a kill-related keyword is present in the text extracted from the ndarray.

    Parameters:
    -----------
        - `img (ndarray)` : The image to extract text from
        - `rio (tuple)` : The region of interest to extract text from

    """
    x, y, w, h = rio
    # Crop the frame to the region of interest (rio)
    img_rio = img[y : y + h, x : x + w]
    gray_frame = cv2.cvtColor(img_rio, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray_frame, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(threshold, lang="eng")
    cv2.imshow("THRESH", threshold)
    # Check if any kill-related keyword is present in the extracted text
    if any(keyword.lower() in text.lower() for keyword in keywords):
        return True, text.lower()
    return False, text.lower()


def detect_frames(
    video_path: str,
    output_video_path: str,
    buffer: FrameBuffer,
    keywords: list[str],
) -> list[str]:
    """Create a new video with frames that contain <keywords>.

    Parameters:
    -----------
        - `interval` (int): How often to check for <keywords> in the video (in frames)
        - `buffer` (int): An instance of a FrameBuffer to cache a limited number of frames
        - `keywords` (list): A list of keywords to to look for in the killfeed.

    Returns:
    --------
    - `log` (list): A Debug log
    """
    log = []
    log_template = "[{}] {} - Frame {}"  # <LOG-LEVEL> - <MESSAGE TEXT> <CURRENT FRAME>
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        msg = log_template.format("ERROR", "Failed to open ", video_path)
        cprint(msg, fg.red)
        log.append(msg)
        cap.release()
        return log

    # Extract vars from video
    count = 0
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define region of interest
    roi = (width - ROI_W, 0, ROI_W, ROI_H)

    # Define output video writer object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Temporary placeholder vars
    kill_detected = False
    name = ""
    written_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        buffer.add_frame((frame, count))
        # Check for killfeed every <INTERVAL> frames instead of each frame to save time/resources
        if count % INTERVAL == 0:
            kill_detected, name = name_in_killfeed(frame, roi, keywords)
            if kill_detected is True:
                msg = log_template.format("DETECT", "Kill found @", count)  # DEBUG
                print(f"{msg:>60}", end="\r")  # DEBUG
                log.append("\t".join([msg, name]))  # DEBUG
                # Write the past <INTERVAL> frames to the output video
                for buffered_frame, index in buffer.get_frames():
                    # This is a check to ensure that we don't write duplicate frames
                    if index not in written_frames:
                        msg = log_template.format("WRITE", "Wrote frame", index)  # DEBUG
                        cprint(msg, fg.green, end="\r")  # DEBUG
                        log.append(msg)  # DEBUG
                        out.write(buffered_frame)
                        written_frames.append(index)

                        # cv2.imshow("FRAME", buffered_frame)
                    # Debug logging
                    else:
                        msg = log_template.format("SKIPPED", "Duplicate frame", index)  # DEBUG
                        # print(msg, end="\r")  # DEBUG
                        log.append(msg)  # DEBUG
            # Debug logging
            else:
                msg = log_template.format("SKIPPED", "No kill", count)  # DEBUG
                log.append("\t".join([msg, name]))  # DEBUG
        # Debug logging
        else:
            msg = log_template.format("INFO", "Current", count)  # DEBUG
            log.append(msg)  # DEBUG

        # # DEBUG
        # cv2.putText(
        #     frame,
        #     f"Frame: {count}",
        #     (10, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.6,
        #     (0, 255, 0),
        #     2,
        # )
        # cv2.imshow("FRAME", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        count += 1
    # Flag for garbage collection
    cap.release()
    out.release()
    return log


def main(input_path: str, keywords: list[str], debug: bool) -> None:
    """Process all videos in `input_paht`.

    Parameters:
    ------------
        - `input_path (str)` : path to input videos folder
        - `keywords (liststr)` : list of keywords to search for in killfeed messages
        - `debug (bool)` : whether or not to print debug information
    """

    videos = Dir(input_path).videos
    # Error handling
    if not videos:
        raise Exception("Error: No videos found in the provided directory.")
    output_folder = os.path.join(input_path, "opencv-output")
    log_folder = os.path.join(output_folder, "logs")
    # Error handling
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)
    # Create a buffer of size BUFFER to store frames temporarily
    # <BUFFER> determines the how the number of frames to write prior to killfeed being detected
    buffer = FrameBuffer(BUFFER)
    for vid in tqdm(videos, desc="Processing Videos"):
        # File path definitions
        output_video = os.path.join(output_folder, f"cv2_{vid.filename}")
        log = detect_frames(vid.path, output_video, buffer, keywords)
        # Write debug information to file
        if debug:
            with open(os.path.join(log_folder, f"cv2_{vid.filename}.log"), "w") as logfile:
                logfile.write("\n".join(log))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, usage=f"./main.py OPTIONS PATH { {'|'.join(KEYWORDS.keys())}}"
    )
    parser.add_argument(
        "PATH",
        type=str,
        help="Path to dir containing videos.",
    )
    parser.add_argument(
        "keywords",
        choices=KEYWORDS.keys(),
        help="Keywords to search for",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    keywords = KEYWORDS[args.keywords]
    print(keywords)
    main(args.PATH, keywords, args.debug)
