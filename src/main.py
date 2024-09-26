#!/usr/bin/env python3
"""Create a new new video containing only relevant frames determined by the killfeed."""

import argparse
import os

import cv2
from Color import cprint, fg, style
from fsutils import Dir
from ProgressBar import ProgressBar

from FrameBuffer import FrameBuffer

from .config import (
    FPS as WRITER_FPS,
    INTERVAL,
    KEYWORDS,
    ROI,
)
from .utils import name_in_killfeed


def detect_frames(
    vid_path: str, output_path: str, buffer: FrameBuffer, keywords: list[str]
) -> list[str]:
    """Create a new video with frames that contain <keywords>.

    Parameters:
    -----------
        - `interval` (int): How often to check for <keywords> in the video (in frames)
        - `buffer` (int): An instance of a FrameBuffer to cache a limited number of frames
        - `keywords` (list): A list of keywords to to look for in the killfeed.

    Returns:
    --------
        - `log (list)` : A Debug log
    """
    log = []
    log_template = "[{}] {} - Frame {}"  # <LOG-LEVEL> - <MESSAGE TEXT> <CURRENT FRAME>
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        msg = log_template.format("ERROR", "Failed to open ", vid_path)
        cprint(msg, fg.red)
        log.append(msg)
        cap.release()
        return log

    # Extract vars from video(\w+)\s=\sR
    count = 0
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define output video writer object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Temporary placeholder vars
    kill_detected = False
    name = ""
    written_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        buffer.add_frame((frame, count))
        # Check for killfeed every <CONSTS["INTERVAL"]> frames instead of each frame to save time/resources
        if count % CONSTS["INTERVAL"] == 0:
            kill_detected, name = name_in_killfeed(frame, keywords)
            if kill_detected is True:
                msg = log_template.format("DETECT", "Kill found @", count)  # DEBUG
                print(f"{msg:>60}", end="\r")  # DEBUG
                log.append("\t".join([msg, name]))  # DEBUG
                # Write the past <CONSTS["INTERVAL"]> frames to the output video
                for buffered_frame, index in buffer.get_frames():
                    # Check to ensure that we don't write duplicate frames
                    if index not in written_frames:
                        msg = log_template.format(
                            f"{fg.green}WRITE{style.reset}", "Wrote frame", index
                        )  # DEBUG
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
                print(f"{msg:60}", end="\r")  # DEBUG
        # Debug logging
        else:
            msg = log_template.format("INFO", "Current", count)  # DEBUG
            log.append(msg)  # DEBUG
            # print(msg, end="\r")  # DEBUG

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
        - `keywords (list[str])` : list of keywords to search for in killfeed messages
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
    buffer = FrameBuffer(CONSTS["BUFFER"])
    pb = ProgressBar(len(videos))
    for vid in videos:
        # File path definitions
        output_video = os.path.join(output_folder, f"cv2_{vid.basename}")
        log = detect_frames(vid.path, output_video, buffer, keywords)
        # Write debug information to file
        if debug:
            with open(os.path.join(log_folder, f"cv2_{vid.filename}.log"), "w") as logfile:
                logfile.write("\n".join(log))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        usage=f"./main.py OPTIONS PATH { {'|'.join(CONSTS["KEYWORDS"].keys())}}",
    )
    parser.add_argument(
        "PATH",
        type=str,
        help="Path to dir containing videos.",
    )
    parser.add_argument(
        "--keywords",
        choices=CONSTS["KEYWORDS"].keys(),
        help="Keywords to search for",
        type=str,
        # default=KEYWORDS["hoff"],
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    keywords = CONSTS["KEYWORDS"][args.keywords]
    print(keywords)
    main(args.PATH, keywords, args.debug)
