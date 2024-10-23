#!/usr/bin/env python3
"""Create a new new video containing only relevant frames determined by the killfeed."""

import argparse
import os
from pathlib import Path

import cv2
from Color import cprint, fg, style
from config import (
    BUFFER,
    FPS as WRITER_FPS,
    INTERVAL,
    KEYWORDS,
    ROI,
)
from FrameBuffer import FrameBuffer
from fsutils import Dir
from ThreadPoolHelper import Pool

from .utils import name_in_killfeed

ROI_W, ROI_H = ROI


def detect_frames(
    vid_path: str, output_path: str, buffer: FrameBuffer, keywords: list[str]
) -> list[str]:
    """Create a new video with frames that contain <keywords>.

    Parameters
    -----------
        - `interval` (int): How often to check for <keywords> in the video (in frames)
        - `buffer` (int): An instance of a FrameBuffer to cache a limited number of frames
        - `keywords` (list): A list of keywords to to look for in the killfeed.

    Returns
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

    # Extract vars from video
    count = 0
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    roi = (width - ROI_W, 0, ROI_W, ROI_H)
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
        # Check for killfeed every <INTERVAL> frames instead of each frame, to save time/resources
        if count % INTERVAL == 0:
            kill_detected, name = name_in_killfeed(frame, keywords, roi)
            if kill_detected is True:
                msg = log_template.format("DETECT", "Kill found @", count)  # DEBUG
                print(f"{msg:>100}", end="\r")  # DEBUG
                log.append("\t".join([msg, name]))  # DEBUG
                # Write the past <CONSTS["INTERVAL"]> frames to the output video
                for buffered_frame, index in buffer.get_frames():
                    # Check to ensure that we don't write duplicate frames
                    if index not in written_frames:
                        msg = log_template.format(
                            f"{fg.green}WRITE{style.reset}", "Wrote frame", index
                        )  # DEBUG
                        cprint(f"{msg:40}", fg.green, end="\r")  # DEBUG
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
                print(f"{msg:>100}", end="\r")  # DEBUG
        # Debug logging
        else:
            msg = log_template.format("INFO", "Current", f"{count}/{total_frames}")  # DEBUG
            log.append(msg)  # DEBUG
            print(f"{msg:<40}", end="\r")  # DEBUG

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
        # cv2.waitKey(10)
        count += 1
    # Flag for garbage collection
    cap.release()
    out.release()
    return log


def main(input_path: str, keywords: list[str], debug: bool) -> None:
    """Process all videos in `input_paht`.

    Parameters
    ------------
        - `input_path (str)` : path to input videos folder
        - `keywords (list[str])` : list of keywords to search for in killfeed messages
        - `debug (bool)` : whether or not to print debug information
    """

    videos = Dir(input_path).videos
    # Error handling
    if not videos:
        raise Exception("Error: No videos found in the provided directory.")
    output_folder = Path(input_path, "opencv-output")
    log_folder = Path(output_folder, "logs")
    # Error handling
    output_folder.mkdir(parents=True, exist_ok=True)
    log_folder.mkdir(parents=True, exist_ok=True)

    # Create a buffer of size BUFFER to store frames temporarily
    # <BUFFER> determines the how the number of frames to write prior to killfeed being detected
    buffer = FrameBuffer(BUFFER)
    # pb = ProgressBar(len(videos))
    buffer = FrameBuffer(BUFFER)
    # pb = ProgressBar(len(videos))
    for vid in videos:
        # File path definitions
        output_video = Path(output_folder, f"cv2_{vid.name}")
        log = detect_frames(vid.path, output_video, buffer, keywords)
        # Write debug information to file
        if debug:
            logfile = Path(log_folder, f"cv2_{vid.name}.log")
            logfile.write_text("\n".join(log))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        usage=f"./main.py OPTIONS PATH { {'|'.join(KEYWORDS.keys())}}",
    )
    parser.add_argument(
        "PATH",
        type=str,
        help="Path to dir containing videos.",
    )
    parser.add_argument(
        "--keywords",
        choices=KEYWORDS.keys(),
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
    keywords = KEYWORDS[args.keywords]
    print(keywords)
    main(args.PATH, keywords, args.debug)
