#!/usr/bin/env python3

import argparse
import os

import cv2
import ExecutionTimer
import pytesseract
from Color import cprint, fg
from fsutils import Dir
from numpy import ndarray
from ProgressBar import ProgressBar

from FrameBuffer import FrameBuffer

# Constants
INTERVAL = 120
WRITER_FPS = 60
BUFFER = 120
# List of keywords related to kill feeds
KEYWORDS = [
    # "Hoffman",
    # "itsgroovybabe",
    # "Mesofunny",
    # "Bartarded",
    "you",
]


def name_in_killfeed(img: ndarray) -> bool:  # , cuda_image: cv2.cuda.GpuMat) -> bool:
    """Check if a kill-related keyword is present in the text extracted from an image frame."""
    # cuda_image.upload(img)
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray_frame, 150, 255, cv2.THRESH_BINARY)
    # Download image from GPU to CPU
    # cpu_threshold = threshold.download()
    text = pytesseract.image_to_string(threshold)

    # Check if any kill-related keyword is present in the extracted text
    return any(keyword.lower() in text.lower() for keyword in KEYWORDS)


def detect_frames(video_path: str, buffer: FrameBuffer) -> list[ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cprint(f"Error opening {video_path}", fg.red)
        return []
    # frame_gpu = cv2.cuda.GpuMat()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    killframes: list[ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Check for killfeed every `INTERVAL` frames instead of each frame to save time/resources
        buffer.add_frame(frame)
        if count % INTERVAL == 0 and name_in_killfeed(frame):
            killframes.extend(buffer.get_frames())
            print(f"Kill found @ {count}/{total_frames}")

        count += 1
    cap.release()
    # buffer.release()

    return killframes


def video_from_frames(frame_list: list[ndarray], video_path: str, output_path: str):
    # Get the total number of frames in the original video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        for frame in frame_list:
            out.write(frame)
    cap.release()
    out.release()


def main(input_path: str) -> None:
    videos = Dir(input_path).videos
    if not videos:
        raise Exception("Error: No videos found in the provided directory.")
    output_folder = os.path.join(input_path, "opencv-output")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    buffer = FrameBuffer(BUFFER)
    with ProgressBar(len(videos)) as progress:
        with ExecutionTimer.ExecutionTimer():
            for vid in videos:
                frames_of_interest = detect_frames(vid.path, buffer)
                num_interesting_frames = len(frames_of_interest)
                if num_interesting_frames == 0:
                    continue
                try:
                    print(f"Found {num_interesting_frames} FOI. Writing to file...")
                    output_path = os.path.join(output_folder, f"cvTrimmed_{vid.basename}")
                    video_from_frames(frames_of_interest, vid.path, output_path)
                except Exception as e:
                    cprint(f"Failed processing {vid.basename}\n{e!r}", fg.red)
            progress.increment()
            print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "VIDEO",
        type=str,
        help="Path to dir containing videos.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.VIDEO)
