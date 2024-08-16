import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from Color import cprint, fg, style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames and encode them in parallel")
    parser.add_argument("VIDEO", type=str, help="Path to the video file")
    parser.add_argument("PATH", type=str, help="Directory to save the encoded frames")
    args = parser.parse_args()
    return args


def encode_frame(frame: np.ndarray) -> str:
    """Encodes a frame into a base64 string.

    Parameters:
        frame (np.ndarray): The frame to be encoded.

    Returns:
        str: A base64 encoded string representing the frame in JPEG format.
    """
    encoded_frame = cv2.imencode(".jpg", frame)[1].tobytes()
    base64_frame = base64.b64encode(encoded_frame).decode("utf-8")
    return base64_frame


def main(video_path: str, output_dir: str | Path) -> int:
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    cap.release()
    with ThreadPoolExecutor() as executer:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                executer.submit(encode_frame, frame).add_done_callback(
                    lambda future: write_frame(future.result(), output_dir, frame_count)
                )
                frame_count += 1
            else:
                break

    cap.release()
    return 0


def write_frame(base64_frame: str, output_dir: str | Path, frame_count: int) -> int:
    filename = f"{output_dir}/frame_{frame_count}.jpg"
    try:
        with open(filename, "w") as f:
            f.write(base64_frame)
        return 0
    except Exception as e:
        print(f"Error writing frame {frame_count} to file {filename}: {e}")
        return 1


if __name__ == "__main__":
    args = parse_args()
    main(args.VIDEO, args.PATH)
