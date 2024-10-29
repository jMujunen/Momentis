#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import moviepy.editor as mp
import pytesseract
from numpy import ndarray
from utils import FrameBuffer, find_continuous_segments

# Constants
INTERVAL = 30
WRITER_FPS = 60
BUFFER = 240
ROI_W, ROI_H = (800, 200)
ALT_W, ALT_H = (800, 200)

MONITOR_DIMS = (1920, 1080)


def name_in_killfeed(img: ndarray, keywords: list[str], *args: tuple[int, ...]) -> tuple[bool, str]:
    """Check if a kill-related keyword is present in the text extracted from the ndarray.

    Parameters
    -----------
        - `img (ndarray)` : The image to extract text from
        - `keywords (list[str])` : The keywords to search for in the text
        - `*args (tuple[int])` : The region of interest(s) to extract text from in the format: x, y, w, h
    """
    preprocessed_frames = []
    if args is not None:
        for arg in args:
            x, y, w, h = arg  # type: ignore
            roi = img[y : y + h, x : x + w]
            # Crop the frame to the region of interest (rio)
            gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            preprocessed_frames.append(cv2.threshold(gray_frame, 175, 255, cv2.THRESH_BINARY)[1])
            preprocessed_frames.append(
                cv2.threshold(gray_frame, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            )

    concatted_img = cv2.hconcat(preprocessed_frames)
    text = pytesseract.image_to_string(concatted_img, lang="eng")

    # Overlay ROIs on the original frame for debugging
    # for arg in args:
    # x, y, w, h = arg  # type: ignore
    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # if img.shape[:2] > MONITOR_DIMS:
    # img = cv2.resize(img, MONITOR_DIMS)
    # cv2.imshow("Frame", img)

    # Check if any kill-related keyword is present in the extracted text
    if any(keyword.lower() in text.lower() for keyword in keywords):
        return True, text.lower()
    return False, text.lower()


def relevant_frames(video_path: Path, buffer: FrameBuffer, keywords: list) -> tuple[list[int], int]:
    """Process a video and extracts frames that contain kill-related information.

    Parameters
        - video_path (str): Path to the input video file.
        - buffer (FrameBuffer): Buffer object to store recently processed frames.
        - keywords (list): List of keywords related to kill feeds.

    Returns
        A tuple containing a list of frame indices that contain kill-related information and the original FPS of the video.

    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error loading {video_path}")
        cap.release()
        raise FileNotFoundError

    log_template = "[{}] {} - Frame {}"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define region of interest
    killfeed_roi = (width - ROI_W, 75, ROI_W, ROI_H)
    alt_x, alt_y = (round(width * 0.35), round(height * 0.6))
    alternative_roi = (alt_x, alt_y, ALT_W, ALT_H)

    # Extract vars from video
    count = 0
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    written_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        buffer.add_frame((frame, count))
        # Check for killfeed every INTERVAL frames instead of each frame to save time/resources
        if count % INTERVAL == 0:
            kill_detected, name = name_in_killfeed(frame, keywords, alternative_roi, killfeed_roi)
            cv2.waitKey(1)
            if kill_detected is True:
                msg = log_template.format("DETECT", "Kill found @", count)
                print(f"{msg:>100}", end="\r")
                # Write the past INTERVAL frames to the output video
                for _buffered_frame, index in buffer.get_frames():
                    if index not in written_frames:
                        msg = log_template.format("WRITE", "Wrote frame", index)
                        written_frames.append(index)
                        msg = log_template.format("SKIPPED", "Duplicate frame", index)
                        print(f"{msg:<50}", end="\r")
            else:
                msg = log_template.format("SKIPPED", "No kill", count)
        else:
            msg = log_template.format("INFO", "Current", count)
            print(f"{msg:<40}", end="\r")

        count += 1
    cv2.destroyAllWindows()
    return written_frames, fps


def main(input_path: str, keywords: list[str], debug=False) -> None:
    """Process videos and extract frames containing kill-related information.

    Parameters
        - input_path (str): Path to the directory containing video files.
        - keywords (list): List of keywords related to kill feeds.
        - debug (bool): If True, output a json file containing the frames written and their indices.
    """
    exts = [".mp4", ".avi", ".MOV", ".mkv"]
    input_dir = Path(input_path)
    videos = [
        Path(root, file)
        for root, _, files in input_dir.walk()
        for file in files
        if file[-4:].lower() in exts
    ]
    # Error handling
    if not videos:
        raise Exception("Error: No videos found in the provided directory.")

    # Folder creation
    output_folder = Path(input_path, "opencv-output")
    output_folder.mkdir(parents=True, exist_ok=True)

    # Create a buffer of size BUFFER to store frames temporarily
    # <BUFFER> determines the how many frames to write prior to killfeed being detected
    buffer = FrameBuffer(BUFFER)
    for vid in videos:
        try:
            # File path definitions
            output_video = Path(output_folder, f"cv2_{vid.name}")
            try:
                continuous_frames, fps = relevant_frames(vid, buffer, keywords)
                if len(continuous_frames) == 0:
                    continue
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"Error processing continuous frames {vid.name}: {e}")
                continue
            # Extract continuous frame sequences from set of frames
            segments = sorted(find_continuous_segments(continuous_frames))
            clip = mp.VideoFileClip(str(vid))
            # Get the audio from the original video
            audio = clip.audio

            # Create a single video clip for each segment of continuous frames
            subclips = [clip.subclip(segment[0] / fps, segment[-1] / fps) for segment in segments]
            if len(subclips) > 0:
                try:
                    # Write the video and audio to a new file
                    final_clip = mp.concatenate_videoclips(subclips, method="compose")
                    final_clip.write_videofile(
                        str(output_video), codec="libx264", audio_codec="aac", remove_temp=True
                    )
                except Exception as e:
                    print(f"Error writing subclips {vid.name}: {e}")
            clip.close()
            if debug:
                # Write sequence to file for debugging
                log_file = Path(output_folder, f"{vid.name}-frames.log")
                log_file.write_text(json.dumps(segments))
        except Exception as e:
            print(f"Error processing video {vid.name}: {e}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("PATH", help="Path to videos", type=str)
    parser.add_argument("--debug", help="Enable debug mode", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    module_path = Path(__file__)
    args = parse_args()
    keywords_file = Path(module_path.parent, "keywords.txt")
    keywords = keywords_file.read_text().splitlines()
    main(args.PATH, keywords, args.debug)
