#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import cv2
import moviepy
import pytesseract
from _FrameBuffer cimport FrameBuffer


from .utils import find_continuous_segments  # type: ignore

ctypedef unsigned char[:, :, :] Frame
ctypedef (unsigned short int, unsigned short int, unsigned short int, unsigned short int) ROI
cdef public tuple[unsigned int, unsigned int] MONITOR_DIMS
cdef public set[str] VIDEO_EXTENSIONS

# Constants
cdef public unsigned int INTERVAL, WRITER_FPS, BUFFER, ROI_W, ROI_H, ALT_W, ALT_H

WRITER_FPS = 60
BUFFER = 240
ROI_W, ROI_H = (800,200)
ALT_W, ALT_H = (800, 200)

MONITOR_DIMS = (1920, 1080)
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


cdef bint name_in_killfeed(frame, list[str] keywords, roi_1):
    """Check if a kill-related keyword is present in the text extracted from the np.ndarray.

    Parameters
    -----------
        - `img (np.ndarray)` : The image to extract text from
        - `keywords (list[str])` : The keywords to search for in the text
        - `*args (tuple[int])` : The region of interest(s) to extract text from in the format: x, y, w, h
    """
    cdef str text, keyword
    # cdef Frame gray_frame, cropped_roi, concatted_img, thresh_1, thresh_2
    cdef unsigned short int x, t, w, h

    x, y, w, h = roi_1
    cropped_roi = frame[y : y + h, x : x + w] # type: ignore
    # Crop the frame to the region of interest (rio)
    gray_frame = cv2.cvtColor(cropped_roi, cv2.COLOR_BGR2GRAY)
    thresh_1 = cv2.threshold(gray_frame, 175, 255, cv2.THRESH_BINARY)[1]
    # thresh_2 = cv2.threshold(gray_frame, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


    # concatted_img = cv2.hconcat((thresh_1, thresh_2))
    text = pytesseract.image_to_string(thresh_1, lang="eng")
    # Check if any kill-related keyword is present in the extracted text
    if any(keyword.lower() in text.lower() for keyword in keywords):
        return True # type: ignore
    return False # type: ignore

cdef tuple[list[int], unsigned int] relevant_frames(str video_path, FrameBuffer buffer, list[str] keywords): # -> tuple[list[int], int]:
    """Process a video and extracts frames that contain kill-related information.

    Parameters
        - video_path (str): Path to the input video file.
        - buffer (FrameBuffer): Buffer object to store recently processed frames.
        - keywords (list): List of keywords related to kill feeds.

    Returns
        tuple: a list of continue frame sequences that contain the desired frames

    """
    cdef str log_template = "[{}] {} - Frame {}"
    cdef unsigned int width, height, fps, alt_x, alt_y
    cdef unsigned short int count = 0
    cdef bint kill_detected
    cdef list[int] written_frames = []
    cdef ROI killfeed_roi, alternative_roi
    cdef tuple[str, int] name
    cdef Frame frame
    cdef str msg

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error loading {video_path}")
        cap.release()
        raise FileNotFoundError



    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))

    # Define region of interest
    killfeed_roi = (width - ROI_W, 75, ROI_W, ROI_H)
    alt_x, alt_y = (round(width * 0.35), round(height * 0.6))
    alternative_roi = (alt_x, alt_y, ALT_W, ALT_H)

    # Extract vars from video
    count = 0
    # cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    written_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        buffer.add_frame(frame, count)
        # Check for killfeed every INTERVAL frames instead of each frame to save time/resources
        if count % INTERVAL == 0:
            # cv2.waitKey(1)
            if name_in_killfeed(frame, keywords, killfeed_roi) is True:
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


cdef inline bint is_video(str filepath):
    """Check if the given file path points to a video file.

    Parameters
    ----------
        filepath (str | Path): The file path to check.
    """
    return any(filepath.endswith(ext) for ext in VIDEO_EXTENSIONS) # type: ignore


def main(
    str input_path,
    list[str] keywords,
    bint debug=False,
    str output_path = "./opencv_output",
) -> None:
    """Process videos and extract frames containing kill-related information.

    Parameters
        - input_path (str): Path to the directory containing video files.
        - keywords (list): List of keywords related to kill feeds.
        - debug (bool): If True, output a json file containing the frames written and their indices.
        - output_path (str): Output directory path to write the processed video and frames.
    """
    cdef FrameBuffer buffer
    cdef list[object] segments
    cdef object clip
    cdef object final_clip
    cdef object log_file
    cdef object output_video
    cdef int fps
    cdef list[object] continuous_frames
    cdef list[object] subclips
    cdef list[object] frames


    input_dir = Path(input_path)
    videos = [
        Path(root, file) for root, _, files in os.walk(input_path) for file in files if is_video(file)
    ]
    if not videos:
        print("Error: No videos found in the provided directory.")
        return

    # Folder creation
    output_folder = Path(output_path).resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    # Create a buffer of size BUFFER to store frames temporarily
    # <BUFFER> determines the how many frames to write prior to killfeed being detected
    buffer = FrameBuffer(BUFFER)
    for vid in videos:
        try:
            # File path definitions
            output_video = Path(output_path, f"cv2_{vid.name}")
            # Check if video is already processed or corrupted
            if output_video.exists() and output_video.stat().st_size > 300:
                print("Skipping existing video...")
                continue
            if output_video.exists() and output_video.stat().st_size < 300:
                output_video.unlink()
            try:
                continuous_frames, fps = relevant_frames(str(vid), buffer, keywords)
                if len(continuous_frames) == 0:
                    continue
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"Error processing continuous frames {vid.name}: {e}")
                continue
            # Extract continuous frame sequences from set of frames
            segments = sorted(find_continuous_segments(continuous_frames))
            clip = moviepy.VideoFileClip(str(vid))
            # Get the audio from the original video
            audio = clip.audio
            # Create a single video clip for each segment of continuous frames
            subclips = [
                clip.subclipped(segment[0] / fps, segment[-1] / fps) for segment in segments
            ]
            if len(subclips) > 0:
                try:
                    # Write the video and audio to a new file
                    final_clip = moviepy.concatenate_videoclips(subclips, method="compose")
                    final_clip.write_videofile(
                        str(output_video),
                        codec="libx264",
                        audio_codec="aac",
                        remove_temp=True,
                        audio=True,
                    )
                except Exception as e:
                    print(f"Error writing subclips {vid.name}: {e}")
            clip.close()
            if debug:
                # Write sequence to file for debugging
                log_file = Path(output_folder, f".{output_video.name}.log")
                log_file.write_text(json.dumps(segments))
        except Exception as e:
            print(f"Error processing video {vid.name}: {e}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("PATH", help="Path to videos", type=str)
    parser.add_argument("--output", help="Output folder")
    parser.add_argument(
        "--archive",
        help="Move original videos to this path after processing",
    )
    parser.add_argument("--debug", help="Enable debug mode", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    module_path = Path(__file__)
    args = parse_args()
    keywords_file = Path(module_path.parent, "keywords.txt")
    if any((not keywords_file.exists(), not keywords_file.read_text())):
        print("Error: keywords file not found")
        raise FileNotFoundError

    keywords = [
        word for word in keywords_file.read_text().splitlines() if word[0] not in {"#", ";", "/"}
    ]

    input_path = Path(args.PATH)
    archive_path = Path(str(input_path).replace("ssd", "hdd"))

    main(input_path=input_path, keywords=keywords, debug=args.debug)
