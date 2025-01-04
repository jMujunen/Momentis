#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from collections import namedtuple
import cv2
import moviepy

from numpy cimport ndarray
from cpython cimport bool
from ._FrameBuffer cimport FrameBuffer

ctypedef unsigned char[:, :, :] Frame

# Constants
cdef int INTERVAL = 120
cdef int WRITER_FPS = 60
cdef int BUFFER = 240
cdef int ROI_W, ROI_H, ALT_W, ALT_H, NULLSIZE
cdef public str log_template = "[{}] {} - Frame {}"

ROI_W, ROI_H = (800, 200)
ALT_W, ALT_H = (800, 200)

MONITOR_DIMS = (1920, 1080)
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

NULLSIZE = 300

dimensions = namedtuple("dimensions", ["w", "h"])
video_props = namedtuple("video_props", ["w", "h", "fps"])

ROI = dimensions(w=800, h=200)

region_of_interest = namedtuple("region_of_interest", ["x", "y", "w", "h"])



cpdef tuple[list[int], int] relevant_frames(str video_path, FrameBuffer buffer, list[str] keywords):# -> tuple[list[int], int]:
    """Process a video and extracts frames that contain kill-related information.

    Parameters
        - video_path (str): Path to the input video file.
        - buffer (FrameBuffer): Buffer object to store recently processed frames.
        - keywords (list): List of keywords related to kill feeds.

    Returns
        tuple: a list of continue frame sequences that contain the desired frames

    """

    cdef int count = 0
    cdef int i = 0
    cdef list[int] written_frames = []
    cdef int ret
    cdef Frame frame
    cdef int _index
    cdef tuple[ndarray, int] item

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error loading {video_path}")
        cap.release()
        raise FileNotFoundError

    # Define video properties
    props = video_props(
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=round(cap.get(cv2.CAP_PROP_FPS)),
    )
    # Define region of interest
    killfeed = region_of_interest(props.w - ROI.w, 75, ROI.w, ROI.h)
    alt_roi = region_of_interest((round(props.w * 0.35)), round(props.h * 0.6), ROI.w, ROI.h)


    # Extract vars from video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        buffer.add_frame(count)
        # Check for killfeed every INTERVAL frames instead of each frame to save time/resources
        if i == INTERVAL:
            i = 0
            for _index in buffer.exec(keywords, killfeed):
                msg = log_template.format("DETECT", "Kill found @", count)
                print(f"{msg:>100}", end="\r")
                # Write the past INTERVAL frames to the output video
                if _index not in written_frames:
                    msg = log_template.format("WRITE", "Wrote frame", _index)
                    written_frames.append(_index)
                else:
                    msg = log_template.format("SKIPPED", "Duplicate frame", _index)
                print(f"{msg:<40}", end="\r")
        else:
            msg = log_template.format("INFO", "Current", count)
            print(f"{msg:<40}", end="\r")

        count += 1
        i +=1
    cv2.destroyAllWindows()
    return written_frames, props.fps


def is_video(filepath: str) -> bool:
    """Check if the given file path points to a video file.

    Parameters
    ----------
        filepath (str | Path): The file path to check.
    """
    return any(filepath.endswith(ext) for ext in VIDEO_EXTENSIONS) # type: ignore


def main(
    str input_path,
    list[str] keywords,
    bool debug=False, # type: ignore
    str output_path="./opencv_output",
) -> None:
    """Process videos and extract frames containing kill-related information.

    Parameters
        - input_path (str): Path to the directory containing video files.
        - keywords (list): List of keywords related to kill feeds.
        - debug (bool): If True, output a json file containing the frames written and their indices.
        - output_path (str): Output directory path to write the processed video and frames.
    """
    cdef list[object] videos
    cdef FrameBuffer buffer
    cdef list[int] continuous_frames
    cdef int fps
    cdef list[list[int]] segments

    input_dir = Path(input_path)
    videos = [
        Path(root, file) for root, _, files in input_dir.walk() for file in files if is_video(file)
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
            output_video = Path(output_folder, f"cv2_{vid.name}")
            # Check if video is already processed or corrupted
            if output_video.exists() and output_video.stat().st_size > NULLSIZE:
                print("Skipping existing video...")
                continue
            if output_video.exists() and output_video.stat().st_size < NULLSIZE:
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
            # # Extract continuous frame sequences from set of frames
            # segments = sorted(find_continuous_segments(continuous_frames))
            # clip = moviepy.VideoFileClip(str(vid))
            # # Get the audio from the original video
            # audio = clip.audio
            # # Create a single video clip for each segment of continuous frames
            # subclips = [
            #     clip.subclipped(segment[0] / fps, segment[-1] / fps) for segment in segments
            # ]
            # if len(subclips) > 0:
            #     try:
            #         # Write the video and audio to a new file
            #         final_clip = moviepy.concatenate_videoclips(subclips, method="compose")
            #         final_clip.write_videofile(
            #             str(output_video),
            #             codec="libx265",
            #             audio_codec="aac",
            #             remove_temp=True,
            #             audio=True,
            #         )
            #     except Exception as e:
            #         print(f"Error writing subclips {vid.name}: {e}")
            # clip.close()
            # if debug:
            #     # Write sequence to file for debugging
            #     log_file = Path(output_folder, f".{output_video.name}.log")
            #     log_file.write_text(json.dumps(segments))
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
        word
        for word in keywords_file.read_text().splitlines()
        if not word.startswith(("#", ";", "/"))
    ]

    input_path = Path(args.PATH)
    archive_path = Path(str(input_path).replace("ssd", "hdd"))

    main(input_path=str(input_path), keywords=keywords, debug=args.debug)


cdef list[list[int]] find_continuous_segments(list[int] frames_index):# -> list[list[int]]:
    """Find continuous segments of frames.

    Args:
        frames (list[int]): A list of integers representing frames.

    Returns:

        list[list[int]]: A list of lists, where each sublist represents a continuous segment of frames.
    """
    if not frames_index:
        return []

    segments = [[frames_index[0]]]
    for i in range(1, len(frames_index)):
        if frames_index[i] == frames_index[i - 1] + 1:
            segments[-1].append(frames_index[i])
        else:
            segments.append([frames_index[i]])
    return segments
