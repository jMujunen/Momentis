#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from collections import namedtuple
import cv2
import moviepy
from numpy import ndarray
from decorators import exectimer
from tqdm import tqdm
from ThreadPoolHelper import Pool

import logging


try:
    from .utils import FrameBuffer, find_continuous_segments, VideoReader
except ImportError:
    from utils import FrameBuffer, find_continuous_segments, VideoReader  # type: ignore

import tesserocr
from tesserocr import PyTessBaseAPI

tesserocr.set_leptonica_log_level(tesserocr.LeptLogLevel.NONE)
tess_api = PyTessBaseAPI()
cv2.setLogLevel(1)

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Constants
INTERVAL = 30
WRITER_FPS = 60
BUFFER = 360
NULLSIZE = 300

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


dimensions = namedtuple("dimensions", ["w", "h"])
video_props = namedtuple("video_props", ["w", "h", "fps"])

ROI_SIZE = dimensions(w=800, h=200)

roi = namedtuple("region_of_interest", ["x", "y", "w", "h"])
log_template = "[{}] {} - Frame {}"

DEBUG = False


def ocr(np_image: ndarray) -> str:
    """Perform OCR on a numpy image using Tesseract.

    Args:
        np_image (ndarray): The numpy image to perform OCR on.
        tess_api (PyTessBaseAPI): The Tesseract API instance.

    Returns:
        str: The OCR result.
    """

    tess_api.SetImageBytes(
        imagedata=np_image.tobytes(),
        width=np_image.shape[1],
        height=np_image.shape[0],
        bytes_per_pixel=1,
        bytes_per_line=np_image.shape[1],
    )
    tess_api.Recognize()
    return tess_api.GetUTF8Text()


def name_in_killfeed(img: ndarray, keywords: list[str], *args: roi) -> tuple[bool, str]:
    """Check if a kill-related keyword is present in the text extracted from the ndarray.

    Parameters
    -----------
        - `img (ndarray)` : The image to extract text from
        - `keywords (list[str])` : The keywords to search for in the text
        - `*args (tuple[int])` : The region of interest(s) to extract text from in the format: x, y, w, h
    """
    threshold_frames = []
    gray_frames = []
    if args is not None:
        for arg in args:
            x, y, w, h = arg  # type: ignore
            roi = img[y : y + h, x : x + w]
            # Crop the frame to the region of interest (rio)
            gray_frames.append(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))

    threshold_frames.extend(
        list(map(lambda x: cv2.threshold(x, 60, 255, cv2.THRESH_BINARY)[1], gray_frames))
    )
    threshold_frames.extend(
        list(map(lambda x: cv2.threshold(x, 225, 255, cv2.THRESH_BINARY)[1], gray_frames))
    )
    text = " ".join(map(ocr, threshold_frames)).replace("\n", "\t").lower()
    if DEBUG:
        cv2.imshow(
            "Thresholded Frames", cv2.vconcat([*threshold_frames[2:], *threshold_frames[:2]])
        )
        cv2.waitKey(1)
    # cv2.destroyAllWindows()

    # momentis/momentis.py
    # Check if any kill-related keyword is present in the extracted text
    if any(keyword.lower() in text for keyword in keywords):
        return True, text
    return False, text


@exectimer
def relevant_frames(
    video_path: Path, buffer: FrameBuffer, keywords: list[str]
) -> tuple[list[int], int]:
    """Process a video and extracts frames that contain kill-related information.

    Parameters
        - video_path (Path): Path to the input video file.
        - buffer (FrameBuffer): Buffer object to store recently processed frames.
        - keywords (list[str]): List of keywords related to kill feeds.

    Returns
        tuple[list[int], int]: a list of continue frame sequences that contain the desired frames

    """
    reader = VideoReader(str(video_path))
    if not reader.cap.isOpened():
        print(f"Error loading {video_path}")
        reader.cap.release()
        raise FileNotFoundError
    # Define video properties
    props = video_props(
        w=int(reader.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        h=int(reader.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=round(reader.cap.get(cv2.CAP_PROP_FPS)),
    )
    num_frames = int(reader.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Define region of interest
    killfeed = roi(props.w - ROI_SIZE.w, 75, ROI_SIZE.w, ROI_SIZE.h)
    alt_roi = roi((round(props.w * 0.35)), round(props.h * 0.6), ROI_SIZE.w, ROI_SIZE.h)

    # Extract vars from video
    count = 0
    # cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    written_frames = []
    frames_to_write = set()
    with tqdm(total=num_frames / INTERVAL) as pb:
        while reader.cap.isOpened():
            ret, frames = reader.read()
            if not ret:
                break
            buffer.add_frame((frames[0], count * INTERVAL))
            logger.debug(f"Count: {count}, {count * INTERVAL}")
            text = " ".join(map(ocr, frames))
            if any(keyword.lower() in text.lower() for keyword in keywords):
                logger.debug(f"\nFOUND KILL @ {count * INTERVAL}\033[0m")
                current_frame = count * INTERVAL
                for _buffer in range(current_frame - 240, current_frame + 120):
                    if _buffer >= 0 and _buffer < num_frames:
                        frames_to_write.add(_buffer)
                    else:
                        print(f"\033[31m{_buffer}/{num_frames} is out of range\033[0m")
                for bufframe, index in buffer.get_frames():
                    if index not in written_frames:
                        written_frames.append(index)
            """
            buffer.add_frame((frame, count))

            # Check for killfeed every INTERVAL frames instead of each frame to save time/resources
            if count % INTERVAL == 0 or num_frames - count == 1:
                if "Counter" in video_path.name:
                    kill_detected, name = name_in_killfeed(frame, keywords, killfeed)
                else:
                    kill_detected, name = name_in_killfeed(frame, keywords, alt_roi, killfeed)
                logger.debug(
                    f"[{count}/{num_frames}] {'\033[32mTrue' if kill_detected else '\033[31mFalse'}\033[0m {name:<15}"
                )
                # print(
                if kill_detected:
                    msg = log_template.format("DETECT", "Kill found @", count)
                    logger.debug(f"{msg:>80}")
                    # Write the past INTERVAL frames to the output video
                    for _buffered_frame, index in buffer.get_frames():
                        if index not in written_frames:
                            msg = log_template.format("WRITE", "Wrote frame", index)
                            written_frames.append(index)
                            msg = log_template.format("SKIPPED", "Duplicate frame", index)
                else:
                    msg = log_template.format("SKIPPED", "No kill", count)
            else:
                msg = log_template.format("INFO", "Current", count)
            """
            count += 1
            pb.update()
        cv2.destroyAllWindows()
    print(f"\033[35mframes_to_write:\033[0m {len(frames_to_write)}")
    print(frames_to_write)
    return list(frames_to_write), props.fps


def is_video(filepath: str) -> bool:
    """Check if the given file path points to a video file.

    Parameters
    ----------
        filepath (str | Path): The file path to check.
    """
    return any(filepath.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)


def main(input_path: str, keywords: list[str], output_path: str | Path, debug=False) -> None:
    """Process videos and extract frames containing kill-related information.

    Args:
    ----
        - input_path (str): Path to the directory containing video files.
        - keywords (list[str]): List of keywords related to kill feeds.
        - output_path (str): Output directory path to write the processed video and frames.
        - debug (bool): If True, output a json file containing the frames written and their indices.

    """
    global DEBUG

    DEBUG = debug
    if debug:
        logger.setLevel(logging.DEBUG)

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

    num_output_vids = 0
    count = 0

    for vid in videos:
        count += 1
        print(
            f"\033[32m============= Processing {vid.name} - {count}/{len(videos)} ===============\033[0m"
        )
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
                continuous_frames, fps = relevant_frames(vid, buffer, keywords)
                if len(continuous_frames) == 0:
                    continue
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"Error processing continuous frames {vid.name}: {e}")
                continue

            # Extract continuous frame sequences from set of frames
            segments = sorted(find_continuous_segments(sorted(continuous_frames)))
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
                        temp_audiofile_path="/tmp/",
                        threads=18,
                        # audio_codec="aac",
                        remove_temp=True,
                        audio=True,
                        ffmpeg_params=[
                            "-y",
                            "-loglevel",
                            "error",
                            "-pix_fmt",
                            "yuv420p",
                        ],
                    )
                except Exception as e:
                    print(f"Error writing subclips {vid.name}: {e}")
            else:
                continue
            num_output_vids += 1
            clip.close()
            if debug:
                # Write sequence to file for debugging
                log_file = Path(output_folder, f".{output_video.name}.log")
                log_file.write_text(json.dumps(segments))
        except Exception as e:
            print(f"Error processing video {vid.name}: {e}")
        print()
    print(f"Num input videos: {len(videos)}")
    print(f"Num output videos: {num_output_vids}")
    print("Done.")


if __name__ == "__main__":
    main(
        input_path="/mnt/win_ssd/Users/Joona/Videos/NVIDIA/PLAYERUNKNOWN'S BATTLEGROUNDS",
        output_path="/tmp/vids/out",
        debug=True,
        keywords=[
            "sofunny",
            "meso",
            "solunny",
            "hoff",
            "ffman",
            "bartard",
            "dankniss",
            "vermeme",
            "nissev",
            "knocked",
            "ocked",
            "headshot",
            "you",
        ],
    )
    main(
        input_path="/tmp/vids/in",
        output_path="/tmp/vids/out",
        debug=True,
        keywords=[
            "sofunny",
            "meso",
            "solunny",
            "hoff",
            "ffman",
            "bartard",
            "dankniss",
            "vermeme",
            "nissev",
            "knocked",
            "ocked",
            "headshot",
            "you",
        ],
    )
