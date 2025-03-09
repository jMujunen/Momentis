#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from collections import namedtuple
import cv2
import moviepy
from numpy import ndarray

try:
    from .utils import FrameBuffer, find_continuous_segments  # type: ignore
except ImportError:
    from utils import FrameBuffer, find_continuous_segments
from ProgressBar import ProgressBar  # Update deps to include this custom module
from tesserocr import PyTessBaseAPI
from decorators import exectimer

tess_api = PyTessBaseAPI()

# Constants
INTERVAL = 60
WRITER_FPS = 60
BUFFER = 240
ROI_W, ROI_H = (800, 200)
ALT_W, ALT_H = (800, 200)

MONITOR_DIMS = (1920, 1080)
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

NULLSIZE = 300

dimensions = namedtuple("dimensions", ["w", "h"])
video_props = namedtuple("video_props", ["w", "h", "fps"])

ROI = dimensions(w=800, h=200)

region_of_interest = namedtuple("region_of_interest", ["x", "y", "w", "h"])
log_template = "[{}] {} - Frame {}"


def ocr(np_image: ndarray) -> str:
    """Perform OCR on a numpy image using Tesseract.

    Args:
        np_image (ndarray): The numpy image to perform OCR on.
        tess_api (PyTessBaseAPI): The Tesseract API instance.

    Returns:
        str: The OCR result.
    """

    # bpp = 3 if len(np_image.shape) > 2 else 1
    # bpl = bpp * w

    tess_api.SetImageBytes(
        imagedata=np_image.tobytes(),
        width=np_image.shape[1],
        height=np_image.shape[0],
        bytes_per_pixel=1,
        bytes_per_line=np_image.shape[1],
        # bytes_per_line=bpl,
    )
    tess_api.Recognize()
    return tess_api.GetUTF8Text()


def name_in_killfeed(
    img: ndarray, keywords: list[str], *args: region_of_interest
) -> tuple[bool, str]:
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
    # text = pytesseract.image_to_string(concatted_img, lang="eng")
    text = ocr(concatted_img)
    # Check if any kill-related keyword is present in the extracted text
    if any(keyword.lower() in text.lower() for keyword in keywords):
        return True, text.lower()
    return False, text.lower()


@exectimer
def relevant_frames(video_path: Path, buffer: FrameBuffer, keywords: list) -> tuple[list[int], int]:
    """Process a video and extracts frames that contain kill-related information.

    Parameters
        - video_path (str): Path to the input video file.
        - buffer (FrameBuffer): Buffer object to store recently processed frames.
        - keywords (list): List of keywords related to kill feeds.

    Returns
        tuple: a list of continue frame sequences that contain the desired frames

    """
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
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pb = ProgressBar(num_frames)
    # Define region of interest
    killfeed = region_of_interest(props.w - ROI.w, 75, ROI.w, ROI.h)
    alt_roi = region_of_interest((round(props.w * 0.35)), round(props.h * 0.6), ROI.w, ROI.h)

    # Extract vars from video
    count = 0
    # cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    written_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        buffer.add_frame((frame, count))
        # Check for killfeed every INTERVAL frames instead of each frame to save time/resources
        if count % INTERVAL == 0:
            kill_detected, name = name_in_killfeed(frame, keywords, alt_roi, killfeed)

            # cv2.waitKey(1)
            if kill_detected is True:
                msg = log_template.format("DETECT", "Kill found @", count)
                print(f"{msg:>100}", end="\r")
                # Write the past INTERVAL frames to the output video
                for _buffered_frame, index in buffer.get_frames():
                    if index not in written_frames:
                        msg = log_template.format("WRITE", "Wrote frame", index)
                        written_frames.append(index)
                        msg = log_template.format("SKIPPED", "Duplicate frame", index)
                        # print(f"{msg:<50}", end="\r") # debug
            else:
                msg = log_template.format("SKIPPED", "No kill", count)
        else:
            msg = log_template.format("INFO", "Current", count)
            # print(f"{msg:<40}", end="\r") # debug

        count += 1
        pb.increment()
    cv2.destroyAllWindows()
    return written_frames, props.fps


def is_video(filepath: str) -> bool:
    """Check if the given file path points to a video file.

    Parameters
    ----------
        filepath (str | Path): The file path to check.
    """
    return any(filepath.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)


def main(
    input_path: str,
    keywords: list[str],
    output_path: str | Path,
    debug=False,
) -> None:
    """Process videos and extract frames containing kill-related information.

    Parameters
        - input_path (str): Path to the directory containing video files.
        - keywords (list): List of keywords related to kill feeds.
        - debug (bool): If True, output a json file containing the frames written and their indices.
        - output_path (str): Output directory path to write the processed video and frames.
    """

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
                        codec="hevc_nvenc",
                        temp_audiofile_path="/tmp/",
                        threads=18,
                        # audio_codec="aac",
                        remove_temp=True,
                        audio=True,
                        ffmpeg_params=[
                            "-c:v",
                            "hevc_nvenc",
                            "-b:v",
                            "12000k",
                            "-y",
                            "-loglevel",
                            "error",
                            "-pix_fmt",
                            "yuv420p",
                        ],
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
    parser.add_argument(
        "--output", help="Output folder. Default is {PATH}/opencv_output", default=None
    )
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
    output_path = Path(args.output) if args.output is not None else input_path / "opencv_output"

    main(input_path=input_path, keywords=keywords, debug=args.debug, output_path=args.output)
