#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import moviepy.editor as mp
import pytesseract
from numpy import ndarray

from .utils import FrameBuffer, find_continuous_segments  # type: ignore

# Constants
INTERVAL = 60
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

    # # Overlay ROIs on the original frame for debugging
    # for arg in args:
    #     x, y, w, h = arg  # type: ignore
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     if img.shape[-2::-1] > MONITOR_DIMS:
    #         img = cv2.resize(img, MONITOR_DIMS)
    #     cv2.imshow("Frame", img)

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
    # cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    cap = cv2.VideoCapture(video_path)
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


def main(
    input_path: str, keywords: list[str], debug=False, output_path: str = "../opencv_output"
) -> None:
    """Process videos and extract frames containing kill-related information.

    Parameters
        - input_path (str): Path to the directory containing video files.
        - keywords (list): List of keywords related to kill feeds.
        - debug (bool): If True, output a json file containing the frames written and their indices.
        - output_path (str): Output directory path to write the processed video and frames.
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
    output_folder = Path(output_path).resolve()
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


def cleanup(
    original_path: str | Path,
    processed_path: str | Path,
    archive_path: str | Path,
    final_output_path: str | Path,
) -> None:
    """Housekeeping. Move the original and result videos to their respective folders.

    Parameters
        - original_path (str): Path to the directory containing the original videos.
        - processed_path (str): Path to the directory containing successfully processed videos.
        - archive_path (str): Path to the directory where original videos are moved after processing.
        - final_output_path (str): Path to the directory where processed videos are moved for final output.

    The function moves successfully processed videos from the processed path to the final output path,
    and original videos from the original path to the archive path. It also handles errors if files do not exist.
    """
    valid_extentions = [".mp4", ".mov", ".mkv"]
    # Define path objects
    original_dir = Path(original_path)  # Original videos
    processed_dir = Path(processed_path)  # Successfully processed videos get moved here
    final_out_dir = Path(final_output_path)  # Final output. Processed videos end up here at the end
    archive_path = Path(archive_path)  # Originals get placed here after processing is complete

    for processed_video in processed_dir.rglob("*.*"):
        if processed_video.suffix in valid_extentions and processed_video.stat().st_size > 300:
            # Move the processed video to the final output directory (if frames were written)
            # If frames were not written, size of the file is 257 bytes
            processed_video.rename(final_out_dir / processed_video.name)
            # Find the original
            original_video = Path(original_dir, processed_video.name.removeprefix("cv2_"))
            if original_video.exists():
                # Move the original video to the archive directory
                original_video.rename(archive_path / original_video.name)
            else:
                print(
                    f"\x1b[31m>>> Error: {original_video.name} does not exsist!\x1b[0m\n  {processed_video!s} \n  {original_video!s}"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("PATH", help="Path to videos", type=str)
    parser.add_argument("--output", help="Output folder")
    parser.add_argument(
        "--remove_originals", help="Remove original videos after processing", action="store_true"
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
        word for word in keywords_file.read_text().splitlines() if word[0] not in ["#", ";", "/"]
    ]

    input_path = Path(args.PATH)

    main(input_path=input_path, keywords=keywords, debug=args.debug)
    if args.remove_originals:
        archive_path = Path(str(input_path).replace("ssd", "hdd"))
        cleanup(
            original_path=args.PATH,
            processed_path=args.output,
            archive_path=archive_path,
            final_output_path=args.output,
        )
