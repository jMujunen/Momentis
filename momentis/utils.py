from collections import deque
from numpy import ndarray
from pathlib import Path
import cv2
import threading
from collections.abc import Generator
from collections import namedtuple
import queue
import numpy as np

# Constants
INTERVAL = 30
WRITER_FPS = 60
BUFFER = 360
NULLSIZE = 300

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


dimensions = namedtuple("dimensions", ["w", "h"])
video_props = namedtuple("video_props", ["w", "h", "fps"])

ROI = dimensions(w=800, h=200)

roi = namedtuple("region_of_interest", ["x", "y", "w", "h"])


def parse_keywords() -> list[str]:
    """Parse keywords from a file and return them as a list."""
    module_path = Path(__file__)
    keywords_file = Path(module_path.parent, "keywords.txt")
    if any((not keywords_file.exists(), not keywords_file.read_text())):
        print("Error: keywords file not found")
        raise FileNotFoundError

    return [
        word
        for word in keywords_file.read_text().splitlines()
        if not word.startswith(("#", ";", "/"))
    ]


def find_continuous_segments(frames: list[int]) -> list[list[int]]:
    """Find continuous segments of frames.

    Args:
        frames (list[int]): A list of integers representing frames.

    Returns:

        list[list[int]]: A list of lists, where each sublist represents a continuous segment of frames.
    """
    if not frames:
        return []

    segments = [[frames[0]]]
    for i in range(1, len(frames)):
        if frames[i] == frames[i - 1] + 1:
            segments[-1].append(frames[i])
        else:
            segments.append([frames[i]])
    return segments


class FrameBuffer:
    def __init__(self, max_size: int) -> None:
        """Initialize the frame buffer.

        ### Paramters
        -----------------
            - `max_size (int)`: Maximum number of frames to store.
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.index = deque(maxlen=max_size)

    def add_frame(self, frame: tuple[ndarray, int]) -> None:
        """Add a new frame to the buffer.

        ### Parameters
        ---------------
            - `frame tuple[ndarray, int]`: The (frame, index) to add.
        """
        if len(self.buffer) < self.max_size:
            self.buffer.append(frame)
        else:
            # If the buffer is full, remove the oldest frame
            self.buffer.popleft()
            self.buffer.append(frame)

    def get_frames(self) -> list[ndarray]:
        """Get all frames currently in the buffer.

        ### Returns
        ------------
        - `list[ndarray]`: The current frames in the buffer as a list.
        """
        return list(self.buffer)

    def get_recent_frames(self, num_frames: int) -> list[ndarray]:
        """Get a specified number of recent frames from the buffer.

        ### Parameters
        --------------
            - `num_frames (int)`: The number of recent frames to return.
        """
        num_frames = min(num_frames, len(self.buffer))
        return list(self.buffer)[num_frames:]

    def get_future_frames(self, num_frames: int) -> list[ndarray]:
        """Get a specified number of older frames from the buffer.

        ### Parameters
        ---------------
            - `num_frames (int)`: The number of older frames to return.
        """
        num_frames = min(num_frames, len(self.buffer))
        return list(self.buffer)[:num_frames]

    def release(self) -> None:
        """Release the buffer by emptying it."""
        self.buffer = deque(maxlen=0)
        del self.buffer

    def __len__(self) -> int:
        return len(self.buffer)


class VideoReader:
    def __init__(self, video_path, buffer_size=256):
        self.video_path = video_path
        self.buffer_size = buffer_size
        self.cap = cv2.VideoCapture(video_path)

        # Get cap properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # ROI settings
        self.killfeed = roi(x=self.width - ROI.w, y=75, w=ROI.w, h=ROI.h)
        self.alt_killfeed = roi(
            x=round(self.width * 0.35), y=round(self.height * 0.6), w=ROI.w, h=ROI.h
        )

        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._read_frames)
        self.thread.start()

    def _read_frames(self):
        count = 0
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                break
            count += 1
            if count % INTERVAL == 0:
                processed_frames = []
                for roi in self._extract_roi(frame, self.killfeed, self.alt_killfeed):
                    processed_frames.extend(self._process_frame(roi))
                self.frame_queue.put(processed_frames)
        self.stop_event.set()

    @staticmethod
    def _process_frame(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh_a = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
        thresh_b = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)[1]
        return thresh_a, thresh_b

    @staticmethod
    def _extract_roi(frame: np.ndarray, *args: roi) -> Generator[np.ndarray, None, None]:
        for r in args:
            x, y, w, h = r
            yield frame[y : y + h, x : x + w]

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.frame_queue.empty() and self.stop_event.is_set():
            return False, None
        return True, self.frame_queue.get()

    def release(self) -> None:
        self.stop_event.set()
        self.thread.join()
        self.cap.release()
