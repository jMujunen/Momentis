from collections import deque
from numpy import ndarray
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable, Generator
import cv2
import pytesseract


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

    def exec(
        self, keywords: list[str], *args: tuple[int, ...]
    ):  # -> Generator[ndarray, None, None]:
        """Execute a function on the aggregated data.

        Args:
            func (Callable): The function to apply to each frame.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.
        """

        def func(
            frame: ndarray, keywords: list[str], region_of_interst: tuple[int, int, int, int]
        ) -> bool:
            preprocessed_frames = []
            x, y, w, h = region_of_interst  # type: ignore
            roi = frame[y : y + h, x : x + w]
            # Crop the frame to the region of interest (rio)
            gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            preprocessed_frames.append(cv2.threshold(gray_frame, 175, 255, cv2.THRESH_BINARY)[1])
            preprocessed_frames.append(
                cv2.threshold(gray_frame, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            )

            concatted_img = cv2.hconcat(preprocessed_frames)
            text = pytesseract.image_to_string(concatted_img, lang="eng")
            # Check if any kill-related keyword is present in the extracted text
            return bool(any(keyword.lower() in text.lower() for keyword in keywords))

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(func, frame, keywords, *args) for _, frame in self.get_frames()
            ]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    return self.get_frames()
        return []
