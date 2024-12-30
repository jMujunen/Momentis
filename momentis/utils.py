from collections import deque
from numpy import ndarray


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
