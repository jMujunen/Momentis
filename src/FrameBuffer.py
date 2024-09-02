from collections import deque

from numpy import ndarray


class FrameBuffer:
    def __init__(self, max_size: int) -> None:
        """
        Initialize the frame buffer.

        ### Paramters:
        -----------------
            - `max_size (int)`: Maximum number of frames to store.
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.index = deque(maxlen=max_size)

    def add_frame(self, frame: tuple[ndarray, int]) -> None:
        """
        Add a new frame to the buffer.

        ### Parameters:
        ---------------
            - `frame (ndarray)`: The frame to add.
        """
        if len(self.buffer) < self.max_size:
            self.buffer.append(frame)
        else:
            # If the buffer is full, remove the oldest frame
            self.buffer.popleft()
            self.buffer.append(frame)

    def get_frames(self) -> list[ndarray]:
        """Get all frames currently in the buffer.

        ### Returns:
        ------------
        - `list[ndarray]`: The current frames in the buffer as a list.
        """
        return list(self.buffer)

    def get_recent_frames(self, num_frames: int) -> list[ndarray]:
        """Get a specified number of recent frames from the buffer.

        ### Parameters:
        -----------------
            - `num_frames (int)`: The number of recent frames to return.
        """
        if num_frames > len(self.buffer):
            num_frames = len(self.buffer)
        return list(self.buffer)[num_frames:]

    def get_future_frames(self, num_frames: int) -> list[ndarray]:
        """
        Get a specified number of older frames from the buffer.

        ### Parameters:
        -----------------
            - `num_frames (int)`: The number of older frames to return.
        """
        if num_frames > len(self.buffer):
            num_frames = len(self.buffer)
        return list(self.buffer)[:num_frames]

    def release(self) -> None:
        """Release the buffer by emptying it."""
        self.buffer = deque(maxlen=0)
        del self.buffer
