# cython: language_level=3, boundscheck=False

from numpy import ndarray
from collections import deque
from typing import Generator

from libc.stdlib cimport malloc, free

cdef public unsigned int NDIM = 3


cdef class FrameBuffer:
    def __init__(self, unsigned int max_size) -> None:
        """Initialize the frame buffer.

        ### Paramters
        -----------------
            - `max_size (int)`: Maximum number of frames to store.
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    cpdef void add_frame(self, Frame frame, unsigned int index):
        """Add a new frame to the buffer.

        ### Parameters
        ---------------
            - `frame tuple[ndarray, int]`: The (frame, index) to add.
        """
        if len(self.buffer) < self.max_size: # type: ignore
            self.buffer.append((frame, index)) # type: ignore
        else:
            # If the buffer is full, remove the oldest frame
            self.buffer.popleft() # type: ignore
            self.buffer.append((frame, index)) # type: ignore

    def get_frames(self) -> Generator[tuple, None, None]:
        """Get all frames currently in the buffer.

        ### Returns
        ------------
        - `list[ndarray]`: The current frames in the buffer as a list.
        """

        cdef Frame frame
        cdef unsigned int index

        for frame, index in self.buffer: # type: ignore
            yield (frame, index)

    cpdef void release(self):
        """Release the buffer by emptying it."""
        self.buffer = deque(maxlen=0)
        del self.buffer

    def __len__(self) -> int:
        return len(self.buffer)
