
from typing import Any
from collections import deque
from cpython cimport bool
import pytesseract
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from numpy cimport ndarray
# cimport numpy as cnp

cdef class FrameBuffer:

    def __init__(self, int max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    cpdef void add_frame(self, int index):
        # Add the frame to the buffer at the specified index
        self.buffer.append(index) # type: ignore


    cdef list[int] get_frames(self):
        """Get all frames currently in the buffer."""
        # results = []
        # for i in range(self.max_size):
            # results.append(self.buffer.popleft())
        return self.buffer.popleft()
    def __len__(self) -> int:
        return len(self.buffer) # type: ignore

    cdef list[int] exec(self, list[str] keywords, (int, int, int, int) roi):
        """Execute a function on the aggregated data.

        Args:
            func (Callable): The function to apply to each frame.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.
        """
        cdef list[tuple[ndarray, int]] frames = self.get_frames()
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(name_in_killfeed, frame=frame, keywords=keywords, region_of_interst=roi) for frame, _ in self.get_frames() # type: ignore
            ]
            for future in futures:
                try:
                    result = future.result()
                    if result:
                        return self.get_frames()
                except Exception as e:
                    print(f"An error occurred: {e}")



cdef inline ndarray decode_frame(Frame frame):
    return np.array(frame)


cdef inline bool name_in_killfeed(Frame frame, list[str] keywords, (int, int, int, int) region_of_interst):
        """Check if a frame contains a name in the killfeed."""

        cdef ndarray decoded_frame = np.array(frame)
        cdef list preprocessed_frames = []
        cdef int x, y, w , h
        # cdef ndarray roi
        x, y, w, h = region_of_interst


        roi = decoded_frame[y : y + h, x : x + w]
        # Crop the frame to the region of interest (rio)
        gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        preprocessed_frames.append(cv2.threshold(gray_frame, 175, 255, cv2.THRESH_BINARY)[1])
        preprocessed_frames.append(
            cv2.threshold(gray_frame, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        )

        concatted_img = cv2.hconcat(preprocessed_frames)
        text = pytesseract.image_to_string(concatted_img, lang="eng")
        # Check if any kill-related keyword is present in the extracted text
        return bool(any(keyword.lower() in text.lower() for keyword in keywords)) # type: ignore
