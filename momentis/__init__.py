"""Momentis is a Python library for extracting kill (moments) from videos."""

from .utils import parse_keywords, find_continuous_segments, FrameBuffer
from .momentis import name_in_killfeed, ocr, relevant_frames, INTERVAL, WRITER_FPS, BUFFER

__all__ = [
    "BUFFER",
    "INTERVAL",
    "WRITER_FPS",
    "FrameBuffer",
    "find_continuous_segments",
    "name_in_killfeed",
    "ocr",
    "parse_keywords",
    "relevant_frames",
]
