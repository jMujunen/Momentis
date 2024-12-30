



cpdef list[list[unsigned int]] find_continuous_segments(list[int] frames):# -> list[list[int]]:
    """Find continuous segments of frames.

    Args:
        frames (list[int]): A list of integers representing frames.

    Returns:

        list[list[int]]: A list of lists, where each sublist represents a continuous segment of frames.
    """
    cdef list[list[unsigned int]] segments
    cdef list[unsigned int] segment
    cdef unsigned int i

    if not frames:
        return []



    segments = [[frames[0]]]
    for i in range(1, len(frames)):
        if frames[i] == frames[i - 1] + 1:
            segments[-1].append(frames[i])
        else:
            segments.append([frames[i]])
