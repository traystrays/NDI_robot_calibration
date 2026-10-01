"""
Owned, immutable packets and bounded histories for local live acquisition.

All matching timestamps are host monotonic nanoseconds. ``received_ns`` is
preserved separately from a configurable latency-corrected ``monotonic_ns``.
Arrays are backed by immutable bytes: neither field assignment nor pixel edits
can mutate samples while another thread is reading them.
"""
from collections import deque
from dataclasses import dataclass
from threading import Lock

import numpy as np

from ndi_robot_registration.transforms import is_valid_transform


def immutable_array(value):
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)

# Immutable packets of Video Frame
@dataclass(frozen=True, eq=False)
class VideoFrame:
    sequence: int
    monotonic_ns: int
    received_ns: int
    image: np.ndarray

    def __post_init__(self):
        image = np.asarray(self.image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("VideoFrame requires a uint8 BGR image")
        object.__setattr__(self, "image", immutable_array(image))

# Immutable packets of Transforms, for NDI and robot 
@dataclass(frozen=True, eq=False)
class TransformSample:
    sequence: int
    monotonic_ns: int
    received_ns: int
    transform: np.ndarray | None
    valid: bool = True
    reason: str = ""
    source_frame: int | None = None
    tracking_quality: float | None = None

    def __post_init__(self):
        if self.valid:
            if not is_valid_transform(self.transform, rotation_atol=1e-3):
                raise ValueError("Pose must be a finite rigid 4x4 transform in metres")
            object.__setattr__(self, "transform", immutable_array(
                np.asarray(self.transform, dtype=float)))
        else:
            object.__setattr__(self, "transform", None)


class History:
    """Single-stream, ordered history; snapshots release the lock before use."""
    def __init__(self, capacity=120):
        if capacity <= 0:
            raise ValueError("History capacity must be positive")
        self._items = deque(maxlen=capacity)
        self._lock = Lock()
        self.evicted = 0

    def append(self, sample):
        with self._lock:
            if self._items and sample.monotonic_ns < self._items[-1].monotonic_ns:
                raise ValueError("Stream timestamps must not move backwards")
            if len(self._items) == self._items.maxlen:
                self.evicted += 1
            self._items.append(sample)

    def snapshot(self):
        with self._lock:
            return tuple(self._items)

    def latest(self):
        with self._lock:
            return self._items[-1] if self._items else None
