"""Live camera and NDI workers, plus an adapter for the existing robot receiver.

Each device has one owner. Do not run the video logger against the same devices
simultaneously. Positive latency_ms subtracts an independently measured device
latency from receipt time; defaults make no claim about capture-time alignment.
"""
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .packets import History, VideoFrame, TransformSample


class Worker:
    """One-use acquisition worker with visible failures and bounded shutdown."""
    def __init__(self, name):
        self.name = name
        self.error = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = None

    def start(self, timeout=15):
        if self._thread is not None:
            raise RuntimeError("Create a new receiver to restart acquisition")
        self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            self._stop.set()
            raise TimeoutError(f"{self.name}: startup timed out; device driver may be blocked")
        if self.error:
            self.stop()
            raise RuntimeError(f"{self.name}: {self.error}") from self.error
        return self

    def _run(self):
        try:
            self.acquire()
        except Exception as error:
            self.error = error
        finally:
            self._ready.set()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(5)
            if self._thread.is_alive():
                raise RuntimeError(f"{self.name}: driver blocked; worker did not stop")

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()


def latency_ns(milliseconds):
    if not math.isfinite(milliseconds) or milliseconds < 0:
        raise ValueError("Measured latency must be finite and nonnegative")
    return int(milliseconds * 1_000_000)


class VideoReceiver(Worker):
    def __init__(self, camera, *, resolution=(1280, 720), fps=30, backend=cv2.CAP_ANY,
                 latency_ms=0, capacity=30, capture_factory=None):
        super().__init__(f"video-{camera}")
        self.history = History(capacity)
        self.camera, self.resolution, self.fps, self.backend = camera, resolution, fps, backend
        self.delay = latency_ns(latency_ms)
        self.capture_factory = capture_factory or cv2.VideoCapture

    def acquire(self):
        capture = self.capture_factory(self.camera, self.backend)
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Could not open camera {self.camera}")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            capture.set(cv2.CAP_PROP_FPS, self.fps)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Best effort; backend may ignore.
            sequence = 0
            while not self._stop.is_set():
                ok, image = capture.read()
                receipt = time.monotonic_ns()
                if not ok:
                    raise RuntimeError(f"Camera {self.camera} stopped returning frames")
                if (image.shape[1], image.shape[0]) != tuple(self.resolution):
                    raise ValueError(f"Camera {self.camera} delivered {image.shape[1]}x{image.shape[0]}, "
                                     f"expected {self.resolution}; check capture/calibration settings")
                sequence += 1
                self.history.append(VideoFrame(sequence, receipt - self.delay, receipt, image))
                self._ready.set()
        finally:
            capture.release()


class NDIReceiver(Worker):
    def __init__(self, rom_path, serial_port, *, tracker_type="polaris", tool_index=0,
                 latency_ms=0, capacity=300, tracker_factory=None):
        super().__init__("ndi-tracker")
        self.history = History(capacity)
        self.settings = {"tracker type": tracker_type, "romfiles": [str(rom_path)],
                         "serial port": serial_port}
        if not Path(rom_path).is_file():
            raise FileNotFoundError(rom_path)
        if tool_index != 0:
            raise ValueError("One probe ROM is configured; tool_index must be 0")
        self.tool_index = tool_index
        self.delay = latency_ns(latency_ms)
        self.tracker_factory = tracker_factory

    def acquire(self):
        factory = self.tracker_factory
        if factory is None:
            from sksurgerynditracker.nditracker import NDITracker
            factory = NDITracker
        tracker = factory(self.settings)
        try:
            tracker.use_quaternions = False
            tracker.start_tracking()
            self._ready.set()
            sequence = 0
            previous_frame = None
            while not self._stop.is_set():
                frame = tracker.get_frame()
                receipt = time.monotonic_ns()
                source_frame = None
                tracking_quality = None
                try:
                    handles, timestamps, numbers, tracking, quality = frame
                    source_frame = int(numbers[self.tool_index])
                    tracking_quality = float(quality[self.tool_index])
                    if source_frame == previous_frame:
                        # get_frame() can return the most recent device frame faster
                        # than the tracker produces new measurements.  A duplicate is
                        # not a failed measurement and must not replace the latest
                        # valid pose with an artificial ``ndi_invalid`` sample.
                        continue
                    transform = np.asarray(tracking[self.tool_index], dtype=float).copy()
                    transform[:3, 3] *= 0.001  # NDI mm -> common metres.
                    if not np.isfinite(np.asarray(quality[self.tool_index], dtype=float)).all():
                        raise ValueError("Nonfinite tracking quality")
                    sample = TransformSample(sequence + 1, receipt-self.delay, receipt, transform,
                                             source_frame=source_frame, tracking_quality=tracking_quality)
                except (ValueError, IndexError, TypeError) as error:
                    sample = TransformSample(sequence + 1, receipt-self.delay, receipt, None,
                                             False, str(error), source_frame, tracking_quality)
                sequence += 1
                previous_frame = source_frame
                self.history.append(sample)
        finally:
            try:
                tracker.stop_tracking()
            finally:
                tracker.close()


class RobotPoseHistory:
    """Convert ECM ID 2 into base_T_ecm, matching clean_si_data's row-major order."""
    def __init__(self, receiver, latency_ms=0):
        self.receiver = receiver
        self.delay = latency_ns(latency_ms)
        self._cache = {}

    def snapshot(self):
        converted = []
        samples = self.receiver.snapshot()
        cache = {}
        for sample in samples:
            previous = self._cache.get(sample.sequence)
            if previous is not None and previous[0] is sample:
                converted.append(previous[1])
                cache[sample.sequence] = previous
                continue
            values = next(values for identifier, values in sample.poses if identifier == 2)
            matrix = np.eye(4)
            matrix[:3, 3] = values[:3]  # Robot translation already in metres.
            matrix[:3, :3] = np.asarray(values[3:]).reshape(3, 3)
            try:
                pose = TransformSample(sample.sequence, sample.monotonic_ns-self.delay,
                                       sample.monotonic_ns, matrix)
            except ValueError as error:
                pose = TransformSample(sample.sequence, sample.monotonic_ns-self.delay,
                                       sample.monotonic_ns, None, False, str(error))
            converted.append(pose)
            cache[sample.sequence] = (sample, pose)
        self._cache = cache  # Drop entries no longer in the robot's bounded history.
        return tuple(converted)
