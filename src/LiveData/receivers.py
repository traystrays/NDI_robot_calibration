"""Live camera and NDI workers, plus robot-packet to ECM-transform conversion.

Each device has one owner. Do not run the video logger against the same devices
simultaneously. Positive latency_ms subtracts an independently measured device
latency from receipt time; defaults make no claim about capture-time alignment.
"""
import math
import multiprocessing
from pathlib import Path

import cv2
import numpy as np

from .packets import History, VideoFrame, TransformSample
from .worker import Worker
from .clock import clock_ns


def latency_ns(milliseconds):
    """Convert a measured latency in milliseconds to nanoseconds."""

    if not math.isfinite(milliseconds) or milliseconds < 0:
        raise ValueError("Measured latency must be finite and nonnegative")
    return int(milliseconds * 1_000_000)


class VideoReceiver(Worker):
    def __init__(self, camera_idx, *, resolution=(1280, 720), fps=30, backend=cv2.CAP_ANY,
                 latency_ms=0, capacity=30):
        super().__init__(f"video-{camera_idx}")
        self.history = History(capacity) # init bounded history
        self.camera, self.resolution, self.fps, self.backend = camera_idx, resolution, fps, backend
        self.delay = latency_ns(latency_ms) # configured device latency, but currently dont have it configured

    def acquire(self):
        capture = cv2.VideoCapture(self.camera, self.backend)
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Could not open camera {self.camera}")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            capture.set(cv2.CAP_PROP_FPS, self.fps)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # try to keep as little latency as possible
            sequence = 0
            while not self._stop.is_set():
                ok, image = capture.read()
                receipt = clock_ns()
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


def _ndi_tracking_process(settings, tool_index, delay, stop, send, tracker_factory=None):
    """Keep blocking ndicapy calls outside the GUI's Python interpreter."""
    tracker = None
    try:
        if tracker_factory is None:
            from sksurgerynditracker.nditracker import NDITracker
            tracker_factory = NDITracker
        tracker = tracker_factory(settings)
        tracker.use_quaternions = False
        tracker.start_tracking()
        send(("ready", None))
        sequence = 0
        previous_frame = None
        while not stop.is_set():
            frame = tracker.get_frame()
            receipt = clock_ns()
            source_frame = tracking_quality = None
            try:
                _, _, numbers, tracking, quality = frame
                source_frame = int(numbers[tool_index])
                tracking_quality = float(quality[tool_index])
                if source_frame == previous_frame:
                    continue
                transform = np.asarray(tracking[tool_index], dtype=float).copy()
                transform[:3, 3] *= .001
                if not np.isfinite(tracking_quality):
                    raise ValueError("Nonfinite tracking quality")
                sample = TransformSample(sequence + 1, receipt-delay, receipt, transform,
                                         source_frame=source_frame, tracking_quality=tracking_quality)
            except (ValueError, IndexError, TypeError) as error:
                sample = TransformSample(sequence + 1, receipt-delay, receipt, None,
                                         False, str(error), source_frame, tracking_quality)
            sequence += 1
            previous_frame = source_frame
            send(("sample", (sample.sequence, sample.monotonic_ns, sample.received_ns,
                             sample.transform, sample.valid, sample.reason,
                             sample.source_frame, sample.tracking_quality)))
    except Exception as error:
        send(("error", f"{type(error).__name__}: {error}"))
    finally:
        if tracker is not None:
            try:
                tracker.stop_tracking()
            finally:
                tracker.close()


def _ndi_process_entry(settings, tool_index, delay, stop, connection):
    try:
        _ndi_tracking_process(settings, tool_index, delay, stop, connection.send)
    except (BrokenPipeError, EOFError):
        if not stop.is_set():
            raise
    finally:
        connection.close()


class NDIReceiver(Worker):
    def __init__(self, rom_path, serial_port, *, tracker_type="polaris", tool_index=0,
                 latency_ms=0, capacity=300, tracker_factory=None):
        super().__init__("ndi-tracker")
        self.history = History(capacity) # init bounded history
        self.settings = {"tracker type": tracker_type, "romfiles": [str(rom_path)],
                         "serial port": serial_port}
        if not Path(rom_path).is_file():
            raise FileNotFoundError(rom_path)
        if tool_index != 0:
            raise ValueError("One probe ROM is configured; tool_index must be 0")
        self.tool_index = tool_index
        self.delay = latency_ns(latency_ms)
        self._tracker_factory = tracker_factory
        self._process = None
        self._shutdown_error = None

    def stop(self):
        super().stop()
        if self._shutdown_error is not None:
            raise RuntimeError(self._shutdown_error)

    def acquire(self):
        def receive(message):
            kind, value = message
            if kind == "ready":
                self._ready.set()
            elif kind == "error":
                raise RuntimeError(value)
            elif kind == "sample":
                # Recreate immutable arrays after IPC; preserve child receipt time.
                self.history.append(TransformSample(*value))

        if self._tracker_factory is not None:
            # Injected fake trackers support hardware-independent unit tests only.
            _ndi_tracking_process(self.settings, self.tool_index, self.delay,
                                  self._stop, receive, self._tracker_factory)
            return
        context = multiprocessing.get_context("spawn")
        incoming, outgoing = context.Pipe(duplex=False)
        stop = context.Event()
        process = context.Process(target=_ndi_process_entry,
            args=(self.settings, self.tool_index, self.delay, stop, outgoing),
            name="ndi-device", daemon=True)
        self._process = process
        try:
            process.start()
            outgoing.close()
            while not self._stop.is_set():
                if incoming.poll(.05):
                    try:
                        receive(incoming.recv())
                    except EOFError:
                        raise RuntimeError(f"NDI process exited (code {process.exitcode})") from None
                elif not process.is_alive():
                    raise RuntimeError(f"NDI process exited (code {process.exitcode})")
        finally:
            stop.set()
            outgoing.close()
            try:
                if process.pid is not None:
                    process.join(3)
                    if process.is_alive():
                        process.terminate()
                        process.join(1)
                        self._shutdown_error = "NDI driver blocked during shutdown; device process terminated"
                    process.close()
                    if self._shutdown_error is not None:
                        raise RuntimeError(self._shutdown_error)
            finally:
                incoming.close()
                self._process = None


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
