"""Receive DVPControl poses into a thread-safe history, or print them directly.

Start DVPControl first, then run: python src/LiveData/robot_receiver.py
"""

import socket
import struct
import math
import time
from collections import deque
from dataclasses import dataclass
import threading

# EpiLogger's DVAPI_MIP: float pos[3], float orientation[9], int type.
# Three 52-byte records form the 156-byte pose packet on this Windows setup.
POSE_RECORD = struct.Struct("<12fi")
MANIPULATORS = {0: "PSM1", 1: "PSM2", 2: "ECM"}


def decode_poses(data: bytes) -> list[tuple[int, tuple[float, ...]]]:
    """Decode the layout expected by EpiLogger's callbackDVAPI()."""
    expected = 3 * POSE_RECORD.size
    if len(data) != expected:
        raise ValueError(f"Expected {expected} pose bytes, got {len(data)}")

    poses = []
    for values in POSE_RECORD.iter_unpack(data):
        manipulator_id = values[12]
        if manipulator_id not in MANIPULATORS:
            raise ValueError(f"Unexpected manipulator ID: {manipulator_id}")
        if not all(math.isfinite(value) for value in values[:12]):
            raise ValueError("Packet contains non-finite position/orientation values")
        poses.append((manipulator_id, values[:12]))
    if len({identifier for identifier, _ in poses}) != 3:
        raise ValueError("Packet contains duplicate manipulator IDs")
    return poses


@dataclass(frozen=True)
class PoseSample:
    """One immutable packet; timestamps describe local receipt, not robot capture."""

    sequence: int
    monotonic_ns: int
    received_at_us: int
    poses: tuple[tuple[int, tuple[float, ...]], ...]
    sender: tuple[str, int]


class RobotReceiver:
    """Receive on a worker thread; read bounded history from the application thread.

    Call start()/stop() from the application thread. Data access methods are
    thread-safe. Restart after restarting DVPControl to register again.
    """

    def __init__(self, publisher=("127.0.0.1", 60000), buffer_size=300):
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        self.publisher = publisher
        self._buffer = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._socket = None
        self._error = None
        self._invalid_packets = 0
        self.receiver_port = None

    def start(self):
        """Bind and register, then return immediately while reception continues."""
        if self._thread is not None:
            raise RuntimeError("Call stop() before starting the receiver again")
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            receiver.bind(("127.0.0.1", 0))
            receiver.settimeout(0.2)  # Allows stop() to join the worker promptly.
            self.receiver_port = receiver.getsockname()[1]
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as registration:
                registration.sendto(struct.pack("<H", self.receiver_port), self.publisher)
        except BaseException:
            receiver.close()
            raise
        with self._lock:
            self._buffer.clear()
            self._error = None
            self._invalid_packets = 0
        self._stop.clear()
        self._socket = receiver
        self._thread = threading.Thread(
            target=self._receive, args=(receiver,), name="robot-udp", daemon=True
        )
        try:
            self._thread.start()
        except BaseException:
            receiver.close()
            self._socket = None
            self._thread = None
            raise
        return self

    def _receive(self, receiver):
        sequence = 0
        try:
            while not self._stop.is_set():
                try:
                    data, sender = receiver.recvfrom(65535)
                    monotonic_ns = time.monotonic_ns()
                    received_at_us = time.time_ns() // 1_000
                except socket.timeout:
                    continue
                try:
                    poses = tuple(decode_poses(data))
                except ValueError:
                    with self._lock:
                        self._invalid_packets += 1
                    continue
                sequence += 1
                sample = PoseSample(sequence, monotonic_ns, received_at_us, poses, sender)
                # Never hold the lock while waiting for a packet or decoding it.
                with self._lock:
                    self._buffer.append(sample)
        except OSError as error:
            with self._lock:
                self._error = str(error)
        finally:
            receiver.close()

    def latest(self):
        """Return the newest sample or None. It may be stale; check its timestamp."""
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def snapshot(self):
        """Return an immutable copy of the current history, oldest first."""
        with self._lock:
            return tuple(self._buffer)

    def nearest(self, timestamp_ns, max_delta_ms=40.0):
        """Find a sample near a time.monotonic_ns() timestamp, or return None.

        Searches existing history only; does not wait for a future sample.
        """
        if not math.isfinite(max_delta_ms) or max_delta_ms < 0:
            raise ValueError("max_delta_ms must be finite and nonnegative")
        samples = self.snapshot()
        if not samples:
            return None
        sample = min(samples, key=lambda item: abs(item.monotonic_ns - timestamp_ns))
        if abs(sample.monotonic_ns - timestamp_ns) > max_delta_ms * 1_000_000:
            return None
        return sample

    @property
    def error(self):
        """Socket failure, if any; a failed worker must be stopped before restart."""
        with self._lock:
            return self._error

    @property
    def invalid_packets(self):
        with self._lock:
            return self._invalid_packets

    def stop(self):
        """Wait for the receive worker to exit and close its socket."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
            self._socket = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()


def main() -> None:
    """Terminal monitor using the same public interface as a future overlay."""
    with RobotReceiver() as robot:
        print(f"Registered with {robot.publisher}; receiving on {robot.receiver_port}.")
        print("Fields: manipulator, id, x, y, z, o0, o1, o2, o3, o4, o5, o6, o7, o8")
        print("Positions: metres. Orientation: EpiLogger wire order. Ctrl+C stops.")
        last_sequence = 0
        last_packet_time = time.monotonic()
        while True:
            if robot.error is not None:
                raise OSError(robot.error)
            # A snapshot lets this monitor print multiple packets received since
            # its previous iteration, without blocking the networking worker.
            for sample in robot.snapshot():
                if sample.sequence <= last_sequence:
                    continue
                lines = [f"received at {sample.received_at_us} us: packet #{sample.sequence}"]
                for identifier, values in sample.poses:
                    numbers = ", ".join(f"{value:.5f}" for value in values)
                    lines.append(f"{MANIPULATORS[identifier]}, {identifier}, {numbers}")
                print("\n".join(lines), flush=True)
                last_sequence = sample.sequence
                last_packet_time = time.monotonic()
            if time.monotonic() - last_packet_time >= 5:
                print("No new valid packet in 5 seconds. Check DVPControl; restart "
                      f"this receiver if the publisher restarted. Invalid packets: {robot.invalid_packets}",
                      flush=True)
                last_packet_time = time.monotonic()
            time.sleep(0.01)


if __name__ == "__main__":
    # GUI automation is only needed when running this file directly.
    from pywinauto.application import Application
    import pyautogui

    exe_path = r"c:\Users\rcl\Documents\Linghao\eye_gaze_epilogger_2\DVPControl_precompiled\DVPControl.exe"
    app = Application(backend="win32").start(exe_path)
    time.sleep(5)
    window = app.top_window()
    
    try:
        window["NO"].click_input() 
        print("Clicked 'NO' button to dismiss the dialog.")
        
        time.sleep(1)
        window = app.top_window()
        window["Save SUJ"].click_input()
        print("Clicked 'Save SUJ' button to save the SUJ file.")
        
        time.sleep(5)
        pyautogui.keyDown('alt')
        pyautogui.press('tab')
        pyautogui.keyUp('alt')
        time.sleep(1)

        main()
    except KeyboardInterrupt:
        print("\nStopped.")
    except OSError as error:
        raise SystemExit(f"UDP error: {error}") from error
