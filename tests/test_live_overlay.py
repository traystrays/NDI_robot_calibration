"""Hardware-independent synchronization, acquisition, and rendering regression tests."""
import json
from pathlib import Path
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import numpy as np

from LiveData.packets import History, VideoFrame, TransformSample
from LiveData.synchronizer import FrameSynchronizer
from LiveData.receivers import VideoReceiver, NDIReceiver, RobotPoseHistory
from LiveData.robot_receiver_class import RobotReceiver, POSE_RECORD
from scripts.live_ultrasound_overlay import demo_setup, demo_feed, OverlayRenderer


def video(n, t):
    return VideoFrame(n, t, t, np.zeros((4, 5, 3), np.uint8))


def pose(n, t, valid=True):
    return TransformSample(n, t, t, np.eye(4) if valid else None, valid)


class MatchingTests(unittest.TestCase):
    def setUp(self):
        self.h = [History(5) for _ in range(4)]
        self.sync = FrameSynchronizer(*self.h)
        self.t = 1_000_000_000

    def add(self, us=5, robot=-2, ndi=3, valid=True):
        self.h[0].append(video(1, self.t))
        self.h[1].append(video(1, self.t+int(us*1e6)))
        self.h[2].append(pose(1, self.t+int(robot*1e6)))
        self.h[3].append(pose(1, self.t+int(ndi*1e6), valid))

    def test_wait_then_match_and_emit_once(self):
        self.add()
        self.assertEqual(self.sync.poll(self.t+10_000_000), [])
        result = self.sync.poll(self.t+60_000_000)[0]
        self.assertEqual(result.status, "matched")
        self.assertEqual(result.errors_ms, (5, -2, 3))
        self.assertEqual(self.sync.poll(self.t+80_000_000), [])

    def test_wait_selects_later_closer_sample(self):
        self.add(us=-15)
        self.assertFalse(self.sync.poll(self.t))
        self.h[1].append(video(2, self.t+2_000_000))
        self.assertEqual(self.sync.poll(self.t+60_000_000)[0].ultrasound.sequence, 2)

    def test_invalid_sample_does_not_fall_back_to_good_pose(self):
        self.h[3].append(pose(0, self.t-10_000_000))
        self.add(valid=False)
        self.assertIn("ndi_invalid", self.sync.poll(self.t+60_000_000)[0].status)

    def test_tolerance_and_missing_stream(self):
        self.add(ndi=50)
        result = self.sync.poll(self.t+60_000_000)[0]
        self.assertEqual(result.status, "ndi_unmatched")
        self.assertIsNone(result.ndi)
        empty = [History() for _ in range(4)]
        empty[0].append(video(1, self.t))
        self.assertIn("robot_unmatched", FrameSynchronizer(*empty).poll(self.t+60_000_000)[0].status)

    def test_stale_rejected(self):
        self.add()
        self.assertIn("stale", self.sync.poll(self.t+300_000_000)[0].status)

    def test_bounded_history_drop_count(self):
        for n in range(1, 10):
            self.h[0].append(video(n, self.t+n))
        self.assertEqual(len(self.h[0].snapshot()), 5)
        self.sync.poll(self.t+100_000_000)
        self.assertEqual(self.sync.dropped, 4)
        self.assertEqual(self.h[0].evicted, 4)

    def test_images_are_owned_and_immutable(self):
        original = np.zeros((2, 2, 3), np.uint8)
        frame = VideoFrame(1, 1, 1, original)
        original[:] = 100
        self.assertEqual(frame.image.max(), 0)
        with self.assertRaises(ValueError):
            frame.image[:] = 1
        with self.assertRaises(ValueError):
            frame.image.setflags(write=True)

    def test_reject_bad_transform_and_timing(self):
        with self.assertRaises(ValueError):
            pose(1, 1).__class__(1, 1, 1, np.zeros((4, 4)))
        with self.assertRaises(ValueError):
            FrameSynchronizer(*self.h, wait_ms=float("nan"))


class RenderingTests(unittest.TestCase):
    def test_unknown_calibration_resolution_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "calibration_resolution"):
            OverlayRenderer.from_config({"calibration_resolution": None})

    def test_known_projection_and_unmatched_suppression(self):
        h = [History() for _ in range(4)]
        demo_feed(h, 1, 1_000_000_000)
        bundle = FrameSynchronizer(*h).poll(1_000_000_000)[0]
        image, status = demo_setup().render(bundle)
        self.assertEqual(status, "rendered")
        self.assertGreater(np.abs(image[200:280].astype(float)-bundle.ecm.image[200:280]).sum(), 0)
        from dataclasses import replace
        image, status = demo_setup().render(replace(bundle, status="ndi_unmatched", ndi=None))
        np.testing.assert_array_equal(image[100:], bundle.ecm.image[100:])
        self.assertEqual(status, "ndi_unmatched")
        behind = np.eye(4)
        behind[2, 3] = -1
        _, status = demo_setup().render(replace(bundle, ndi=pose(1, 1).__class__(1, 1, 1, behind)))
        self.assertEqual(status, "behind_camera")

    def test_existing_calibration_loads(self):
        config = json.loads((ROOT / "scripts/live_overlay_config.json").read_text())
        # Test loading only; this does NOT verify actual calibration image dimensions.
        config["overlay"]["calibration_resolution"] = [1920, 1080]
        renderer = OverlayRenderer.from_config(config["overlay"])
        self.assertEqual(renderer.corners.shape, (4, 3))


class AcquisitionTests(unittest.TestCase):
    def test_camera_start_failure_releases_device(self):
        class Capture:
            released = False
            def isOpened(self): return False
            def release(self): self.released = True
        capture = Capture()
        worker = VideoReceiver(0, capture_factory=lambda *_: capture)
        with self.assertRaisesRegex(RuntimeError, "Could not open"):
            worker.start()
        self.assertTrue(capture.released)
        self.assertFalse(worker._thread.is_alive())

    def test_camera_ownership_timestamp_and_cleanup(self):
        class Capture:
            released = False
            def isOpened(self): return True
            def set(self, *args): pass
            def read(self):
                time.sleep(.005)
                return True, np.zeros((4, 5, 3), np.uint8)
            def release(self): self.released = True
        capture = Capture()
        worker = VideoReceiver(0, resolution=(5, 4), latency_ms=10,
                               capture_factory=lambda *_: capture)
        with worker:
            sample = worker.history.latest()
            self.assertEqual(sample.received_ns-sample.monotonic_ns, 10_000_000)
        self.assertTrue(capture.released)
        self.assertFalse(worker._thread.is_alive())

    def test_ndi_mm_conversion_and_lost_tracking(self):
        class Tracker:
            closed = False
            calls = 0
            def start_tracking(self): pass
            def get_frame(self):
                time.sleep(.005)
                self.calls += 1
                matrix = np.eye(4)
                matrix[0, 3] = 100
                if self.calls > 1: matrix[:] = np.nan
                return [1], [123], [self.calls], [matrix], [0.1]
            def stop_tracking(self): pass
            def close(self): self.closed = True
        tracker = Tracker()
        with tempfile.NamedTemporaryFile() as rom:
            worker = NDIReceiver(rom.name, "FAKE", tracker_factory=lambda _: tracker)
            with worker:
                deadline = time.monotonic()+1
                while len(worker.history.snapshot()) < 2 and time.monotonic() < deadline:
                    time.sleep(.005)
            samples = worker.history.snapshot()
        self.assertAlmostEqual(samples[0].transform[0, 3], .1)
        self.assertFalse(samples[1].valid)
        self.assertTrue(tracker.closed)

    def test_repeated_ndi_device_frame_is_ignored(self):
        class Tracker:
            closed = False
            calls = 0
            def start_tracking(self): pass
            def get_frame(self):
                time.sleep(.005)
                self.calls += 1
                source_frame = 1 if self.calls < 3 else self.calls - 1
                matrix = np.eye(4)
                return [1], [123], [source_frame], [matrix], [0.1]
            def stop_tracking(self): pass
            def close(self): self.closed = True
        tracker = Tracker()
        with tempfile.NamedTemporaryFile() as rom:
            worker = NDIReceiver(rom.name, "FAKE", tracker_factory=lambda _: tracker)
            with worker:
                deadline = time.monotonic()+1
                while len(worker.history.snapshot()) < 2 and time.monotonic() < deadline:
                    time.sleep(.005)
            samples = worker.history.snapshot()
        self.assertTrue(all(sample.valid for sample in samples))
        self.assertEqual(len({sample.source_frame for sample in samples}), len(samples))
        self.assertTrue(tracker.closed)

    def test_real_udp_registration_decode_and_stop(self):
        publisher = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            publisher.bind(("127.0.0.1", 0))
            publisher.settimeout(1)
            with RobotReceiver(publisher=publisher.getsockname()) as receiver:
                registration, _ = publisher.recvfrom(16)
                port, = struct.unpack("<H", registration)
                retry, _ = publisher.recvfrom(16)
                self.assertEqual(retry, registration)
                packet = b"".join(POSE_RECORD.pack(
                    0., 0., .1, *np.eye(3).ravel(), i) for i in range(3))
                publisher.sendto(b"bad", ("127.0.0.1", port))
                publisher.sendto(packet, ("127.0.0.1", port))
                deadline = time.monotonic()+1
                while receiver.latest() is None and time.monotonic() < deadline:
                    time.sleep(.005)
                self.assertIsNotNone(receiver.latest())
                self.assertEqual(receiver.invalid_packets, 1)
                converted = RobotPoseHistory(receiver).snapshot()[0]
                self.assertAlmostEqual(converted.transform[2, 3], .1)
            self.assertIsNone(receiver._thread)
        finally:
            publisher.close()


if __name__ == "__main__":
    unittest.main()
