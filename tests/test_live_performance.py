"""Regression checks for isolated NDI acquisition and cropped rendering."""
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import cv2
import numpy as np
from LiveData.receivers import NDIReceiver
from LiveData.clock import clock_ns
from scripts.reproject_ultrasound import overlay_slice


def fake_device(settings, tool_index, delay, stop, connection):
    try:
        connection.send(("ready", None))
        seq = 0
        while not stop.is_set():
            seq += 1
            receipt = clock_ns()
            connection.send(("sample", (seq, receipt-delay, receipt, np.eye(4),
                                        True, "", seq, .1)))
            stop.wait(.01)
    finally:
        connection.close()


def failed_device(settings, tool_index, delay, stop, connection):
    connection.send(("error", "test serial failure"))
    connection.close()


def blocked_device(settings, tool_index, delay, stop, connection):
    connection.send(("ready", None))
    time.sleep(60)


class IsolatedTrackerTests(unittest.TestCase):
    def test_process_preserves_receipt_time_latency_and_immutable_pose(self):
        with patch("LiveData.receivers._ndi_process_entry", fake_device):
            worker = NDIReceiver(__file__, "FAKE", latency_ms=12)
            with worker:
                self.assertIsNotNone(worker._process)
                deadline = time.monotonic() + 2
                while worker.history.latest() is None and time.monotonic() < deadline:
                    time.sleep(.01)
                sample = worker.history.latest()
                self.assertIsNotNone(sample)
                self.assertEqual(sample.received_ns - sample.monotonic_ns, 12_000_000)
                self.assertLess(abs(clock_ns() - sample.received_ns), 500_000_000)
                with self.assertRaises(ValueError):
                    sample.transform[0, 0] = 2
            self.assertIsNone(worker._process)
            self.assertFalse(worker._thread.is_alive())

    def test_child_failure_reaches_startup_and_cleans_up(self):
        with patch("LiveData.receivers._ndi_process_entry", failed_device):
            worker = NDIReceiver(__file__, "FAKE")
            with self.assertRaisesRegex(RuntimeError, "test serial failure"):
                worker.start()
            self.assertIsNone(worker._process)
            self.assertFalse(worker._thread.is_alive())

    def test_blocked_driver_is_terminated_and_shutdown_error_is_visible(self):
        with patch("LiveData.receivers._ndi_process_entry", blocked_device):
            worker = NDIReceiver(__file__, "FAKE").start()
            with self.assertRaisesRegex(RuntimeError, "device process terminated"):
                worker.stop()
            self.assertIsNone(worker._process)
            self.assertFalse(worker._thread.is_alive())


class CroppedOverlayTests(unittest.TestCase):
    def test_matches_full_frame_warp_for_visible_and_clipped_slices(self):
        rng = np.random.default_rng(42)
        background = rng.integers(0, 256, (180, 240, 3), np.uint8)
        scan = rng.integers(0, 256, (50, 60, 3), np.uint8)
        source = np.array([[0, 0], [59, 0], [59, 49], [0, 49]], np.float32)
        for corners in ([(40, 30), (170, 45), (180, 140), (60, 130)],
                        [(-30, 20), (110, -20), (160, 140), (-10, 130)],
                        [(300, 20), (400, 20), (400, 80), (300, 80)]):
            pixels = np.array(corners, np.float32)
            homography = cv2.getPerspectiveTransform(source, pixels)
            warped = cv2.warpPerspective(scan, homography, (240, 180))
            mask = cv2.warpPerspective(np.full((50, 60), 255, np.uint8),
                                       homography, (240, 180)).astype(np.float32) / 255
            alpha = (mask * .65)[:, :, None]
            reference = np.clip(warped.astype(np.float32) * alpha
                                + background.astype(np.float32) * (1-alpha), 0, 255).astype(np.uint8)
            result = overlay_slice(background, scan, pixels, opacity=.65)
            # OpenCV rounds where the former numpy blend truncated to uint8.
            self.assertLessEqual(np.max(np.abs(result.astype(int)-reference.astype(int))), 1)
            np.testing.assert_array_equal(result[mask == 0], background[mask == 0])


if __name__ == "__main__":
    unittest.main()
