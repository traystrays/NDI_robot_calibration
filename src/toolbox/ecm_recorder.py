"""Compatibility layer for the former standalone ECM camera recorder."""

from .imaging import VideoRecorder, list_cameras


def record_video(device_index, filename, resolution=(1280, 720), fps=30):
    """Record until interrupted; retained for callers of the old API."""
    timestamp_path = str(filename).replace(".mp4", "_timestamps.txt")
    recorder = VideoRecorder(
        device_index, filename, timestamp_path, resolution=resolution, fps=fps
    )
    recorder.start()
    try:
        recorder._thread.join()  # Compatibility function is intentionally blocking.
    except KeyboardInterrupt:
        recorder.stop()


__all__ = ["VideoRecorder", "list_cameras", "record_video"]
