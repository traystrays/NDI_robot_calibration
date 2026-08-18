"""Compatibility exports for camera tools.

Camera implementation lives in :mod:`toolbox.imaging`; keeping this module
avoids two independent recording loops while preserving older imports.
"""

from .imaging import VideoCapture, VideoRecorder, list_cameras, read_camera_frame

__all__ = ["VideoCapture", "VideoRecorder", "list_cameras", "read_camera_frame"]
