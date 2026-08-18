"""
Script for collecting NDI position data, ECM video, and US video in one.
Outputs 5 files: NDI csv, ECM video, ECM timestamp, US video, and US timestamp
"""

from toolbox.imaging import VideoCapture
from toolbox.ecm_recorder import list_cameras, record_video


