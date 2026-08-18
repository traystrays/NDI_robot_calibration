from toolbox.imaging import VideoCapture


import cv2
from pathlib import Path
import time

output_folder = Path(r"data/20260815_test")

camera = VideoCapture(output_folder)

index = camera.find_camera()


camera.record_video(index)