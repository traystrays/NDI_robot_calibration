"""
Opening and reading image live feed from cameras, saving timestamp and video frames.
TODO: live feed open, time synconize frames
fake 
"""

import cv2
from pathlib import Path

class ImageCapture:

    def __init__(self, output_vid: Path, output_ts: Path):

        self.output_vid = output_vid
        self.output_ts = output_ts

    def find_camera(self):

            # let max cameras index be 10
        max = 10

        for i in range(max):

            cap = cv2.VideoCapture(i)
              
            if not cap.isOpened():
                continue

            print

            while True:
                cv2.

